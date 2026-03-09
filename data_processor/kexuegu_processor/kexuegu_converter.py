import argparse
import json
import os
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from kexuegu_helpers import (
    build_undistort_maps,
    discover_camera_mappings,
    filter_frames_by_range,
    get_pose_for_timestamp,
    load_camera_calibrations,
    load_traj_lidar,
    read_matched_frames,
    save_track_placeholders,
    timestamp_from_filename,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_pose(
    root_dir: str,
    scene_dir: str,
    frames: List[Dict],
    camera_mappings: Dict[int, Dict[str, str]],
    skip_existing: bool,
) -> None:
    cam_ids = sorted(camera_mappings.keys())
    pose_dir = os.path.join(scene_dir, "ego_pose")

    if skip_existing and len(frames) > 0:
        last_frame = int(frames[-1]["id"])
        last_pose = os.path.join(pose_dir, f"{last_frame:06d}.txt")
        if os.path.exists(last_pose):
            print("Pose exists, skipping pose processing.")
            return

    _ensure_dir(pose_dir)

    pose_map, traj_timestamps, traj_poses = load_traj_lidar(root_dir)

    timestamps = {"FRAME": {}}
    for cam_id in cam_ids:
        timestamps[camera_mappings[cam_id]["name"]] = {}

    for frame_data in tqdm(frames, desc="Processing poses"):
        frame_id = int(frame_data["id"])
        frame_key = f"{frame_id:06d}"

        lidar_name = frame_data["ruby"]
        lidar_ts_str = os.path.splitext(lidar_name)[0]
        pose = get_pose_for_timestamp(lidar_ts_str, pose_map, traj_timestamps, traj_poses)

        np.savetxt(os.path.join(pose_dir, f"{frame_key}.txt"), pose)
        for cam_id in cam_ids:
            np.savetxt(os.path.join(pose_dir, f"{frame_key}_{cam_id}.txt"), pose)

        timestamps["FRAME"][frame_key] = float(lidar_ts_str)
        for cam_id in cam_ids:
            mapping = camera_mappings[cam_id]
            if mapping["image_folder"] not in frame_data:
                raise RuntimeError(
                    f"Missing image key '{mapping['image_folder']}' in lidar_camera_matched frame id={frame_id}"
                )
            image_name = frame_data[mapping["image_folder"]]
            timestamps[mapping["name"]][frame_key] = timestamp_from_filename(image_name)

    with open(os.path.join(scene_dir, "timestamps.json"), "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=1)


def process_calib(
    root_dir: str,
    scene_dir: str,
    camera_mappings: Dict[int, Dict[str, str]],
    undistort_balance: float,
    skip_existing: bool,
):
    intrinsics_dir = os.path.join(scene_dir, "intrinsics")
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")

    cam_ids = sorted(camera_mappings.keys())
    if skip_existing and len(cam_ids) > 0:
        cam0 = cam_ids[0]
        if os.path.exists(os.path.join(intrinsics_dir, f"{cam0}.txt")) and os.path.exists(
            os.path.join(extrinsics_dir, f"{cam0}.txt")
        ):
            print("Calibration exists, skipping calibration processing.")
            return load_camera_calibrations(root_dir, camera_mappings, balance=undistort_balance)

    _ensure_dir(intrinsics_dir)
    _ensure_dir(extrinsics_dir)

    calibrations = load_camera_calibrations(root_dir, camera_mappings, balance=undistort_balance)

    calib_summary = {}
    for cam_id in cam_ids:
        calib = calibrations[cam_id]
        fx, fy = float(calib["new_K"][0, 0]), float(calib["new_K"][1, 1])
        cx, cy = float(calib["new_K"][0, 2]), float(calib["new_K"][1, 2])

        np.savetxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"), np.array([fx, fy, cx, cy], dtype=np.float64))
        np.savetxt(os.path.join(extrinsics_dir, f"{cam_id}.txt"), calib["cam_to_ego"])

        calib_summary[cam_id] = {
            "raw_K": calib["K"].tolist(),
            "distortion": calib["D"].tolist(),
            "new_K": calib["new_K"].tolist(),
            "cam_to_ego": calib["cam_to_ego"].tolist(),
            "resolution": calib["resolution"].tolist(),
            "source_image_folder": camera_mappings[cam_id]["image_folder"],
            "source_config_folder": camera_mappings[cam_id]["config_folder"],
        }

    with open(os.path.join(scene_dir, "camera_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib_summary, f, indent=1)

    return calibrations


def process_images(
    root_dir: str,
    scene_dir: str,
    frames: List[Dict],
    camera_mappings: Dict[int, Dict[str, str]],
    calibrations,
    skip_existing: bool,
) -> None:
    image_dir = os.path.join(scene_dir, "images")
    cam_ids = sorted(camera_mappings.keys())

    expected = len(frames) * len(cam_ids)
    if skip_existing and os.path.exists(image_dir):
        num_png = len([x for x in os.listdir(image_dir) if x.endswith(".png")])
        if num_png >= expected:
            print("Images exist, skipping image processing.")
            return

    _ensure_dir(image_dir)

    undistort_maps = build_undistort_maps(calibrations)
    image_timestamps = {}

    for frame_data in tqdm(frames, desc="Processing images"):
        frame_id = int(frame_data["id"])
        frame_key = f"{frame_id:06d}"

        for cam_id in cam_ids:
            mapping = camera_mappings[cam_id]
            if mapping["image_folder"] not in frame_data:
                raise RuntimeError(
                    f"Missing image key '{mapping['image_folder']}' in lidar_camera_matched frame id={frame_id}"
                )

            image_name = frame_data[mapping["image_folder"]]
            src_path = os.path.join(root_dir, "matched_images", mapping["image_folder"], image_name)
            dst_path = os.path.join(image_dir, f"{frame_key}_{cam_id}.png")

            image = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {src_path}")

            map1, map2 = undistort_maps[cam_id]
            undistorted = cv2.remap(
                image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            cv2.imwrite(dst_path, undistorted)

            image_timestamps[f"{frame_key}_{cam_id}"] = timestamp_from_filename(image_name)

    with open(os.path.join(image_dir, "timestamps.json"), "w", encoding="utf-8") as f:
        json.dump(image_timestamps, f, indent=1)


def process_track(scene_dir: str, frames: List[Dict], cam_ids: List[int], skip_existing: bool) -> None:
    track_dir = os.path.join(scene_dir, "track")
    if skip_existing and os.path.exists(os.path.join(track_dir, "track_info.pkl")):
        print("Track files exist, skipping track processing.")
        return

    frame_ids = [int(frame["id"]) for frame in frames]
    save_track_placeholders(track_dir=track_dir, frame_ids=frame_ids, cam_ids=cam_ids)


def process_dynamic_masks(scene_dir: str, frames: List[Dict], cam_ids: List[int], calibrations, skip_existing: bool) -> None:
    dynamic_dir = os.path.join(scene_dir, "dynamic_mask")

    expected = len(frames) * len(cam_ids)
    if skip_existing and os.path.exists(dynamic_dir):
        num_png = len([x for x in os.listdir(dynamic_dir) if x.endswith(".png")])
        if num_png >= expected:
            print("Dynamic masks exist, skipping dynamic mask processing.")
            return

    _ensure_dir(dynamic_dir)

    zero_masks = {}
    for cam_id in cam_ids:
        width, height = calibrations[cam_id]["resolution"].tolist()
        zero_masks[cam_id] = np.zeros((int(height), int(width)), dtype=np.uint8)

    for frame_data in tqdm(frames, desc="Processing dynamic masks"):
        frame_id = int(frame_data["id"])
        frame_key = f"{frame_id:06d}"
        for cam_id in cam_ids:
            mask_path = os.path.join(dynamic_dir, f"{frame_key}_{cam_id}.png")
            cv2.imwrite(mask_path, zero_masks[cam_id])


def parse_args():
    parser = argparse.ArgumentParser(description="Convert kexuegu data to Waymo-like processed layout.")
    parser.add_argument(
        "--process_list",
        type=str,
        nargs="+",
        default=["pose", "calib", "image", "track", "dynamic"],
        help="Subset of processing steps.",
    )
    parser.add_argument("--root_dir", type=str, default="data/20260113")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/20260113/waymo_format/training_set_processed",
        help="Directory that stores sequence folders like 000/001.",
    )
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--cams", type=int, nargs="+", default=None, help="Camera ids to process, default: all discovered cams")
    parser.add_argument("--start_frame_id", type=int, default=None)
    parser.add_argument("--end_frame_id", type=int, default=None)
    parser.add_argument("--undistort_balance", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    process_list = set(args.process_list)
    camera_mappings = discover_camera_mappings(args.root_dir, cams=args.cams)
    cam_ids = sorted(camera_mappings.keys())

    print(f"Selected cams: {cam_ids}")
    frames = read_matched_frames(args.root_dir)
    frames = filter_frames_by_range(frames, args.start_frame_id, args.end_frame_id)
    if len(frames) == 0:
        raise RuntimeError("No frames selected after frame range filtering.")
    print(f"Selected frame range: {int(frames[0]['id'])} -> {int(frames[-1]['id'])}, total={len(frames)}")

    scene_dir = os.path.join(args.save_dir, f"{args.scene_id:03d}")
    _ensure_dir(scene_dir)

    calibrations = None

    if "pose" in process_list:
        process_pose(args.root_dir, scene_dir, frames, camera_mappings, args.skip_existing)
    else:
        print("Skipping pose processing.")

    if "calib" in process_list:
        calibrations = process_calib(args.root_dir, scene_dir, camera_mappings, args.undistort_balance, args.skip_existing)
    else:
        print("Skipping calibration processing.")

    if "image" in process_list:
        if calibrations is None:
            calibrations = load_camera_calibrations(args.root_dir, camera_mappings, balance=args.undistort_balance)
        process_images(args.root_dir, scene_dir, frames, camera_mappings, calibrations, args.skip_existing)
    else:
        print("Skipping image processing.")

    if "track" in process_list:
        process_track(scene_dir, frames, cam_ids, args.skip_existing)
    else:
        print("Skipping track processing.")

    if "dynamic" in process_list:
        if calibrations is None:
            calibrations = load_camera_calibrations(args.root_dir, camera_mappings, balance=args.undistort_balance)
        process_dynamic_masks(scene_dir, frames, cam_ids, calibrations, args.skip_existing)
    else:
        print("Skipping dynamic mask processing.")


if __name__ == "__main__":
    main()
