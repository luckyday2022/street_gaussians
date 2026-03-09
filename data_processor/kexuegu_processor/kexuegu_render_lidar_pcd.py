import argparse
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from kexuegu_helpers import (
    get_lane_shift_direction,
    load_calibration,
    load_ego_poses,
    load_scene_cam_ids,
)
from pcd_utils import fetch_ply


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_background_world(scene_dir: str, frame_id: int, ego_frame_pose: np.ndarray) -> np.ndarray:
    ply_path = os.path.join(scene_dir, "lidar", "background", f"{frame_id:06d}.ply")
    if not os.path.exists(ply_path):
        return np.empty((0, 6), dtype=np.float32)

    ply = fetch_ply(ply_path)
    mask = ply.mask.astype(np.bool_)
    if mask.sum() == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyz_vehicle = ply.points[mask]
    xyz_vehicle_h = np.concatenate([xyz_vehicle, np.ones_like(xyz_vehicle[:, :1])], axis=1)
    xyz_world = xyz_vehicle_h @ ego_frame_pose.T
    xyz_world = xyz_world[:, :3]
    rgb = ply.colors[mask]
    return np.concatenate([xyz_world, rgb], axis=1).astype(np.float32)


def _render_pointcloud_zbuffer(points_world_rgb: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, height: int, width: int):
    rgb_out = np.zeros((height * width, 3), dtype=np.float32)
    mask_out = np.zeros((height * width,), dtype=np.uint8)
    depth_out = np.zeros((height * width,), dtype=np.float32)

    if points_world_rgb.shape[0] == 0:
        return rgb_out.reshape(height, width, 3), mask_out.reshape(height, width), depth_out.reshape(height, width)

    xyz_world = points_world_rgb[:, :3]
    rgb = points_world_rgb[:, 3:]

    w2c = np.linalg.inv(c2w)
    xyz_cam = xyz_world @ w2c[:3, :3].T + w2c[:3, 3]
    depth = xyz_cam[:, 2]

    uvw = xyz_cam @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)

    valid = depth > 1e-2
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < width
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < height

    idx = np.where(valid)[0]
    if idx.shape[0] == 0:
        return rgb_out.reshape(height, width, 3), mask_out.reshape(height, width), depth_out.reshape(height, width)

    u = np.clip(np.round(uv[idx, 0]), 0, width - 1).astype(np.int32)
    v = np.clip(np.round(uv[idx, 1]), 0, height - 1).astype(np.int32)
    d = depth[idx].astype(np.float32)
    c = rgb[idx].astype(np.float32)

    pixel_idx = v.astype(np.int64) * int(width) + u.astype(np.int64)
    order = np.argsort(d)
    pixel_sorted = pixel_idx[order]
    _, first = np.unique(pixel_sorted, return_index=True)
    selected = order[first]

    selected_pixels = pixel_idx[selected]
    rgb_out[selected_pixels] = c[selected]
    mask_out[selected_pixels] = 255
    depth_out[selected_pixels] = d[selected]

    return rgb_out.reshape(height, width, 3), mask_out.reshape(height, width), depth_out.reshape(height, width)


def _check_existing(scene_dir: str, save_dir_name: str, cam: int, frame_ids: List[int]) -> bool:
    render_dir = os.path.join(scene_dir, "lidar", save_dir_name)
    for frame_id in frame_ids:
        rgb_path = os.path.join(render_dir, f"{frame_id:06d}_{cam}.png")
        mask_path = os.path.join(render_dir, f"{frame_id:06d}_{cam}_mask.png")
        if not (os.path.exists(rgb_path) and os.path.exists(mask_path)):
            return False
    return True


def _save_dir_name(base_save_dir: str, shift: float) -> str:
    if abs(shift) < 1e-8:
        return base_save_dir
    return f"{base_save_dir}_shift_{shift:.2f}"


def _resolve_cam_ids(scene_dir: str, requested_cams: Optional[List[int]], intrinsics: Dict[int, np.ndarray], extrinsics: Dict[int, np.ndarray], ego_cam_pose_map: Dict[int, Dict[int, np.ndarray]]) -> List[int]:
    available = sorted(set(load_scene_cam_ids(scene_dir)) & set(intrinsics.keys()) & set(extrinsics.keys()) & set(ego_cam_pose_map.keys()))
    if requested_cams is None:
        return available

    missing = sorted([c for c in requested_cams if c not in available])
    if len(missing) > 0:
        raise RuntimeError(f"Requested cams not available in scene: {missing}, available cams: {available}")
    return sorted(requested_cams)


def render_scene(
    scene_dir: str,
    cams: Optional[List[int]],
    delta_frames: int,
    shifts: List[float],
    base_save_dir: str,
    lane_shift_sign: float,
    skip_existing: bool,
    save_depth: bool,
    start_frame_id: Optional[int],
    end_frame_id: Optional[int],
):
    ego_frame_pose_map, ego_cam_pose_map, frame_ids_all = load_ego_poses(scene_dir)
    intrinsics, extrinsics = load_calibration(scene_dir)

    cam_ids = _resolve_cam_ids(scene_dir, cams, intrinsics, extrinsics, ego_cam_pose_map)
    if len(cam_ids) == 0:
        raise RuntimeError("No valid cameras found for rendering.")

    start = start_frame_id if start_frame_id is not None else frame_ids_all[0]
    end = end_frame_id if end_frame_id is not None else frame_ids_all[-1]
    if end < start:
        raise RuntimeError(f"Invalid frame range: start_frame_id={start}, end_frame_id={end}")

    frame_ids = [f for f in frame_ids_all if start <= f <= end]
    if len(frame_ids) == 0:
        raise RuntimeError("No frames selected for rendering after frame range filtering.")

    image_shapes = {}
    for cam in cam_ids:
        found = False
        for frame_id in frame_ids:
            img_path = os.path.join(scene_dir, "images", f"{frame_id:06d}_{cam}.png")
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image_shapes[cam] = image.shape[:2]
            found = True
            break
        if not found:
            raise RuntimeError(f"No readable image found for cam={cam} in selected frame range")

    print(f"Selected cams: {cam_ids}")
    print(f"Selected frame range: {frame_ids[0]} -> {frame_ids[-1]}, total={len(frame_ids)}")

    for shift in shifts:
        save_dir_name = _save_dir_name(base_save_dir, shift)
        render_dir = os.path.join(scene_dir, "lidar", save_dir_name)
        _ensure_dir(render_dir)

        active_cams = []
        for cam in cam_ids:
            if skip_existing and _check_existing(scene_dir, save_dir_name, cam, frame_ids):
                print(f"Skipping cam={cam}, shift={shift:.2f}, all outputs exist.")
                continue
            active_cams.append(cam)

        if len(active_cams) == 0:
            continue

        cache: Dict[int, np.ndarray] = {}
        for idx, frame_id in enumerate(tqdm(frame_ids, desc=f"Render shift {shift:.2f}")):
            left_idx = max(0, idx - delta_frames)
            right_idx = min(len(frame_ids) - 1, idx + delta_frames)
            needed_ids = frame_ids[left_idx : right_idx + 1]
            needed_set = set(needed_ids)

            drop_keys = [k for k in cache.keys() if k not in needed_set]
            for k in drop_keys:
                cache.pop(k)

            for f_id in needed_ids:
                if f_id not in cache:
                    cache[f_id] = _read_background_world(scene_dir, f_id, ego_frame_pose_map[f_id])

            ordered = [cache[f_id] for f_id in needed_ids if f_id in cache and cache[f_id].shape[0] > 0]
            points_world = np.concatenate(ordered, axis=0) if len(ordered) > 0 else np.empty((0, 6), dtype=np.float32)

            lane_dir = get_lane_shift_direction(ego_frame_pose_map, frame_ids, frame_id)
            for cam in active_cams:
                h, w = image_shapes[cam]
                if frame_id not in ego_cam_pose_map[cam]:
                    continue

                ego_pose = ego_cam_pose_map[cam][frame_id].copy()
                if abs(shift) > 1e-8:
                    ego_pose[:3, 3] += lane_shift_sign * lane_dir * shift

                c2w = ego_pose @ extrinsics[cam]
                rgb, mask, depth = _render_pointcloud_zbuffer(points_world, c2w, intrinsics[cam], h, w)

                rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

                rgb_path = os.path.join(render_dir, f"{frame_id:06d}_{cam}.png")
                mask_path = os.path.join(render_dir, f"{frame_id:06d}_{cam}_mask.png")
                cv2.imwrite(rgb_path, rgb_bgr)
                cv2.imwrite(mask_path, mask)

                if save_depth:
                    depth_path = os.path.join(render_dir, f"{frame_id:06d}_{cam}_depth.npy")
                    np.save(depth_path, depth)


def parse_args():
    parser = argparse.ArgumentParser(description="Render aggregated lidar conditions in Waymo-style layout.")
    parser.add_argument("--data_dir", type=str, default="data/20260113/waymo_format/training_set_processed")
    parser.add_argument("--scene_ids", type=int, nargs="+", default=None)
    parser.add_argument("--delta_frames", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=None, help="Camera ids to render, default: all cams in scene")
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0])
    parser.add_argument("--save_dir", type=str, default="color_render")
    parser.add_argument("--lane_shift_sign", type=float, default=1.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--start_frame_id", type=int, default=None)
    parser.add_argument("--end_frame_id", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.scene_ids is None:
        scene_ids = sorted([int(x) for x in os.listdir(args.data_dir) if x.isdigit()])
    else:
        scene_ids = args.scene_ids

    for scene_id in scene_ids:
        scene_dir = os.path.join(args.data_dir, f"{scene_id:03d}")
        if not os.path.exists(scene_dir):
            print(f"Scene {scene_id:03d} not found, skipping.")
            continue

        print(f"Rendering scene {scene_id:03d} ...")
        render_scene(
            scene_dir=scene_dir,
            cams=args.cams,
            delta_frames=args.delta_frames,
            shifts=args.shifts,
            base_save_dir=args.save_dir,
            lane_shift_sign=args.lane_shift_sign,
            skip_existing=args.skip_existing,
            save_depth=args.save_depth,
            start_frame_id=args.start_frame_id,
            end_frame_id=args.end_frame_id,
        )


if __name__ == "__main__":
    main()
