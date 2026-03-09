import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from kexuegu_helpers import (
    discover_camera_mappings,
    filter_frames_by_range,
    load_calibration,
    read_matched_frames,
)
from pcd_utils import read_xyz_from_ply, store_ply


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def project_points_to_image(
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    cam_to_ego: np.ndarray,
    height: int,
    width: int,
):
    points_h = np.concatenate([points_ego, np.ones((points_ego.shape[0], 1), dtype=np.float32)], axis=1)
    ego_to_cam = np.linalg.inv(cam_to_ego)

    points_cam = points_h @ ego_to_cam.T
    points_cam = points_cam[:, :3]

    depth = points_cam[:, 2]
    valid = depth > 1e-2

    uvw = points_cam @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)

    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < width
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < height

    indices = np.where(valid)[0]
    if indices.shape[0] == 0:
        return indices, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    u = np.clip(np.round(uv[indices, 0]), 0, width - 1).astype(np.int32)
    v = np.clip(np.round(uv[indices, 1]), 0, height - 1).astype(np.int32)
    d = depth[indices].astype(np.float32)
    return indices, u, v, d


def save_sparse_depth(depth_path: str, u: np.ndarray, v: np.ndarray, d: np.ndarray, height: int, width: int):
    depth_flat = np.full((height * width,), np.inf, dtype=np.float32)
    pixel_idx = v.astype(np.int64) * int(width) + u.astype(np.int64)
    np.minimum.at(depth_flat, pixel_idx, d)

    mask_flat = np.isfinite(depth_flat)
    values = depth_flat[mask_flat].astype(np.float32)
    mask = mask_flat.reshape(height, width)
    np.savez_compressed(depth_path, mask=mask, value=values)


def process_lidar(
    root_dir: str,
    scene_dir: str,
    frames: List[Dict],
    cam_ids: List[int],
    skip_existing: bool,
):
    lidar_dir = os.path.join(scene_dir, "lidar")
    background_dir = os.path.join(lidar_dir, "background")
    actor_dir = os.path.join(lidar_dir, "actor")
    depth_dir = os.path.join(lidar_dir, "depth")

    _ensure_dir(lidar_dir)
    _ensure_dir(background_dir)
    _ensure_dir(actor_dir)
    _ensure_dir(depth_dir)

    cam_ids = sorted(cam_ids)
    if skip_existing and len(frames) > 0 and len(cam_ids) > 0:
        last_frame = int(frames[-1]["id"])
        last_cam = cam_ids[-1]
        last_ply = os.path.join(background_dir, f"{last_frame:06d}.ply")
        last_depth = os.path.join(depth_dir, f"{last_frame:06d}_{last_cam}.npz")
        if os.path.exists(last_ply) and os.path.exists(last_depth):
            print("LiDAR outputs exist, skipping LiDAR processing.")
            return

    intrinsics, extrinsics = load_calibration(scene_dir)
    for cam_id in cam_ids:
        if cam_id not in intrinsics or cam_id not in extrinsics:
            raise RuntimeError(f"Missing calibration for cam {cam_id} in {scene_dir}")

    print("Generating LiDAR background point clouds and sparse depths...")
    for frame_data in tqdm(frames, desc="LiDAR frames"):
        frame_id = int(frame_data["id"])
        frame_key = f"{frame_id:06d}"

        lidar_filename = frame_data["ruby"]
        lidar_path = os.path.join(root_dir, "deskewed_scans", lidar_filename)
        points_ego = read_xyz_from_ply(lidar_path)

        num_points = points_ego.shape[0]
        colors = np.zeros((num_points, 3), dtype=np.float32)
        visible_any = np.zeros((num_points,), dtype=np.bool_)

        for cam_id in cam_ids:
            image_path = os.path.join(scene_dir, "images", f"{frame_key}_{cam_id}.png")
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            height, width = image_rgb.shape[:2]

            idx, u, v, d = project_points_to_image(
                points_ego=points_ego,
                intrinsic=intrinsics[cam_id],
                cam_to_ego=extrinsics[cam_id],
                height=height,
                width=width,
            )

            if idx.shape[0] == 0:
                save_sparse_depth(
                    os.path.join(depth_dir, f"{frame_key}_{cam_id}.npz"),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                    height,
                    width,
                )
                continue

            rgb_vals = image_rgb[v, u]
            new_points = idx[~visible_any[idx]]
            if new_points.shape[0] > 0:
                keep = ~visible_any[idx]
                colors[new_points] = rgb_vals[keep]

            visible_any[idx] = True
            save_sparse_depth(
                os.path.join(depth_dir, f"{frame_key}_{cam_id}.npz"),
                u,
                v,
                d,
                height,
                width,
            )

        store_ply(
            os.path.join(background_dir, f"{frame_key}.ply"),
            points_ego.astype(np.float32),
            colors.astype(np.float32),
            visible_any[:, None],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Waymo-style lidar background ply and sparse depth.")
    parser.add_argument("--root_dir", type=str, default="data/20260113")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/20260113/waymo_format/training_set_processed",
        help="Directory containing sequence folders like 000/001.",
    )
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--cams", type=int, nargs="+", default=None, help="Camera ids to process, default: all discovered cams")
    parser.add_argument("--start_frame_id", type=int, default=None)
    parser.add_argument("--end_frame_id", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    scene_dir = os.path.join(args.save_dir, f"{args.scene_id:03d}")
    if not os.path.exists(scene_dir):
        raise RuntimeError(f"Scene directory not found: {scene_dir}. Run kexuegu_converter.py first.")

    camera_mappings = discover_camera_mappings(args.root_dir, cams=args.cams)
    cam_ids = sorted(camera_mappings.keys())
    print(f"Selected cams: {cam_ids}")

    frames = read_matched_frames(args.root_dir)
    frames = filter_frames_by_range(frames, args.start_frame_id, args.end_frame_id)
    if len(frames) == 0:
        raise RuntimeError("No frames selected after frame range filtering.")
    print(f"Selected frame range: {int(frames[0]['id'])} -> {int(frames[-1]['id'])}, total={len(frames)}")

    process_lidar(args.root_dir, scene_dir, frames, cam_ids, args.skip_existing)


if __name__ == "__main__":
    main()
