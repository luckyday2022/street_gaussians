import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from kexuegu_helpers import load_calibration
from pcd_utils import fetch_ply, store_ply


def project_points_to_image(
    points_ego: np.ndarray,
    intrinsic: np.ndarray,
    cam_to_ego: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    idx = np.where(valid)[0]
    if idx.shape[0] == 0:
        return idx, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    u = np.clip(np.round(uv[idx, 0]), 0, width - 1).astype(np.int32)
    v = np.clip(np.round(uv[idx, 1]), 0, height - 1).astype(np.int32)
    return idx, u, v


def list_frame_ids(input_dir: str) -> List[int]:
    frame_ids = []
    for name in os.listdir(input_dir):
        if not name.endswith(".ply"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            frame_ids.append(int(stem))
    return sorted(set(frame_ids))


def filter_frames(frame_ids: List[int], start_frame_id: Optional[int], end_frame_id: Optional[int]) -> List[int]:
    if len(frame_ids) == 0:
        return []

    start = frame_ids[0] if start_frame_id is None else int(start_frame_id)
    end = frame_ids[-1] if end_frame_id is None else int(end_frame_id)

    if end < start:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")

    return [x for x in frame_ids if start <= x <= end]


def is_road_pixel(mask: np.ndarray, args) -> np.ndarray:
    if args.mask_mode == "binary":
        return mask >= args.mask_threshold

    # label mode
    return np.isin(mask, np.asarray(args.road_label, dtype=np.int32))


def read_mask(mask_path: str, args) -> Optional[np.ndarray]:
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None

    if mask.ndim == 3:
        # For color-coded masks, use the first channel by convention.
        # For class-index masks, save as single-channel PNG to avoid ambiguity.
        mask = mask[..., 0]

    return mask.astype(np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Split per-frame LiDAR background PLY into ground/non-ground using image masks.")

    parser.add_argument("--scene_dir", type=str, required=True, help="Processed scene dir, e.g. .../training_set_processed/000")
    parser.add_argument("--input_dir", type=str, default=None, help="Input per-frame PLY dir, default: <scene_dir>/lidar/background")
    parser.add_argument("--mask_dir", type=str, default=None, help="Mask dir, default: <scene_dir>/road_mask")
    parser.add_argument("--ground_dir", type=str, default=None, help="Output ground ply dir, default: <scene_dir>/lidar/ground")
    parser.add_argument("--non_ground_dir", type=str, default=None, help="Output non-ground ply dir, default: <scene_dir>/lidar/non_ground")

    parser.add_argument("--cams", type=int, nargs="+", default=None, help="Camera IDs to use, default: all calibrated cams")
    parser.add_argument("--start_frame_id", type=int, default=None)
    parser.add_argument("--end_frame_id", type=int, default=None)

    parser.add_argument("--mask_mode", type=str, choices=["binary", "label"], default="binary")
    parser.add_argument("--mask_threshold", type=int, default=128, help="Binary mode: pixel >= threshold is road")
    parser.add_argument("--road_label", type=int, nargs="+", default=[1], help="Label mode: road class id(s)")

    parser.add_argument("--min_votes", type=int, default=2, help="Minimum valid camera votes to decide ground")
    parser.add_argument("--positive_ratio", type=float, default=0.5, help="Road votes / valid votes threshold")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_stats", action="store_true", help="Save per-frame vote stats json")

    return parser.parse_args()


def main():
    args = parse_args()

    scene_dir = args.scene_dir
    input_dir = args.input_dir or os.path.join(scene_dir, "lidar", "background")
    mask_dir = args.mask_dir or os.path.join(scene_dir, "road_mask")
    ground_dir = args.ground_dir or os.path.join(scene_dir, "lidar", "ground")
    non_ground_dir = args.non_ground_dir or os.path.join(scene_dir, "lidar", "non_ground")

    if not os.path.exists(input_dir):
        raise RuntimeError(f"Input PLY dir not found: {input_dir}")
    if not os.path.exists(mask_dir):
        raise RuntimeError(f"Mask dir not found: {mask_dir}")

    os.makedirs(ground_dir, exist_ok=True)
    os.makedirs(non_ground_dir, exist_ok=True)

    intrinsics, extrinsics = load_calibration(scene_dir)
    available_cams = sorted(set(intrinsics.keys()) & set(extrinsics.keys()))
    if len(available_cams) == 0:
        raise RuntimeError(f"No valid intrinsics/extrinsics found in {scene_dir}")

    if args.cams is None:
        cams = available_cams
    else:
        cams = sorted(args.cams)
        missing = [c for c in cams if c not in available_cams]
        if len(missing) > 0:
            raise RuntimeError(f"Requested cams not available in calibration: {missing}; available={available_cams}")

    frame_ids = list_frame_ids(input_dir)
    frame_ids = filter_frames(frame_ids, args.start_frame_id, args.end_frame_id)
    if len(frame_ids) == 0:
        raise RuntimeError(f"No frame ply files found in {input_dir} for requested range")

    print(f"Input dir: {input_dir}")
    print(f"Mask dir: {mask_dir}")
    print(f"Cameras: {cams}")
    print(f"Frames: {frame_ids[0]} -> {frame_ids[-1]} (total={len(frame_ids)})")

    stats: Dict[str, Dict] = {}

    for frame_id in tqdm(frame_ids, desc="Split ground"):
        frame_key = f"{frame_id:06d}"
        input_ply = os.path.join(input_dir, f"{frame_key}.ply")
        out_ground = os.path.join(ground_dir, f"{frame_key}.ply")
        out_non_ground = os.path.join(non_ground_dir, f"{frame_key}.ply")

        if args.skip_existing and os.path.exists(out_ground) and os.path.exists(out_non_ground):
            continue

        pcd = fetch_ply(input_ply)
        valid = pcd.mask.astype(np.bool_)
        points = pcd.points[valid].astype(np.float32)
        colors = pcd.colors[valid].astype(np.float32)

        if points.shape[0] == 0:
            stats[frame_key] = {
                "points_total": int(pcd.points.shape[0]),
                "points_valid": 0,
                "points_ground": 0,
                "points_non_ground": 0,
                "cams_with_mask": 0,
            }
            continue

        vote_total = np.zeros((points.shape[0],), dtype=np.int16)
        vote_road = np.zeros((points.shape[0],), dtype=np.int16)

        cams_with_mask = 0
        for cam_id in cams:
            mask_path = os.path.join(mask_dir, f"{frame_key}_{cam_id}.png")
            if not os.path.exists(mask_path):
                continue

            mask = read_mask(mask_path, args)
            if mask is None:
                continue

            h, w = mask.shape[:2]
            idx, u, v = project_points_to_image(
                points_ego=points,
                intrinsic=intrinsics[cam_id],
                cam_to_ego=extrinsics[cam_id],
                height=h,
                width=w,
            )
            if idx.shape[0] == 0:
                continue

            cams_with_mask += 1
            road = is_road_pixel(mask[v, u], args)
            vote_total[idx] += 1
            vote_road[idx] += road.astype(np.int16)

        enough = vote_total >= max(1, int(args.min_votes))
        ratio = np.zeros_like(vote_road, dtype=np.float32)
        ratio[enough] = vote_road[enough].astype(np.float32) / np.maximum(vote_total[enough], 1)

        ground_mask = np.logical_and(enough, ratio >= float(args.positive_ratio))
        non_ground_mask = np.logical_not(ground_mask)

        if np.any(ground_mask):
            ones = np.ones((int(np.sum(ground_mask)), 1), dtype=np.bool_)
            store_ply(out_ground, points[ground_mask], colors[ground_mask], ones)
        elif os.path.exists(out_ground):
            os.remove(out_ground)

        if np.any(non_ground_mask):
            ones = np.ones((int(np.sum(non_ground_mask)), 1), dtype=np.bool_)
            store_ply(out_non_ground, points[non_ground_mask], colors[non_ground_mask], ones)
        elif os.path.exists(out_non_ground):
            os.remove(out_non_ground)

        stats[frame_key] = {
            "points_total": int(pcd.points.shape[0]),
            "points_valid": int(points.shape[0]),
            "points_ground": int(np.sum(ground_mask)),
            "points_non_ground": int(np.sum(non_ground_mask)),
            "cams_with_mask": int(cams_with_mask),
            "mean_votes": float(vote_total.mean()) if vote_total.size > 0 else 0.0,
        }

    if args.save_stats:
        stats_path = os.path.join(scene_dir, "lidar", "ground_split_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved stats: {stats_path}")

    print("Done.")


if __name__ == "__main__":
    main()
