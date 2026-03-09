import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def image_filename_to_cam(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return int(stem.split("_")[-1])


def image_filename_to_frame(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return int(stem.split("_")[0])


def timestamp_from_filename(filename: str) -> float:
    return float(os.path.splitext(os.path.basename(filename))[0])


def parse_fakra_index(name: str) -> Optional[int]:
    if "fakra" not in name:
        return None
    suffix = name.split("fakra", 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if len(digits) == 0:
        return None
    return int("".join(digits))


def discover_camera_mappings(root_dir: str, cams: Optional[List[int]] = None) -> Dict[int, Dict[str, str]]:
    matched_root = os.path.join(root_dir, "matched_images")
    config_root = os.path.join(root_dir, "config")

    if not os.path.exists(matched_root):
        raise RuntimeError(f"matched_images dir does not exist: {matched_root}")
    if not os.path.exists(config_root):
        raise RuntimeError(f"config dir does not exist: {config_root}")

    image_folders = {}
    for name in os.listdir(matched_root):
        path = os.path.join(matched_root, name)
        if not os.path.isdir(path):
            continue
        idx = parse_fakra_index(name)
        if idx is None:
            continue
        image_folders[idx] = name

    config_folders = {}
    for name in os.listdir(config_root):
        path = os.path.join(config_root, name)
        if not os.path.isdir(path):
            continue
        idx = parse_fakra_index(name)
        if idx is None:
            continue
        # prefer directory containing calibration yaml files
        intr_path = os.path.join(path, "intrinsics.yaml")
        ext_path = os.path.join(path, "extrinsics.yaml")
        if os.path.exists(intr_path) and os.path.exists(ext_path):
            config_folders[idx] = name

    available_ids = sorted(set(image_folders.keys()) & set(config_folders.keys()))
    if len(available_ids) == 0:
        raise RuntimeError("No valid camera folders found (matched_images/config mismatch).")

    if cams is not None:
        cam_set = set(cams)
        missing = sorted([c for c in cam_set if c not in available_ids])
        if len(missing) > 0:
            raise RuntimeError(f"Requested cams not available: {missing}, available cams: {available_ids}")
        available_ids = [c for c in available_ids if c in cam_set]

    mappings = {}
    for cam_id in available_ids:
        mappings[cam_id] = {
            "name": f"CAM_{cam_id}",
            "image_folder": image_folders[cam_id],
            "config_folder": config_folders[cam_id],
        }

    return mappings


def quaternion_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw.astype(np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_intrinsics_from_yaml(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cam0 = data["cam0"]
    fx, fy, cx, cy = cam0["intrinsics"]
    dist = np.asarray(cam0["distortion_coeffs"], dtype=np.float64)
    width, height = int(cam0["resolution"][0]), int(cam0["resolution"][1])

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K, dist, (width, height)


def load_extrinsics_from_yaml(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    t = data["transform"]["translation"]
    q = data["transform"]["rotation"]
    quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)

    cam_to_lidar = np.eye(4, dtype=np.float64)
    cam_to_lidar[:3, :3] = quaternion_xyzw_to_matrix(quat)
    cam_to_lidar[:3, 3] = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
    return cam_to_lidar


def load_camera_calibrations(
    root_dir: str,
    camera_mappings: Dict[int, Dict[str, str]],
    balance: float = 0.0,
) -> Dict[int, Dict[str, np.ndarray]]:
    config_root = os.path.join(root_dir, "config")
    calibrations: Dict[int, Dict[str, np.ndarray]] = {}

    for cam_id, mapping in camera_mappings.items():
        intr_path = os.path.join(config_root, mapping["config_folder"], "intrinsics.yaml")
        ext_path = os.path.join(config_root, mapping["config_folder"], "extrinsics.yaml")

        K, D, resolution = load_intrinsics_from_yaml(intr_path)
        cam_to_ego = load_extrinsics_from_yaml(ext_path)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            resolution,
            np.eye(3, dtype=np.float64),
            balance=float(balance),
            new_size=resolution,
        )

        calibrations[cam_id] = {
            "K": K,
            "D": D,
            "new_K": new_K,
            "cam_to_ego": cam_to_ego,
            "resolution": np.array(resolution, dtype=np.int32),
        }

    return calibrations


def build_undistort_maps(calibrations: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    maps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cam_id, calib in calibrations.items():
        width, height = calib["resolution"].tolist()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            calib["K"],
            calib["D"],
            np.eye(3, dtype=np.float64),
            calib["new_K"],
            (int(width), int(height)),
            cv2.CV_16SC2,
        )
        maps[cam_id] = (map1, map2)
    return maps


def read_matched_frames(root_dir: str) -> List[Dict]:
    matched_path = os.path.join(root_dir, "lidar_camera_matched.json")
    with open(matched_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = sorted(data["frames"], key=lambda x: int(x["id"]))
    return frames


def filter_frames_by_range(frames: List[Dict], start_frame_id: Optional[int], end_frame_id: Optional[int]) -> List[Dict]:
    if len(frames) == 0:
        return frames

    start = start_frame_id if start_frame_id is not None else int(frames[0]["id"])
    end = end_frame_id if end_frame_id is not None else int(frames[-1]["id"])

    if end < start:
        raise RuntimeError(f"Invalid frame range: start_frame_id={start}, end_frame_id={end}")

    filtered = [f for f in frames if start <= int(f["id"]) <= end]
    return filtered


def load_traj_lidar(root_dir: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    traj_path = os.path.join(root_dir, "traj_lidar.txt")

    pose_map: Dict[str, np.ndarray] = {}
    timestamps: List[float] = []
    poses: List[np.ndarray] = []

    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            toks = line.split()
            if len(toks) != 8:
                continue

            ts_str = toks[0]
            tx, ty, tz = map(float, toks[1:4])
            qx, qy, qz, qw = map(float, toks[4:8])

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = quaternion_xyzw_to_matrix(np.array([qx, qy, qz, qw], dtype=np.float64))
            pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

            pose_map[ts_str] = pose
            timestamps.append(float(ts_str))
            poses.append(pose)

    ts_arr = np.array(timestamps, dtype=np.float64)
    pose_arr = np.stack(poses, axis=0) if len(poses) > 0 else np.empty((0, 4, 4), dtype=np.float64)
    return pose_map, ts_arr, pose_arr


def get_pose_for_timestamp(
    timestamp_str: str,
    pose_map: Dict[str, np.ndarray],
    traj_timestamps: np.ndarray,
    traj_poses: np.ndarray,
) -> np.ndarray:
    if timestamp_str in pose_map:
        return pose_map[timestamp_str].copy()

    if traj_timestamps.shape[0] == 0:
        raise RuntimeError("No trajectory pose found in traj_lidar.txt")

    timestamp = float(timestamp_str)
    nearest = int(np.argmin(np.abs(traj_timestamps - timestamp)))
    return traj_poses[nearest].copy()


def save_track_placeholders(track_dir: str, frame_ids: List[int], cam_ids: List[int]) -> None:
    os.makedirs(track_dir, exist_ok=True)

    track_info = {}
    track_camera_visible = {}
    trajectory = {}

    cam_ids = sorted(cam_ids)
    for frame_id in frame_ids:
        frame_key = f"{frame_id:06d}"
        track_info[frame_key] = {}
        track_camera_visible[frame_key] = {cam: [] for cam in cam_ids}

    with open(os.path.join(track_dir, "track_info.pkl"), "wb") as f:
        pickle.dump(track_info, f)

    with open(os.path.join(track_dir, "track_camera_visible.pkl"), "wb") as f:
        pickle.dump(track_camera_visible, f)

    with open(os.path.join(track_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(trajectory, f)

    with open(os.path.join(track_dir, "track_ids.json"), "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)


def load_track(scene_dir: str):
    track_dir = os.path.join(scene_dir, "track")
    with open(os.path.join(track_dir, "track_info.pkl"), "rb") as f:
        track_info = pickle.load(f)
    with open(os.path.join(track_dir, "track_camera_visible.pkl"), "rb") as f:
        track_camera_visible = pickle.load(f)
    with open(os.path.join(track_dir, "trajectory.pkl"), "rb") as f:
        trajectory = pickle.load(f)
    return track_info, track_camera_visible, trajectory


def load_ego_poses(scene_dir: str):
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    if not os.path.exists(ego_pose_dir):
        raise RuntimeError(f"ego_pose dir does not exist: {ego_pose_dir}")

    ego_frame_pose_map: Dict[int, np.ndarray] = {}
    ego_cam_pose_map: Dict[int, Dict[int, np.ndarray]] = {}

    for pose_name in sorted(os.listdir(ego_pose_dir)):
        if not pose_name.endswith(".txt"):
            continue
        pose_path = os.path.join(ego_pose_dir, pose_name)
        stem = os.path.splitext(pose_name)[0]
        pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float64)

        if "_" not in stem:
            frame_id = int(stem)
            ego_frame_pose_map[frame_id] = pose
        else:
            frame_str, cam_str = stem.split("_")
            frame_id = int(frame_str)
            cam_id = int(cam_str)
            ego_cam_pose_map.setdefault(cam_id, {})[frame_id] = pose

    frame_ids = sorted(ego_frame_pose_map.keys())
    if len(frame_ids) == 0:
        raise RuntimeError(f"No frame poses found in {ego_pose_dir}")

    frame_pose_stack = np.stack([ego_frame_pose_map[f] for f in frame_ids], axis=0)
    center_point = np.mean(frame_pose_stack[:, :3, 3], axis=0)

    for frame_id in frame_ids:
        ego_frame_pose_map[frame_id] = ego_frame_pose_map[frame_id].copy()
        ego_frame_pose_map[frame_id][:3, 3] -= center_point

    for cam_id, cam_pose_dict in ego_cam_pose_map.items():
        for frame_id, pose in cam_pose_dict.items():
            pose_ = pose.copy()
            pose_[:3, 3] -= center_point
            cam_pose_dict[frame_id] = pose_

    return ego_frame_pose_map, ego_cam_pose_map, frame_ids


def load_calibration(scene_dir: str):
    intrinsics_dir = os.path.join(scene_dir, "intrinsics")
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")

    if not os.path.exists(intrinsics_dir) or not os.path.exists(extrinsics_dir):
        raise RuntimeError(f"Missing intrinsics/extrinsics under {scene_dir}")

    intr_files = sorted([x for x in os.listdir(intrinsics_dir) if x.endswith(".txt")])
    ext_files = sorted([x for x in os.listdir(extrinsics_dir) if x.endswith(".txt")])

    intr_cam_ids = sorted([int(os.path.splitext(x)[0]) for x in intr_files])
    ext_cam_ids = sorted([int(os.path.splitext(x)[0]) for x in ext_files])

    cam_ids = sorted(set(intr_cam_ids) & set(ext_cam_ids))
    if len(cam_ids) == 0:
        raise RuntimeError(f"No calibration files found in {scene_dir}")

    intrinsics = {}
    extrinsics = {}
    for cam in cam_ids:
        intrinsic_raw = np.loadtxt(os.path.join(intrinsics_dir, f"{cam}.txt")).reshape(-1)
        fx, fy, cx, cy = intrinsic_raw[:4]
        intrinsic = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        intrinsics[cam] = intrinsic

        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{cam}.txt")).reshape(4, 4).astype(np.float64)
        extrinsics[cam] = cam_to_ego

    return intrinsics, extrinsics


def get_lane_shift_direction(ego_frame_pose_map: Dict[int, np.ndarray], ordered_frame_ids: List[int], frame_id: int) -> np.ndarray:
    if len(ordered_frame_ids) < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if frame_id not in ego_frame_pose_map:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    idx = ordered_frame_ids.index(frame_id)
    if idx == 0:
        f0, f1 = ordered_frame_ids[0], ordered_frame_ids[1]
    else:
        f0, f1 = ordered_frame_ids[idx - 1], ordered_frame_ids[idx]

    delta = ego_frame_pose_map[f1][:3, 3] - ego_frame_pose_map[f0][:3, 3]
    delta = delta[:2]
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    delta = delta / norm
    return np.array([delta[1], -delta[0], 0.0], dtype=np.float64)


def load_scene_frame_ids(scene_dir: str) -> List[int]:
    image_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(image_dir):
        raise RuntimeError(f"images dir does not exist: {image_dir}")

    frame_ids = set()
    for name in os.listdir(image_dir):
        if not name.endswith(".png"):
            continue
        try:
            frame_ids.add(image_filename_to_frame(name))
        except Exception:
            continue

    if len(frame_ids) == 0:
        raise RuntimeError(f"No image frames found in {image_dir}")

    return sorted(frame_ids)


def load_scene_cam_ids(scene_dir: str) -> List[int]:
    intrinsics_dir = os.path.join(scene_dir, "intrinsics")
    if not os.path.exists(intrinsics_dir):
        raise RuntimeError(f"intrinsics dir does not exist: {intrinsics_dir}")

    cam_ids = []
    for name in os.listdir(intrinsics_dir):
        if not name.endswith(".txt"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            cam_ids.append(int(stem))

    cam_ids = sorted(set(cam_ids))
    if len(cam_ids) == 0:
        raise RuntimeError(f"No camera intrinsics found in {intrinsics_dir}")

    return cam_ids


def load_camera_image_shapes(scene_dir: str, cam_ids: Optional[List[int]] = None) -> Dict[int, Tuple[int, int]]:
    if cam_ids is None:
        cam_ids = load_scene_cam_ids(scene_dir)

    frame_ids = load_scene_frame_ids(scene_dir)
    shapes = {}

    for cam in cam_ids:
        found = False
        for frame_id in frame_ids:
            image_path = os.path.join(scene_dir, "images", f"{frame_id:06d}_{cam}.png")
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w = image.shape[:2]
            shapes[cam] = (h, w)
            found = True
            break
        if not found:
            raise RuntimeError(f"Failed to find readable image for cam={cam} in {scene_dir}/images")

    return shapes
