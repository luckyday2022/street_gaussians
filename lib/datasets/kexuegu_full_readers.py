from lib.utils.graphics_utils import focal2fov
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_Sphere_Norm, storePly
from lib.config import cfg

from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

import cv2
import glob
import json
import numpy as np
import os
import shutil
import sys

sys.path.append(os.getcwd())


def image_filename_to_cam(name: str) -> int:
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(stem.split("_")[-1])


def image_filename_to_frame(name: str) -> int:
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(stem.split("_")[0])


def _list_image_files(image_dir: str):
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files += glob.glob(os.path.join(image_dir, "*.jpg"))
    return sorted(image_files)


def _discover_frames_and_cams(image_files):
    frames = set()
    cams = set()
    file_map = {}
    for path in image_files:
        frame = image_filename_to_frame(path)
        cam = image_filename_to_cam(path)
        frames.add(frame)
        cams.add(cam)
        file_map[(frame, cam)] = path
    return sorted(frames), sorted(cams), file_map


def _select_frame_ids(all_frames, selected_frames=None):
    if len(all_frames) == 0:
        return []

    if selected_frames is None:
        start_frame = all_frames[0]
        end_frame = all_frames[-1]
    else:
        start_frame = int(selected_frames[0])
        end_frame = int(selected_frames[1])

    if end_frame < start_frame:
        raise RuntimeError(f"Invalid selected_frames: {selected_frames}")

    frame_ids = [f for f in all_frames if start_frame <= f <= end_frame]
    return frame_ids


def _load_intrinsics_extrinsics(datadir: str, cameras):
    intrinsics_dir = os.path.join(datadir, "intrinsics")
    extrinsics_dir = os.path.join(datadir, "extrinsics")

    intrinsics = {}
    extrinsics = {}

    for cam in cameras:
        ixt_path = os.path.join(intrinsics_dir, f"{cam}.txt")
        ext_path = os.path.join(extrinsics_dir, f"{cam}.txt")
        if not os.path.exists(ixt_path):
            raise RuntimeError(f"Missing intrinsic: {ixt_path}")
        if not os.path.exists(ext_path):
            raise RuntimeError(f"Missing extrinsic: {ext_path}")

        intrinsic_raw = np.loadtxt(ixt_path).reshape(-1)
        fx, fy, cx, cy = intrinsic_raw[:4]
        intrinsics[cam] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        extrinsics[cam] = np.loadtxt(ext_path).reshape(4, 4).astype(np.float64)

    return intrinsics, extrinsics


def _load_ego_poses(datadir: str):
    ego_pose_dir = os.path.join(datadir, "ego_pose")
    if not os.path.exists(ego_pose_dir):
        raise RuntimeError(f"Missing ego_pose dir: {ego_pose_dir}")

    frame_pose_map = {}
    cam_pose_map = {}

    for name in sorted(os.listdir(ego_pose_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(ego_pose_dir, name)
        pose = np.loadtxt(path).reshape(4, 4).astype(np.float64)
        stem = os.path.splitext(name)[0]
        if "_" not in stem:
            frame = int(stem)
            frame_pose_map[frame] = pose
        else:
            frame_str, cam_str = stem.split("_")
            frame = int(frame_str)
            cam = int(cam_str)
            cam_pose_map.setdefault(cam, {})[frame] = pose

    if len(frame_pose_map) == 0:
        raise RuntimeError(f"No frame poses in {ego_pose_dir}")

    return frame_pose_map, cam_pose_map


def _center_ego_poses(frame_pose_map, cam_pose_map, frame_ids):
    pose_stack = np.stack([frame_pose_map[f] for f in frame_ids], axis=0)
    center = np.mean(pose_stack[:, :3, 3], axis=0)

    frame_pose_centered = {}
    for frame in frame_pose_map.keys():
        pose = frame_pose_map[frame].copy()
        pose[:3, 3] -= center
        frame_pose_centered[frame] = pose

    cam_pose_centered = {}
    for cam, pose_dict in cam_pose_map.items():
        cam_pose_centered[cam] = {}
        for frame, pose in pose_dict.items():
            p = pose.copy()
            p[:3, 3] -= center
            cam_pose_centered[cam][frame] = p

    return frame_pose_centered, cam_pose_centered


def _read_timestamps(datadir: str):
    timestamp_path = os.path.join(datadir, "timestamps.json")
    if not os.path.exists(timestamp_path):
        return {}
    with open(timestamp_path, "r") as f:
        return json.load(f)


def _get_frame_timestamp(timestamps, frame: int):
    if "FRAME" in timestamps:
        key = f"{frame:06d}"
        if key in timestamps["FRAME"]:
            return float(timestamps["FRAME"][key])
    return float(frame)


def _get_cam_timestamp(timestamps, cam: int, frame: int):
    key = f"{frame:06d}"

    # preferred key from kexuegu converter
    cam_key = f"CAM_{cam}"
    if cam_key in timestamps and key in timestamps[cam_key]:
        return float(timestamps[cam_key][key])

    # compatible fallback from waymo style names
    waymo_name_map = {
        0: "FRONT",
        1: "FRONT_LEFT",
        2: "FRONT_RIGHT",
        3: "SIDE_LEFT",
        4: "SIDE_RIGHT",
    }
    if cam in waymo_name_map:
        name = waymo_name_map[cam]
        if name in timestamps and key in timestamps[name]:
            return float(timestamps[name][key])

    # final fallback
    return _get_frame_timestamp(timestamps, frame)


def _read_background_ply_points(path: str):
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    names = set(vertices.data.dtype.names)

    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    else:
        rgb = np.ones((xyz.shape[0], 3), dtype=np.float32)

    if "mask" in names:
        mask = vertices["mask"].astype(np.bool_)
    else:
        mask = np.ones((xyz.shape[0],), dtype=np.bool_)

    return xyz[mask], rgb[mask]


def _build_background_pointcloud(datadir: str, frame_ids, frame_pose_map):
    lidar_background_dir = os.path.join(datadir, "lidar", "background")
    if not os.path.exists(lidar_background_dir):
        raise RuntimeError(f"Missing lidar background dir: {lidar_background_dir}")

    all_xyz = []
    all_rgb = []

    for frame in frame_ids:
        ply_path = os.path.join(lidar_background_dir, f"{frame:06d}.ply")
        if not os.path.exists(ply_path):
            continue

        xyz_vehicle, rgb = _read_background_ply_points(ply_path)
        if xyz_vehicle.shape[0] == 0:
            continue

        xyz_h = np.concatenate([xyz_vehicle, np.ones_like(xyz_vehicle[:, :1])], axis=-1)
        xyz_world = xyz_h @ frame_pose_map[frame].T
        xyz_world = xyz_world[:, :3]

        all_xyz.append(xyz_world.astype(np.float32))
        all_rgb.append(rgb.astype(np.float32))

    if len(all_xyz) == 0:
        raise RuntimeError("No valid lidar background points found to initialize point cloud")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    max_points = int(cfg.data.get("max_background_points", 1500000))
    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def _make_empty_tracklets(num_frames: int):
    obj_tracklets = np.ones((num_frames, 1, 8), dtype=np.float32) * -1.0
    obj_info = {}
    return obj_tracklets, obj_info


def _load_dynamic_bound(path: str, height: int, width: int):
    if not os.path.exists(path):
        return np.zeros((height, width), dtype=np.bool_)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((height, width), dtype=np.bool_)
    return (m > 0)


def readKexueguFullInfo(path, images="images", split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get("selected_frames", None)
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get("load_pcd_from", False) and (cfg.mode == "train"):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, "input_ply")
        save_dir = os.path.join(cfg.model_path, "input_ply")
        os.system(f"rm -rf {save_dir}")
        shutil.copytree(load_dir, save_dir)

        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, "colmap")
        save_dir = os.path.join(cfg.model_path, "colmap")
        os.system(f"rm -rf {save_dir}")
        shutil.copytree(colmap_dir, save_dir)

    image_dir = os.path.join(path, images)
    if not os.path.exists(image_dir):
        raise RuntimeError(f"Image dir does not exist: {image_dir}")

    image_files = _list_image_files(image_dir)
    all_frames, all_cams, image_file_map = _discover_frames_and_cams(image_files)
    if len(all_frames) == 0:
        raise RuntimeError(f"No images found under {image_dir}")

    selected_frame_ids = _select_frame_ids(all_frames, selected_frames)
    if len(selected_frame_ids) == 0:
        raise RuntimeError(f"No frames selected from {selected_frames} in {image_dir}")

    selected_cams = sorted(cfg.data.get("cameras", all_cams))
    for cam in selected_cams:
        if cam not in all_cams:
            raise RuntimeError(f"Camera {cam} not found in image set. Available: {all_cams}")

    frame_to_idx = {f: i for i, f in enumerate(selected_frame_ids)}
    num_frames = len(selected_frame_ids)

    intrinsics, extrinsics = _load_intrinsics_extrinsics(path, selected_cams)
    frame_pose_map_raw, cam_pose_map_raw = _load_ego_poses(path)

    for frame in selected_frame_ids:
        if frame not in frame_pose_map_raw:
            raise RuntimeError(f"Missing ego frame pose for frame {frame}")

    frame_pose_map, cam_pose_map = _center_ego_poses(frame_pose_map_raw, cam_pose_map_raw, selected_frame_ids)

    timestamps = _read_timestamps(path)
    frame_timestamps = np.array([_get_frame_timestamp(timestamps, f) for f in selected_frame_ids], dtype=np.float64)

    dynamic_mask_dir = os.path.join(path, "dynamic_mask")
    load_dynamic_mask = os.path.exists(dynamic_mask_dir)

    sky_mask_dir = os.path.join(path, "sky_mask")
    load_sky_mask = (cfg.mode == "train") and os.path.exists(sky_mask_dir)

    lidar_depth_dir = os.path.join(path, "lidar", "depth")
    load_lidar_depth = (cfg.mode == "train") and os.path.exists(lidar_depth_dir)

    exts = []
    ixts = []
    poses = []
    c2ws = []
    frames = []
    cams = []
    frames_idx = []
    image_filenames = []
    cams_timestamps = []
    obj_bounds = []

    for frame in selected_frame_ids:
        for cam in selected_cams:
            key = (frame, cam)
            if key not in image_file_map:
                continue

            image_path = image_file_map[key]
            pose = cam_pose_map.get(cam, {}).get(frame, frame_pose_map[frame])
            ext = extrinsics[cam]
            ixt = intrinsics[cam]
            c2w = pose @ ext

            frames.append(frame)
            cams.append(cam)
            frames_idx.append(frame_to_idx[frame])
            image_filenames.append(image_path)

            exts.append(ext)
            ixts.append(ixt)
            poses.append(pose)
            c2ws.append(c2w)
            cams_timestamps.append(_get_cam_timestamp(timestamps, cam, frame))

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            h, w = img.shape[:2]
            if load_dynamic_mask:
                mask_path = os.path.join(dynamic_mask_dir, f"{frame:06d}_{cam}.png")
                obj_bounds.append(_load_dynamic_bound(mask_path, h, w))
            else:
                obj_bounds.append(np.zeros((h, w), dtype=np.bool_))

    if len(image_filenames) == 0:
        raise RuntimeError("No camera images matched selected frames/cameras")

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(float(np.min(cams_timestamps)), float(np.min(frame_timestamps)))
    cams_timestamps = np.array(cams_timestamps, dtype=np.float64) - timestamp_offset
    tracklet_timestamps = frame_timestamps - timestamp_offset

    # currently no object annotations are required for kexuegu pipeline
    obj_tracklets, obj_info = _make_empty_tracklets(num_frames)

    bkgd_ply_path = os.path.join(cfg.model_path, "input_ply", "points3D_bkgd.ply")
    build_pointcloud = (cfg.mode == "train") and (not os.path.exists(bkgd_ply_path) or cfg.data.get("regenerate_pcd", False))

    if build_pointcloud:
        pointcloud_dir = os.path.join(cfg.model_path, "input_ply")
        os.makedirs(pointcloud_dir, exist_ok=True)

        points_xyz, points_rgb = _build_background_pointcloud(path, selected_frame_ids, frame_pose_map)
        storePly(bkgd_ply_path, points_xyz, points_rgb)

        # keep a second copy for compatibility with utilities that look up lidar ply first
        lidar_ply_path = os.path.join(pointcloud_dir, "points3D_lidar.ply")
        storePly(lidar_ply_path, points_xyz, points_rgb)

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata["obj_tracklets"] = obj_tracklets
    scene_metadata["tracklet_timestamps"] = tracklet_timestamps
    scene_metadata["obj_meta"] = obj_info
    scene_metadata["num_images"] = len(exts)
    scene_metadata["num_cams"] = len(selected_cams)
    scene_metadata["num_frames"] = num_frames

    camera_timestamps = {}
    for cam in selected_cams:
        camera_timestamps[cam] = {"train_timestamps": [], "test_timestamps": []}

    cam_infos = []
    for i in tqdm(range(len(exts)), desc="Preparing kexuegu cameras"):
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]

        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        fov_y = focal2fov(fx, height)
        fov_x = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]

        metadata = dict()
        metadata["frame"] = frames[i]
        metadata["cam"] = cams[i]
        metadata["frame_idx"] = frames_idx[i]
        metadata["ego_pose"] = pose
        metadata["extrinsic"] = ext
        metadata["timestamp"] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata["is_val"] = False
            camera_timestamps[cams[i]]["train_timestamps"].append(cams_timestamps[i])
        else:
            metadata["is_val"] = True
            camera_timestamps[cams[i]]["test_timestamps"].append(cams_timestamps[i])

        guidance = dict()
        guidance["obj_bound"] = Image.fromarray(obj_bounds[i])

        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f"{image_name}.npz")
            if os.path.exists(depth_path):
                depth_npz = np.load(depth_path)
                mask = depth_npz["mask"].astype(np.bool_)
                value = depth_npz["value"].astype(np.float32)
                depth = np.zeros_like(mask, dtype=np.float32)
                depth[mask] = value
                guidance["lidar_depth"] = depth

        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f"{image_name}.png")
            if os.path.exists(sky_mask_path):
                sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
                guidance["sky_mask"] = Image.fromarray(sky_mask)

        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=fov_y,
            FovX=fov_x,
            K=ixt.copy(),
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)

    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata["is_val"]]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata["is_val"]]

    for cam in selected_cams:
        camera_timestamps[cam]["train_timestamps"] = sorted(camera_timestamps[cam]["train_timestamps"])
        camera_timestamps[cam]["test_timestamps"] = sorted(camera_timestamps[cam]["test_timestamps"])
    scene_metadata["camera_timestamps"] = camera_timestamps

    novel_view_cam_infos = []

    # scene normalization
    if cfg.mode == "novel_view":
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    nerf_normalization["radius"] = max(nerf_normalization["radius"], 10)
    if cfg.data.get("extent", False):
        nerf_normalization["radius"] = cfg.data.extent

    cfg.data.extent = float(nerf_normalization["radius"])
    scene_metadata["scene_center"] = nerf_normalization["center"]
    scene_metadata["scene_radius"] = nerf_normalization["radius"]
    print(f"Scene extent: {nerf_normalization['radius']}")

    pcd = fetchPly(bkgd_ply_path)
    sphere_normalization = get_Sphere_Norm(pcd.points)
    scene_metadata["sphere_center"] = sphere_normalization["center"]
    scene_metadata["sphere_radius"] = sphere_normalization["radius"]
    print(f"Sphere extent: {sphere_normalization['radius']}")

    if cfg.mode == "train":
        point_cloud = pcd
    else:
        point_cloud = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )

    return scene_info
