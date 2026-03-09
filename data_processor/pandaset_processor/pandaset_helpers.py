import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Type


def image_filename_to_cam(x): return int(x.split('.')[0][-1])
def image_filename_to_frame(x): return int(x.split('.')[0][:3])


PANDA_LABELS = [
    'Animals - Other', 'Bicycle', 'Bus', 'Car',
    'Cones', 'Construction Signs', 'Emergency Vehicle', 'Medium-sized Truck',
    'Motorcycle', 'Motorized Scooter', 'Other Vehicle - Construction Vehicle', 'Other Vehicle - Pedicab',
    'Other Vehicle - Uncommon', 'Pedestria\n', 'Pedestrian with Object', 'Personal Mobility Device',
    'Pickup Truck', 'Pylons', 'Road Barriers', 'Rolling Containers',
    'Semi-truck', 'Signs', 'Temporary Construction Barriers', 'Towed Object',
    'Train', 'Tram / Subway'
]

PANDA_NONRIGID_DYNAMIC_CLASSES = [
    'Pedestrian', 'Pedestrian with Object', 'Bicycle', 'Animals - Other'
]

PANDA_RIGID_DYNAMIC_CLASSES = [
    'Bus', 'Car', 'Emergency Vehicle', 'Medium-sized Truck',
    'Motorcycle', 'Motorized Scooter', 'Other Vehicle - Construction Vehicle', 'Other Vehicle - Pedicab',
    'Other Vehicle - Uncommon', 'Personal Mobility Device', 'Pickup Truck',
    'Semi-truck', 'Train', 'Tram / Subway'
]

PANDA_DYNAMIC_CLASSES = PANDA_NONRIGID_DYNAMIC_CLASSES + PANDA_RIGID_DYNAMIC_CLASSES


PANDA_CAMERA2ID = {
    'front_camera': 0,
    'front_left_camera': 1,
    'front_right_camera': 2,
    'left_camera': 3,
    'right_camera': 4,
    'back_camera': 5
}

PANDA_ID2CAMERA = {
    0: 'front_camera',
    1: 'front_left_camera',
    2: 'front_right_camera',
    3: 'left_camera',
    4: 'right_camera',
    5: 'back_camera'
}

LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
        "001": -1,
        "011": 1,
        "016": 1,
        "053": -1, 
        "158": -1,
    }
)


# PandaSet Camera List:
# 0: front_camera
# 1: front_left_camera
# 2: front_right_camera
# 3: left_camera
# 4: right_camera
# 5: back_camera
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4, 5]
CAM2NAME = {0: "front_camera", 1: "front_left_camera", 2: "front_right_camera", 3: "left_camera", 4: "right_camera", 5: "back_camera"}

NUM_FRAMES = 80
NUM_CAMS = len(AVAILABLE_CAM_LIST)
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920


def load_camera_info(datadir):
    intrinsics = []
    for i in range(NUM_CAMS):
        intrinsic = np.loadtxt(os.path.join(datadir, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    cam_poses = []
    for i in range(NUM_FRAMES):
        for j in range(NUM_CAMS):
            c2w = np.loadtxt(os.path.join(datadir, "extrinsics", f"{i:03d}_{j}.txt"))
            cam_poses.append(c2w)

    cam_poses = np.array(cam_poses).reshape(NUM_FRAMES, NUM_CAMS, 4, 4)

    return cam_poses, intrinsics


def load_track(datadir):
    instance_dir = os.path.join(datadir, 'instances')
    frame_instances_path = os.path.join(instance_dir, "frame_instances.json")
    with open(frame_instances_path, "r") as f:
        frame_instances = json.load(f)

    instances_info_path = os.path.join(instance_dir, "instances_info.json")
    with open(instances_info_path, "r") as f:
        instances_info = json.load(f)

    return frame_instances, instances_info


def inter_two_poses(pose_a, pose_b, alpha):
    ret = np.zeros([3, 4], dtype=np.float64)
    key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret


def get_obj_info(frame_annotations, box_timestamps, timestamp):
    frame_idxs = frame_annotations['frame_idx']
    obj_to_worlds = frame_annotations['obj_to_world']
    box_sizes = frame_annotations['box_size']
    box_timestamps = np.array(box_timestamps)[frame_idxs]

    if timestamp >= box_timestamps[0] and timestamp <= box_timestamps[-1]:
        if len(box_timestamps) > 1:
            delta_timestamps = np.abs(box_timestamps - timestamp)
            idx1, idx2 = np.argsort(delta_timestamps)[:2]
            obj_to_world1 = np.array(obj_to_worlds[idx1]).astype(np.float32)
            obj_to_world2 = np.array(obj_to_worlds[idx2]).astype(np.float32)
            box_size1 = np.array(box_sizes[idx1]).astype(np.float32)
            box_size2 = np.array(box_sizes[idx2]).astype(np.float32)
            timestamp1 = box_timestamps[idx1]
            timestamp2 = box_timestamps[idx2]
            
            alpha = (timestamp2 - timestamp) / (timestamp2 - timestamp1)
            obj_to_world = inter_two_poses(obj_to_world1, obj_to_world2, alpha)
            box_size = box_size1 * alpha + box_size2 * (1. - alpha)
        else:
            obj_to_world = np.array(obj_to_worlds[0]).astype(np.float32)
            box_size = np.array(box_sizes[0]).astype(np.float32)

    elif timestamp < box_timestamps[0] and timestamp >= box_timestamps[0] - 0.1:
        obj_to_world = np.array(obj_to_worlds[0]).astype(np.float32)
        box_size = np.array(box_sizes[0]).astype(np.float32)

    elif timestamp > box_timestamps[-1] and timestamp <= box_timestamps[-1] + 0.1:
        obj_to_world = np.array(obj_to_worlds[-1]).astype(np.float32)
        box_size = np.array(box_sizes[-1]).astype(np.float32)

    else:
        return None

    ret = dict()
    ret['pose'] = obj_to_world
    ret['box'] = box_size

    return ret


def get_lane_shift_direction(cam_poses, cam, frame):
    cam_poses = cam_poses[:, cam]  # (80, 4, 4)
    velocities = cam_poses[1:, :3, 3] - cam_poses[:-1, :3, 3]  # (79, 3)
    velocities = np.concatenate([velocities, velocities[-1:]], axis=0)  # (80, 3)
    driving_direction = velocities[frame]
    driving_direction = driving_direction / np.linalg.norm(driving_direction)
    orth_right_direction = np.cross(driving_direction, np.array([0, 0, 1]).astype(np.float32))
    orth_right_direction = orth_right_direction / np.linalg.norm(orth_right_direction)
    return orth_right_direction
