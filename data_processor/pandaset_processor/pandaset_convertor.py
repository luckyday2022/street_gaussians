import json
import os
import argparse
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from pandaset import DataSet as PandaSet, geometry
from pandaset.sequence import Sequence

import sys
import cv2
sys.path.append(os.getcwd())
from utils.multiprocess_utils import track_parallel_progress
from utils.visualization_utils import color_mapper, dump_3d_bbox_on_image
from utils.img_utils import visualize_depth_numpy
from utils.pcd_utils import storePly, fetchPly
from utils.box_utils import bbox_to_corner3d, inbbox_points
from pandaset_helpers import PANDA_CAMERA2ID, PANDA_ID2CAMERA, PANDA_LABELS, PANDA_NONRIGID_DYNAMIC_CLASSES, PANDA_RIGID_DYNAMIC_CLASSES, PANDA_DYNAMIC_CLASSES
EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")


class PandaSetProcessor(object):
    """Process PandaSet.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # PandaSet Provides 6 cameras and 2 lidars
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "front_camera",        # "xxx_0.jpg"
            "front_left_camera",   # "xxx_1.jpg"
            "front_right_camera",  # "xxx_2.jpg"
            "left_camera",         # "xxx_3.jpg"
            "right_camera",        # "xxx_4.jpg"
            "back_camera"          # "xxx_5.jpg"
        ]
        # 0: mechanical 360° LiDAR, 1: front-facing LiDAR, -1: All LiDARs
        self.lidar_list = [-1]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}"
        self.workers = int(workers)
        self.pandaset = PandaSet(load_dir)
        self.create_folder()
        
        self.extrinsics = yaml.load(open(EXTRINSICS_FILE_PATH, "r"), Loader=yaml.FullLoader)

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_idx):
        """Convert action for single file.

        Args:
            scene_idx (str): Scene index.
        """
        scene_data = self.pandaset[scene_idx]
        scene_data.load()
        num_frames = sum(1 for _ in scene_data.timestamps)
        
        if "timestamps" in self.process_keys:
            self.save_timestamp(scene_data, scene_idx)
        for frame_idx in tqdm(range(num_frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True):
            if "images" in self.process_keys:
                self.save_image(scene_data, scene_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(scene_data, scene_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(scene_data, scene_idx, frame_idx, folder_name='lidar')
            if "lidar_forward" in self.process_keys:
                self.save_lidar(scene_data, scene_idx, frame_idx, folder_name='lidar_forward')
            if "pose" in self.process_keys:
                self.save_pose(scene_data, scene_idx, frame_idx)
            if "3dbox_vis" in self.process_keys:
                # visualize 3d box, debug usage
                self.visualize_3dbox(scene_data, scene_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='all')
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='human')
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='vehicle')
        
        # make full ply
        if 'lidar' in self.process_keys:
            lidar_actor_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/actor"
            for instance in os.listdir(lidar_actor_dir):
                instance_dir = os.path.join(lidar_actor_dir, instance)
                ply_files = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if f.endswith('.ply')]
                ply_xyz, ply_rgb = [], []
                for ply_file in ply_files:
                    ply = fetchPly(ply_file)
                    mask = ply.mask
                    ply_xyz.append(ply.points[mask])
                    ply_rgb.append(ply.colors[mask])
                ply_xyz = np.concatenate(ply_xyz, axis=0)   
                ply_rgb = np.concatenate(ply_rgb, axis=0)
                ply_mask = np.ones((ply_xyz.shape[0])).astype(np.bool_)
                storePly(os.path.join(instance_dir, 'full.ply'), ply_xyz, ply_rgb, ply_mask[:, None])

        # save instances info
        if "objects" in self.process_keys:
            instances_info = self.save_objects(scene_data, num_frames)
            
            # solve duplicated objects from different lidars
            duplicated_id_pairs = []
            for k, v in instances_info.items():
                if v["sibling_id"] != '-':
                    # find if the pair is already in the list
                    if (v["id"], v["sibling_id"]) in duplicated_id_pairs or (v["sibling_id"], v["id"]) in duplicated_id_pairs:
                        continue
                    else:
                        duplicated_id_pairs.append((v["id"], v["sibling_id"]))
            
            for pair in duplicated_id_pairs:
                # check if all in the pair are in the instances_info
                if pair[0] not in instances_info:
                    # print(f"WARN: {pair[0]} not in instances_info")
                    continue
                elif pair[1] not in instances_info:
                    # print(f"WARN: {pair[1]} not in instances_info")
                    continue
                else:
                    # keep the longer one in pairs
                    if len(instances_info[pair[0]]['frame_annotations']['frame_idx']) > \
                        len(instances_info[pair[1]]['frame_annotations']['frame_idx']):
                        instances_info.pop(pair[1])
                    else:
                        instances_info.pop(pair[0])
            
            # rough filter stationary objects
            # if all the annotations of an object are stationary, remove it
            static_ids = []
            for k, v in instances_info.items():
                if all(v['frame_annotations']['stationary']):
                    static_ids.append(v['id'])
            print(f"INFO: {len(static_ids)} static objects removed")
            for static_id in static_ids:
                instances_info.pop(static_id)
            print(f"INFO: Final number of objects: {len(instances_info)}")
            
            frame_instances = {}
            # update frame_instances
            for frame_idx in range(num_frames):
                # must ceate a object for each frame
                frame_instances[frame_idx] = []
                for k, v in instances_info.items():
                    if frame_idx in v['frame_annotations']['frame_idx']:
                        frame_instances[frame_idx].append(v["id"])
            
            # verbose: visualize the instances on the image (Debug Usage)
            if "objects_vis" in self.process_keys:
                self.visualize_dynamic_objects(
                    scene_data, scene_idx,
                    instances_info=instances_info,
                    frame_instances=frame_instances
                )
            
            # correct id
            id_map = {}
            for i, (k, v) in enumerate(instances_info.items()):
                id_map[v["id"]] = i
            # update keys in instances_info
            new_instances_info = {}
            for k, v in instances_info.items():
                new_instances_info[id_map[v["id"]]] = v
            # update keys in frame_instances
            new_frame_instances = {}
            for k, v in frame_instances.items():
                new_frame_instances[k] = [id_map[i] for i in v]
                
            # write as json
            instances_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
            with open(f"{instances_dir}/instances_info.json", "w") as fp:
                json.dump(new_instances_info, fp, indent=4)
            with open(f"{instances_dir}/frame_instances.json", "w") as fp:
                json.dump(new_frame_instances, fp, indent=4)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_timestamp(self, scene_data: Sequence, scene_idx):
        """Parse and save the timestamp data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
        """
        timestamps = dict()
        frame_timestamps = scene_data.timestamps
        timestamps['frame'] = frame_timestamps.data
        for idx, cam in enumerate(self.cam_list):
            cam_timestamps = scene_data.camera[cam].timestamps
            timestamps[cam] = cam_timestamps

        with open(f"{self.save_dir}/{str(scene_idx).zfill(3)}/timestamps.json", "w") as f:
            json.dump(timestamps, f, indent=1)


    def get_lidar(self, scene_data: Sequence, frame_idx, lidar_idx=0):
        """Get the lidar data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.

        Returns:
            np.ndarray: Lidar data.
        """
        import pyquaternion
        def _pandaset_pose_to_matrix(pose):
            translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
            quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
            pose = np.eye(4)
            pose[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
            pose[:3, 3] = translation
            return pose
        
        # we only use the Pandar64 lidar 
        pc_world = scene_data.lidar[frame_idx].to_numpy()
        pc_world = pc_world[pc_world[:, -1] == lidar_idx]
        
        # the pose information in self.sequence.lidar.poses is not correct, so we compute it from the camera pose and extrinsics
        # remove ego points
        pcd_world = pc_world[:, :3]
        front_cam = scene_data.camera["front_camera"]
        front_cam2w = _pandaset_pose_to_matrix(front_cam.poses[frame_idx])
        front_cam_extrinsics = self.extrinsics["front_camera"]
        front_cam_extrinsics["position"] = front_cam_extrinsics["extrinsic"]["transform"]["translation"]
        front_cam_extrinsics["heading"] = front_cam_extrinsics["extrinsic"]["transform"]["rotation"]
        l2front_cam = _pandaset_pose_to_matrix(front_cam_extrinsics)
        l2w = front_cam2w @ l2front_cam
        
        pcd_world = np.concatenate([pcd_world, np.ones_like(pcd_world[..., :1])], axis=-1)
        pcd_ego = (pcd_world @ np.linalg.inv(l2w).T)[..., :3]
        mask = (np.abs(pcd_ego[:, :3]) >= np.array([1.0, 2.0, 2.0])).any(-1)
        
        pc_world = pc_world[mask]
        
        # remove outlier points 
        # pc_world_all = o3d.geometry.PointCloud()
        # pc_world_all.points = o3d.utility.Vector3dVector(pc_world[..., :3])
        # _, ind_inliers = pc_world_all.remove_radius_outlier(nb_points=10, radius=0.5)
        # ind_inliers = np.asarray(ind_inliers).astype(np.int32)
        # print(ind_inliers.shape[0], ind_inliers.max())
        # pc_world = pc_world[ind_inliers]
                
        return pc_world

    def save_image(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        
        lidar_depth_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/depth"
        os.makedirs(lidar_depth_dir, exist_ok=True)
        
        for idx, cam in enumerate(self.cam_list):
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            # write PIL Image to jpg
            image = scene_data.camera[cam][frame_idx]
            image.save(img_path)

                        
    def save_calib(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        for idx, cam in enumerate(self.cam_list):
            camera = scene_data.camera[cam]
            poses = camera.poses[frame_idx]
            c2w = geometry._heading_position_to_mat(poses['heading'], poses['position'])
            K = camera.intrinsics
            intrinsics = [K.fx, K.fy, K.cx, K.cy, 0.0, 0.0, 0.0, 0.0, 0.0] 
    
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.txt",
                c2w,
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                + f"{str(idx)}.txt",
                intrinsics,
            )

    def save_lidar(self, scene_data: Sequence, scene_idx, frame_idx, folder_name='lidar'):
        """Parse and save the lidar data in psd format.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        
        frame_instance_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/frame_instances.json"
        instances_info_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/instances_info.json"
        assert os.path.exists(frame_instance_path), f"ERROR: {frame_instance_path} not found"
        assert os.path.exists(instances_info_path), f"ERROR: {instances_info_path} not found"
        with open(frame_instance_path, "r") as f:
            frame_instances = json.load(f)
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        
        lidar_background_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/background"
        lidar_actor_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/actor"
        lidar_depth_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/depth"
        os.makedirs(lidar_background_dir, exist_ok=True)
        os.makedirs(lidar_actor_dir, exist_ok=True)
        os.makedirs(lidar_depth_dir, exist_ok=True)
        
        current_instance_info = dict()
        for instance in frame_instances[str(frame_idx)]:
            current_instance_info[instance] = dict()
            lidar_instance_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/actor/{str(instance).zfill(3)}"
            os.makedirs(lidar_instance_dir, exist_ok=True)
            save_path = f"{lidar_instance_dir}/{str(frame_idx).zfill(3)}.ply"   
            current_instance_info[instance]['save_path'] = save_path
            
            frame_annotations = instances_info[str(instance)]['frame_annotations']
            idx = frame_annotations['frame_idx'].index(frame_idx)
            current_instance_info[instance]['obj_to_world'] = frame_annotations['obj_to_world'][idx]
            current_instance_info[instance]['box_size'] = frame_annotations['box_size'][idx]
            current_instance_info[instance]['class_name'] = instances_info[str(instance)]['class_name']
                    
        # index        x           y         z        i         t       d                                                     
        # 0       -75.131138  -79.331690  3.511804   7.0  1.557540e+09  0
        # 1      -112.588306 -118.666002  1.423499  31.0  1.557540e+09  0
        # - `i`: `float`: Reflection intensity in a range `[0,255]`
        # - `t`: `float`: Recorded timestamp for specific point
        # - `d`: `int`: Sensor ID. `0` -> mechnical 360° LiDAR, `1` -> forward-facing LiDAR
        
        # paint the lidar points
        lidar_idx = 1 if folder_name == 'lidar_forward' else 0
        pc_world = self.get_lidar(scene_data, frame_idx, lidar_idx)
        pcd_world = pc_world[:, :3]
        pcd_mask = np.zeros((pcd_world.shape[0])).astype(np.bool_)
        pcd_color = np.zeros((pcd_world.shape[0], 3)).astype(np.uint8)      
        
        # the lidar scans are synced such that the middle of a scan is at the same time as the front camera image
        timestamp = scene_data.camera["front_camera"].timestamps[frame_idx]

        for i, cam in enumerate(self.cam_list):
            camera = scene_data.camera[cam]
            cam_timestamps = np.array(scene_data.camera[cam].timestamps)
            cam_frame_idx = np.argmin(np.abs(cam_timestamps - timestamp)).astype(np.int32)
            cam_frame_idx = frame_idx
            points2d_camera, points3d_camera, inliner_indices_arr = geometry.projection(
                lidar_points=pcd_world,                
                camera_data=camera[cam_frame_idx], # type: ignore
                camera_pose=camera.poses[cam_frame_idx],
                camera_intrinsics=camera.intrinsics,
                filter_outliers=True
            )
            
            image = np.asarray(camera[cam_frame_idx]) # type: ignore            
            h, w = image.shape[:2]
            u_depth, v_depth = points2d_camera[:, 0], points2d_camera[:, 1]
            u_depth = np.clip(u_depth, 0, w-1).astype(np.int32)
            v_depth = np.clip(v_depth, 0, h-1).astype(np.int32)
            color_value = image[v_depth, u_depth]
            
            # ignore the points that have been painted         
            paint_mask = np.logical_not(pcd_mask[inliner_indices_arr])
            paint_inlinear_indices_arr = inliner_indices_arr[paint_mask]
            paint_color = color_value[paint_mask]
            pcd_color[paint_inlinear_indices_arr] = paint_color
            pcd_mask[paint_inlinear_indices_arr] = True
            
            # save lidar depth
            depth_value = points3d_camera[:, 2]
            depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
            indices = v_depth * w + u_depth
            np.minimum.at(depth, indices, depth_value)
            depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
            valid_depth_pixel = (depth != 0)
            valid_depth_value = depth[valid_depth_pixel]
            valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
            
            depth_filename = f"{lidar_depth_dir}/{str(frame_idx).zfill(3)}_{str(i)}.npz"
            depth_vis_filename = f"{lidar_depth_dir}/{str(frame_idx).zfill(3)}_{str(i)}.png"
            np.savez_compressed(depth_filename, mask=valid_depth_pixel, value=valid_depth_value)

            if i == 0:
                depth = depth.reshape(h, w).astype(np.float32)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_on_img = np.asarray(image)[..., [2, 1, 0]]
                depth_on_img[depth > 0] = depth_vis[depth > 0]
                cv2.imwrite(depth_vis_filename, depth_on_img)      

        pcd_instance_mask = np.zeros((pcd_world.shape[0])).astype(np.bool_)
        for instance in current_instance_info.keys():
            obj_to_world = current_instance_info[instance]['obj_to_world']
            length, width, height = current_instance_info[instance]['box_size']
            
            # padding the box
            if current_instance_info[instance]['class_name'] in PANDA_RIGID_DYNAMIC_CLASSES:
                length = length * 1.5
                width = width * 1.5
            
            pcd_world_homo = np.concatenate([pcd_world, np.ones_like(pcd_world[..., :1])], axis=-1)
            pcd_instance = (pcd_world_homo @ np.linalg.inv(obj_to_world).T)[..., :3]
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(pcd_instance, corners3d)
            pcd_instance_mask = np.logical_or(pcd_instance_mask, inbbox_mask)

            if inbbox_mask.sum() > 0:
                save_path = current_instance_info[instance]['save_path']
                storePly(save_path, pcd_instance[inbbox_mask], pcd_color[inbbox_mask], pcd_mask[inbbox_mask][:, None])

        pcd_bkgd_xyz = pcd_world[~pcd_instance_mask]
        pcd_bkgd_color = pcd_color[~pcd_instance_mask]
        pcd_bkgd_mask = pcd_mask[~pcd_instance_mask]
        ply_path = f"{lidar_background_dir}/{str(frame_idx).zfill(3)}.ply"
        storePly(ply_path, pcd_bkgd_xyz, pcd_bkgd_color, pcd_bkgd_mask[:, None])
            
    def save_pose(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the pose data.

        Since pandaset does not provide the ego pose, we use the lidar pose as the ego pose.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        lidar_poses = scene_data.lidar.poses[frame_idx]
        lidar_to_world = geometry._heading_position_to_mat(lidar_poses['heading'], lidar_poses['position'])

        np.savetxt(
            f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            lidar_to_world,
        )
        
    def visualize_3dbox(self, scene_data: Sequence, scene_idx, frame_idx):
        """DEBUG: Visualize the 3D bounding box on the image.
        Visualize the 3D bounding box all with the same COLOR.
        If you want to visualize the 3D bounding box with different colors, please use the `visualize_dynamic_objects` function.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        for idx, cam in enumerate(self.cam_list):
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            canvas = np.array(Image.open(img_path))
    
            camera = scene_data.camera[cam]
            cuboids = scene_data.cuboids[frame_idx]
            
            lstProj2d = []
            recorded_id = []
            for _, row in cuboids.iterrows():
                if row["label"] not in PANDA_DYNAMIC_CLASSES or row["stationary"]:
                    continue
                if not row["cuboids.sensor_id"] == -1:
                    if row["cuboids.sibling_id"] in recorded_id:
                        continue
                recorded_id.append(row["uuid"])
                box = [
                        row[  "position.x"], row[  "position.y"], row[  "position.z"],
                        row["dimensions.x"], row["dimensions.y"], row["dimensions.z"],
                        row["yaw"]
                    ]
                corners = geometry.center_box_to_corners(box)
                
                projected_points2d, _, _ = geometry.projection(
                    lidar_points=corners,                
                    camera_data=camera[frame_idx],
                    camera_pose=camera.poses[frame_idx],
                    camera_intrinsics=camera.intrinsics,
                    filter_outliers=True
                )
                projected_points2d = projected_points2d.tolist()
                if len(projected_points2d) == 8:
                    lstProj2d.append(projected_points2d)
            
            lstProj2d = np.asarray(lstProj2d)
            
            img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=(255,0,0))
            
            # save
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/3dbox_vis/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            Image.fromarray(img_plotted).save(img_path)
            
    def visualize_dynamic_objects(
        self, scene_data: Sequence, scene_idx,
        instances_info: dict, frame_instances: dict
    ):
        """DEBUG: Visualize the dynamic objects'box with different colors on the image.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            instances_info (dict): Instances information.
            frame_instances (dict): Frame instances.
        """
        output_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/debug_vis"
        
        num_frames = sum(1 for _ in scene_data.timestamps)
        for frame_idx in range(num_frames):
            for idx, cam in enumerate(self.cam_list):
                
                frame_idx_key = str(frame_idx) + '_' + cam
                frame_instances[frame_idx_key] = []
                
                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
                )
                canvas = np.array(Image.open(img_path))
                
                camera = scene_data.camera[cam]
                if frame_idx in frame_instances:                
                    objects = frame_instances[frame_idx]
                    
                    lstProj2d = []
                    color_list = []
                    for obj_id in objects:
                        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
                        o2w = instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
                        o2w = np.array(o2w)
                        length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
                        half_dim_x, half_dim_y, half_dim_z = length/2.0, width/2.0, height/2.0
                        corners = np.array(
                            [[half_dim_x, half_dim_y, -half_dim_z],
                             [half_dim_x, -half_dim_y, -half_dim_z],
                             [-half_dim_x, -half_dim_y, -half_dim_z],
                             [-half_dim_x, half_dim_y, -half_dim_z],
                             [half_dim_x, half_dim_y, half_dim_z],
                             [half_dim_x, -half_dim_y, half_dim_z],
                             [-half_dim_x, -half_dim_y, half_dim_z],
                             [-half_dim_x, half_dim_y, half_dim_z]]
                        )
                        corners = (o2w[:3, :3] @ corners.T + o2w[:3, [3]]).T
                        
                        projected_points2d, _, inliner_indices_arr = geometry.projection(
                            lidar_points=corners,                
                            camera_data=camera[frame_idx],
                            camera_pose=camera.poses[frame_idx],
                            camera_intrinsics=camera.intrinsics,
                            filter_outliers=True
                        )
                        projected_points2d = projected_points2d.tolist()
                        if len(projected_points2d) == 8:
                            lstProj2d.append(projected_points2d)
                            color_list.append(color_mapper(obj_id))

                        # consider as visible if at least 1 corner is visible
                        if len(inliner_indices_arr) > 0:
                            frame_instances[frame_idx_key].append(obj_id)
                        
                    lstProj2d = np.asarray(lstProj2d)
                    img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                
                img_path = (
                    f"{output_path}/"
                    + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
                )
                Image.fromarray(img_plotted).save(img_path)

    def save_dynamic_mask(self, scene_data: Sequence, scene_idx, frame_idx, class_valid='all'):
        """Parse and save the segmentation data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
            class_valid (str): Class valid for dynamic mask.
        """
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = PANDA_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = PANDA_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = PANDA_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            
        for idx, cam in enumerate(self.cam_list):
            # dynamic_mask
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            camera = scene_data.camera[cam]
            cuboids = scene_data.cuboids[frame_idx]
            
            recorded_id = []
            for _, row in cuboids.iterrows():
                if row["label"] not in VALID_CLASSES or row["stationary"]:
                    continue
                if not row["cuboids.sensor_id"] == -1:
                    if row["cuboids.sibling_id"] in recorded_id:
                        continue
                recorded_id.append(row["uuid"])

                box = [
                        row[  "position.x"], row[  "position.y"], row[  "position.z"],
                        row["dimensions.x"], row["dimensions.y"], row["dimensions.z"],
                        row["yaw"]
                    ]
                corners = geometry.center_box_to_corners(box)
                
                projected_points2d, _, _ = geometry.projection(
                    lidar_points=corners,                
                    camera_data=camera[frame_idx],
                    camera_pose=camera.poses[frame_idx],
                    camera_intrinsics=camera.intrinsics,
                    filter_outliers=True
                )
                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not len(projected_points2d) == 8:
                    continue
                u, v= projected_points2d[:, 0], projected_points2d[:, 1]
                u = u.astype(np.int32)
                v = v.astype(np.int32)

                # Clip box to image bounds.
                u = np.clip(u, 0, camera[frame_idx].size[0])
                v = np.clip(v, 0, camera[frame_idx].size[1])

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    1.,
                )
            dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = os.path.join(mask_dir, f"{str(frame_idx).zfill(3)}_{str(idx)}.png")
            dynamic_mask.save(dynamic_mask_path)
        
    def save_objects(self, scene_data: Sequence, num_frames):
        """Parse and save the objects annotation data.
        
        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            num_frames (int): Number of frames.
        """
        instances_info = {}
        
        for frame_idx in range(num_frames):
            cuboids = scene_data.cuboids[frame_idx]
            for _, row in cuboids.iterrows():
                str_id = row["uuid"]
                label = row["label"]
                if label not in PANDA_DYNAMIC_CLASSES:
                    continue
                
                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=str_id,
                        class_name=row["label"],
                        sibling_id=row["cuboids.sibling_id"],
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                            "stationary": [],
                        }
                    )
                
                # Box coordinates in vehicle frame.
                tx, ty, tz = row["position.x"], row["position.y"], row["position.z"]
                
                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(row["yaw"])
                s = np.math.sin(row["yaw"])
                
                # [object to  world] transformation matrix
                o2w = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]]
                
                instances_info[str_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[str_id]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                instances_info[str_id]['frame_annotations']['box_size'].append(dimension)
                instances_info[str_id]['frame_annotations']['stationary'].append(row["stationary"])
        
        return instances_info

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/sky_masks", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/intrinsics", exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
            if "lidar_forward" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar_forward", exist_ok=True)
            if "3dbox_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/3dbox_vis", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
            if "objects_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances/debug_vis", exist_ok=True)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="split of the dataset, e.g. training, validation, testing, please specify the split name for different dataset",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    # priority: scene_ids > split_file > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--max_frame_limit",
        type=int,
        default=300,
        help="maximum number of frames to be processed in a dataset, in nuplan dataset, \
            the scene duration super long, we can limit the number of frames to be processed, \
                this argument is used only for nuplan dataset",
    )
    parser.add_argument(
        "--start_frame_idx",
        type=int,
        default=1000,
        help="We skip the first start_frame_idx frames to avoid ego static frames",
    )
    parser.add_argument(
        "--interpolate_N",
        type=int,
        default=0,
        help="Interpolate to get frames at higher frequency, this is only used for nuscene dataset",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
    )
    args = parser.parse_args()
    
    scene_ids_list = [
        1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 23, 24, 27, 28, 29, 30, 32, 33, 34, 35, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 77, 78, 79, 80, 84, 85, 86, 88, 89, 90, 91,
        92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
        109, 110, 112, 113, 115, 116, 117, 119, 120, 122, 123, 124, 139, 149, 158
    ]
    
    scene_ids_list = [str(scene_id).zfill(3) for scene_id in scene_ids_list]
    dataset_processor = PandaSetProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        process_keys=args.process_keys,
        process_id_list=scene_ids_list,
        workers=args.workers,
    )

    for scene_id in scene_ids_list:
        dataset_processor.convert_one(scene_id)



    # dataset_processor.convert()

