import sys
import os
import numpy as np
import cv2
import argparse
import imageio
import open3d as o3d
import open3d as o3d
from tqdm import tqdm

sys.path.append(os.getcwd())
from pandaset_helpers import *
from utils.box_utils import inbbox_points, bbox_to_corner3d
from utils.pcd_utils import BasicPointCloud, fetchPly, storePly
from utils.multiprocess_utils import track_parallel_progress
from utils.render_utils import render_pointcloud_pytorch3d, render_pointcloud_diff_point_rasterization

class PandasetLiDARRenderer(object):
    def __init__(self, args, scene_ids):
        self.data_dir = args.data_dir
        self.skip_existing = args.skip_existing
        self.delta_frames = args.delta_frames
        self.cams = args.cams
        self.scene_ids = scene_ids
        self.gpus = args.gpus
        self.save_dir = args.save_dir
        self.workers = args.workers
        self.shifts = args.shifts
        
        self.flip_axis = 1

    def read_lidar_ply(self, lidar_dir):
        ply_dict = dict()
        # read background ply
        lidar_background_dir = os.path.join(lidar_dir, 'background')
        ply_dict_background = dict()

        bkgd_ply_list = sorted([os.path.join(lidar_background_dir, f) for f in os.listdir(lidar_background_dir) if f.endswith('.ply')])
        for i, bkgd_ply_path in enumerate(tqdm(bkgd_ply_list, desc='Reading Background ply')):
            bkgd_ply = fetchPly(bkgd_ply_path)
            frame = image_filename_to_frame(os.path.basename(bkgd_ply_path))
            xyz_world = bkgd_ply.points
            mask = bkgd_ply.mask 
            xyz_world = bkgd_ply.points[mask]
            rgb = bkgd_ply.colors[mask]
            ply_dict_background[frame] = np.concatenate([xyz_world, rgb], axis=-1)

        ply_dict['background'] = ply_dict_background

        # read actor lidar
        lidar_actor_dir = os.path.join(lidar_dir, 'actor')
        for track_id in os.listdir(lidar_actor_dir):
            ply_dict_actor = dict()
            
            lidar_actor_dir_ = os.path.join(lidar_actor_dir, track_id)
            actor_ply_list = sorted([os.path.join(lidar_actor_dir_, f) for f in os.listdir(lidar_actor_dir_) if f.endswith('.ply') and f != 'full.ply'])
            for actor_ply_path in actor_ply_list:
                frame = image_filename_to_frame(os.path.basename(actor_ply_path))
                actor_ply = fetchPly(actor_ply_path)
                mask = actor_ply.mask

                if mask.sum() == 0:
                    continue

                xyz = actor_ply.points[mask]
                rgb = actor_ply.colors[mask]
                ply_dict_actor[frame] = np.concatenate([xyz, rgb], axis=-1)
            
            ply_dict[int(track_id)] = ply_dict_actor

        return ply_dict

    def make_lidar_ply(self, ply_dict, start_frame, end_frame):
        ply_frame_dict = dict()
        bkgd_ply = []
        for frame in range(start_frame, end_frame + 1):
            bkgd_ply.append(ply_dict['background'][frame])
        bkgd_ply = np.concatenate(bkgd_ply, axis=0)  # [N, xyz + rgb]

        # filter bkgd outliers
        # points_bkgd = o3d.geometry.PointCloud()
        # points_bkgd.points = o3d.utility.Vector3dVector(bkgd_ply[..., :3])
        # points_bkgd.colors = o3d.utility.Vector3dVector(bkgd_ply[..., 3:])
        # points_bkgd_inliers = points_bkgd.remove_radius_outlier(nb_points=10, radius=0.5)
        # bkgd_ply_xyz = np.asarray(points_bkgd_inliers.points).astype(np.float32)
        # bkgd_ply_rgb = np.asarray(points_bkgd_inliers.colors).astype(np.float32)
        # bkgd_ply = np.concatenate([bkgd_ply_xyz, bkgd_ply_rgb], axis=-1)

        ply_frame_dict['background'] = bkgd_ply

        for track_id in ply_dict.keys():
            if track_id == 'background':
                continue
            actor_ply = []
            for frame in range(start_frame, end_frame + 1):
                if frame not in ply_dict[track_id]:
                    continue
                actor_ply.append(ply_dict[track_id][frame])
            # empty actor ply
            if len(actor_ply) == 0:
                continue
            actor_ply = np.concatenate(actor_ply, axis=0)
            ply_frame_dict[track_id] = actor_ply
        return ply_frame_dict

    def transform_lidar_ply(self, lidar_ply, obj_info):
        pose, box = obj_info['pose'], obj_info['box']
        xyz, rgb = lidar_ply[..., :3], lidar_ply[..., 3:]

        # filter points in bbox
        length, width, height = box
        bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
        corners3d = bbox_to_corner3d(bbox)
        inbbox_mask = inbbox_points(xyz, corners3d)
        xyz = xyz[inbbox_mask]
        rgb = rgb[inbbox_mask]

        xyz_homo = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz = (xyz_homo @ pose.T)[..., :3]
        lidar_ply = np.concatenate([xyz, rgb], axis=-1)
        return lidar_ply

    def check_existing(self, scene_dir, cam, num_frames):
        for frame in range(num_frames):
            image_proj_save_path = os.path.join(scene_dir, 'lidar_forward', self.save_dir, f'{str(frame).zfill(3)}_{cam}.png')
            mask_proj_save_path = os.path.join(scene_dir, 'lidar_forward', self.save_dir, f'{str(frame).zfill(3)}_{cam}_mask.png')
            if not (os.path.exists(image_proj_save_path) and os.path.exists(mask_proj_save_path)):
                return False
        return True

    def render(self):
        """Convert action."""
        print("Start rendering ...")
        id_list = self.scene_ids
        render_fn = self.render_one

        chunk = 16
        for i in range(0, len(id_list), chunk):
            print(f"\nProcessing scenes {i:03d} to {min(i+chunk-1, len(id_list)-1):03d} ...")
            cur_id_list = id_list[i:i + chunk]
            track_parallel_progress(render_fn, cur_id_list, self.workers)
        print("\nFinished ...")

    def get_save_dir(self, scene_dir, shift):
        save_dir = os.path.join(scene_dir, 'lidar_forward', self.save_dir) if shift == 0 \
                else os.path.join(scene_dir, 'lidar_forward', f'{self.save_dir}_shift_{shift:.2f}')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def render_one(self, scene_idx):
        """Project action for single file."""
        scene_dir = os.path.join(self.data_dir, str(scene_idx).zfill(3))

        # load timestamps
        timestamp_path = os.path.join(scene_dir, 'timestamps.json')
        with open(timestamp_path, 'r') as f:
            timestamps = json.load(f)

        # load cameras
        cam_poses, intrinsics = load_camera_info(datadir=scene_dir)

        # load tracks
        frame_instances, instances_info = load_track(datadir=scene_dir)
        lidar_dir = os.path.join(scene_dir, 'lidar_forward')
        lidar_render_dir = os.path.join(lidar_dir, self.save_dir)

        os.makedirs(lidar_render_dir, exist_ok=True)

        # gpu_id
        gpu_id = self.gpus[scene_idx % len(self.gpus)]
        for cam in self.cams:            
            print(f'Processing scene {scene_idx:03d}, camera {cam}')
            ply_dict = self.read_lidar_ply(lidar_dir=lidar_dir)
            
            if self.skip_existing and self.check_existing(scene_dir, cam, NUM_FRAMES):
                print(f'Skipping scene {scene_idx:03d}, camera {cam}')
                continue

            cam_name = CAM2NAME[cam]
            cam_timestamps = timestamps[cam_name]
            box_timestamps = timestamps[PANDA_ID2CAMERA[0]]

            for shift in self.shifts:
                ply_render_result_rgb = []
                render_save_dir = self.get_save_dir(scene_dir, shift)
                
                if self.skip_existing and self.check_existing(scene_dir, cam, render_save_dir, NUM_FRAMES): # type: ignore
                    print(f'Skipping scene {scene_idx:03d}, camera {cam}, shift {shift}')
                    continue

                print(f'Processing scene {scene_idx:03d}, camera {cam}, shift {shift}')

                for frame in tqdm(range(NUM_FRAMES), desc='Rendering'):
                    start_frame = max(0, frame - self.delta_frames)
                    end_frame = min(NUM_FRAMES - 1, frame + self.delta_frames)

                    # step1: get point cloud xyz and rgb
                    ply_frame_dict = self.make_lidar_ply(ply_dict, start_frame=start_frame, end_frame=end_frame)
                    ply_frame = [ply_frame_dict.pop('background')]

                    for track_id, ply_actor_frame in ply_frame_dict.items():
                        frame_annotations = instances_info[str(track_id)]['frame_annotations']
                        timestamp = cam_timestamps[frame]
                        obj_info = get_obj_info(frame_annotations=frame_annotations, box_timestamps=box_timestamps, timestamp=timestamp)
                        if obj_info is None:
                            continue

                        ply_actor_frame = self.transform_lidar_ply(ply_actor_frame, obj_info)
                        ply_frame.append(ply_actor_frame)
                    ply_frame = np.concatenate(ply_frame, axis=0)
                    ply_xyz, ply_rgb = ply_frame[..., :3], ply_frame[..., 3:]

                    # step2: modify camera pose
                    c2w = cam_poses[frame, cam].copy()
                    ixt = intrinsics[cam]
                    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
                    lane_shift_direction = get_lane_shift_direction(cam_poses, cam, frame)
                    lane_shift_sign = LANE_SHIFT_SIGN[f'{scene_idx:03d}']

                    c2w[:2, 3] = c2w[:2, 3] + lane_shift_sign * shift * lane_shift_direction[:2]
                    w2c = np.linalg.inv(c2w)

                    ply_xyz_cam = np.dot(ply_xyz, w2c[:3, :3].T) + w2c[:3, 3:].T
                    ply_depth = ply_xyz_cam[:, 2]
                    ply_pixel = np.dot(ply_xyz_cam, ixt.T)
                    ply_pixel = ply_pixel[:, :2] / ply_pixel[:, 2:]
                    ply_valid = (
                        (ply_depth > 1e-3)
                        & (ply_pixel[:, 0] >= 0)
                        & (ply_pixel[:, 0] < w)
                        & (ply_pixel[:, 1] >= 0)
                        & (ply_pixel[:, 1] < h),
                    )[0]
                    ply_pixel = ply_pixel[ply_valid]
                    ply_pixel = np.round(ply_pixel).astype(np.int32)
                    valid_ply_depth = ply_depth[ply_valid]
                    valid_ply_rgb = ply_rgb[ply_valid]
                    valid_ply_mask = np.ones_like(valid_ply_depth)
                    valid_ply_feature = np.concatenate([valid_ply_rgb, valid_ply_depth[:, None], valid_ply_mask[:, None]], axis=-1)
                    valid_ply_xyz = ply_xyz[ply_valid]

                    ply_render = render_pointcloud_diff_point_rasterization(c2w, ixt, valid_ply_xyz, valid_ply_feature, h, w, gpu_id, use_ndc_scale=True, scale=0.01)
                    # ply_render = render_pointcloud_pytorch3d(c2w, ixt, valid_ply_xyz, valid_ply_feature, h, w, gpu_id)          
                    ply_render_rgb, ply_render_mask = ply_render[0, ..., :3], ply_render[0, ..., 3]
                    ply_render_rgb = ply_render_rgb.detach().cpu().numpy()
                    ply_render_mask = ply_render_mask.detach().cpu().numpy()

                    image_render_save_path = os.path.join(render_save_dir, f'{str(frame).zfill(3)}_{cam}.png')
                    mask_render_save_path = os.path.join(render_save_dir, f'{str(frame).zfill(3)}_{cam}_mask.png')

                    image_render = (ply_render_rgb * 255).astype(np.uint8)
                    cv2.imwrite(image_render_save_path, image_render[..., [2, 1, 0]])
                    mask_render = (ply_render_mask * 255).astype(np.uint8)
                    cv2.imwrite(mask_render_save_path, mask_render)

                    ply_render_result_rgb.append(ply_render_rgb)

                image_video_save_path = os.path.join(render_save_dir, f'render_rgb_{cam}.mp4')
                imageio.mimwrite(image_video_save_path, [(x * 255).astype(np.uint8) for x in ply_render_result_rgb], fps=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./waymo_processor/training_set_processed')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing files')
    parser.add_argument('--delta_frames', type=int, default=10)
    parser.add_argument('--cams', type=int, nargs='+', default=[0])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='color_render')
    parser.add_argument('--shifts', type=int, nargs='+', default=[0])

    args = parser.parse_args()
    data_dir = args.data_dir
    scene_ids = sorted([x for x in os.listdir(data_dir)])
    scene_ids = [int(x) for x in scene_ids]
    scene_ids = [x for x in scene_ids if x != 54]
    dataset_projector = PandasetLiDARRenderer(args, scene_ids)
    if args.workers == 1:
        for scene_idx in scene_ids:
            dataset_projector.render_one(scene_idx)
    else:
        dataset_projector.render()


if __name__ == '__main__':
    main()
