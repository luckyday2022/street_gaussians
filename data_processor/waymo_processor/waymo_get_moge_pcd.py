import os
import numpy as np
import cv2
import argparse
import open3d as o3d
import torch
import sys
sys.path.append(os.getcwd())
sys.path.append('/lpai/volumes/jointmodel/yanyunzhi/code/MoGe')
from waymo_helpers import load_calibration, load_track, image_filename_to_cam, image_filename_to_frame
from utils.pcd_utils import storePly, fetchPly
from utils.box_utils import bbox_to_corner3d, inbbox_points
from utils.base_utils import transform_points_numpy
from moge.model import MoGeModel # type: ignore
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda()

from tqdm import tqdm

def recover_metric_depth(pred, gt, mask0):
    mask = (gt > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        mask0 = mask0 > 0
        mask = mask & mask0
        
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    weight = 1.0 / gt_mask
    try:
        a, b = np.polyfit(x=pred_mask, y=gt_mask, w=weight, deg=1)
    except:
        a, b = 1.0, 0.0
        print(f"num of valid preds:{(pred > 1e-8).sum()}, num of valid gts:{(gt > 1e-8).sum()}")

    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)

    return pred_metric, a, b

def save_lidar(seq_save_dir):
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f'Processing scene {seq_save_dir}...')
    print(f'Saving to {seq_save_dir}')

    os.makedirs(seq_save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_depth_dir = os.path.join(seq_save_dir, 'lidar/depth')
    num_frames = len(sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')])) // 5
    
    moge_dir = os.path.join(seq_save_dir, 'moge')
    os.makedirs(moge_dir, exist_ok=True)
    moge_dir_background = os.path.join(moge_dir, 'background')
    os.makedirs(moge_dir_background, exist_ok=True)
    moge_dir_actor = os.path.join(moge_dir, 'actor')
    os.makedirs(moge_dir_actor, exist_ok=True)
    lidar_dir_actor = os.path.join(seq_save_dir, 'lidar/actor')
    assert os.path.exists(lidar_dir_actor)
    
    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(moge_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []
    
    
    for frame_id in tqdm(range(num_frames)):
        image_path = os.path.join(image_dir, f'{frame_id:06d}_0.png')
        lidar_depth_path = os.path.join(lidar_depth_dir, f'{frame_id:06d}_0.npz')
        lidar_depth = np.load(lidar_depth_path)
        lidar_depth_mask = lidar_depth['mask'].astype(np.bool_)
        lidar_depth_value = lidar_depth['value'].astype(np.float32)

        intrinsic = intrinsics[0]
        extrinsic = extrinsics[0]
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.
        image = torch.tensor(image).float().cuda().permute(2, 0, 1).unsqueeze(0) # (b, 3, h, w)
        _, _, orig_h, orig_w = image.shape
        area = orig_h * orig_w
        expected_area = 700 * 700
        expected_height, expected_width = int(orig_h * (expected_area / area) ** 0.5), int(orig_w * (expected_area / area) ** 0.5)
        image = torch.nn.functional.interpolate(image, (expected_height, expected_width), mode="bicubic", align_corners=False, antialias=True)        
        image = torch.clamp(image, 0, 1)

        fov_x = 2 * np.arctan(0.5 * orig_w / fx) / np.pi * 180

        output = moge_model.infer(image, fov_x=fov_x)
        
        pred_points, pred_depth, pred_intrinsics, pred_mask = output['points'], output['depth'], output['intrinsics'], output['mask']        
        pred_depth_original = torch.nn.functional.interpolate(pred_depth.unsqueeze(1), (orig_h, orig_w), mode='bilinear', align_corners=False, antialias=False).squeeze(1)
        pred_depth_original = pred_depth_original.squeeze(0).cpu().numpy() # (h, w)
        pred_depth_invalid = np.isnan(pred_depth_original) | np.isinf(pred_depth_original) 
        pred_depth_valid = np.logical_not(pred_depth_invalid)

        gt_depth = np.zeros_like(pred_depth_original).astype(np.float32)
        gt_depth[lidar_depth_mask] = lidar_depth_value
        pred_depth_aligned, a, b = recover_metric_depth(pred_depth_original, gt_depth, pred_depth_valid)
        xyzs = (pred_points * a + b).squeeze(0).cpu().numpy().reshape(-1, 3)
        rgbs = image.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        pred_mask = pred_mask.squeeze(0).cpu().numpy().reshape(-1)
        xyzs = xyzs[pred_mask]
        xyzs = transform_points_numpy(xyzs, extrinsic) # transform from camera space to vehicle space
        
        rgbs = rgbs[pred_mask]
        masks = np.ones_like(xyzs[:, 0]).astype(np.bool_)
        
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f'{frame_id:06d}']
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue
            
            ply_actor_path_lidar =  os.path.join(lidar_dir_actor, track_id, f'{frame_id:06d}.ply')
            if not os.path.exists(ply_actor_path_lidar):
                continue            
            
            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(frame_id)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
        
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            
            if np.sum(inbbox_mask) > 10:
                
                xyzs_inbbox = xyzs_actor[inbbox_mask]
                rgbs_inbbox = rgbs[inbbox_mask]
                masks_inbbox = masks[inbbox_mask]
                
                pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
                pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
                pointcloud_actor[track_id]['mask'].append(masks_inbbox)
                
                masks_inbbox = masks_inbbox[..., None]
                ply_actor_path = os.path.join(moge_dir_actor, track_id, f'{frame_id:06d}.ply')
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, masks_inbbox)
  
        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        masks_background = masks_background[..., None]
        ply_background_path = os.path.join(moge_dir_background, f'{frame_id:06d}.ply')
        
        storePly(ply_background_path, xyzs_background, rgbs_background, masks_background)

    for track_id, pointcloud in pointcloud_actor.items():
        try:
            xyzs = np.concatenate(pointcloud['xyz'], axis=0)
            rgbs = np.concatenate(pointcloud['rgb'], axis=0)
            masks = np.concatenate(pointcloud['mask'], axis=0)
            masks = masks[..., None]
            ply_actor_path_full = os.path.join(moge_dir_actor, track_id, 'full.ply')
            storePly(ply_actor_path_full, xyzs, rgbs, masks)
        except:
            pass # No pcd

def check_existing(scene_dir):
    image_dir = os.path.join(scene_dir, 'images')
    num_frames = len(os.listdir(image_dir)) // 5
    moge_background_dir = os.path.join(scene_dir, 'moge/background')
    num_pcds = len(os.listdir(moge_background_dir))
    if num_frames == num_pcds:
        return True
    else:
        return False
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./waymo_processor/training_set_processed')
    parser.add_argument('--skip_existing', action='store_true')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    scene_ids = sorted([x for x in os.listdir(data_dir)])
    for scene_id in scene_ids:
        print(f'Processing scene {scene_id}...')
        scene_dir = os.path.join(data_dir, scene_id)
        if args.skip_existing and check_existing(scene_dir):
            print(f'moge pcd exists for {scene_id}, skipping...')
            continue
        save_lidar(scene_dir)
        
if __name__ == '__main__':
    main()