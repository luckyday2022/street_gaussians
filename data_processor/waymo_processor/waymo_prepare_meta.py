import sys
import os
import numpy as np
import cv2
import argparse
import imageio
from tqdm import tqdm
sys.path.append(os.getcwd())
from waymo_helpers import load_ego_poses, load_calibration, load_track, \
    image_filename_to_cam, image_filename_to_frame, image_heights, image_widths
import json

# build training set
# sample data at 2Hz (5frames)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/lpai/volumes/jointmodel/yanyunzhi/data/waymo')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--scene_ids', type=int, nargs='+', default=None)
    parser.add_argument('--postfix', type=str, default=None)
    
    args = parser.parse_args()

    root_dir = args.root_dir
    if args.split == 'train':
        data_dir = os.path.join(root_dir, 'training_set_processed')
        save_path = 'meta_info_train.json'
    elif args.split == 'val':
        data_dir = os.path.join(root_dir, 'validation_set_processed')
        save_path = 'meta_info_val.json'
    else:
        raise ValueError(f'Unknown split {args.split}') # test set not used
    
    
    if args.postfix is not None:
        save_path = save_path.replace('.json', f'_{args.postfix}.json')
    
    if args.scene_ids is None:
        scene_ids = sorted([x for x in os.listdir(data_dir)])
        scene_ids = [int(x) for x in scene_ids]
    else:
        scene_ids = args.scene_ids
    
    meta_infos = list()
    for scene_idx in scene_ids:
        scene_dir = os.path.join(data_dir, str(scene_idx).zfill(3))
        num_frames = len(os.listdir(os.path.join(scene_dir, 'images'))) // 5
                
        lidar_render_dir = os.path.join(scene_dir, 'lidar/color_render')
        if args.postfix is not None:
            lidar_render_dir = lidar_render_dir.replace('color_render', f'color_render_{args.postfix}')
        
        image_dir = os.path.join(scene_dir, 'images')
        for start_frame in range(0, num_frames, 5):
            end_frame = start_frame + 25
            if end_frame >= num_frames:
                continue
            
            samples = dict()
            samples['frames'] = list()
            samples['guidances'] = list()
            samples['guidances_mask'] = list()

            for frame in range(start_frame, end_frame):
                image_path = os.path.join(image_dir, f'{frame:06d}_0.png')
                guidance_path = os.path.join(lidar_render_dir, f'{frame:06d}_0.png')
                guidance_mask_path = os.path.join(lidar_render_dir, f'{frame:06d}_0_mask.png')
                assert os.path.exists(image_path), f'{image_path} does not exist'
                assert os.path.exists(guidance_path), f'{guidance_path} does not exist'
                assert os.path.exists(guidance_mask_path), f'{guidance_mask_path} does not exist'
                samples['frames'].append(os.path.relpath(image_path, root_dir))
                samples['guidances'].append(os.path.relpath(guidance_path, root_dir))
                samples['guidances_mask'].append(os.path.relpath(guidance_mask_path, root_dir))      
            meta_infos.append(samples)
        
    meta_info_path = os.path.join(root_dir, save_path)
    json.dump(meta_infos, open(meta_info_path, 'w'), indent=1)

            
if __name__ == '__main__':
    main()