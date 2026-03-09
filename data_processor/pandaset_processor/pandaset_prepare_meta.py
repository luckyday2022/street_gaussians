import sys
import os
import numpy as np
import cv2
import argparse
import imageio
from tqdm import tqdm
sys.path.append(os.getcwd())
import json
from pandaset_helpers import *


# build training set
# sample data at 2Hz (5frames)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/lpai/volumes/jointmodel/yanyunzhi/data/pandaset/pandaset_processed')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--postfix', type=str, default=None)

    args = parser.parse_args()
    root_dir = args.root_dir
    if args.split == 'train':
        data_dir = os.path.join(root_dir, 'pandaset_processed')
        save_path = 'meta_info_train.json'
    elif args.split == 'val':
        data_dir = os.path.join(root_dir, 'pandaset_processed')
        save_path = 'meta_info_val.json'
    else:
        raise ValueError(f'Unknown split {args.split}') # test set not used
    
    if args.postfix is not None:
        save_path = save_path.replace('.json', f'_{args.postfix}.json')
    
    if args.split == 'train':
        scene_ids = sorted([x for x in os.listdir(data_dir)])
        scene_ids = [int(x) for x in scene_ids if x != '054']
        scene_ids = [x for x in scene_ids if x not in [1, 11, 16, 53, 158]]
    elif args.split == 'val':
        scene_ids = [1, 11, 16, 53, 158]
    else:
        raise ValueError(f'Unknown split {args.split}') # test set not used

    meta_infos = list()
    for scene_idx in scene_ids:
        scene_dir = os.path.join(data_dir, str(scene_idx).zfill(3))
        lidar_render_dir = os.path.join(scene_dir, 'lidar_forward/color_render')
        if args.postfix is not None:
            lidar_render_dir = lidar_render_dir.replace('color_render', f'color_render_{args.postfix}')
        image_dir = os.path.join(scene_dir, 'images')
        for start_frame in range(0, NUM_FRAMES, 5):
            end_frame = start_frame + 25
            if end_frame > NUM_FRAMES:
                end_frame = NUM_FRAMES
                start_frame = end_frame - 25

            samples = dict()
            samples['frames'] = list()
            samples['guidances'] = list()
            samples['guidances_mask'] = list()

            for frame in range(start_frame, end_frame):
                image_path = os.path.join(image_dir, f'{frame:03d}_0.jpg')
                guidance_path = os.path.join(lidar_render_dir, f'{frame:03d}_0.png')
                guidance_mask_path = os.path.join(lidar_render_dir, f'{frame:03d}_0_mask.png')
                assert os.path.exists(image_path)
                assert os.path.exists(guidance_path)
                assert os.path.exists(guidance_mask_path)
                samples['frames'].append(os.path.relpath(image_path, root_dir))
                samples['guidances'].append(os.path.relpath(guidance_path, root_dir))
                samples['guidances_mask'].append(os.path.relpath(guidance_mask_path, root_dir))      
            meta_infos.append(samples)
      
    meta_info_path = os.path.join(root_dir, save_path)
    json.dump(meta_infos, open(meta_info_path, 'w'), indent=1)

            
if __name__ == '__main__':
    main()