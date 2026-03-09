import argparse
import json
import os
from typing import List, Optional, Tuple


def parse_image_name(name: str) -> Optional[Tuple[int, int]]:
    if not name.endswith(".png"):
        return None
    stem = os.path.splitext(name)[0]
    toks = stem.split("_")
    if len(toks) != 2:
        return None
    if not toks[0].isdigit() or not toks[1].isdigit():
        return None
    return int(toks[0]), int(toks[1])


def collect_frame_cam_pairs(image_dir: str):
    pairs = []
    for name in os.listdir(image_dir):
        parsed = parse_image_name(name)
        if parsed is None:
            continue
        pairs.append(parsed)
    return pairs


def filter_range(frame_ids: List[int], start_frame_id: Optional[int], end_frame_id: Optional[int]) -> List[int]:
    if len(frame_ids) == 0:
        return frame_ids

    start = start_frame_id if start_frame_id is not None else frame_ids[0]
    end = end_frame_id if end_frame_id is not None else frame_ids[-1]
    if end < start:
        raise RuntimeError(f"Invalid frame range: start_frame_id={start}, end_frame_id={end}")

    return [f for f in frame_ids if start <= f <= end]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare StreetCrafter meta info json for kexuegu data.")
    parser.add_argument("--root_dir", type=str, default="data/20260113/waymo_format")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--scene_ids", type=int, nargs="+", default=None)
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--cam_ids", type=int, nargs="+", default=None, help="Camera ids in meta, default: all cams found")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--start_frame_id", type=int, default=None)
    parser.add_argument("--end_frame_id", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.split == "train":
        data_dir = os.path.join(args.root_dir, "training_set_processed")
        save_name = "meta_info_train.json"
    else:
        data_dir = os.path.join(args.root_dir, "validation_set_processed")
        save_name = "meta_info_val.json"

    if args.postfix:
        save_name = save_name.replace(".json", f"_{args.postfix}.json")

    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory does not exist: {data_dir}")

    if args.scene_ids is None:
        scene_ids = sorted([int(x) for x in os.listdir(data_dir) if x.isdigit()])
    else:
        scene_ids = args.scene_ids

    meta_infos = []

    for scene_id in scene_ids:
        scene_dir = os.path.join(data_dir, f"{scene_id:03d}")
        image_dir = os.path.join(scene_dir, "images")
        lidar_render_dir = os.path.join(scene_dir, "lidar", "color_render")
        if args.postfix:
            lidar_render_dir = lidar_render_dir.replace("color_render", f"color_render_{args.postfix}")

        if not os.path.exists(image_dir):
            print(f"Warning: missing image_dir, skip scene {scene_id:03d}")
            continue
        if not os.path.exists(lidar_render_dir):
            print(f"Warning: missing lidar_render_dir, skip scene {scene_id:03d}")
            continue

        pairs = collect_frame_cam_pairs(image_dir)
        if len(pairs) == 0:
            print(f"Warning: no images found, skip scene {scene_id:03d}")
            continue

        cam_to_frames = {}
        for frame_id, cam_id in pairs:
            cam_to_frames.setdefault(cam_id, []).append(frame_id)

        available_cam_ids = sorted(cam_to_frames.keys())
        if args.cam_ids is None:
            cam_ids = available_cam_ids
        else:
            missing = sorted([c for c in args.cam_ids if c not in available_cam_ids])
            if len(missing) > 0:
                raise RuntimeError(
                    f"Scene {scene_id:03d} missing requested cams {missing}, available cams: {available_cam_ids}"
                )
            cam_ids = sorted(args.cam_ids)

        for cam_id in cam_ids:
            frame_ids = sorted(set(cam_to_frames[cam_id]))
            frame_ids = filter_range(frame_ids, args.start_frame_id, args.end_frame_id)
            if len(frame_ids) < args.window:
                continue

            for start_idx in range(0, len(frame_ids), args.stride):
                end_idx = start_idx + args.window
                if end_idx > len(frame_ids):
                    continue

                window_frames = frame_ids[start_idx:end_idx]
                sample = {"frames": [], "guidances": [], "guidances_mask": []}

                valid = True
                for frame_id in window_frames:
                    image_path = os.path.join(image_dir, f"{frame_id:06d}_{cam_id}.png")
                    guidance_path = os.path.join(lidar_render_dir, f"{frame_id:06d}_{cam_id}.png")
                    guidance_mask_path = os.path.join(lidar_render_dir, f"{frame_id:06d}_{cam_id}_mask.png")

                    if not (os.path.exists(image_path) and os.path.exists(guidance_path) and os.path.exists(guidance_mask_path)):
                        valid = False
                        break

                    sample["frames"].append(os.path.relpath(image_path, args.root_dir))
                    sample["guidances"].append(os.path.relpath(guidance_path, args.root_dir))
                    sample["guidances_mask"].append(os.path.relpath(guidance_mask_path, args.root_dir))

                if valid:
                    meta_infos.append(sample)

    save_path = os.path.join(args.root_dir, save_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(meta_infos, f, indent=1)

    print(f"Saved {len(meta_infos)} samples to {save_path}")


if __name__ == "__main__":
    main()
