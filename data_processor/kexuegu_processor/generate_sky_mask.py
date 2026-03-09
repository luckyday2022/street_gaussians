import os
import sys
from glob import glob

sys.path.append(os.getcwd())

import argparse
import imageio
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from termcolor import colored
from tqdm import tqdm

from kexuegu_helpers import image_filename_to_cam, image_filename_to_frame

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict

# Segment Anything
from segment_anything import build_sam, SamPredictor


def load_groundingdino(device: str):
    print(colored("Load Grounding DINO model", "green"))

    repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    cfg_filename = "GroundingDINO_SwinB.cfg.py"

    cache_cfg = hf_hub_download(repo_id=repo_id, filename=cfg_filename)
    model_args = SLConfig.fromfile(cache_cfg)
    model_args.device = device
    model = build_model(model_args)

    cache_ckpt = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
    checkpoint = torch.load(cache_ckpt, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded from {cache_ckpt}\n => {log}")
    model.eval()
    return model


def load_sam(sam_checkpoint: str):
    print(colored("Load SAM model", "green"))
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    return SamPredictor(sam)


def list_image_files(datadir: str):
    image_dir = os.path.join(datadir, "images")
    files = glob(os.path.join(image_dir, "*.jpg"))
    files += glob(os.path.join(image_dir, "*.png"))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No images found under {image_dir}")
    return files


def discover_cams(image_files):
    cams = sorted({image_filename_to_cam(path) for path in image_files})
    if len(cams) == 0:
        raise RuntimeError("No camera ids discovered from image filenames.")
    return cams


def build_cam_thresholds(cam_ids, box_thresholds):
    if len(box_thresholds) == 1:
        return {cam: float(box_thresholds[0]) for cam in cam_ids}

    if len(box_thresholds) != len(cam_ids):
        raise RuntimeError(
            f"--box_threshold expects 1 value or {len(cam_ids)} values for cams {cam_ids}, "
            f"but got {len(box_thresholds)} values."
        )

    return {cam: float(thr) for cam, thr in zip(cam_ids, box_thresholds)}


def update_mask_dict(masks_dict, mask_path):
    name = os.path.basename(mask_path)
    frame = image_filename_to_frame(name)
    cam = image_filename_to_cam(name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    if frame not in masks_dict:
        masks_dict[frame] = {}
    masks_dict[frame][cam] = mask


def save_preview_video(save_dir: str, masks_dict, ordered_cams, fps: int):
    if len(masks_dict) == 0:
        return

    frames = []
    for frame_id in sorted(masks_dict.keys()):
        row = []
        for cam in ordered_cams:
            mask = masks_dict[frame_id].get(cam, None)
            if mask is None:
                continue
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, f"cam={cam}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            row.append(vis)

        if len(row) == 0:
            continue

        merged = np.concatenate(row, axis=1)
        cv2.putText(
            merged,
            f"frame={frame_id}",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )
        frames.append(merged)

    if len(frames) == 0:
        return

    out_path = os.path.join(save_dir, "mask.mp4")
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Saved preview video: {out_path}")


def generate_sky_masks(
    datadir: str,
    cam_ids,
    box_thresholds,
    text_threshold: float,
    top_margin: int,
    ignore_exists: bool,
    skip_preview: bool,
    preview_fps: int,
    groundingdino_model,
    sam_predictor,
):
    save_dir = os.path.join(datadir, "sky_mask")
    os.makedirs(save_dir, exist_ok=True)

    image_files = list_image_files(datadir)
    all_cams = discover_cams(image_files)
    if cam_ids is None or len(cam_ids) == 0:
        selected_cams = all_cams
    else:
        selected_cams = sorted(set(cam_ids))
        missing = [cam for cam in selected_cams if cam not in all_cams]
        if len(missing) > 0:
            raise RuntimeError(f"Requested cams not found in dataset: {missing}, available: {all_cams}")

    threshold_map = build_cam_thresholds(selected_cams, box_thresholds)
    print(f"Selected cams: {selected_cams}")
    print(f"Per-cam box_threshold: {threshold_map}")

    masks_dict = {}
    processed = 0
    skipped = 0

    for image_path in tqdm(image_files, desc="Sky mask"):
        image_name = os.path.basename(image_path)
        cam = image_filename_to_cam(image_name)
        if cam not in selected_cams:
            continue

        output_mask = os.path.join(save_dir, image_name)
        if ignore_exists and os.path.exists(output_mask):
            update_mask_dict(masks_dict, output_mask)
            skipped += 1
            continue

        box_threshold = threshold_map[cam]
        image_source, image = load_image(image_path)
        boxes, logits, _ = predict(
            model=groundingdino_model,
            image=image,
            caption="sky",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if boxes.shape[0] != 0:
            h, w, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
            boxes_xyxy = boxes_xyxy[boxes_xyxy[:, 1] < float(top_margin)]
        else:
            boxes_xyxy = []

        if len(boxes_xyxy) == 0:
            mask = np.zeros_like(image_source[..., 0], dtype=np.uint8)
        else:
            sam_predictor.set_image(image_source)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            mask_union = torch.zeros_like(masks[0, 0]).bool()
            for m in masks[:, 0]:
                mask_union = mask_union | m.bool()
            mask = (mask_union.cpu().numpy().astype(np.uint8)) * 255

        cv2.imwrite(output_mask, mask)
        update_mask_dict(masks_dict, output_mask)
        processed += 1

    if not skip_preview:
        print("Saving sky mask preview video")
        save_preview_video(save_dir, masks_dict, selected_cams, preview_fps)

    print(f"Done. generated={processed}, skipped_existing={skipped}, output={save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sky masks for kexuegu processed data.")
    parser.add_argument("--datadir", required=True, type=str, help="Scene dir, e.g. .../training_set_processed/000")
    parser.add_argument("--sam_checkpoint", required=True, type=str, help="Path to SAM checkpoint .pth")
    parser.add_argument(
        "--cams",
        nargs="*",
        type=int,
        default=None,
        help="Camera ids to process (default: auto all cams from images).",
    )
    parser.add_argument(
        "--box_threshold",
        nargs="+",
        type=float,
        default=[0.3],
        help="One value for all cams, or one value per selected cam in cam order.",
    )
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--top_margin", type=int, default=100, help="Only keep detected sky boxes near image top.")
    parser.add_argument("--ignore_exists", action="store_true")
    parser.add_argument("--skip_preview", action="store_true")
    parser.add_argument("--preview_fps", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    groundingdino_model = load_groundingdino(device="cpu")
    sam_predictor = load_sam(args.sam_checkpoint)
    generate_sky_masks(
        datadir=args.datadir,
        cam_ids=args.cams,
        box_thresholds=args.box_threshold,
        text_threshold=args.text_threshold,
        top_margin=args.top_margin,
        ignore_exists=args.ignore_exists,
        skip_preview=args.skip_preview,
        preview_fps=args.preview_fps,
        groundingdino_model=groundingdino_model,
        sam_predictor=sam_predictor,
    )


if __name__ == "__main__":
    main()
