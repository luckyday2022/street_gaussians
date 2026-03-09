## Kexuegu Dataset Processor (Waymo-style Pipeline)

该目录是一个**独立处理器**，流程对齐 `waymo_processor`：
1. `kexuegu_converter.py`：生成 `ego_pose / intrinsics / extrinsics / images / track / dynamic_mask / timestamps`
2. `kexuegu_get_lidar_pcd.py`：生成 `lidar/background/*.ply` 与 `lidar/depth/*.npz`
3. `kexuegu_render_lidar_pcd.py`：渲染聚合 LiDAR 到 `lidar/color_render`
4. `kexuegu_prepare_meta.py`：生成 `meta_info_train.json`
5. `generate_sky_mask.py`：可选生成 `sky_mask/*.png`

特性：
- 默认自动发现并处理**全部相机**（当前数据为 8 相机：`0..7`）。
- 支持按原始帧号范围处理（如 `600-700`）。
- 全部逻辑在本目录，不从原 `waymo_processor` 直接导入。

## 默认全量（8 相机）
### 1) Convert
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_converter.py \
  --root_dir data/20260113 \
  --save_dir data/20260113/waymo_format/training_set_processed \
  --process_list pose calib image track dynamic
```

### 2) LiDAR PCD + Depth
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_get_lidar_pcd.py \
  --root_dir data/20260113 \
  --save_dir data/20260113/waymo_format/training_set_processed
```

### 3) Render LiDAR
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_render_lidar_pcd.py \
  --data_dir data/20260113/waymo_format/training_set_processed \
  --delta_frames 10 \
  --shifts 0
```

### 4) Prepare Meta
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_prepare_meta.py \
  --root_dir data/20260113/waymo_format \
  --split train
```

### 5) Generate Sky Mask (Optional)
该步骤仅在需要 sky supervision 时使用（例如在 `street_gaussians` 中设置 `include_sky: true` 且 `lambda_sky > 0`）。

先安装 GroundingDINO，并下载 SAM checkpoint，然后执行：

```bash
python street_crafter/data_processor/kexuegu_processor/generate_sky_mask.py \
  --datadir data/20260113/waymo_format/training_set_processed/000 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --cams 0 1 2 3 4 5 6 7 \
  --box_threshold 0.3
```

说明：
- `--box_threshold` 传 1 个值时对全部相机生效；也可按相机顺序传多个值（数量需等于 `--cams` 数量）。
- 结果输出到 `.../sky_mask/*.png`，文件名与 `images/*.png` 同名。

## 指定 8 相机 + 指定帧范围（示例：600-700）
### 1) Convert
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_converter.py \
  --root_dir data/20260113 \
  --save_dir data/20260113/waymo_format/training_set_processed \
  --cams 0 1 2 3 4 5 6 7 \
  --start_frame_id 600 --end_frame_id 700 \
  --process_list pose calib image track dynamic
```

### 2) LiDAR PCD + Depth
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_get_lidar_pcd.py \
  --root_dir data/20260113 \
  --save_dir data/20260113/waymo_format/training_set_processed \
  --cams 0 1 2 3 4 5 6 7 \
  --start_frame_id 600 --end_frame_id 700
```

### 3) Render LiDAR
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_render_lidar_pcd.py \
  --data_dir data/20260113/waymo_format/training_set_processed \
  --cams 0 1 2 3 4 5 6 7 \
  --delta_frames 10 \
  --shifts 0 \
  --start_frame_id 600 --end_frame_id 700
```

### 4) Prepare Meta（8 相机 + 600-700）
```bash
python street_crafter/data_processor/kexuegu_processor/kexuegu_prepare_meta.py \
  --root_dir data/20260113/waymo_format \
  --split train \
  --cam_ids 0 1 2 3 4 5 6 7 \
  --start_frame_id 600 --end_frame_id 700
```

## 参数说明
- `--cams`: 处理/渲染的相机 ID 列表，默认全部自动发现相机。
- `--cam_ids`: `prepare_meta` 使用的相机 ID 列表，默认全部相机。
- `--start_frame_id --end_frame_id`: 按原始帧号筛选范围（闭区间）。

## 输出结构
```text
data/20260113/waymo_format/
├── training_set_processed/
│   └── 000/
│       ├── images/
│       ├── ego_pose/
│       ├── intrinsics/
│       ├── extrinsics/
│       ├── dynamic_mask/
│       ├── sky_mask/   # optional
│       ├── track/
│       └── lidar/
│           ├── background/
│           ├── actor/
│           ├── depth/
│           └── color_render/
└── meta_info_train.json
```

## Notes
- 当前 `track` 为占位空轨迹（无 3D 跟踪标注时的兼容方案）。
- 当前 `dynamic_mask` 为全零（可后续替换）。
- 图像先做鱼眼去畸变，再写入 `images/*.png`，与保存的 `intrinsics/*.txt` 对应。

## 一键执行
- bash data_processor/kexuegu_processor/run_kexuegu_pipeline.sh \
  --root_dir data/20260113 \
  --save_dir data/20260113/waymo_format/training_set_processed \
  --cams "0 1 2 3 4 5 6 7" \
  --start_frame_id 650 --end_frame_id 700

