#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=""
OUT_ROOT=""
SAVE_DIR_OVERRIDE=""
SCENE_ID="0"
CAMS_STR="0 1 2 3 4 5 6 7"
START_FRAME_ID=""
END_FRAME_ID=""
DELTA_FRAMES="10"
SHIFTS_STR="0"
SPLIT="train"
UNDISTORT_BALANCE="0.0"
POSTFIX=""
SKIP_EXISTING=0

usage() {
  cat <<USAGE
Usage:
  bash $0 [options]

Options:
  --root_dir PATH           Raw dataset root. Default: auto-detect data/20260113
  --out_root PATH           Output root. Default: <root_dir>/waymo_format
  --save_dir PATH           Full processed dir (e.g. .../training_set_processed), overrides --out_root
  --scene_id INT            Scene id folder name. Default: 0
  --cams "LIST"             Camera ids (space-separated). Default: "0 1 2 3 4 5 6 7"
  --start_frame_id INT      Optional start frame id (inclusive)
  --end_frame_id INT        Optional end frame id (inclusive)
  --delta_frames INT        Render aggregation window. Default: 10
  --shifts "LIST"           Render shift values (space-separated). Default: "0"
  --split train|val         Meta split. Default: train
  --undistort_balance FLOAT Fisheye undistort balance. Default: 0.0
  --postfix STR             Optional postfix for meta json name
  --skip_existing           Skip already-generated outputs when possible
  -h, --help               Show this help

Examples:
  1) Full 8-camera pipeline on all frames:
     bash $0

  2) 8-camera frames 600-700:
     bash $0 --cams "0 1 2 3 4 5 6 7" --start_frame_id 600 --end_frame_id 700
USAGE
}

to_abs_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    echo "$p"
  else
    echo "$PWD/$p"
  fi
}

auto_detect_root_dir() {
  local candidates=(
    "$PWD/data/20260113"
    "$SCRIPT_DIR/../../data/20260113"
    "$SCRIPT_DIR/../../../data/20260113"
    "$SCRIPT_DIR/../../../../data/20260113"
  )
  for c in "${candidates[@]}"; do
    if [[ -d "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --scene_id)
      SCENE_ID="$2"
      shift 2
      ;;
    --cams)
      CAMS_STR="$2"
      shift 2
      ;;
    --start_frame_id)
      START_FRAME_ID="$2"
      shift 2
      ;;
    --end_frame_id)
      END_FRAME_ID="$2"
      shift 2
      ;;
    --delta_frames)
      DELTA_FRAMES="$2"
      shift 2
      ;;
    --shifts)
      SHIFTS_STR="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --undistort_balance)
      UNDISTORT_BALANCE="$2"
      shift 2
      ;;
    --postfix)
      POSTFIX="$2"
      shift 2
      ;;
    --skip_existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ROOT_DIR" ]]; then
  if ROOT_DIR="$(auto_detect_root_dir)"; then
    :
  else
    echo "Failed to auto-detect root_dir. Please pass --root_dir explicitly."
    exit 1
  fi
fi
ROOT_DIR="$(to_abs_path "$ROOT_DIR")"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "root_dir does not exist: $ROOT_DIR"
  exit 1
fi

if [[ -n "$SAVE_DIR_OVERRIDE" ]]; then
  SAVE_DIR="$(to_abs_path "$SAVE_DIR_OVERRIDE")"
  OUT_ROOT="$(dirname "$SAVE_DIR")"
else
  if [[ -z "$OUT_ROOT" ]]; then
    OUT_ROOT="$ROOT_DIR/waymo_format"
  fi
  OUT_ROOT="$(to_abs_path "$OUT_ROOT")"
  SAVE_DIR="$OUT_ROOT/training_set_processed"
fi

read -r -a CAMS <<< "$CAMS_STR"
read -r -a SHIFTS <<< "$SHIFTS_STR"

RANGE_ARGS=()
if [[ -n "$START_FRAME_ID" ]]; then
  RANGE_ARGS+=(--start_frame_id "$START_FRAME_ID")
fi
if [[ -n "$END_FRAME_ID" ]]; then
  RANGE_ARGS+=(--end_frame_id "$END_FRAME_ID")
fi

SKIP_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  SKIP_ARGS+=(--skip_existing)
fi

META_POSTFIX_ARGS=()
if [[ -n "$POSTFIX" ]]; then
  META_POSTFIX_ARGS+=(--postfix "$POSTFIX")
fi

echo "[1/4] Converting images/poses/calib/track/dynamic"
python "$SCRIPT_DIR/kexuegu_converter.py" \
  --root_dir "$ROOT_DIR" \
  --save_dir "$SAVE_DIR" \
  --scene_id "$SCENE_ID" \
  --cams "${CAMS[@]}" \
  "${RANGE_ARGS[@]}" \
  --undistort_balance "$UNDISTORT_BALANCE" \
  --process_list pose calib image track dynamic \
  "${SKIP_ARGS[@]}"

echo "[2/4] Building LiDAR background/depth"
python "$SCRIPT_DIR/kexuegu_get_lidar_pcd.py" \
  --root_dir "$ROOT_DIR" \
  --save_dir "$SAVE_DIR" \
  --scene_id "$SCENE_ID" \
  --cams "${CAMS[@]}" \
  "${RANGE_ARGS[@]}" \
  "${SKIP_ARGS[@]}"

echo "[3/4] Rendering LiDAR conditions"
python "$SCRIPT_DIR/kexuegu_render_lidar_pcd.py" \
  --data_dir "$SAVE_DIR" \
  --scene_ids "$SCENE_ID" \
  --cams "${CAMS[@]}" \
  --delta_frames "$DELTA_FRAMES" \
  --shifts "${SHIFTS[@]}" \
  "${RANGE_ARGS[@]}" \
  "${SKIP_ARGS[@]}"

echo "[4/4] Preparing meta json"
python "$SCRIPT_DIR/kexuegu_prepare_meta.py" \
  --root_dir "$OUT_ROOT" \
  --split "$SPLIT" \
  --scene_ids "$SCENE_ID" \
  --cam_ids "${CAMS[@]}" \
  "${RANGE_ARGS[@]}" \
  "${META_POSTFIX_ARGS[@]}"

echo "Done. Outputs under: $OUT_ROOT"
