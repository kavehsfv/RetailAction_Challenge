#!/usr/bin/env bash
# ==============================================================================
# Full End-to-End Training Recipe for RetailAction V4 (0.7942 Public Leaderboard)
# Authors: Kaveh Safavigerdini, Kannappan Palaniappan
# Affiliation: EECS Department, University of Missouri
# ==============================================================================
# Usage:
#   # Run final V4 stage (from v3b_256 checkpoint) if weights/v3b_256_best.pt exists:
#   bash train_full_pipeline.sh final_v4
#
#   # Or run the complete multi-stage progressive curriculum from scratch:
#   bash train_full_pipeline.sh all
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/src"

MODE="${1:-final_v4}"
PYTHON="${PYTHON:-python}"

echo "[RetailAction V4 Training Pipeline] Mode: $MODE | Working dir: $(pwd)"

if [[ "$MODE" == "all" ]]; then
  echo "=== Stage 1: Base MultiViewActionDETR + Quality Head (run2q, 12 epochs, 224x224) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage1_run2q \
    --epochs 12 --batch 4 --accum 4 \
    --size 224 --grid-hw 7 --frames 16 \
    --lr 1e-4 --backbone-lr 1e-5 --wd 1e-4 \
    --no-obj --no-aux --all-cls \
    --seed 42

  echo "=== Stage 2: Temporal DIoU + ModelEma + WD-Hygiene (ab_diou_ema, 4 epochs, 224x224) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage2_ab_diou_ema \
    --init outputs/stage1_run2q/best.pt \
    --epochs 4 --batch 8 --accum 4 \
    --size 224 --grid-hw 7 --frames 16 \
    --lr 5e-5 --backbone-lr 5e-6 --wd 1e-4 \
    --diou --ema --ema-decay 0.999 --wd-hygiene \
    --no-obj --no-aux --all-cls \
    --seed 1

  echo "=== Stage 3: STMixer Objectness/Class Split + Per-Layer Fresh Re-Matching (stmixer_ft, 3 epochs, 224x224) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage3_stmixer_ft \
    --init outputs/stage2_ab_diou_ema/best.pt \
    --epochs 3 --batch 8 --accum 4 \
    --size 224 --grid-hw 7 --frames 16 \
    --lr 5e-5 --backbone-lr 5e-6 --wd 1e-4 \
    --diou --ema --ema-decay 0.999 --wd-hygiene \
    --obj-weight 2.0 --aux-weight 0.4 --new-head-lr 5e-4 \
    --seed 1

  echo "=== Stage 4: PointTAD Dense Per-Frame Supervision + Quality Calibration (combine_v2a, 4 epochs, 224x224) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage4_combine_v2a \
    --init outputs/stage3_stmixer_ft/last.pt --init-ema \
    --epochs 4 --batch 8 --accum 4 \
    --size 224 --grid-hw 7 --frames 16 \
    --lr 4e-5 --backbone-lr 4e-6 --wd 1e-4 \
    --diou --ema --ema-decay 0.9985 --wd-hygiene \
    --obj-weight 2.0 --aux-weight 0.4 \
    --dense-head --dense-weight 3.0 --q-weight 4.0 --new-head-lr 5e-4 \
    --val-rank "obj*q" \
    --seed 42

  echo "=== Stage 5a: Hand-Pose Tokens + All-Query IoU Head + Log-Duration Loss (combine_v3a, 4 epochs, 224x224) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage5a_combine_v3a \
    --init outputs/stage4_combine_v2a/best.pt --init-ema \
    --epochs 4 --batch 16 --accum 2 \
    --size 224 --grid-hw 7 --frames 16 \
    --lr 3e-5 --backbone-lr 3e-6 --wd 1e-4 \
    --diou --ema --ema-decay 0.998 --wd-hygiene \
    --obj-weight 2.0 --aux-weight 0.4 \
    --dense-head --dense-weight 3.0 --q-weight 4.0 \
    --iou-weight 3.0 --iou-rank-weight 1.0 \
    --use-pose-tokens --log-w-weight 0.2 \
    --new-head-lr 5e-4 --grad-ckpt \
    --val-rank "obj*q" \
    --seed 42

  echo "=== Stage 5b: High-Resolution 256x256 (grid_hw=8) Curriculum (combine_v3b_256, 2 epochs) ==="
  $PYTHON -m retailaction.train \
    --out outputs/stage5b_combine_v3b_256 \
    --init outputs/stage5a_combine_v3a/best.pt --init-ema \
    --epochs 2 --batch 16 --accum 2 \
    --size 256 --grid-hw 8 --frames 16 \
    --lr 1.5e-5 --backbone-lr 2e-6 --wd 1e-4 \
    --diou --ema --ema-decay 0.998 --wd-hygiene \
    --obj-weight 2.0 --aux-weight 0.4 \
    --dense-head --dense-weight 3.0 --q-weight 4.0 \
    --iou-weight 3.0 --iou-rank-weight 1.0 \
    --use-pose-tokens --log-w-weight 0.2 \
    --new-head-lr 1e-4 --grad-ckpt \
    --val-rank "obj*q" \
    --seed 43

  V3B_INIT="outputs/stage5b_combine_v3b_256/best.pt"
else
  V3B_INIT="${V3B_INIT:-outputs/combine_v3b_256/best.pt}"
fi

echo "=== Stage 6a (V4): Paired 256x256 (grid_hw=8) Continuation (v4_control256, 2 epochs) ==="
$PYTHON -m retailaction.train \
  --out outputs/v4_control256 \
  --init "$V3B_INIT" \
  --epochs 2 --batch 16 --accum 2 --workers 2 \
  --size 256 --grid-hw 8 --frames 16 \
  --lr 1e-5 --backbone-lr 1.5e-6 --new-head-lr 5e-5 --wd 1e-4 \
  --diou --ema --ema-decay 0.998 --wd-hygiene \
  --obj-weight 2.0 --aux-weight 0.4 \
  --dense-head --dense-weight 3.0 --q-weight 4.0 \
  --iou-weight 3.0 --iou-rank-weight 1.0 \
  --use-pose-tokens --log-w-weight 0.2 \
  --grad-ckpt --val-limit 400 --val-rank "obj*q" \
  --seed 44

echo "=== Stage 6b (V4): Paired 288x288 (grid_hw=9) Continuation (v4_high288, 2 epochs) ==="
$PYTHON -m retailaction.train \
  --out outputs/v4_high288 \
  --init "$V3B_INIT" \
  --epochs 2 --batch 16 --accum 2 --workers 2 \
  --size 288 --grid-hw 9 --frames 16 \
  --lr 1e-5 --backbone-lr 1.5e-6 --new-head-lr 5e-5 --wd 1e-4 \
  --diou --ema --ema-decay 0.998 --wd-hygiene \
  --obj-weight 2.0 --aux-weight 0.4 \
  --dense-head --dense-weight 3.0 --q-weight 4.0 \
  --iou-weight 3.0 --iou-rank-weight 1.0 \
  --use-pose-tokens --log-w-weight 0.2 \
  --grad-ckpt --val-limit 400 --val-rank "obj*q" \
  --seed 44

echo "=== Stage 7 (V4): Generate Submission CSV ==="
cd "$ROOT"
$PYTHON reproduce_submission.py \
  --ckpt-256 src/outputs/v4_control256/best.pt \
  --ckpt-288 src/outputs/v4_high288/best.pt \
  --out submission.csv
