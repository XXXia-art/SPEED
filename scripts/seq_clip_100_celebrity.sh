#!/bin/bash

set -e

baseline="SPEED"
erase_type="celebrity"
root="logs/${baseline}/seq_multi_100_coco/100_celebrity"

# 使用一张空闲 GPU 来算 CLIP
export CUDA_VISIBLE_DEVICES=3

echo "[INFO] Start scoring for 100 celebrity..."
echo "[INFO] Root path: ${root}"

python src/clip_score_cal.py \
  --contents "coco" \
  --root_path "${root}" \
  --pretrained_path "coco"

echo "[DONE] Scoring finished for 100 celebrity."
