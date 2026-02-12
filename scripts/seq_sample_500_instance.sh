#!/bin/bash

baseline="SPEED"
save_root="logs/${baseline}/seq_500_instance"
erase_type="instance"

STEP_INTERVAL=10
MAX_STEP=99        # 你训练到的最大 step（自己改）

EVAL_GPUS=(0 1 2)
NUM_EVAL_GPUS=${#EVAL_GPUS[@]}
EVAL_BATCH_SIZE=10


for ((step_id=0; step_id<=MAX_STEP; step_id+=STEP_INTERVAL)); do
  step_name=$(printf "step_%03d" "$step_id")
  ckpt="${save_root}/${erase_type}/${step_name}/weight.pt"

  # 如果该 step 没训练完，直接跳过（非常重要）
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] ${step_name} not found"
    continue
  fi


  gpu=${EVAL_GPUS[$(( (step_id / STEP_INTERVAL) % NUM_EVAL_GPUS ))]}

  echo "[SAMPLE] step=${step_name}  GPU=${gpu}"

  CUDA_VISIBLE_DEVICES=$gpu python sample2.py \
    --erase_type "instance" \
    --contents "coco" \
    --mode "edit" \
    --num_samples 1 \
    --batch_size "$EVAL_BATCH_SIZE" \
    --save_root "${save_root}/${erase_type}/${step_name}" \
    --edit_ckpt "$ckpt" &
  
  # 控制并行
  if (( (step_id / STEP_INTERVAL + 1) % NUM_EVAL_GPUS == 0 )); then
    wait
  fi
done

wait
echo "✅ Sampling finished"
