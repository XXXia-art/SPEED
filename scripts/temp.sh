#!/bin/bash

baseline="SPEED"
params="V"
aug_num=10
threshold="1e-1"
save_root="logs/${baseline}/seq_500_instance"
erase_type="instance"
anchor=" "

TRAIN_GPU=1
EVAL_GPU_IDX=('2' '3')
NUM_EVAL_GPUS=${#EVAL_GPU_IDX[@]}
EVAL_INTERVAL=50
EVAL_BATCH_SIZE=10






for i in "${!instances[@]}"; do
  current="${instances[$i]}"
  idx=$((i + 1))
  current_dir="${current}${idx}"
  
  targets=$(printf "%s, " "${instances[@]:0:$((i + 1))}")
  targets=${targets%, }

  echo "Sequential edit step $((i + 1)) / ${#instances[@]} : ${current}"

  # ---------- 构造 edit_ckpt 参数（用 array，避免空格炸掉） ----------
  EXTRA_ARGS=()
  if [ "$i" -gt 0 ]; then
    prev="${instances[$((i - 1))]}"
    prev_ckpt="${save_root}/${erase_type}/${prev}/weight.pt"
    echo "[INFO] Load previous edit: ${prev_ckpt}"
    EXTRA_ARGS+=(--edit_ckpt "$prev_ckpt")
  fi
  # ------------------------------------------------------------------

  CUDA_VISIBLE_DEVICES=$TRAIN_GPU python train_erase_null.py \
    --target_concepts "$targets" \
    --anchor_concepts "$anchor" \
    --retain_path "data/${erase_type}.csv" \
    --heads "concept" \
    --save_path "${save_root}/${erase_type}/${current}" \
    --file_name "weight" \
    --params "$params" \
    --aug_num "$aug_num" \
    --threshold "$threshold" \
    "${EXTRA_ARGS[@]}"

  if (((i + 1) % EVAL_INTERVAL == 0)); then
    eval_gpu=${EVAL_GPU_IDX[$(( (i / EVAL_INTERVAL) % NUM_EVAL_GPUS ))]}
    (
      CUDA_VISIBLE_DEVICES=$eval_gpu python sample2.py \
        --erase_type "coco" \
        --target_concept "$current" \
        --contents "coco" \
        --mode "edit" \
        --num_samples 1 \
        --batch_size "$EVAL_BATCH_SIZE" \
        --save_root "${save_root}/${erase_type}" \
        --edit_ckpt "${save_root}/${erase_type}/${current}/weight.pt"

      CUDA_VISIBLE_DEVICES=$eval_gpu python src/clip_score_cal.py \
        --contents "coco" \
        --root_path "${save_root}/${erase_type}"
    ) &
  fi
done

wait
