#!/bin/bash

baseline="SPEED"
params="V"
aug_num=10
threshold="1e-4"
retain_scale="0.05"
save_root="logs/${baseline}/seq_multi_100_coco"
erase_type="100_celebrity"
anchor="person"

TRAIN_GPU=1
EVAL_GPU_IDX=('2' '3')
NUM_EVAL_GPUS=${#EVAL_GPU_IDX[@]}
EVAL_INTERVAL=10
EVAL_BATCH_SIZE=10


celebrities=(
  "Adam Driver"
  "Adriana Lima"
  "Amber Heard"
  "Amy Adams"
  "Andrew Garfield"
  "Angelina Jolie"
  "Anjelica Huston"
  "Anna Faris"
  "Anna Kendrick"
  "Anne Hathaway"
  "Arnold Schwarzenegger"
  "Barack Obama"
  "Beth Behrs"
  "Bill Clinton"
  "Bob Dylan"
  "Bob Marley"
  "Bradley Cooper"
  "Bruce Willis"
  "Bryan Cranston"
  "Cameron Diaz"
  "Channing Tatum"
  "Charlie Sheen"
  "Charlize Theron"
  "Chris Evans"
  "Chris Hemsworth"
  "Chris Pine"
  "Chuck Norris"
  "Courteney Cox"
  "Demi Lovato"
  "Drake"
  "Drew Barrymore"
  "Dwayne Johnson"
  "Ed Sheeran"
  "Elon Musk"
  "Elvis Presley"
  "Emma Stone"
  "Frida Kahlo"
  "George Clooney"
  "Glenn Close"
  "Gwyneth Paltrow"
  "Harrison Ford"
  "Hillary Clinton"
  "Hugh Jackman"
  "Idris Elba"
  "Jake Gyllenhaal"
  "James Franco"
  "Jared Leto"
  "Jason Momoa"
  "Jennifer Aniston"
  "Jennifer Lawrence"
  "Jennifer Lopez"
  "Jeremy Renner"
  "Jessica Biel"
  "Jessica Chastain"
  "John Oliver"
  "John Wayne"
  "Johnny Depp"
  "Julianne Hough"
  "Justin Timberlake"
  "Kate Bosworth"
  "Kate Winslet"
  "Leonardo Dicaprio"
  "Margot Robbie"
  "Mariah Carey"
  "Melania Trump"
  "Meryl Streep"
  "Mick Jagger"
  "Mila Kunis"
  "Milla Jovovich"
  "Morgan Freeman"
  "Nick Jonas"
  "Nicolas Cage"
  "Nicole Kidman"
  "Octavia Spencer"
  "Olivia Wilde"
  "Oprah Winfrey"
  "Paul Mccartney"
  "Paul Walker"
  "Peter Dinklage"
  "Philip Seymour Hoffman"
  "Reese Witherspoon"
  "Richard Gere"
  "Ricky Gervais"
  "Rihanna"
  "Robin Williams"
  "Ronald Reagan"
  "Ryan Gosling"
  "Ryan Reynolds"
  "Shia Labeouf"
  "Shirley Temple"
  "Spike Lee"
  "Stan Lee"
  "Theresa May"
  "Tom Cruise"
  "Tom Hanks"
  "Tom Hardy"
  "Tom Hiddleston"
  "Whoopi Goldberg"
  "Zac Efron"
  "Zayn Malik"
)


for i in "${!celebrities[@]}"; do
  current="${celebrities[$i]}"

  targets=$(printf "%s, " "${celebrities[@]:0:$((i + 1))}")
  targets=${targets%, }

  echo "Sequential edit step $((i + 1)) / ${#celebrities[@]} : ${current}"

  # ---------- 构造 edit_ckpt 参数（用 array，避免空格炸掉） ----------
  EXTRA_ARGS=()
  if [ "$i" -gt 0 ]; then
    prev="${celebrities[$((i - 1))]}"
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
    --retain_scale "$retain_scale" \
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
