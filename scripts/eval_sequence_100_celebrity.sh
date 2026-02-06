#!/bin/bash

# Sequentially edit 100 celebrity concepts and evaluate on COCO after each edit.

baseline="SPEED"
params="V"
aug_num=10
threshold="1e-4"
retain_scale="0.05"
save_root="logs/${baseline}/seq_multi_100_coco"
erase_type="100_celebrity"
anchor="person"
TRAIN_GPU=0
EVAL_GPU_IDX=('1' '2' '3')
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

limit_target_name() {
  local target="$1"
  local num
  num=$(echo "$target" | tr -cd ',' | wc -c)
  num=$((num + 1))
  local limited_target
  limited_target=$(echo "$target" | awk -F', ' '{for (i=1; i<=NF && i<=5; i++) printf (i<NF && i<5 ? $i "_": $i)}')
  if [ "$num" -gt 5 ]; then
    limited_target="${limited_target}_${num}"
  fi
  echo "$limited_target"
}

for i in "${!celebrities[@]}"; do
  targets=$(printf "%s, " "${celebrities[@]:0:$((i + 1))}")
  targets=${targets%, }
  limited_target=$(limit_target_name "$targets")

  echo "Sequential edit step $((i + 1))/100: ${limited_target}"

  CUDA_VISIBLE_DEVICES=$TRAIN_GPU python train_erase_null.py \
    --baseline "$baseline" \
    --target_concepts "$targets" --anchor_concepts "$anchor" \
    --retain_path "data/${erase_type}.csv" --heads "concept" \
    --save_path "${save_root}/${erase_type}/${limited_target}" --file_name "weight" \
    --params "$params" --aug_num "$aug_num" --threshold "$threshold" \
    --retain_scale "$retain_scale" --disable_filter

  if (((i + 1) % EVAL_INTERVAL == 0)); then
    eval_gpu=${EVAL_GPU_IDX[$(( (i / EVAL_INTERVAL) % NUM_EVAL_GPUS ))]}
    (
      CUDA_VISIBLE_DEVICES=$eval_gpu python sample2.py \
        --erase_type "coco" \
        --target_concept "$limited_target" \
        --contents "coco" \
        --mode "edit" \
        --num_samples 1 --batch_size "$EVAL_BATCH_SIZE" \
        --save_root "${save_root}/${erase_type}" \
        --edit_ckpt "${save_root}/${erase_type}/${limited_target}/weight.pt"

      CUDA_VISIBLE_DEVICES=$eval_gpu python src/clip_score_cal.py \
        --contents "coco" \
        --root_path "${save_root}/${erase_type}"
    ) &
  fi
done

wait
