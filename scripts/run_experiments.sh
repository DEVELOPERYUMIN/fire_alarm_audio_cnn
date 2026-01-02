#!/bin/bash
# run_experiments.sh
# 축소 그리드: 7개 변수 × 2가지씩 → 128 실험

# ─── 1) threshold, pos_weight는 고정 ──────────────────────────
threshold=0.5
# pos_weight 자동 계산을 train_experiments.py 쪽에서 구현했다 가정

# ─── 2) 축소된 실험용 파라미터 (각 2개씩) ───────────────────
dropouts=(0.3 0.5)
channels=(16 32 64)
freq_masks=(10 20)
time_masks=(10 20)
num_masks=(1 2)
mixup_alphas=(0 0.2 0.4)
lrs=(1e-3 1e-4)
pos_weights_arr=(1.0 1.5 2.0)
idx=1

for dropout in "${dropouts[@]}"; do
  for channel in "${channels[@]}"; do
    for freq_mask in "${freq_masks[@]}"; do
      for time_mask in "${time_masks[@]}"; do
        for num_mask in "${num_masks[@]}"; do
          for mixup in "${mixup_alphas[@]}"; do
            for lr in "${lrs[@]}"; do
              for seed in "${seeds_arr[@]}"; do
                for pos_weight in "${pos_weights_arr[@]}"; do

              echo "=== Experiment $idx ==="
              echo "dropout=$dropout, channels=$channel, freq_mask=$freq_mask, time_mask=$time_mask"
              echo "num_masks=$num_mask, mixup_alpha=$mixup, lr=$lr"

              python3 scripts/train_experiments.py \
                  --model_name lightcnn \
                  --exp_idx "$idx" \
                  --epochs 30 \
                  --batch_size 32 \
                  --lr "$lr" \
                  --seed "$seed" \
                  --threshold "$threshold" \
                  --pos_weight "$pos_w" \
                  --dropout "$dropout" \
                  --channels "$ch" \
                  --freq_mask "$fmask" \
                  --time_mask "$tmask" \
                  --num_masks "$nm" \
                  --mixup_alpha "$mix"

              idx=$((idx+1))

                done
              done
            done
          done
        done
      done
    done
  done
done


echo "총 실험 횟수: $((idx-1))"
