#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_v592.log
mkdir -p logs
echo "=== Stage 1 v5.9.2 Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000 275000 300000 325000 350000 375000 400000 425000 450000 475000 500000; do
    ckpt="checkpoints/stage1_v592/checkpoint_step${step}.pt"
    if [ ! -f "$ckpt" ]; then
        echo "--- step $step --- SKIPPED (no checkpoint)" | tee -a $LOG
        continue
    fi
    echo "--- step $step ---" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=16 conda run --no-capture-output -n hybridbid python -u \
        -m src.evaluation.evaluate_stage1 --checkpoint "$ckpt" --device cuda 2>&1 | tee -a $LOG
done

echo "=== Sweep complete ===" | tee -a $LOG
