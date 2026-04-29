#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_stage2_v1.log
echo "=== Stage 2 v1 Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000; do
    ckpt="checkpoints/stage2/checkpoint_step${step}.pt"
    echo "--- step $step ---" | tee -a $LOG
    conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage2 --checkpoint "$ckpt" --device cuda 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "--- Stage 1 baseline (300k on post-RTC+B) ---" | tee -a $LOG
conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage2 --stage1-baseline --device cuda 2>&1 | tee -a $LOG

echo "=== Sweep complete ===" | tee -a $LOG
