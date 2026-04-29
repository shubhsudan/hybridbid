#!/bin/bash
set -e
cd ~/hybridbid
LOG=logs/eval_sweep_v58.log
echo "=== v5.8 Eval Sweep ===" | tee $LOG
date | tee -a $LOG

for step in 100000 150000 200000 250000 300000 350000 400000 450000 500000 550000 600000 650000 700000 750000 800000 850000 900000 950000 1000000; do
    ckpt="checkpoints/stage1/checkpoint_step${step}.pt"
    echo "--- step $step ---" | tee -a $LOG
    conda run --no-capture-output -n hybridbid python -u -m src.evaluation.evaluate_stage1 --checkpoint "$ckpt" --device cuda 2>&1 | tee -a $LOG
done

echo "=== Sweep complete ===" | tee -a $LOG
