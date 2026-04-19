#!/bin/bash
# Smoke test for Stage 1 v5.9.2 — runs 5k steps on M4, validates stability fixes.
set -e
cd ~/hybridbid

LOG=logs/smoke_v592.log
mkdir -p logs checkpoints/stage1_v592

echo "=== Stage 1 v5.9.2 Smoke Test (5k steps) ===" | tee $LOG
date | tee -a $LOG
echo "" | tee -a $LOG

python -u -m src.training.train_stage1 \
    --v592 \
    --steps 5000 \
    --log-interval 1000 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Smoke test complete ===" | tee -a $LOG

# Quick stability summary
echo "" | tee -a $LOG
echo "--- Alpha trajectory ---" | tee -a $LOG
grep "alpha=" $LOG | grep "^Step" | awk -F'alpha=' '{print $1, "alpha="$2}' | awk '{print $1, $2, $4}' | tee -a /dev/null | \
    grep -oP 'Step\s+\K\d+.*alpha=\K[0-9.]+' | paste - - 2>/dev/null || \
    grep "^Step" $LOG | grep -oP 'alpha=\K[0-9.]+' | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- grad_c pre→post clip ---" | tee -a $LOG
grep "^Step" $LOG | grep -oP 'grad_c=\K[0-9.→]+' | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- mode_batch distribution at final step ---" | tee -a $LOG
grep "^Step" $LOG | tail -1 | grep -oP 'mode_batch=\K\S+' | tee -a $LOG
