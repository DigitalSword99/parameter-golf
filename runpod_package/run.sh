#!/bin/bash
set -e
cd /workspace/parameter-golf

echo "=== Training (10 min cap on 8xH100) ==="
RUN_ID=fugue_swiglu_10L_kv2 \
NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=2 \
MLP_MULT=2 USE_SWIGLU=1 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
NUM_RECURRENCE_PASSES=1 WARMDOWN_ITERS=1800 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log

echo ""
echo "=== DONE ==="
echo "If val_bpb < 1.2194 you beat the leaderboard!"
