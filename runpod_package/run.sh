#!/bin/bash
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Downloading data + tokenizer ==="
python data/cached_challenge_fineweb.py

echo "=== Training Hydra BitNet (10 min cap on 8xH100) ==="
RUN_ID=hydra_bitnet_12L_768d \
NUM_LAYERS=12 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
MLP_MULT=3 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 ROPE_BASE=200000 \
SWA_ENABLED=1 SWA_EVERY=100 EVAL_STRIDE=64 \
WARMDOWN_ITERS=1200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log

echo ""
echo "=== DONE ==="
echo "Check val_bpb in the output above!"
