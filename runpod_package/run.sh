#!/bin/bash
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== Installing dependencies ==="
pip install zstd 2>/dev/null || true

echo "=== Downloading data + tokenizer ==="
python data/cached_challenge_fineweb.py

echo "=== Training Hydra v2 (10 min cap on 8xH100) ==="
RUN_ID=hydra_v2_11L_512d \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 ROPE_BASE=10000 \
BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
SWA_ENABLED=0 \
EVAL_STRIDE=64 \
WARMDOWN_ITERS=3000 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_WD=0.04 GRAD_CLIP_NORM=0.3 \
torchrun --standalone --nproc_per_node=8 runpod_package/train_gpt.py 2>&1 | tee train.log

echo ""
echo "=== DONE ==="
echo "Check val_bpb in the output above!"
