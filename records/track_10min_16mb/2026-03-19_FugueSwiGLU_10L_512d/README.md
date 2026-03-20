# Fugue SwiGLU 10L

## Summary

Replaces the baseline relu^2 MLP activation with SwiGLU (used in LLaMA, Gemma, and other modern architectures). Adds one additional transformer layer (10 vs 9) using headroom freed by improved compression characteristics of SwiGLU weights.

## Key Changes from Baseline

1. **SwiGLU activation**: Replaces `relu(x)^2` with `silu(gate(x)) * fc(x)`. SwiGLU adds a gating linear layer per block but provides significantly better per-token learning efficiency (~0.05 BPB improvement at matched training steps in local testing).

2. **10 layers** (vs 9 baseline): The SwiGLU model compresses more efficiently under int8+zlib, freeing ~2-3 MB of headroom. An extra transformer layer uses this budget for additional model depth.

3. **Environment variable controlled**: `USE_SWIGLU=1` enables SwiGLU, `NUM_RECURRENCE_PASSES=1` (default) for standard forward pass. The script remains backward-compatible with all baseline configurations.

## Architecture

- 10 transformer blocks, 512 model dim
- 8 attention heads, 2 KV heads (GQA 4:1)
- SwiGLU MLP with 2x expansion (hidden=1024, 3 linear layers per block)
- 1024 vocab (SentencePiece BPE), tied embeddings
- ~24M parameters, estimated ~14 MB compressed

## Local Testing Results (500 steps, RTX 5090, batch=262144)

| Config | val_bpb @500 steps | Step time |
|--------|-------------------|-----------|
| Baseline 9L relu^2 | 1.5304 | 351ms |
| **9L SwiGLU** | **1.4793** | 438ms |

SwiGLU showed -0.051 BPB improvement at 500 steps with 25% step time overhead.

## Run Command (8xH100)

```bash
RUN_ID=fugue_swiglu_10L \
NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=2 USE_SWIGLU=1 NUM_KV_HEADS=2 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
NUM_RECURRENCE_PASSES=1 WARMDOWN_ITERS=1800 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
