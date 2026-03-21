# Hydra: BitNet b1.58 Transformer for Parameter Golf

## Problem
Current submission (SwiGLU 10L) compresses to 22.4MB — over the 16MB limit. Leaderboard has moved to ~1.13 BPB using int6 QAT with ~21M params. We need a new lever.

## Solution
BitNet b1.58 ternary weights ({-1, 0, +1}) fit ~65M params in 15MB — 3x more parameters than int6 competitors. Combine with all proven top techniques.

## Architecture
- **12 layers**, 896 model dim, 14 attention heads (head_dim=64), 2 KV heads (GQA 7:1)
- **3x MLP expansion** (hidden=2688), relu^2 activation
- **1024 vocab**, tied embeddings, seq_len=1024
- **BitNet b1.58**: ternary weights with STE QAT during training
- **SmearGate**: per-layer previous-token blending
- **U-Net skip connections**: preserved from baseline
- **~65M params**, ~14.8 MB compressed

## BitNet b1.58 Quantization
- Forward: `w_ternary = sign(w) * mean(|w|)`, quantized to {-scale, 0, +scale}
- Backward: STE passes gradients through to latent float weights
- Serialization: 2-bit packing (4 weights per byte) + per-row fp16 scales + zstd
- Zero quantization degradation at eval (model trained with exact eval weights)

## Additional Techniques
- **SWA**: average last 50 checkpoints for better generalization
- **Sliding window eval**: stride=64 overlapping windows at eval time
- **Zstd compression**: replaces zlib for ~10% better ratios
- **Weight decay 0.04**: constrains magnitudes, reduces quant gap
- **Orthogonal init**: faster convergence in 10-min budget

## Parameter Budget (15MB target)
| Component | Params | Compressed |
|---|---|---|
| Embedding (FP16 tied) | 917K | 1.83 MB |
| Attention ×12 | 18.5M | 4.2 MB |
| MLP (3x) ×12 | 38.7M | 8.7 MB |
| SmearGate ×12 | 21K | 0.01 MB |
| Scales/gates/skips | ~130K | 0.05 MB |
| **Total model** | **~58.3M** | **~14.8 MB** |
| Code | — | ~0.08 MB |
| **Grand total** | | **~14.9 MB** |

## Optimizer
- Latent float weights updated by Muon (matrix params) and Adam (scalars/embeddings)
- STE quantization in forward pass only — optimizer sees smooth gradients
- Weight decay 0.04 on matrix params

## Validation
1. CPU: verify model instantiates, shapes correct, compression fits 15MB
2. Local 5090: 50-step smoke test, verify loss decreases
3. Local 5090: 500-step run for early BPB signal
4. RunPod 8xH100: official 10-min submission run
