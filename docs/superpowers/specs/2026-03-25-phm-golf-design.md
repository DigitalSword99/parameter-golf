# PHM-Golf: Kronecker-Factored Transformer for Parameter Golf

**Date:** 2026-03-25
**Author:** John (DigitalSword99) + Claude
**Target:** Beat SOTA 1.1194 BPB → aim for 1.095-1.115 BPB

## Problem

The Parameter Golf leaderboard has converged on a single stack: 11L/512d/3xMLP with int6 quantization + LZMA. Every top entry uses the same architecture and fights over marginal quantization improvements. Nobody is using structural matrix factorization to fundamentally change the parameter efficiency equation.

## Core Innovation: PHM Layers

**Parameterized Hypercomplex Multiplication (PHM)** replaces every linear layer with a Kronecker-factored equivalent:

```
W = Σᵢ₌₁ⁿ Aᵢ ⊗ Sᵢ     (n=4)
```

Where:
- `Aᵢ` are learned 4×4 "algebra rule" matrices (64 params total per layer)
- `Sᵢ` are learned (d_out/4 × d_in/4) "filter" matrices (bulk of params)

**Benefits:**
1. **4x parameter reduction** on all linear layers
2. **Same param count buys 18L/768d** instead of 11L/512d — nearly 2x deeper, 50% wider
3. **Unique architectural bet** — nobody else uses structural matrix factorization

**Note on FLOPs:** PHM does NOT reduce FLOPs. The two-step einsum computes the same O(d²) work as a standard linear layer (the contraction over the filter dimension dominates). Training step speed is comparable to baseline. The advantage is purely in parameter efficiency: more model capacity per stored byte.

**Efficient forward pass** (avoids materializing the full weight matrix):
```python
# x: (B, seq, d_in) → reshape to (B, seq, n, d_in//n)
# Step 1: T = einsum('bkl,iol->bkio', X, S)  — apply filters within blocks
# Step 2: y = einsum('ijk,bkio->bjo', A, T)  — mix blocks via algebra
# FLOPs: O(d²) same as standard, but stores only O(d²/n) parameters
```

## Architecture: PHM-Golf

```
Model:     18 layers, 768 dim, 12 heads, 4 KV heads (GQA 3:1)
MLP:       3x expansion (2304 hidden), LeakyReLU(0.5)²
PHM:       n=4 on all attention (Q/K/V/Proj) and MLP (fc/proj) layers
Embedding: 1024 vocab × 768d, tied with output head, fp16
```

### Parameter Budget

| Component | Params | Storage |
|-----------|--------|---------|
| Embedding (1024×768, tied) | 786K | fp16 (1.57 MB) |
| BigramHash (1536×768) | 1,180K | fp16 (2.36 MB) |
| PHM Attention ×18 (Q/K/V/Proj) | 7,082K | S_i: int6, A_i: fp32 |
| PHM MLP ×18 (fc + proj) | 15,927K | S_i: int6, A_i: fp32 |
| Norms, scales, skips, gates | ~100K | fp32 |
| **Total** | **~25.1M** | |

### Size Estimate

```
S_i matrices:     ~22.5M × 0.875 bytes (7-bit packed: 6-bit index + sign)  = 19.7 MB
  Per-row scales:  ~117K  × 2 bytes (fp16)                                 =  0.23 MB
A_i matrices:     ~7K    × 4 bytes (fp32)                                  =  0.03 MB
Embedding:        786K   × 2 bytes (fp16)                                  =  1.57 MB
BigramHash:       1,180K × 2 bytes (fp16)                                  =  2.36 MB
Control tensors:  ~100K  × 4 bytes (fp32)                                  =  0.40 MB
                                            ─────────────
Pre-compression:                              ~24.3 MB
LZMA compression (~0.55x on packed indices):  ~13.4 MB
Code:                                         + 0.10 MB
                                            ─────────────
Estimated artifact:                           ~13.5 MB
Headroom:                                      2.5 MB

Note: 7-bit packing stores 8 values in 7 bytes. LZMA ratio of 0.55x is
conservative — APoT indices cluster near zero (non-uniform), giving LZMA
better compression than uniform random data (which achieves ~0.75x).
```

## Novel Techniques (6 innovations)

### 1. PHM(n=4) Factored Layers

All linear layers use Kronecker-factored parameterization. The A_i matrices (4×4) learn algebraic structure; the S_i matrices hold the bulk of parameters and are quantized.

**Initialization:**
- A_0 = identity(4), A_1..3 = zeros (start as single standard-like linear)
- S_i: Kaiming normal scaled by 1/√n

### 2. APoT (Additive Powers-of-Two) Quantization

Replace uniform int6 with APoT 6-bit. Each weight represented as sum of powers of two:
```
w = ±(2^a + 2^b + 2^c)    where a > b > c ≥ 0
```

Quantization levels are denser near zero, matching the bell-shaped weight distribution.

**Serialization:** Weights are stored as packed 7-bit values (6-bit APoT level index + 1 sign bit), NOT as float32. A shared codebook of ~26 non-negative APoT levels is stored once. Per-row fp16 scales allow reconstruction: `weight = sign * levels[index] * scale[row]`. The 7-bit packing (8 values per 7 bytes) plus non-uniform index distribution gives excellent LZMA compression.

**QAT integration:** Fake-quantize S_i matrices during forward pass with STE. A_i matrices stay fp32.

**GPTQ-lite enhancement:** At export time, per-row optimal APoT clip search (5 candidates, pick min MSE) — same principle as leaderboard GPTQ-lite but with APoT levels.

### 3. Learned-Frequency RoPE

Replace Partial RoPE's fixed 16/64 split with learnable frequencies per head dimension:

```python
# Init: standard RoPE frequencies
self.inv_freq = nn.Parameter(1.0 / (base ** (torch.arange(0, dim, 2) / dim)))
# Gradient-based learning finds optimal frequencies
# Some dims → ~0 frequency (position-free, like Partial RoPE)
# Others → optimal rotation speed for the task
```

Subsumes Partial RoPE as a special case. 64 extra learnable floats per layer.

### 4. K-FAC Test-Time Training

PHM layers are already Kronecker-factored, so the Fisher information matrix naturally factorizes:

```
F ≈ F_A ⊗ F_S

K-FAC update:  ΔS = -lr · F_S⁻¹ @ grad_S @ F_A⁻¹
```

This gives approximate natural gradient TTT at the cost of SGD. Faster convergence means better adaptation in limited eval time.

**Protocol (same as Legal Score-First):**
1. Split val tokens into non-overlapping 32K-token chunks
2. Score each chunk under `torch.inference_mode()` (record loss)
3. Adapt with SGD: lr=0.002, 3 epochs, cosine decay, all S_i unfrozen
4. A_i matrices: zero out gradients after backward (not requires_grad=False, to avoid torch.compile graph invalidation)
5. Last chunk scored but never trained on
6. TTT uses uncompiled base_model directly to avoid torch.compile conflicts

### 5. Multi-Resolution Skip Connections

Each decoder layer receives weighted input from multiple encoder layers:

```python
# Standard U-Net: decoder[i] += skip_w[i] * encoder[N-1-i]
# Multi-res:      decoder[i] += Σⱼ skip_w[i,j] * encoder[selected_j]
```

Selection strategy: each decoder layer connects to its symmetric partner AND every 4th deeper encoder layer. For 18 layers (9 encoder, 9 decoder):
- Decoder layer 9 ← encoder[8] (symmetric) + encoder[4] (deep)
- Decoder layer 10 ← encoder[7] + encoder[3]
- etc.

~180 extra learnable parameters (negligible cost).

### 6. Learnable Layer Scaling + Cosine Warmdown

**Layer scaling:** Initialize at 1/√(layer+1), learn during training:
```python
self.ln_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(layer_idx + 1)))
```

**Cosine warmdown:** Replace linear LR decay with cosine for smoother convergence:
```python
scale = 0.5 * (1 + cos(π * progress))  # progress: 0→1 during warmdown
```

**EMA temperature annealing:** Increase EMA decay from 0.997 → 0.9999 during warmdown (sharper averaging as model converges).

## Proven Techniques (from leaderboard)

| Technique | Config | Source |
|-----------|--------|--------|
| BigramHash | 1536 buckets, additive to embedding | Multiple top entries |
| XSA | Cross-sequence attention on last 4 layers | Top 4 entries |
| LeakyReLU(0.5)² | Drop-in replacement for relu² in MLP | SOTA (#1) |
| EMA | 0.997 decay, every step, stacks with SWA | Top 3 entries |
| Tight SWA | Checkpoint every 50 steps when LR scale < 0.2 | Top 2 entries |
| GQA | 12 heads, 4 KV heads (3:1 ratio) | All entries |
| Logit softcap | 30.0 | Baseline |

## Training Configuration

```
Optimizer:
  S_i matrices     → Muon, lr=0.04, WD=0.04 (decoupled, uses base_lr), momentum 0.85→0.99 (warmup 500 steps)
  Note: Muon applies NS5 per-slice on each (s_out, s_in) factor independently, NOT concatenated
  A_i matrices     → Adam, lr=0.01, β1=0.9, β2=0.95
  Embedding        → Adam, lr=0.05
  BigramHash       → Adam, lr=0.05
  Scalars/control  → Adam, lr=0.04

Schedule:
  Compile warmup:    20 steps (reset model + optimizer state after)
  LR warmup:         50 steps
  Main training:     ~8,000-10,000 steps (same step speed as baseline)
  Cosine warmdown:   last 3500 steps
  EMA:               0.997 → 0.9999 (annealed during warmdown)
  SWA:               every 50 steps when LR scale < 0.2

Data:
  Sequence length:   2048
  Batch tokens:      524,288
  Grad accumulation: 8 / world_size

Hardware:
  8× H100 SXM via RunPod
  torchrun --standalone --nproc_per_node=8
  DDP with NCCL backend
  torch.compile(dynamic=False, fullgraph=True)
```

## Evaluation Configuration

```
1. Weight averaging: EMA + SWA merge
2. APoT int6 quantization with GPTQ-lite clip search on S_i
3. A_i kept fp32, embedding fp16, control fp32
4. LZMA compression → artifact (target: <14.5 MB)
5. Roundtrip: decompress → dequantize → load model
6. K-FAC Score-First TTT:
   - 32K-token chunks, non-overlapping
   - Score first (inference_mode), then adapt (K-FAC SGD)
   - lr=0.002, 3 epochs, cosine decay per chunk
   - All S_i unfrozen, A_i frozen
   - Last chunk scored only
7. Report final val_bpb
```

## Expected Outcomes

| Scenario | BPB | vs SOTA (1.1194) | Probability |
|----------|-----|-------------------|-------------|
| Best case | 1.095-1.105 | -0.015 to -0.025 | ~20% |
| Expected case | 1.105-1.120 | -0.000 to -0.015 | ~45% |
| Worst case | 1.130-1.150 | +0.010 to +0.030 | ~35% |

Note: PHM does not reduce FLOPs (corrected from earlier analysis). Training
step count is comparable to baseline. The advantage is purely architectural:
more depth and width per parameter stored.

**Fallback plan:** If PHM underperforms, revert to standard linear layers with the full proven technique stack. The novel quantization (APoT) and evaluation (K-FAC TTT) innovations still apply.

## File Structure

```
records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/
├── README.md           # Submission description
├── submission.json     # Metadata
├── train.log           # Training output
├── train_gpt.py        # Full training script
└── requirements.txt    # Dependencies
```

## Implementation Order

1. **PHMLinear module** — core Kronecker layer with efficient forward
2. **APoT quantization** — QAT fake-quantize + export serialize/deserialize
3. **Architecture assembly** — 18L/768d GPT with PHM, BigramHash, XSA, skips
4. **Learned RoPE + layer scaling** — small modifications to attention/norm
5. **Training loop** — Muon on S_i, Adam on A_i, cosine warmdown, EMA annealing
6. **K-FAC TTT** — test-time adaptation with factored natural gradient
7. **Serialization** — APoT pack/unpack + LZMA compression pipeline
8. **Smoke test** — local GPU verification (RTX 5090)
9. **8×H100 run** — RunPod official submission

## Key Risks

| Risk | Mitigation |
|------|-----------|
| PHM layers hurt LM quality | Fallback to standard linear + proven stack |
| APoT quantization degrades more than int6 | Fall back to GPTQ-lite int6 |
| torch.compile doesn't optimize einsum well | Pre-compute W = Σ kron(A,S) and use standard matmul |
| K-FAC TTT is unstable | Fall back to SGD TTT |
| Artifact exceeds 16MB | Reduce to 16L or shrink BigramHash |
| Training too slow (>10 min) | Reduce to 16L or increase batch size |

## References

- [PHM Layer (ICLR 2021)](https://arxiv.org/abs/2102.08597)
- [APoT Quantization (ICLR 2020)](https://arxiv.org/abs/1909.13144)
- [Krony-PT: GPT-2 Kronecker (2024)](https://arxiv.org/abs/2412.12351)
- [K-FAC (ICML 2015)](https://arxiv.org/abs/1503.05671)
- [Learned RoPE frequencies](https://arxiv.org/abs/2104.09864)
