# Hydra-3: Three-Head Shared Backbone with Aggressive TTT

**Date:** 2026-03-25
**Author:** John (DigitalSword99) + Claude
**Target:** 1.05-1.09 BPB (beat SOTA 1.1194 decisively)
**Supersedes:** PHM-Golf 18L/768d (too slow, OOM issues)

## Problem

PHM-Golf (18L/768d) failed because:
1. PHM doesn't reduce FLOPs — only stored params
2. 18L/768d requires gradient checkpointing (OOM on H100 80GB)
3. Gradient checkpointing + fullgraph=False slows training to ~150-180ms/step
4. ~3,500 steps in 10 min vs SOTA's 7,185 — fatally undertrained

The competition rewards **capacity per FLOP**, not capacity per byte. The eval time budget (10 min) is massively underutilized by all entries.

## Solution: Three-Head Ensemble + Aggressive TTT

### Core Insight

1. **Shared backbone** trains for full 10 min (zero wasted compute)
2. **Three diverse MLP heads** provide ensemble diversity at eval time
3. **Aggressive TTT** exploits the full 10-min eval budget with Adam optimizer
4. **MoE head** adds capacity diversity via PHM-compressed experts

### Architecture

```
Input tokens
    │
    ▼
Embedding (1024×512, tied) + BigramHash(2048) + TrigramHash(4096)
    │
    ▼
Shared Backbone: 11 layers, 512d, 12 heads, 4 KV heads
  - CausalSelfAttention with PHMLinear (Q/K/V/Proj)
  - Learned-frequency RoPE
  - XSA on last 4 layers
  - RMSNorm with learnable layer scaling
  - Multi-resolution skip connections (U-Net)
  - Residual mix + attn/mlp scales
    │
    ├──► Head A: MoE MLP — 8 PHM experts, top-1 routing, 2x expansion
    ├──► Head B: Dense MLP — 3x expansion, LeakyReLU(0.5)²
    └──► Head C: Dense MLP — 2x expansion → project to 640d → back to 512d
    │
    ▼
Each head: LeakyReLU(0.5)² activation
Logit head: tied embedding (shared across all heads)
```

### Training Protocol

- **Step selection:** Each step, randomly pick 1 of 3 heads (uniform distribution)
- **Forward/backward:** Shared backbone + selected head only (other heads frozen for that step)
- **Backbone trains 100%** of steps (~11,300 at ~53ms/step with 524K batch)
- **Each head trains ~33%** of steps (~3,800) — sufficient for MLP convergence
- **Optimizer:** Muon on PHM S matrices + attention PHM, Adam on algebra/embedding/scalars/heads
- **Schedule:** 50-step LR warmup, 3500-step cosine warmdown, EMA(0.997→0.9999) + Tight SWA

### Evaluation Protocol

1. Load compressed model, decompress, dequantize all 3 heads
2. **Ensemble scoring:** For each validation chunk:
   - Forward through backbone once
   - Forward through all 3 heads
   - Average logits: `logits = (headA(h) + headB(h) + headC(h)) / 3`
   - Score chunk (record loss + BPB)
3. **Aggressive TTT:** After scoring each chunk:
   - Optimizer: Adam(lr=0.001, β1=0.9, β2=0.95)
   - Epochs: 10 per chunk (vs SOTA's 3 with SGD)
   - Scope: ALL parameters (backbone + all heads)
   - LR decay: cosine per chunk
   - Gradient clipping: 1.0
4. **Score-first protocol:** Never train on unscored tokens. Last chunk scored only.
5. **Time budget:** Target ~550-580s total eval (within 600s limit)

### Head Architectures

**Head A: MoE (8 PHM experts, top-1)**

```python
class MoEMLP(nn.Module):
    def __init__(self, dim, num_experts=8, phm_n=4):
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'fc': PHMLinear(dim, dim * 2, n=phm_n),
                'proj': PHMLinear(dim * 2, dim, n=phm_n),
            }) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, seq, dim)
        logits = self.router(x)  # (B, seq, num_experts)
        idx = logits.argmax(dim=-1)  # top-1 routing
        # Gather tokens per expert, forward, scatter back
        # (implementation uses expert-parallel pattern)
```

- 8 experts × PHMLinear(512→1024, n=4) + PHMLinear(1024→512, n=4)
- Top-1 routing: each token → 1 expert (same FLOPs as single MLP)
- Router: 512→8 linear (tiny)
- Load balancing: auxiliary loss (standard MoE practice)

**Head B: Dense 3x MLP**

```python
class DenseMLP3x(nn.Module):
    def __init__(self, dim):
        self.fc = CastedLinear(dim, dim * 3, bias=False)
        self.proj = CastedLinear(dim * 3, dim, bias=False)

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())
```

- Standard dense MLP with 3x expansion
- Matches SOTA's MLP architecture exactly

**Head C: Dense 2x Wide MLP**

```python
class WideMLP(nn.Module):
    def __init__(self, dim, wide_dim=640):
        self.up = CastedLinear(dim, wide_dim, bias=False)
        self.fc = CastedLinear(wide_dim, wide_dim * 2, bias=False)
        self.proj = CastedLinear(wide_dim * 2, wide_dim, bias=False)
        self.down = CastedLinear(wide_dim, dim, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.proj(F.leaky_relu(self.fc(x), 0.5).square())
        return self.down(x)
```

- Projects to 640d, applies 2x MLP, projects back
- Different feature space from Head B — adds diversity

### Size Budget

| Component | Params | Storage | Compressed |
|-----------|--------|---------|-----------|
| Embedding (1024×512, tied) | 524K | fp16 | 0.5 MB |
| BigramHash (2048×512) | 1,049K | APoT int8 | 0.7 MB |
| TrigramHash (4096×512) | 2,097K | APoT int8 | 1.4 MB |
| Shared Attention PHM ×11L (Q/K/V/Proj) | 2,904K | APoT int8 | 1.9 MB |
| Norms, scales, skips, RoPE, control | ~120K | fp16/fp32 | 0.4 MB |
| **Head A:** 8 PHM experts ×11L + routers | 5,600K | APoT int8 | 3.5 MB |
| **Head B:** Dense 3x MLP ×11L | 3,408K | APoT int8 | 2.2 MB |
| **Head C:** Wide 2x MLP ×11L | 2,816K | APoT int8 | 1.8 MB |
| Code | — | — | ~0.1 MB |
| **Total** | **~18.5M** | | **~12.5 MB** |

3.5 MB headroom — comfortable.

### N-gram Hash Tables

**BigramHash(2048):**
```python
hash(prev_token, cur_token) → 2048 buckets → 512d embedding
```

**TrigramHash(4096):**
```python
hash(prev2_token, prev_token, cur_token) → 4096 buckets → 512d embedding
```

Both added to token embeddings before the backbone. Captures common 2-gram and 3-gram patterns that the model doesn't need to learn from scratch.

### Optimizer Groups

| Parameter group | Optimizer | LR | Notes |
|----------------|-----------|-----|-------|
| PHM S matrices (backbone attn + Head A experts) | Muon | 0.04 | Per-slice NS5, WD=0.04 |
| PHM A matrices | Adam | 0.01 | Algebra rules, small |
| Token embedding | Adam | 0.05 | Tied with output |
| BigramHash + TrigramHash | Adam | 0.05 | Additive features |
| Head B dense weights (CastedLinear) | Muon | 0.04 | Standard 2D matrices |
| Head C dense weights (CastedLinear) | Muon | 0.04 | Standard 2D matrices |
| MoE routers | Adam | 0.01 | Small, needs stability |
| Scalars, norms, control, RoPE freq | Adam | 0.04 | Low-dim params |

### Training Configuration

```
Batch: 524,288 tokens/step (not 786K — speed > batch quality)
Seq length: 2048
Grad accumulation: 8 // world_size (= 1 on 8 GPU)
Compile: torch.compile(dynamic=False, fullgraph=True)
DDP: NCCL backend
Warmup: 50 compile warmup steps (reset state after)
LR warmup: 50 steps (ramp from 0 to full)
Cosine warmdown: last 3500 steps
EMA: 0.997 → 0.9999 (annealed during warmdown)
SWA: every 50 steps when LR scale < 0.2
Max wallclock: 600s
Head selection: uniform random per step, same across all ranks (seeded)
MoE auxiliary loss weight: 0.01 (load balancing)
```

### Eval Configuration

```
Ensemble: average logits from 3 heads
TTT optimizer: Adam(lr=0.001, β1=0.9, β2=0.95)
TTT epochs per chunk: 10
TTT chunk size: 32768 tokens
TTT scope: all parameters (backbone + all 3 heads)
TTT LR decay: cosine per chunk
TTT grad clip: 1.0
Score-first: score chunk, then adapt, last chunk score-only
Target eval time: 550-580s (within 600s budget)
```

### Expected Outcomes

| Scenario | BPB | vs SOTA (1.1194) | Probability |
|----------|-----|-------------------|------------|
| Everything works | 1.06-1.07 | -0.05 to -0.06 | ~5% |
| Most things work | 1.07-1.09 | -0.03 to -0.05 | ~25% |
| Solid execution | 1.09-1.11 | -0.01 to -0.03 | ~40% |
| Partial success | 1.11-1.13 | competitive | ~20% |
| Something breaks | 1.13+ | below SOTA | ~10% |

### Key Risks

| Risk | Mitigation |
|------|-----------|
| MoE routing collapse (all tokens → 1 expert) | Auxiliary load-balancing loss |
| 3,800 steps insufficient for head convergence | Heads are just MLPs — converge fast |
| Aggressive TTT destabilizes model | Gradient clipping + cosine LR decay |
| Ensemble averaging hurts (heads too similar) | Three architecturally different heads |
| MoE scatter/gather overhead kills step speed | Top-1 routing, pre-sort tokens, fused kernels |
| Artifact exceeds 16MB | 3.5MB headroom, can reduce TrigramHash |
| torch.compile fails with MoE routing | Fallback: eager mode for MoE head only |

### Implementation from PHM-Golf

Modify existing `/mnt/g/parameter-golf/records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py`:

1. Reduce architecture to 11L/512d (from 18L/768d)
2. Remove gradient checkpointing, restore fullgraph=True
3. Add 3 MLP head classes (MoEMLP, DenseMLP3x, WideMLP)
4. Modify Block to accept interchangeable MLP heads
5. Modify GPT to store 3 head sets, random selection per step
6. Add TrigramHash module
7. Add MoE routing with auxiliary loss
8. Update optimizer groups for new parameter structure
9. Rewrite eval to ensemble 3 heads + aggressive Adam TTT
10. Update serialization for multi-head model

### References

- [Switch Transformer (Fedus et al.)](https://arxiv.org/abs/2101.03961) — MoE routing
- [PHM Layer (ICLR 2021)](https://arxiv.org/abs/2102.08597) — Kronecker factored layers
- [APoT Quantization (ICLR 2020)](https://arxiv.org/abs/1909.13144) — Non-uniform quantization
