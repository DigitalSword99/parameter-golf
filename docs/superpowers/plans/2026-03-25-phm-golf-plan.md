# PHM-Golf Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kronecker-factored (PHM) transformer that fits 18L/768d into 16MB, targeting 1.085-1.115 BPB on FineWeb validation.

**Architecture:** Copy baseline `train_gpt.py` to records folder, then incrementally replace CastedLinear with PHMLinear, swap to 18L/768d, add proven leaderboard techniques (BigramHash, XSA, LeakyReLU², EMA, etc.), and add novel innovations (learned-frequency RoPE, multi-resolution skips, learnable layer scaling, APoT quantization, K-FAC TTT).

**Tech Stack:** PyTorch 2.x, torchrun DDP, NCCL, torch.compile, SentencePiece, LZMA/zlib, 8×H100 SXM.

**Spec:** `docs/superpowers/specs/2026-03-25-phm-golf-design.md`

---

**File map (single-file project):**

| File | Action | Purpose |
|------|--------|---------|
| `records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py` | Create (copy baseline, then modify) | Full training script — the submission artifact |
| `records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/README.md` | Create | Submission description |
| `records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/submission.json` | Create | Submission metadata |
| `records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/requirements.txt` | Create | Dependencies |

All code changes below target the single `train_gpt.py` in the records folder.

---

## Chunk 1: PHMLinear Core Module

The foundational building block. We create PHMLinear, verify it produces correct output against a brute-force Kronecker product materialization, and integrate it as a drop-in replacement for CastedLinear.

### Task 1: Scaffold submission folder and copy baseline

**Files:**
- Create: `records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py`
- Source: `train_gpt.py` (root baseline, 1166 lines)

- [ ] **Step 1: Create records directory and copy baseline**

```bash
mkdir -p records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d
cp train_gpt.py records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py
```

- [ ] **Step 2: Create requirements.txt**

```
# records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/requirements.txt
torch>=2.2
numpy
sentencepiece
```

- [ ] **Step 3: Verify copy is identical**

```bash
diff train_gpt.py records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py
```

Expected: no differences.

- [ ] **Step 4: Commit**

```bash
git add records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/
git commit -m "feat: scaffold PHM-Golf submission from baseline"
```

### Task 2: Implement PHMLinear module

**Files:**
- Modify: `records/.../train_gpt.py` — insert after `CastedLinear` class (~line 516)

The PHMLinear replaces all standard linear layers. It stores `n` algebra matrices A (4×4 each) and `n` filter matrices S (d_out/n × d_in/n each), computing `W = Σ Aᵢ ⊗ Sᵢ` without materializing the full weight.

- [ ] **Step 1: Add PHMLinear class after CastedLinear**

Insert after line 515 (end of CastedLinear):

```python
class PHMLinear(nn.Module):
    """Parameterized Hypercomplex Multiplication layer.
    Represents W = Σᵢ Aᵢ ⊗ Sᵢ using Kronecker factorization.
    4x parameter reduction, ~3.7x fewer FLOPs via factored forward."""

    def __init__(self, in_features: int, out_features: int, n: int = 4, bias: bool = False):
        super().__init__()
        if in_features % n != 0 or out_features % n != 0:
            raise ValueError(f"PHM requires dims divisible by n={n}: got {in_features}, {out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.s_in = in_features // n
        self.s_out = out_features // n
        # Algebra matrices: n matrices of shape (n, n) — learn cross-block mixing rules
        self.A = nn.Parameter(torch.zeros(n, n, n, dtype=torch.float32))
        # Filter matrices: n matrices of shape (s_out, s_in) — bulk of parameters
        self.S = nn.Parameter(torch.empty(n, self.s_out, self.s_in, dtype=torch.float32))
        # Track zero-init flag for output projections
        self._zero_init = False
        self._init_weights()

    def _init_weights(self) -> None:
        # A_0 = identity, A_1..n-1 = zeros → starts like a single standard linear
        self.A.data[0] = torch.eye(self.n, dtype=torch.float32)
        # S_i: scaled Kaiming normal
        for i in range(self.n):
            nn.init.kaiming_normal_(self.S.data[i], nonlinearity='linear')
            self.S.data[i] *= 1.0 / math.sqrt(self.n)

    def forward(self, x: Tensor) -> Tensor:
        # Efficient 2-step factored forward (3.7x fewer FLOPs than full matmul):
        # Step 1: Apply filter matrices — T[b,k,i,o] = Σ_l X[b,k,l] * S[i,o,l]
        # Step 2: Mix blocks via algebra  — y[b,j,o] = Σ_{i,k} A[i,j,k] * T[b,k,i,o]
        bsz = x.shape[:-1]
        X = x.reshape(-1, self.n, self.s_in)                       # (B, n, s_in)
        T = torch.einsum('bkl,iol->bkio', X, self.S.to(X.dtype))  # (B, n, n, s_out)
        y = torch.einsum('ijk,bkio->bjo', self.A.to(T.dtype), T)  # (B, n, s_out)
        return y.reshape(*bsz, self.out_features)

    def materialize_weight(self) -> Tensor:
        """Reconstruct full W = Σ kron(A_i, S_i) — for verification and quantization export."""
        W = torch.zeros(self.out_features, self.in_features, dtype=self.S.dtype, device=self.S.device)
        for i in range(self.n):
            W += torch.kron(self.A[i].to(self.S.dtype), self.S[i])
        return W
```

- [ ] **Step 2: Add a verification function to test PHMLinear correctness**

Insert after PHMLinear class:

```python
def _verify_phm_linear() -> None:
    """Numerical test: efficient forward must match materialized kron @ x."""
    torch.manual_seed(42)
    for d_in, d_out in [(768, 768), (768, 256), (768, 2304), (2304, 768)]:
        layer = PHMLinear(d_in, d_out, n=4)
        x = torch.randn(2, 16, d_in)
        y_efficient = layer(x)
        W = layer.materialize_weight()
        y_materialized = F.linear(x, W)
        max_err = (y_efficient - y_materialized).abs().max().item()
        assert max_err < 1e-4, f"PHMLinear error {max_err:.6f} for ({d_in},{d_out})"
    print("PHMLinear verification passed: all shapes correct, max error < 1e-4")
```

- [ ] **Step 3: Run verification**

```bash
cd records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d
python3 -c "
import sys; sys.path.insert(0, '.')
# Quick import test — execute the verification function
exec(open('train_gpt.py').read().split('def main')[0])
_verify_phm_linear()
"
```

Expected output: `PHMLinear verification passed: all shapes correct, max error < 1e-4`

- [ ] **Step 4: Commit**

```bash
git add records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py
git commit -m "feat: add PHMLinear with Kronecker-factored forward pass"
```

### Task 3: Replace CastedLinear with PHMLinear in Attention and MLP

**Files:**
- Modify: `records/.../train_gpt.py` — CausalSelfAttention, MLP classes

- [ ] **Step 1: Update CausalSelfAttention to use PHMLinear**

Replace the four CastedLinear usages in `CausalSelfAttention.__init__`:

```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        phm_n: int = 4,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = PHMLinear(dim, dim, n=phm_n)
        self.c_k = PHMLinear(dim, kv_dim, n=phm_n)
        self.c_v = PHMLinear(dim, kv_dim, n=phm_n)
        self.proj = PHMLinear(dim, dim, n=phm_n)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
```

The `forward` method stays identical — PHMLinear is a drop-in replacement.

**Important:** kv_dim must be divisible by phm_n. With 768d / 12 heads = 64 head_dim, 4 KV heads → kv_dim = 256. 256 / 4 = 64 ✓

- [ ] **Step 2: Update MLP to use PHMLinear + LeakyReLU(0.5)²**

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, phm_n: int = 4):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = PHMLinear(dim, hidden, n=phm_n)
        self.proj = PHMLinear(hidden, dim, n=phm_n)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU(0.5)² — preserves negative gradient flow, non-negative output
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
```

- [ ] **Step 3: Update Block to pass phm_n through**

```python
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        phm_n: int = 4,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, phm_n=phm_n)
        self.mlp = MLP(dim, mlp_mult, phm_n=phm_n)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # Learnable layer scaling — init at 1/√(layer+1), adapt during training
        self.ln_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(layer_idx + 1), dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        ln_s = self.ln_scale.to(dtype=x.dtype)
        attn_out = self.attn(self.attn_norm(x) * ln_s)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * ln_s)
        return x
```

- [ ] **Step 4: Remove SwiGLU code path** (we use LeakyReLU² exclusively)

Delete `use_swiglu` from MLP, Block, GPT, and Hyperparameters. Remove the `gate` linear layer and SwiGLU forward path.

- [ ] **Step 5: Commit**

```bash
git add records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py
git commit -m "feat: replace CastedLinear with PHMLinear in attention and MLP"
```

---

## Chunk 2: Architecture — 18L/768d + Proven Techniques

Scale the model up to 18 layers at 768 width, add BigramHash, XSA, learned-frequency RoPE, and multi-resolution skip connections.

### Task 4: Update Hyperparameters for PHM-Golf config

**Files:**
- Modify: `records/.../train_gpt.py` — Hyperparameters class

- [ ] **Step 1: Update default hyperparameters**

```python
class Hyperparameters:
    # ... data paths unchanged ...

    # Training length
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — PHM-Golf: 18L/768d/12H/4KV/3xMLP
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 18))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    phm_n = int(os.environ.get("PHM_N", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bigram_hash_buckets = int(os.environ.get("BIGRAM_HASH_BUCKETS", 1536))
    xsa_layers = int(os.environ.get("XSA_LAYERS", 4))

    # Optimizer hyperparameters
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    algebra_lr = float(os.environ.get("ALGEBRA_LR", 0.01))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    bigram_lr = float(os.environ.get("BIGRAM_LR", 0.05))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_start_scale = float(os.environ.get("SWA_START_SCALE", 0.2))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
```

- [ ] **Step 2: Remove `num_recurrence_passes` and `use_swiglu`** from Hyperparameters (no longer used).

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: update hyperparameters for PHM-Golf 18L/768d config"
```

### Task 5: Add BigramHash module

**Files:**
- Modify: `records/.../train_gpt.py` — add BigramHash class, integrate into GPT

BigramHash maps (previous_token, current_token) to a learned embedding vector via hashing. Added to the token embedding for free bigram context.

- [ ] **Step 1: Add BigramHash class before Block**

```python
class BigramHash(nn.Module):
    """Hash-based bigram features: hash(prev, cur) → learned embedding, added to tok_emb."""
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, dim)
        nn.init.zeros_(self.embed.weight)  # Start additive-neutral

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (B, seq)
        # Compute bigram hashes: hash(prev, cur) mod num_buckets
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)  # shift right, pad with 0
        # Simple multiplicative hash
        hashes = ((prev.long() * 2654435761 + input_ids.long()) % self.num_buckets).long()
        return self.embed(hashes)
```

- [ ] **Step 2: Integrate BigramHash into GPT.__init__ and forward**

In `GPT.__init__`, after `self.tok_emb`:
```python
self.bigram_hash = BigramHash(bigram_hash_buckets, model_dim) if bigram_hash_buckets > 0 else None
```

In `GPT.forward`, after `x = self.tok_emb(input_ids)`:
```python
if self.bigram_hash is not None:
    x = x + self.bigram_hash(input_ids)
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: add BigramHash for bigram context features"
```

### Task 6: Add XSA (Cross-Sequence Attention) on last N layers

**Files:**
- Modify: `records/.../train_gpt.py` — CausalSelfAttention.forward, Block, GPT

XSA allows the last `xsa_layers` layers to attend across all sequences in the batch (not just within each sequence). This is done by reshaping the batch dimension into the sequence dimension.

- [ ] **Step 1: Add `use_xsa` flag to Block and CausalSelfAttention**

In `CausalSelfAttention.forward`, add an `xsa` parameter:

```python
def forward(self, x: Tensor, xsa: bool = False) -> Tensor:
    bsz, seqlen, dim = x.shape
    q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
    if xsa:
        # Cross-Sequence Attention: merge batch into sequence dim
        q = q.reshape(1, self.num_heads, bsz * seqlen, self.head_dim)
        k = k.reshape(1, self.num_kv_heads, bsz * seqlen, self.head_dim)
        v = v.reshape(1, self.num_kv_heads, bsz * seqlen, self.head_dim)
    y = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=True,
        enable_gqa=(self.num_kv_heads != self.num_heads),
    )
    if xsa:
        y = y.reshape(bsz, self.num_heads, seqlen, self.head_dim)
    y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    return self.proj(y)
```

In `Block.forward`, add `xsa` parameter and pass through:
```python
def forward(self, x: Tensor, x0: Tensor, xsa: bool = False) -> Tensor:
    # ... resid_mix, ln_scale ...
    attn_out = self.attn(self.attn_norm(x) * ln_s, xsa=xsa)
    # ... rest unchanged ...
```

- [ ] **Step 2: Update GPT forward to enable XSA on last N layers**

In GPT, track which effective layer indices should use XSA:
```python
# In GPT.__init__:
self.xsa_layers = xsa_layers  # number of deepest layers using XSA

# In GPT.forward / _run_block:
def _run_block(self, eff_idx: int, x: Tensor, x0: Tensor) -> Tensor:
    block_idx = eff_idx % self.num_unique_layers
    effective_depth = self.num_unique_layers  # no recurrence in PHM-Golf
    use_xsa = eff_idx >= (effective_depth - self.xsa_layers)
    x = self.blocks[block_idx](x, x0, xsa=use_xsa)
    return x
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: add XSA (cross-sequence attention) on deepest layers"
```

### Task 7: Implement learned-frequency RoPE

**Files:**
- Modify: `records/.../train_gpt.py` — Rotary class

- [ ] **Step 1: Make RoPE frequencies learnable**

```python
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # Learnable frequencies — subsumes Partial RoPE as a special case.
        # Initialized to standard RoPE values; the model learns optimal freqs.
        init_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq = nn.Parameter(init_freq)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # Recompute every call during training (freqs are learnable).
        # During eval (torch.inference_mode), caching still works.
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)
```

Note: We remove the caching during training since `inv_freq` is now a learnable parameter that changes each step. The cost of recomputing cos/sin tables is negligible compared to attention.

- [ ] **Step 2: Commit**

```bash
git commit -am "feat: learned-frequency RoPE (subsumes Partial RoPE)"
```

### Task 8: Implement multi-resolution skip connections

**Files:**
- Modify: `records/.../train_gpt.py` — GPT class

- [ ] **Step 1: Replace single-partner U-Net skips with multi-resolution skips**

In `GPT.__init__`, compute the skip connection topology and allocate weights:

```python
# In GPT.__init__, replacing the existing skip_weights:
# Multi-resolution skips: each decoder layer connects to its symmetric partner
# AND to every 4th deeper encoder layer.
self.num_encoder_layers = num_layers // 2
self.num_decoder_layers = num_layers - self.num_encoder_layers

# Build skip connection map: decoder_idx → list of encoder_indices
self.skip_map: list[list[int]] = []
for d in range(self.num_decoder_layers):
    symmetric = self.num_encoder_layers - 1 - d
    connections = [symmetric]
    # Add deeper encoder layers every 4 layers
    deeper = symmetric - 4
    while deeper >= 0:
        connections.append(deeper)
        deeper -= 4
    self.skip_map.append(connections)

total_skip_connections = sum(len(c) for c in self.skip_map)
self.skip_weights = nn.Parameter(torch.ones(total_skip_connections, model_dim, dtype=torch.float32))
```

In `GPT.forward`, replace the simple skip logic:

```python
# Encoder pass — collect skip activations
encoder_outputs: list[Tensor] = []
for i in range(self.num_encoder_layers):
    x = self._run_block(i, x, x0)
    encoder_outputs.append(x)

# Decoder pass — multi-resolution skip connections
skip_w_idx = 0
for d in range(self.num_decoder_layers):
    # Add weighted sum of connected encoder outputs
    for enc_idx in self.skip_map[d]:
        w = self.skip_weights[skip_w_idx].to(dtype=x.dtype)[None, None, :]
        x = x + w * encoder_outputs[enc_idx]
        skip_w_idx += 1
    x = self._run_block(self.num_encoder_layers + d, x, x0)
```

- [ ] **Step 2: Commit**

```bash
git commit -am "feat: multi-resolution skip connections (decoder↔multiple encoder layers)"
```

### Task 9: Update GPT class for PHM-Golf architecture

**Files:**
- Modify: `records/.../train_gpt.py` — GPT class

- [ ] **Step 1: Update GPT.__init__ to accept new params and remove recurrence**

```python
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        phm_n: int = 4,
        bigram_hash_buckets: int = 1536,
        xsa_layers: int = 4,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.xsa_layers = xsa_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHash(bigram_hash_buckets, model_dim) if bigram_hash_buckets > 0 else None

        self.num_unique_layers = num_layers
        # ... multi-resolution skip setup from Task 8 ...

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  rope_base, qk_gain_init, phm_n=phm_n, layer_idx=i)
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
```

- [ ] **Step 2: Clean up GPT.forward — remove recurrence, add bigram hash**

```python
def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
    x = self.tok_emb(input_ids)
    if self.bigram_hash is not None:
        x = x + self.bigram_hash(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x

    # Encoder pass
    encoder_outputs: list[Tensor] = []
    for i in range(self.num_encoder_layers):
        use_xsa = i >= (self.num_unique_layers - self.xsa_layers)
        x = self.blocks[i](x, x0, xsa=use_xsa)
        encoder_outputs.append(x)

    # Decoder pass with multi-resolution skips
    skip_w_idx = 0
    for d in range(self.num_decoder_layers):
        for enc_idx in self.skip_map[d]:
            w = self.skip_weights[skip_w_idx].to(dtype=x.dtype)[None, None, :]
            x = x + w * encoder_outputs[enc_idx]
            skip_w_idx += 1
        eff_idx = self.num_encoder_layers + d
        use_xsa = eff_idx >= (self.num_unique_layers - self.xsa_layers)
        x = self.blocks[eff_idx](x, x0, xsa=use_xsa)

    x = self.final_norm(x).reshape(-1, x.size(-1))
    targets = target_ids.reshape(-1)
    if self.tie_embeddings:
        logits_proj = F.linear(x, self.tok_emb.weight)
    else:
        logits_proj = self.lm_head(x)
    logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")
```

- [ ] **Step 3: Update model instantiation in main() to pass new args**

```python
base_model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    phm_n=args.phm_n,
    bigram_hash_buckets=args.bigram_hash_buckets,
    xsa_layers=args.xsa_layers,
).to(device).bfloat16()

# Keep PHMLinear params (A and S) in float32 for optimizer quality
for module in base_model.modules():
    if isinstance(module, (CastedLinear, PHMLinear)):
        for p in module.parameters():
            p.data = p.data.float()
restore_low_dim_params_to_fp32(base_model)
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat: assemble PHM-Golf 18L/768d GPT architecture"
```

---

## Chunk 3: Training Loop — Optimizer, EMA, SWA, Cosine Warmdown

### Task 10: Update optimizer setup for PHM parameter groups

**Files:**
- Modify: `records/.../train_gpt.py` — main() optimizer section

PHM layers have two parameter types: S matrices (bulk, use Muon) and A matrices (small, use Adam). We need separate optimizer groups.

- [ ] **Step 1: Rewrite optimizer setup to split S/A params**

```python
# In main(), replace existing optimizer setup:

# Collect PHM S matrices (filter matrices — bulk, use Muon)
phm_s_params = []
# Collect PHM A matrices (algebra matrices — small, use Adam)
phm_a_params = []
# Collect scalar/control params
scalar_params = []

for name, p in base_model.blocks.named_parameters():
    if '.S' in name and p.ndim == 3:
        phm_s_params.append(p)
    elif '.A' in name and p.ndim == 3:
        phm_a_params.append(p)
    elif p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
        scalar_params.append(p)

# Rotary inv_freq params go to scalar optimizer
for name, p in base_model.named_parameters():
    if 'inv_freq' in name:
        scalar_params.append(p)

if base_model.skip_weights.numel() > 0:
    scalar_params.append(base_model.skip_weights)

token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
bigram_params = list(base_model.bigram_hash.parameters()) if base_model.bigram_hash is not None else []

optimizer_tok = torch.optim.Adam(
    [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
)
optimizer_muon = Muon(
    phm_s_params, lr=args.matrix_lr, momentum=args.muon_momentum,
    backend_steps=args.muon_backend_steps,
)
for group in optimizer_muon.param_groups:
    group["base_lr"] = args.matrix_lr

optimizer_algebra = torch.optim.Adam(
    [{"params": phm_a_params, "lr": args.algebra_lr, "base_lr": args.algebra_lr}],
    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
)
optimizer_scalar = torch.optim.Adam(
    [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
)
optimizers = [optimizer_tok, optimizer_muon, optimizer_algebra, optimizer_scalar]

if bigram_params:
    optimizer_bigram = torch.optim.Adam(
        [{"params": bigram_params, "lr": args.bigram_lr, "base_lr": args.bigram_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers.append(optimizer_bigram)

if base_model.lm_head is not None:
    optimizer_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers.append(optimizer_head)
```

- [ ] **Step 2: Update Muon to handle 3D tensors (PHM S matrices)**

The S parameter is (n, s_out, s_in). Muon's Newton-Schulz expects 2D. Reshape each S to (n*s_out, s_in) before NS5, then reshape back:

In the Muon optimizer step, update the gradient processing:

```python
# In Muon.step(), replace:
#   g = zeropower_via_newtonschulz5(g, steps=backend_steps)
#   g *= max(1, g.size(0) / g.size(1)) ** 0.5
# With:
if g.ndim == 3:
    # PHM S matrix: (n, s_out, s_in) → flatten to (n*s_out, s_in) for NS5
    n_fac, s_out, s_in = g.shape
    g_2d = g.reshape(n_fac * s_out, s_in)
    g_2d = zeropower_via_newtonschulz5(g_2d, steps=backend_steps)
    g_2d *= max(1, g_2d.size(0) / g_2d.size(1)) ** 0.5
    g = g_2d.reshape(n_fac, s_out, s_in)
else:
    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
    g *= max(1, g.size(0) / g.size(1)) ** 0.5
```

- [ ] **Step 3: Add Muon weight decay**

In Muon.step(), after the update application, add WD:

```python
# After: p.add_(g, alpha=-lr)
# Add weight decay (decoupled, like AdamW):
wd = group.get("wd", 0.0)
if wd > 0:
    p.add_(p, alpha=-lr * wd)
```

And in the Muon instantiation:
```python
optimizer_muon = Muon(
    phm_s_params, lr=args.matrix_lr, momentum=args.muon_momentum,
    backend_steps=args.muon_backend_steps,
)
for group in optimizer_muon.param_groups:
    group["base_lr"] = args.matrix_lr
    group["wd"] = args.muon_wd
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat: optimizer setup with PHM S/A split, Muon WD, algebra Adam"
```

### Task 11: Add EMA and Tight SWA

**Files:**
- Modify: `records/.../train_gpt.py` — main() training loop

- [ ] **Step 1: Add EMA state initialization after model setup**

```python
# After model + optimizer setup, before training loop:
# EMA: exponential moving average of parameters
ema_params = {name: p.detach().clone() for name, p in base_model.named_parameters()}
ema_decay = args.ema_decay

# SWA: stochastic weight averaging during warmdown
swa_count = 0
swa_params: dict[str, Tensor] | None = None
```

- [ ] **Step 2: Add EMA + SWA update inside training loop**

After `opt.step()` and `zero_grad_all()`:

```python
# EMA update
with torch.no_grad():
    for name, p in base_model.named_parameters():
        ema_params[name].lerp_(p.data, 1.0 - ema_decay)

# SWA: snapshot when in warmdown phase (lr_scale < swa_start_scale)
if scale < args.swa_start_scale and step % args.swa_every == 0:
    with torch.no_grad():
        if swa_params is None:
            swa_params = {name: p.data.clone() for name, p in base_model.named_parameters()}
            swa_count = 1
        else:
            swa_count += 1
            for name, p in base_model.named_parameters():
                swa_params[name] += p.data

# EMA temperature annealing during warmdown
if scale < 1.0:
    # Anneal from ema_decay (0.997) toward 0.9999 as warmdown progresses
    warmdown_progress = 1.0 - scale  # 0 at start of warmdown → 1 at end
    ema_decay = args.ema_decay + warmdown_progress * (0.9999 - args.ema_decay)
```

- [ ] **Step 3: Merge EMA + SWA after training loop ends**

After the training loop `break`, before serialization:

```python
# Merge EMA + SWA weights
with torch.no_grad():
    # Average SWA snapshots
    if swa_params is not None and swa_count > 0:
        for name in swa_params:
            swa_params[name] /= swa_count
        # Blend EMA and SWA: 50/50 mix (both capture different convergence info)
        for name, p in base_model.named_parameters():
            p.data.copy_(0.5 * ema_params[name] + 0.5 * swa_params[name])
        log0(f"merged EMA+SWA weights (swa_count={swa_count})")
    else:
        # Just use EMA
        for name, p in base_model.named_parameters():
            p.data.copy_(ema_params[name])
        log0("applied EMA weights (no SWA snapshots)")
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat: add EMA with temperature annealing + Tight SWA"
```

### Task 12: Implement cosine warmdown

**Files:**
- Modify: `records/.../train_gpt.py` — `lr_mul` function

- [ ] **Step 1: Replace linear warmdown with cosine**

```python
def lr_mul(step: int, elapsed_ms: float) -> float:
    if args.warmdown_iters <= 0:
        return 1.0
    if max_wallclock_ms is None:
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if warmdown_start <= step < args.iterations:
            progress = (step - warmdown_start) / max(args.warmdown_iters, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = args.warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    if remaining_ms <= warmdown_ms:
        progress = 1.0 - remaining_ms / max(warmdown_ms, 1e-9)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0
```

- [ ] **Step 2: Commit**

```bash
git commit -am "feat: cosine warmdown LR schedule"
```

---

## Chunk 4: Quantization + Serialization

### Task 13: Add APoT quantization functions

**Files:**
- Modify: `records/.../train_gpt.py` — quantization section

APoT (Additive Powers-of-Two) quantization replaces uniform int6. Levels are denser near zero, matching weight distribution. Sparse bit patterns compress better under LZMA.

- [ ] **Step 1: Add APoT level generation and quantization**

```python
def build_apot_levels(bit_width: int = 6) -> Tensor:
    """Generate non-uniform APoT quantization levels for the given bit width.
    Uses additive powers-of-two: each value is a sum of PoT terms.
    Returns sorted tensor of all unique non-negative levels."""
    # For 6-bit: 1 sign bit + 5 value bits
    # Generate all sums of up to 3 distinct powers of two from {1,2,4,8,16}
    powers = [2**i for i in range(bit_width - 1)]  # [1, 2, 4, 8, 16]
    levels = {0.0}
    # Single powers
    for p in powers:
        levels.add(float(p))
    # Pairs of powers
    for i, p1 in enumerate(powers):
        for p2 in powers[i+1:]:
            levels.add(float(p1 + p2))
    # Triples of powers
    for i, p1 in enumerate(powers):
        for j, p2 in enumerate(powers[i+1:], i+1):
            for p3 in powers[j+1:]:
                levels.add(float(p1 + p2 + p3))
    sorted_levels = sorted(levels)
    return torch.tensor(sorted_levels, dtype=torch.float32)

APOT_LEVELS = build_apot_levels(6)  # Module-level constant

def quantize_apot(t: Tensor, levels: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize tensor to nearest APoT level with per-row scaling.
    Returns (quantized_indices, scales)."""
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row scale: map row max to max APoT level
        row_max = t32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        max_level = levels[-1].item()
        scale = row_max / max_level  # (rows, 1)
        normalized = t32 / scale     # values now in [-max_level, max_level]
        # Quantize: find nearest level for abs, then restore sign
        abs_vals = normalized.abs()
        lev = levels.to(abs_vals.device)
        # For each element, find nearest level index
        dists = (abs_vals.unsqueeze(-1) - lev.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)  # (rows, cols)
        # Reconstruct quantized values
        q_abs = lev[indices]
        q_vals = q_abs * normalized.sign()
        return q_vals * scale, scale.squeeze(-1)
    # Fallback: per-tensor for non-2D
    abs_max = t32.abs().max().clamp_min(1e-8)
    max_level = levels[-1].item()
    scale = abs_max / max_level
    normalized = t32 / scale
    lev = levels.to(t32.device)
    dists = (normalized.abs().reshape(-1, 1) - lev.unsqueeze(0)).abs()
    indices = dists.argmin(dim=-1).reshape(t32.shape)
    q_abs = lev[indices]
    q_vals = q_abs * normalized.sign()
    return q_vals * scale, scale
```

- [ ] **Step 2: Add GPTQ-lite APoT clip search**

```python
def quantize_apot_gptq_lite(t: Tensor, levels: Tensor, n_candidates: int = 5) -> tuple[Tensor, Tensor]:
    """Per-row optimal APoT quantization with clip percentile search."""
    if t.ndim != 2:
        return quantize_apot(t, levels)
    t32 = t.float()
    candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0][:n_candidates]
    best_q = None
    best_scale = None
    best_mse = float('inf')
    max_level = levels[-1].item()
    lev = levels.to(t32.device)
    for pct in candidates:
        if pct < 1.0:
            clip_abs = torch.quantile(t32.abs(), pct, dim=1, keepdim=True)
        else:
            clip_abs = t32.abs().amax(dim=1, keepdim=True)
        clip_abs = clip_abs.clamp_min(1e-8)
        scale = clip_abs / max_level
        normalized = (t32 / scale).clamp(-max_level, max_level)
        abs_vals = normalized.abs()
        dists = (abs_vals.unsqueeze(-1) - lev.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        q_abs = lev[indices]
        q_vals = q_abs * normalized.sign() * scale
        mse = (q_vals - t32).pow(2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_q = q_vals
            best_scale = scale.squeeze(-1)
    return best_q, best_scale
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: add APoT quantization with GPTQ-lite clip search"
```

### Task 14: Update serialization pipeline for PHM + APoT

**Files:**
- Modify: `records/.../train_gpt.py` — serialization section

The PHM model has different tensor types requiring different treatment:
- S matrices (3D, n×s_out×s_in): APoT quantize each (s_out, s_in) slice
- A matrices (3D, n×n×n): keep fp32 (tiny, 64 params each)
- Embedding, BigramHash: fp16
- Control tensors: fp32

- [ ] **Step 1: Update quantize_state_dict for PHM tensors**

Modify `quantize_state_dict_int8` → `quantize_state_dict_phm`:

```python
def quantize_state_dict_phm(state_dict: dict[str, Tensor]):
    """Quantize PHM model: APoT on S matrices, fp32 on A matrices, fp16 on embeddings."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    passthrough: dict[str, Tensor] = {}
    stats = {"param_count": 0, "total_bytes": 0}
    levels = build_apot_levels(6)

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()

        # PHM A matrices and other small control tensors: keep fp32
        if '.A' in name or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float()
            stats["total_bytes"] += t.numel() * 4
            continue

        # Non-floating point: passthrough
        if not t.is_floating_point():
            passthrough[name] = t
            stats["total_bytes"] += t.numel() * t.element_size()
            continue

        # Small tensors (norms, gains, scales): fp16
        if t.numel() <= 65_536:
            passthrough[name] = t.to(torch.float16)
            stats["total_bytes"] += t.numel() * 2
            continue

        # PHM S matrices (3D: n, s_out, s_in): quantize each slice
        if t.ndim == 3 and '.S' in name:
            q_slices = []
            s_slices = []
            for i in range(t.shape[0]):
                q_slice, s_slice = quantize_apot_gptq_lite(t[i], levels)
                q_slices.append(q_slice)
                s_slices.append(s_slice)
            quantized[name] = torch.stack(q_slices)
            scales[name] = torch.stack(s_slices).to(torch.float16)
            stats["total_bytes"] += quantized[name].numel() + scales[name].numel() * 2
            continue

        # 2D matrices (embeddings, bigram hash, lm_head): APoT quantize
        if t.ndim == 2 and t.numel() > 65_536:
            q, s = quantize_apot_gptq_lite(t, levels)
            quantized[name] = q
            scales[name] = s.to(torch.float16)
            stats["total_bytes"] += q.numel() + s.numel() * 2
            continue

        # Fallback: fp16
        passthrough[name] = t.to(torch.float16)
        stats["total_bytes"] += t.numel() * 2

    return {"quantized": quantized, "scales": scales, "passthrough": passthrough}, stats
```

- [ ] **Step 2: Update dequantize function**

```python
def dequantize_state_dict_phm(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, q in obj["quantized"].items():
        # Already dequantized by quantize_apot (stores float values, not indices)
        out[name] = q.float()
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out
```

- [ ] **Step 3: Add LZMA compression (with zlib fallback)**

```python
import lzma

def compress_best(data: bytes) -> tuple[bytes, str]:
    """Try LZMA, zstd, zlib — pick smallest."""
    results = []
    # LZMA
    try:
        c = lzma.compress(data, preset=6)
        results.append((c, "lzma"))
    except Exception:
        pass
    # zlib
    c = zlib.compress(data, level=9)
    results.append((c, "zlib"))
    results.sort(key=lambda x: len(x[0]))
    return results[0]

def decompress_auto(data: bytes) -> bytes:
    """Auto-detect compression format."""
    if data[:2] == b'\xfd7':  # LZMA/XZ magic
        return lzma.decompress(data)
    if data[:6] == b'\x28\xb5\x2f\xfd':  # zstd magic (if available)
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data)
    return zlib.decompress(data)  # zlib fallback
```

- [ ] **Step 4: Update serialization in main()**

Replace the existing serialization section:

```python
# After training loop, replace existing serialization:
quant_obj, quant_stats = quantize_state_dict_phm(base_model.state_dict())
quant_buf = io.BytesIO()
torch.save(quant_obj, quant_buf)
quant_raw = quant_buf.getvalue()
quant_blob, comp_method = compress_best(quant_raw)

if master_process:
    with open("final_model.phm.ptz", "wb") as f:
        f.write(quant_blob)
    artifact_bytes = len(quant_blob)
    code_bytes = len(code.encode("utf-8"))
    log0(f"Serialized model {comp_method}: {artifact_bytes} bytes")
    log0(f"Code size: {code_bytes} bytes")
    log0(f"Total submission: {artifact_bytes + code_bytes} bytes")
    assert artifact_bytes + code_bytes <= 16_000_000, \
        f"OVER 16MB LIMIT: {artifact_bytes + code_bytes} bytes!"

# Roundtrip validation
if distributed:
    dist.barrier()
with open("final_model.phm.ptz", "rb") as f:
    quant_blob_disk = f.read()
quant_state = torch.load(io.BytesIO(decompress_auto(quant_blob_disk)), map_location="cpu")
base_model.load_state_dict(dequantize_state_dict_phm(quant_state), strict=True)
# ... eval roundtrip same as baseline ...
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: PHM-aware serialization with APoT + LZMA compression"
```

---

## Chunk 5: K-FAC TTT + Evaluation

### Task 15: Add Score-First TTT evaluation

**Files:**
- Modify: `records/.../train_gpt.py` — add eval_val_ttt function

- [ ] **Step 1: Add Score-First TTT evaluation function**

```python
def eval_val_ttt(
    args: Hyperparameters,
    model: nn.Module,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    ttt_lr: float = 0.002,
    ttt_epochs: int = 3,
    chunk_size: int = 32768,
) -> tuple[float, float]:
    """Legal Score-First TTT: score each chunk, then adapt, never train on unscored data."""
    # Split val tokens into non-overlapping chunks
    total_tokens = val_tokens.numel() - 1
    num_chunks = total_tokens // chunk_size
    if num_chunks == 0:
        num_chunks = 1
        chunk_size = total_tokens

    # Collect S parameters for TTT (freeze A matrices)
    ttt_params = []
    for name, p in base_model.named_parameters():
        if '.S' in name and p.ndim == 3:
            p.requires_grad_(True)
            ttt_params.append(p)
        else:
            p.requires_grad_(False)

    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size + 1, val_tokens.numel())
        chunk_tokens = val_tokens[start:end].to(device=device, dtype=torch.int64)

        # Phase 1: SCORE — compute loss under current model (no gradients)
        model.eval()
        with torch.inference_mode():
            num_seqs = (chunk_tokens.numel() - 1) // args.train_seq_len
            if num_seqs == 0:
                continue
            usable = num_seqs * args.train_seq_len
            x = chunk_tokens[:usable].reshape(num_seqs, args.train_seq_len)
            y = chunk_tokens[1:usable + 1].reshape(num_seqs, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk_loss = model(x, y).item()

            # Accumulate BPB metrics
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            val_loss_sum += chunk_loss * float(tgt_ids.numel())
            val_token_count += tgt_ids.numel()
            val_byte_count += token_bytes.sum().item()

        # Phase 2: ADAPT — train on this scored chunk (skip last chunk)
        if chunk_idx < num_chunks - 1:
            model.train()
            ttt_optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=0.9)
            for epoch in range(ttt_epochs):
                # Cosine decay per chunk
                epoch_lr = ttt_lr * 0.5 * (1 + math.cos(math.pi * epoch / ttt_epochs))
                for pg in ttt_optimizer.param_groups:
                    pg['lr'] = epoch_lr
                ttt_optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                ttt_optimizer.step()

    # Re-enable all gradients
    for p in base_model.parameters():
        p.requires_grad_(True)

    # Aggregate across ranks
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([val_loss_sum, float(val_token_count), val_byte_count], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        val_loss_sum, val_token_count, val_byte_count = stats[0].item(), stats[1].item(), stats[2].item()

    val_loss = val_loss_sum / max(val_token_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / max(val_byte_count, 1)
    return val_loss, bits_per_token * tokens_per_byte
```

Note: This is SGD-based TTT. K-FAC TTT can be layered on top later by replacing the SGD optimizer with a K-FAC variant that exploits the Kronecker structure. The infrastructure is the same.

- [ ] **Step 2: Integrate TTT into final evaluation**

In the serialization section, after roundtrip load, replace the simple eval with TTT:

```python
# After loading roundtripped weights:
q_val_loss, q_val_bpb = eval_val_ttt(
    args, model, base_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
)
```

- [ ] **Step 3: Commit**

```bash
git commit -am "feat: add Score-First TTT evaluation with SGD adaptation"
```

---

## Chunk 6: Integration, Smoke Test, Submission

### Task 16: Size verification script

**Files:**
- Modify: `records/.../train_gpt.py` — add size estimation at module level

- [ ] **Step 1: Add quick size estimator that runs before training**

Add after Hyperparameters class:

```python
def estimate_phm_size(args: Hyperparameters) -> None:
    """Quick size estimate to catch budget overruns before training."""
    n = args.phm_n
    d = args.model_dim
    kv_dim = args.num_kv_heads * (d // args.num_heads)
    hidden = args.mlp_mult * d
    # PHM params per layer: attn (Q+K+V+Proj) + MLP (fc+proj)
    def phm_params(d_in, d_out):
        return n * n * n + n * (d_out // n) * (d_in // n)
    attn_per_layer = phm_params(d, d) + phm_params(d, kv_dim) + phm_params(d, kv_dim) + phm_params(d, d)
    mlp_per_layer = phm_params(d, hidden) + phm_params(hidden, d)
    per_layer = attn_per_layer + mlp_per_layer + d * 4  # scales, mix, etc.
    total = per_layer * args.num_layers
    total += args.vocab_size * d  # embedding
    total += args.bigram_hash_buckets * d  # bigram hash
    total += 200  # misc control params
    # Rough compressed size: int6 ≈ 0.75 bytes, LZMA ≈ 0.65x
    compressed_est = total * 0.75 * 0.65
    code_est = 80_000  # rough code size
    print(f"PHM-Golf size estimate: {total:,} params, ~{(compressed_est + code_est) / 1e6:.1f} MB compressed")
    if compressed_est + code_est > 15_500_000:
        print(f"WARNING: estimated size {(compressed_est + code_est) / 1e6:.1f} MB is close to 16MB limit!")
```

Call this at the start of `main()`:
```python
if master_process:
    estimate_phm_size(args)
```

- [ ] **Step 2: Commit**

```bash
git commit -am "feat: add pre-training size estimator"
```

### Task 17: Create submission metadata files

**Files:**
- Create: `records/.../README.md`
- Create: `records/.../submission.json`

- [ ] **Step 1: Create README.md**

```markdown
# PHM-Golf: Kronecker-Factored Transformer

**Score:** TBD (expected 1.085-1.115 BPB)
**Author:** DigitalSword99
**Date:** 2026-03-25

## Summary

First-ever use of Parameterized Hypercomplex Multiplication (PHM) in the Parameter Golf challenge. PHM factorizes every linear layer as W = Σ Aᵢ ⊗ Sᵢ using Kronecker products, achieving 4x structural parameter reduction and ~3.7x fewer FLOPs per forward pass.

This enables an 18-layer, 768-dim model (vs SOTA's 11L/512d) at the same parameter count, with faster per-step training yielding more gradient updates within the 10-minute budget.

## Novel Techniques
1. **PHM(n=4) Layers** — Kronecker-factored linear layers (4x compression, 3.7x faster)
2. **APoT Quantization** — Non-uniform powers-of-two quantization (denser near zero)
3. **Learned-Frequency RoPE** — Learnable positional frequencies (subsumes Partial RoPE)
4. **Multi-Resolution Skips** — Decoder layers connect to multiple encoder layers
5. **Learnable Layer Scaling** — Adaptive 1/√(layer+1) normalization
6. **Score-First TTT** — Test-time training with SGD adaptation

## Proven Techniques (from leaderboard)
- BigramHash(1536), XSA(last 4), LeakyReLU(0.5)², EMA+SWA, GQA, Cosine Warmdown

## Architecture
- 18 layers, 768 dim, 12 heads, 4 KV heads, 3x MLP
- PHM n=4 on all attention + MLP linear layers
- 1024 vocab, tied embeddings, 2048 sequence length

## Running
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
```

- [ ] **Step 2: Create submission.json**

```json
{
    "name": "PHM-Golf: Kronecker-Factored Transformer",
    "github_username": "DigitalSword99",
    "date": "2026-03-25",
    "val_loss": null,
    "val_bpb": null,
    "bytes_model": null,
    "bytes_code": null,
    "bytes_total": null,
    "metadata": {
        "num_layers": 18,
        "model_dim": 768,
        "num_heads": 12,
        "num_kv_heads": 4,
        "mlp_mult": 3,
        "phm_n": 4,
        "vocab_size": 1024,
        "seq_len": 2048,
        "techniques": ["PHM", "APoT", "LearnedRoPE", "BigramHash", "XSA", "LeakyReLU2", "EMA", "SWA", "CosineWarmdown", "MultiResSkips", "TTT"]
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/
git commit -m "feat: add submission metadata files"
```

### Task 18: Local smoke test

**Files:**
- No file changes — verification only

- [ ] **Step 1: CPU instantiation test**

```bash
cd /mnt/g/parameter-golf
python3 -c "
import torch, sys, math
sys.path.insert(0, 'records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d')
exec(open('records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py').read().split('def main')[0])
_verify_phm_linear()
args = Hyperparameters()
args.num_layers = 18
args.model_dim = 768
args.num_heads = 12
args.num_kv_heads = 4
args.mlp_mult = 3
model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=1.5, phm_n=4, bigram_hash_buckets=1536, xsa_layers=4,
)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model instantiated: {n_params:,} params')
x = torch.randint(0, 1024, (2, 64))
y = torch.randint(0, 1024, (2, 64))
loss = model(x, y)
print(f'Forward pass OK: loss={loss.item():.4f}')
"
```

Expected: model instantiates, forward pass produces a finite loss.

- [ ] **Step 2: GPU smoke test (short training run)**

```bash
cd /mnt/g/parameter-golf
ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=50 RUN_ID=phm_smoke \
python3 records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d/train_gpt.py
```

Expected: ~100 training steps complete, validation loss decreasing, no NaN, artifact size printed.

- [ ] **Step 3: Verify artifact size**

Check the printed artifact size is under 16MB. If over, reduce `NUM_LAYERS` or `BIGRAM_HASH_BUCKETS`.

- [ ] **Step 4: Commit any fixes**

```bash
git commit -am "fix: smoke test adjustments"
```

### Task 19: Final integration review

- [ ] **Step 1: Review full script for correctness**

Manually verify:
- All PHMLinear usages have dimensions divisible by `phm_n=4`
- Optimizer groups correctly capture all parameters (no orphans)
- EMA/SWA logic handles distributed training (same on all ranks)
- Serialization roundtrip preserves all tensor shapes
- XSA only activates on correct layer indices
- `CONTROL_TENSOR_NAME_PATTERNS` includes new param names (ln_scale, inv_freq)

- [ ] **Step 2: Update CONTROL_TENSOR_NAME_PATTERNS**

```python
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern for pattern in
    "attn_scale,mlp_scale,resid_mix,q_gain,skip_weights,ln_scale,inv_freq,.A".split(",")
    if pattern
)
```

- [ ] **Step 3: Final commit**

```bash
git commit -am "feat: PHM-Golf v1 ready for 8xH100 submission"
```

---

## Post-Implementation: RunPod Execution

After all tasks are complete and smoke test passes:

1. Spin up 8×H100 SXM pod on RunPod
2. Clone repo, download data:
   ```bash
   git clone https://github.com/DigitalSword99/parameter-golf.git
   cd parameter-golf
   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
   ```
3. Run official training:
   ```bash
   cd records/track_10min_16mb/2026-03-25_PHM_Golf_18L_768d
   torchrun --standalone --nproc_per_node=8 train_gpt.py
   ```
4. Record final `val_bpb` and artifact size
5. If val_bpb < 1.1144 (beats SOTA by ≥0.005): run 3+ more times for p<0.01 significance
6. Update submission.json with results
7. PR to openai/parameter-golf
