# Hydra: BitNet b1.58 Transformer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BitNet b1.58 ternary-weight transformer that fits ~65M params in 15MB, combining with SmearGate, SWA, and sliding window eval to achieve competitive BPB on the Parameter Golf challenge.

**Architecture:** 12-layer transformer at 768 model dim with ternary weights ({-1,0,+1}) trained via QAT with STE. Weights serialized as base-3 packed trits + LZMA compression. Proven techniques (SmearGate, SWA, sliding window eval) stacked on top.

**Tech Stack:** PyTorch, torchrun (distributed), CUDA, LZMA/zlib/zstd compression.

---

## File Structure

Only ONE file is created/modified (challenge requirement: all code lives in `train_gpt.py`):

- **Create:** `records/track_10min_16mb/2026-03-20_Hydra_BitNet158_12L_768d/train_gpt.py` — the submission script
- **Create:** `records/track_10min_16mb/2026-03-20_Hydra_BitNet158_12L_768d/submission.json` — metadata
- **Create:** `records/track_10min_16mb/2026-03-20_Hydra_BitNet158_12L_768d/README.md` — submission docs
- **Modify:** `runpod_package/run.sh` — updated run config for Hydra
- **Modify:** `runpod_package/train_gpt.py` — copy of submission train_gpt.py

The submission `train_gpt.py` is built by modifying the existing `/mnt/g/parameter-golf/train_gpt.py` baseline.

---

## Chunk 1: Core BitNet Modules

### Task 1: Add BitLinear class with STE quantization

**Files:**
- Modify: `train_gpt.py` (add after CastedLinear class, ~line 516)

- [ ] **Step 1: Add BitLinear class**

Replace `CastedLinear` usage in attention/MLP with `BitLinear` — a linear layer that quantizes weights to ternary {-1, 0, +1} in the forward pass using STE.

```python
class BitLinear(nn.Linear):
    """Linear layer with BitNet b1.58 ternary weight quantization.
    Weights are kept in fp32 for optimizer updates. In forward pass,
    weights are quantized to {-1, 0, +1} per group via absmean scaling.
    STE passes gradients through the quantizer unchanged.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size
        self._cached_q: Tensor | None = None
        self._cached_scale: Tensor | None = None
        self._cached_shape: tuple | None = None

    def _quantize_weights(self, w: Tensor) -> Tensor:
        shape = w.shape
        g = self.group_size
        w_flat = w.reshape(-1, g)
        scale = w_flat.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
        q = (w_flat / scale).round().clamp(-1, 1)
        self._cached_q = q.detach().to(torch.int8)
        self._cached_scale = scale.detach().squeeze(-1).half()
        self._cached_shape = shape
        result = q * scale
        return (result.reshape(shape) - w).detach() + w  # STE

    def forward(self, x: Tensor) -> Tensor:
        x_norm = F.rms_norm(x, (x.size(-1),))
        w_q = self._quantize_weights(self.weight)
        return F.linear(x_norm, w_q.to(x_norm.dtype), self.bias)
```

- [ ] **Step 2: Verify BitLinear on CPU**

```python
# Quick CPU sanity check
layer = BitLinear(512, 512, bias=False)
x = torch.randn(2, 10, 512)
y = layer(x)
assert y.shape == (2, 10, 512)
loss = y.sum()
loss.backward()
assert layer.weight.grad is not None
assert layer._cached_q is not None
assert layer._cached_q.unique().tolist() == [-1, 0, 1] or set(layer._cached_q.unique().tolist()).issubset({-1, 0, 1})
print(f"BitLinear OK: output={y.shape}, cached_q={layer._cached_q.shape}")
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add BitLinear with ternary QAT via STE"
```

### Task 2: Add base-3 ternary packing/unpacking

**Files:**
- Modify: `train_gpt.py` (add after BitLinear, before data loading)

- [ ] **Step 1: Add pack_ternary and unpack_ternary functions**

```python
def pack_ternary(q_int8: Tensor) -> bytes:
    """Pack ternary values {-1, 0, +1} as base-3: 5 trits per byte."""
    flat = (q_int8.flatten().to(torch.int16) + 1).numpy().astype(np.uint8)  # {0,1,2}
    # Pad to multiple of 5
    pad = (5 - len(flat) % 5) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    flat = flat.reshape(-1, 5)
    packed = (flat[:, 0] + 3 * flat[:, 1] + 9 * flat[:, 2] + 27 * flat[:, 3] + 81 * flat[:, 4]).astype(np.uint8)
    return packed.tobytes()


def unpack_ternary(data: bytes, numel: int) -> Tensor:
    """Unpack base-3 packed bytes back to ternary {-1, 0, +1}."""
    packed = np.frombuffer(data, dtype=np.uint8)
    trits = np.zeros((len(packed), 5), dtype=np.int8)
    vals = packed.astype(np.uint16)
    for i in range(5):
        trits[:, i] = vals % 3
        vals //= 3
    flat = trits.flatten()[:numel]
    return torch.from_numpy((flat.astype(np.int8) - 1).copy())
```

- [ ] **Step 2: Verify packing roundtrip on CPU**

```python
# Roundtrip test
q = torch.randint(-1, 2, (1024, 768), dtype=torch.int8)
packed = pack_ternary(q)
unpacked = unpack_ternary(packed, q.numel()).reshape(q.shape)
assert torch.equal(q, unpacked), "Pack/unpack roundtrip failed!"
bits_per_param = len(packed) * 8 / q.numel()
print(f"Packing OK: {q.numel()} trits -> {len(packed)} bytes ({bits_per_param:.2f} bits/param)")
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add base-3 ternary packing for BitNet serialization"
```

### Task 3: Add BitNet quantization and serialization pipeline

**Files:**
- Modify: `train_gpt.py` (replace/extend the existing quantize_state_dict_int8)

- [ ] **Step 1: Add quantize_state_dict_bitnet and dequantize functions**

```python
def quantize_state_dict_bitnet(model: nn.Module) -> dict:
    """Serialize BitLinear layers using cached ternary weights + base-3 packing.
    Non-BitLinear params (embeddings, scales, norms) stored as fp16."""
    quantized = {}
    passthrough = {}
    stats = {"param_count": 0, "payload_bytes": 0}

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module._cached_q is not None:
            q = module._cached_q.cpu()
            scale = module._cached_scale.cpu()
            shape = module._cached_shape
            packed = pack_ternary(q)
            quantized[name + ".weight"] = {
                "packed": packed,
                "scale": scale,
                "shape": list(shape),
                "numel": q.numel(),
                "group_size": module.group_size,
            }
            stats["param_count"] += q.numel()
            stats["payload_bytes"] += len(packed) + scale.numel() * 2  # packed + fp16 scales

    for name, param in model.named_parameters():
        if name not in quantized:
            t = param.detach().cpu()
            if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16}:
                t = t.half()
            passthrough[name] = t
            stats["param_count"] += t.numel()
            stats["payload_bytes"] += t.numel() * t.element_size()

    return {"__format__": "bitnet158_base3_v1", "quantized": quantized, "passthrough": passthrough}, stats


def dequantize_state_dict_bitnet(obj: dict) -> dict[str, Tensor]:
    """Reconstruct state dict from BitNet serialized format."""
    out = {}
    for name, entry in obj["quantized"].items():
        q = unpack_ternary(entry["packed"], entry["numel"])
        q = q.reshape(-1, entry["group_size"]).float()
        scale = entry["scale"].float().unsqueeze(-1)
        w = (q * scale).reshape(entry["shape"])
        out[name] = w
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out
```

- [ ] **Step 2: Add multi-compressor selection (LZMA > zstd > zlib)**

```python
def compress_best(raw: bytes) -> tuple[bytes, str]:
    """Try multiple compressors, return smallest result."""
    import lzma as _lzma
    candidates = []
    # zlib
    candidates.append((zlib.compress(raw, 9), "zlib"))
    # lzma
    try:
        candidates.append((_lzma.compress(raw, preset=9), "lzma"))
    except Exception:
        pass
    # zstd (if available)
    try:
        import zstd
        candidates.append((zstd.compress(raw, 22), "zstd"))
    except ImportError:
        pass
    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0]


def decompress_auto(blob: bytes) -> bytes:
    """Auto-detect and decompress."""
    import lzma as _lzma
    # Try lzma first (has magic bytes), then zlib, then zstd
    try:
        return _lzma.decompress(blob)
    except Exception:
        pass
    try:
        return zlib.decompress(blob)
    except Exception:
        pass
    try:
        import zstd
        return zstd.decompress(blob)
    except Exception:
        pass
    raise ValueError("Could not decompress blob with any known compressor")
```

- [ ] **Step 3: Verify full serialization roundtrip on CPU**

Build a small model, run a forward pass (to populate _cached_q), serialize, compress, decompress, dequantize, verify weights match.

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add BitNet serialization with base-3 packing and multi-compressor"
```

---

## Chunk 2: Architecture Modifications

### Task 4: Replace CastedLinear with BitLinear in attention and MLP

**Files:**
- Modify: `train_gpt.py` — CausalSelfAttention and MLP classes

- [ ] **Step 1: Update CausalSelfAttention to use BitLinear**

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        # ... (same structure, but BitLinear instead of CastedLinear)
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = BitLinear(dim, dim, bias=False)
        self.c_k = BitLinear(dim, kv_dim, bias=False)
        self.c_v = BitLinear(dim, kv_dim, bias=False)
        self.proj = BitLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        # ... rest unchanged
```

Note: BitLinear already applies RMSNorm to input, but the attention forward pass also has its own attn_norm. Keep the block-level attn_norm — BitLinear's internal norm is on the activation dimension, while attn_norm normalizes the residual stream. Both are needed.

- [ ] **Step 2: Update MLP to use BitLinear**

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, use_swiglu=False):
        super().__init__()
        hidden = mlp_mult * dim
        self.use_swiglu = use_swiglu
        self.fc = BitLinear(dim, hidden, bias=False)
        if use_swiglu:
            self.gate = BitLinear(dim, hidden, bias=False)
        self.proj = BitLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
```

- [ ] **Step 3: Keep embedding as nn.Embedding (FP16, not ternary)**

The tok_emb stays as regular `nn.Embedding`. It's small (1024*768 = 786K params) and sensitive to quantization. Stored as fp16 in passthrough.

- [ ] **Step 4: CPU test — instantiate full model, verify param count and shapes**

```python
model = GPT(vocab_size=1024, num_layers=12, model_dim=768, num_heads=12,
            num_kv_heads=6, mlp_mult=3, tie_embeddings=True,
            tied_embed_init_std=0.005, logit_softcap=30.0,
            rope_base=200000.0, qk_gain_init=1.5)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params:,}")
# Expected: ~64-65M
```

- [ ] **Step 5: Commit**

```bash
git add train_gpt.py
git commit -m "feat: replace CastedLinear with BitLinear in attention and MLP"
```

### Task 5: Add SmearGate

**Files:**
- Modify: `train_gpt.py` — GPT class

- [ ] **Step 1: Add SmearGate to GPT forward pass**

Add a learned per-dimension sigmoid gate that blends each token with its predecessor, applied once on embeddings before the transformer blocks:

```python
# In GPT.__init__:
self.smear_gate_param = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

# In GPT.forward, after tok_emb and before RMSNorm:
def _apply_smear_gate(self, x: Tensor) -> Tensor:
    g = torch.sigmoid(self.smear_gate_param.to(dtype=x.dtype))[None, None, :]
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    return (1 - g) * x + g * x_prev

# forward():
x = self.tok_emb(input_ids)
x = self._apply_smear_gate(x)
x = F.rms_norm(x, (x.size(-1),))
x0 = x
# ... rest of forward unchanged
```

- [ ] **Step 2: Verify SmearGate on CPU**

```python
model = GPT(...)  # small test model
x = torch.randint(0, 1024, (2, 32))
y = torch.randint(0, 1024, (2, 32))
loss = model(x, y)
loss.backward()
assert model.smear_gate_param.grad is not None
print(f"SmearGate grad: {model.smear_gate_param.grad.abs().mean():.6f}")
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add SmearGate for previous-token blending"
```

### Task 6: Update hyperparameters for Hydra config

**Files:**
- Modify: `train_gpt.py` — Hyperparameters class

- [ ] **Step 1: Update default hyperparameters**

```python
class Hyperparameters:
    # ... data paths unchanged ...

    # Model shape (Hydra BitNet config)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 6))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    num_recurrence_passes = int(os.environ.get("NUM_RECURRENCE_PASSES", 1))
    use_swiglu = bool(int(os.environ.get("USE_SWIGLU", "0")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 200000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))

    # Optimizer (tuned for BitNet)
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))  # LR warmup (not just compile)
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))

    # SWA
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
    swa_every = int(os.environ.get("SWA_EVERY", 100))

    # Sliding window eval
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
```

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: update hyperparameters for Hydra BitNet config"
```

---

## Chunk 3: Training Loop Modifications

### Task 7: Add LR warmup to training loop

**Files:**
- Modify: `train_gpt.py` — lr_mul function and training loop

- [ ] **Step 1: Add linear LR warmup**

Modify `lr_mul` to include a linear warmup phase:

```python
def lr_mul(step: int, elapsed_ms: float) -> float:
    # Linear warmup
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    # Warmdown (unchanged logic)
    if args.warmdown_iters <= 0:
        return 1.0
    # ... rest of existing warmdown code
```

Also change the warmup loop to be a proper LR warmup instead of just compile warmup — remove the weight reset after warmup, or keep compile warmup separate and add LR warmup on top.

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add linear LR warmup for training stability"
```

### Task 8: Add SWA (Stochastic Weight Averaging)

**Files:**
- Modify: `train_gpt.py` — training loop

- [ ] **Step 1: Add SWA state collection during warmdown**

```python
# Before training loop:
swa_state: dict[str, Tensor] | None = None
swa_count = 0

# Inside training loop, after optimizer step:
if args.swa_enabled:
    scale = lr_mul(step, elapsed_ms)
    if scale < args.swa_start_frac and step % args.swa_every == 0:
        if swa_state is None:
            swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
            swa_count = 1
        else:
            for n, t in base_model.state_dict().items():
                swa_state[n] += t.detach().cpu()
            swa_count += 1

# After training loop, before serialization:
if swa_state is not None and swa_count > 1:
    log0(f"swa: averaging {swa_count} checkpoints")
    avg = {n: (t / swa_count).to(dtype=base_model.state_dict()[n].dtype)
           for n, t in swa_state.items()}
    base_model.load_state_dict(avg, strict=True)
```

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add SWA checkpoint averaging during warmdown"
```

### Task 9: Add sliding window evaluation

**Files:**
- Modify: `train_gpt.py` — eval_val function

- [ ] **Step 1: Modify eval_val to use overlapping sliding windows**

Replace the existing non-overlapping eval with stride-based overlapping windows. Each token (except the first window) gets scored with nearly full context.

Key changes:
- Split val tokens into overlapping windows of length `seq_len`, advancing by `eval_stride`
- For each window, only score the last `eval_stride` tokens (first window scores all)
- Distribute windows across GPUs

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add sliding window evaluation with configurable stride"
```

### Task 10: Replace serialization pipeline with BitNet format

**Files:**
- Modify: `train_gpt.py` — serialization section at end of main()

- [ ] **Step 1: Replace int8+zlib with BitNet base-3 + LZMA**

In the serialization section after training:
1. Run one forward pass to populate _cached_q on all BitLinear layers
2. Call quantize_state_dict_bitnet()
3. torch.save() + compress_best()
4. Dequantize and eval for roundtrip validation

```python
# After training, before serialization:
# Run one forward pass to cache quantized weights
with torch.no_grad():
    dummy_x, dummy_y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(dummy_x, dummy_y)

# Serialize with BitNet format
quant_obj, quant_stats = quantize_state_dict_bitnet(base_model)
quant_buf = io.BytesIO()
torch.save(quant_obj, quant_buf)
quant_raw = quant_buf.getvalue()
quant_blob, comp_name = compress_best(quant_raw)

# Save and log
if master_process:
    with open("final_model.bitnet.ptz", "wb") as f:
        f.write(quant_blob)
    file_bytes = len(quant_blob)
    code_bytes = len(code.encode("utf-8"))
    log0(f"Serialized model bitnet+{comp_name}: {file_bytes} bytes")
    log0(f"Total submission size: {file_bytes + code_bytes} bytes")

# Roundtrip validation
quant_state = torch.load(io.BytesIO(decompress_auto(quant_blob)), map_location="cpu")
base_model.load_state_dict(dequantize_state_dict_bitnet(quant_state), strict=True)
# ... eval roundtrip ...
```

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: replace int8+zlib with BitNet base-3 + LZMA serialization"
```

---

## Chunk 4: Validation and Submission

### Task 11: CPU compression size verification

- [ ] **Step 1: Instantiate model, run forward, serialize, check size**

```python
# On CPU, create model with Hydra config
model = GPT(vocab_size=1024, num_layers=12, model_dim=768, ...)
# Run forward to populate caches
x = torch.randint(0, 1024, (1, 64))
y = torch.randint(0, 1024, (1, 64))
_ = model(x, y)
# Serialize and check
obj, stats = quantize_state_dict_bitnet(model)
buf = io.BytesIO(); torch.save(obj, buf)
blob, name = compress_best(buf.getvalue())
code_bytes = 80000  # estimate
total = len(blob) + code_bytes
print(f"Model: {len(blob)} bytes, Code: {code_bytes}, Total: {total}, Under 15MB: {total < 15_000_000}")
```

Expected: ~14.5-15.0 MB total. Must be under 15,000,000.

- [ ] **Step 2: If over budget, reduce model dim or layers**

Fallback configs:
- 12L/768d/12H/6KV/2xMLP → saves ~15M params
- 11L/768d/12H/6KV/3xMLP → saves ~5M params

- [ ] **Step 3: Commit verified config**

### Task 12: Local GPU smoke test (RTX 5090)

- [ ] **Step 1: Run 50-step smoke test**

```bash
RUN_ID=hydra_smoke \
NUM_LAYERS=12 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
MLP_MULT=3 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 ROPE_BASE=200000 \
MAX_WALLCLOCK_SECONDS=120 ITERATIONS=50 \
SWA_ENABLED=0 EVAL_STRIDE=0 \
python train_gpt.py
```

Verify:
- Loss decreases over 50 steps
- No CUDA errors
- Compressed size under 15MB
- BitLinear weights are ternary in cached state

- [ ] **Step 2: Run 500-step test for BPB signal**

```bash
RUN_ID=hydra_500 \
NUM_LAYERS=12 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
MLP_MULT=3 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 ROPE_BASE=200000 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=500 VAL_LOSS_EVERY=100 \
SWA_ENABLED=1 EVAL_STRIDE=64 \
python train_gpt.py
```

Compare BPB at 500 steps against baseline (~1.53 BPB) and SwiGLU (~1.48 BPB).

- [ ] **Step 3: Commit any fixes**

### Task 13: Update run.sh and submission files

**Files:**
- Modify: `runpod_package/run.sh`
- Create: `records/track_10min_16mb/2026-03-20_Hydra_BitNet158_12L_768d/submission.json`
- Create: `records/track_10min_16mb/2026-03-20_Hydra_BitNet158_12L_768d/README.md`

- [ ] **Step 1: Update run.sh**

```bash
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
```

- [ ] **Step 2: Create submission.json (fill after official run)**
- [ ] **Step 3: Create README.md**
- [ ] **Step 4: Copy train_gpt.py to records dir and runpod_package**
- [ ] **Step 5: Commit**

### Task 14: RunPod 8xH100 official run

- [ ] **Step 1: Upload to RunPod and execute**
- [ ] **Step 2: Collect logs, verify val_bpb and compressed size**
- [ ] **Step 3: Run 2-4 more times for p < 0.01 statistical significance**
- [ ] **Step 4: Update submission.json with results**
- [ ] **Step 5: Final commit and PR to openai/parameter-golf**
