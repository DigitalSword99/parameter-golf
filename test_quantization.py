"""Test 4-bit quantization roundtrip: quantize → compress → decompress → dequantize."""
import io, lzma, math, os, sys, zlib
import torch
import torch.nn as nn
from torch import Tensor

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
    "pass_gate", "ln_scale", "inv_freq", ".A",
)

def build_apot_levels(bit_width: int = 4) -> Tensor:
    powers = [2**i for i in range(bit_width - 1)]
    levels = {0.0}
    for p in powers:
        levels.add(float(p))
    for i, p1 in enumerate(powers):
        for p2 in powers[i+1:]:
            levels.add(float(p1 + p2))
    for i, p1 in enumerate(powers):
        for j, p2 in enumerate(powers[i+1:], i+1):
            for p3 in powers[j+1:]:
                levels.add(float(p1 + p2 + p3))
    return torch.tensor(sorted(levels), dtype=torch.float32)

APOT_LEVELS = build_apot_levels(4)

def build_signed_apot_codebook(levels):
    pos = levels[levels > 0]
    neg = -pos.flip(0)
    return torch.cat([neg, torch.zeros(1), pos])

APOT_CODEBOOK = build_signed_apot_codebook(APOT_LEVELS)
print(f"Codebook: {len(APOT_CODEBOOK)} entries, range [{APOT_CODEBOOK[0]:.0f}, {APOT_CODEBOOK[-1]:.0f}]")
print(f"Levels: {APOT_LEVELS.tolist()}")

def pack_nibbles(indices):
    rows, cols = indices.shape
    if cols % 2 != 0:
        indices = torch.cat([indices, torch.zeros(rows, 1, dtype=indices.dtype)], dim=1)
        cols += 1
    even = indices[:, 0::2]
    odd = indices[:, 1::2]
    return ((even << 4) | odd).to(torch.uint8)

def unpack_nibbles(packed, cols):
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    rows = packed.shape[0]
    unpacked = torch.zeros(rows, high.shape[1] * 2, dtype=torch.uint8, device=packed.device)
    unpacked[:, 0::2] = high
    unpacked[:, 1::2] = low
    return unpacked[:, :cols]

def quantize_apot_gptq_lite(t, levels, codebook, n_candidates=5):
    t32 = t.float()
    candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0][:n_candidates]
    best_indices = None
    best_scale = None
    best_mse = float('inf')
    max_level = levels[-1].item()
    cb = codebook
    for pct in candidates:
        if pct < 1.0:
            clip_abs = torch.quantile(t32.abs(), pct, dim=1, keepdim=True)
        else:
            clip_abs = t32.abs().amax(dim=1, keepdim=True)
        clip_abs = clip_abs.clamp_min(1e-8)
        scale = clip_abs / max_level
        normalized = (t32 / scale).clamp(-max_level, max_level)
        dists = (normalized.unsqueeze(-1) - cb.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        q_vals = cb[indices] * scale
        mse = (q_vals - t32).pow(2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_indices = indices.to(torch.uint8)
            best_scale = scale.squeeze(-1).to(torch.float16)
    packed = pack_nibbles(best_indices)
    return packed, best_scale

def dequantize_apot(packed, scales, codebook, cols):
    indices = unpack_nibbles(packed, cols)
    cb = codebook
    q_vals = cb[indices.long()]
    return q_vals * scales.float().unsqueeze(1)

def quantize_state_dict_phm(state_dict):
    q_packed, q_scales, q_cols = {}, {}, {}
    passthrough = {}
    stats = {"param_count": 0, "total_bytes": 0}
    levels, codebook = APOT_LEVELS, APOT_CODEBOOK
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        if '.A' in name or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float()
            stats["total_bytes"] += t.numel() * 4
            continue
        if not t.is_floating_point():
            passthrough[name] = t
            stats["total_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= 65_536:
            passthrough[name] = t.to(torch.float16)
            stats["total_bytes"] += t.numel() * 2
            continue
        if t.ndim == 3 and '.S' in name:
            idx_slices, scale_slices = [], []
            for i in range(t.shape[0]):
                idx, sc = quantize_apot_gptq_lite(t[i], levels, codebook)
                idx_slices.append(idx)
                scale_slices.append(sc)
            q_packed[name] = torch.stack(idx_slices)
            q_scales[name] = torch.stack(scale_slices)
            q_cols[name] = t.shape[2]
            stats["total_bytes"] += q_packed[name].numel() + q_scales[name].numel() * 2
            continue
        if t.ndim == 2 and t.numel() > 65_536:
            idx, sc = quantize_apot_gptq_lite(t, levels, codebook)
            q_packed[name] = idx
            q_scales[name] = sc
            q_cols[name] = t.shape[1]
            stats["total_bytes"] += idx.numel() + sc.numel() * 2
            continue
        passthrough[name] = t.to(torch.float16)
        stats["total_bytes"] += t.numel() * 2
    return {"q_packed": q_packed, "q_scales": q_scales, "q_cols": q_cols,
            "passthrough": passthrough, "apot_codebook": codebook}, stats

def dequantize_state_dict_phm(obj):
    out = {}
    codebook = obj["apot_codebook"]
    q_cols = obj["q_cols"]
    for name in obj["q_packed"]:
        packed = obj["q_packed"][name]
        scales = obj["q_scales"][name]
        cols = q_cols[name]
        if packed.ndim == 3:
            slices = []
            for i in range(packed.shape[0]):
                slices.append(dequantize_apot(packed[i], scales[i], codebook, cols))
            out[name] = torch.stack(slices)
        else:
            out[name] = dequantize_apot(packed, scales, codebook, cols)
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out

def compress_best(data):
    results = []
    try: results.append((lzma.compress(data, preset=6), "lzma"))
    except: pass
    results.append((zlib.compress(data, level=9), "zlib"))
    results.sort(key=lambda x: len(x[0]))
    return results[0]

def decompress_auto(data):
    return lzma.decompress(data) if data[:2] == b'\xfd7' else zlib.decompress(data)

def build_fake_state_dict():
    sd = {}
    d, n = 512, 4
    s = d // n
    kv_s = (d // 2) // n
    sd["tok_emb.weight"] = torch.randn(1024, d).bfloat16()
    sd["bigram_hash.embed.weight"] = torch.randn(2048, d).bfloat16()
    sd["trigram_hash.embed.weight"] = torch.randn(4096, d).bfloat16()
    for layer in range(11):
        p = f"layers.{layer}"
        for sub in ["attn.c_q", "attn.proj"]:
            sd[f"{p}.{sub}.A"] = torch.randn(n, n, n)
            sd[f"{p}.{sub}.S"] = torch.randn(n, s, s) * 0.1
        for sub in ["attn.c_k", "attn.c_v"]:
            sd[f"{p}.{sub}.A"] = torch.randn(n, n, n)
            sd[f"{p}.{sub}.S"] = torch.randn(n, kv_s, s) * 0.1
        sd[f"{p}.attn_scale"] = torch.randn(1)
        sd[f"{p}.mlp_scale"] = torch.randn(1)
        sd[f"{p}.resid_mix"] = torch.randn(2 * d)
        sd[f"{p}.q_gain"] = torch.randn(d)
        sd[f"{p}.ln_scale"] = torch.randn(d)
    for layer in range(11):
        p = f"head_a.{layer}"
        sd[f"{p}.router.weight"] = torch.randn(6, d) * 0.01
        for e in range(6):
            h_a = d * 2
            s_h = h_a // n
            sd[f"{p}.expert_fc.{e}.A"] = torch.randn(n, n, n)
            sd[f"{p}.expert_fc.{e}.S"] = torch.randn(n, s_h, s) * 0.1
            sd[f"{p}.expert_proj.{e}.A"] = torch.randn(n, n, n)
            sd[f"{p}.expert_proj.{e}.S"] = torch.randn(n, s, s_h) * 0.1
    for layer in range(11):
        p = f"head_b.{layer}"
        s_h_b = (d * 3) // n
        sd[f"{p}.fc.A"] = torch.randn(n, n, n)
        sd[f"{p}.fc.S"] = torch.randn(n, s_h_b, s) * 0.1
        sd[f"{p}.proj.A"] = torch.randn(n, n, n)
        sd[f"{p}.proj.S"] = torch.randn(n, s, s_h_b) * 0.1
    for layer in range(11):
        p = f"head_c.{layer}"
        s_w = 640 // n
        sd[f"{p}.fc.A"] = torch.randn(n, n, n)
        sd[f"{p}.fc.S"] = torch.randn(n, s_w, s) * 0.1
        sd[f"{p}.proj.A"] = torch.randn(n, n, n)
        sd[f"{p}.proj.S"] = torch.randn(n, s, s_w) * 0.1
    return sd

def main():
    sd = build_fake_state_dict()
    total_params = sum(t.numel() for t in sd.values())
    print(f"\nState dict: {len(sd)} tensors, {total_params:,} params")

    quant_obj, stats = quantize_state_dict_phm(sd)
    n_q = len(quant_obj["q_packed"])
    n_p = len(quant_obj["passthrough"])
    print(f"Quantized: {n_q}, Passthrough: {n_p}")

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    blob, method = compress_best(raw)
    print(f"Serialized: {len(raw)/1e6:.1f} MB, Compressed ({method}): {len(blob)/1e6:.1f} MB")

    loaded = torch.load(io.BytesIO(decompress_auto(blob)), map_location="cpu")
    recon = dequantize_state_dict_phm(loaded)

    # Quality check
    total_mse, total_n = 0, 0
    worst_name, worst_rel = "", 0
    for name in sd:
        orig = sd[name].float()
        rec = recon[name].float()
        if orig.shape != rec.shape:
            print(f"SHAPE MISMATCH: {name} {orig.shape} vs {rec.shape}")
            continue
        max_err = (orig - rec).abs().max().item()
        rel = max_err / (orig.abs().max().item() + 1e-10)
        mse = (orig - rec).pow(2).mean().item()
        total_mse += mse * orig.numel()
        total_n += orig.numel()
        if rel > worst_rel:
            worst_rel, worst_name = rel, name

    rmse = math.sqrt(total_mse / total_n)
    print(f"\nRoundtrip RMSE: {rmse:.6f}")
    print(f"Worst relative error: {worst_name} ({worst_rel:.2%})")

    # Size check
    packed_bytes = sum(t.numel() * t.element_size() for t in quant_obj["q_packed"].values())
    scale_bytes = sum(t.numel() * t.element_size() for t in quant_obj["q_scales"].values())
    pass_bytes = sum(t.numel() * t.element_size() for t in quant_obj["passthrough"].values())
    print(f"\nPacked indices (uint8): {packed_bytes/1e6:.2f} MB")
    print(f"Scales (fp16):          {scale_bytes/1e6:.2f} MB")
    print(f"Passthrough:            {pass_bytes/1e6:.2f} MB")
    print(f"Compressed total:       {len(blob)/1e6:.2f} MB")
    code_sz = 68000
    total_sub = len(blob) + code_sz
    print(f"Submission total:       {total_sub/1e6:.2f} MB  {'PASS' if total_sub < 16_000_000 else 'FAIL'}")

if __name__ == "__main__":
    main()
