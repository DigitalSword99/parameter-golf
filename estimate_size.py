"""Quick parameter count + compressed size estimator for different configs."""
import sys

def estimate(
    vocab_size=1024, num_layers=9, model_dim=512,
    num_heads=8, num_kv_heads=4, mlp_mult=2, tie_embeddings=True
):
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    hidden = mlp_mult * model_dim

    # Per block: Q, K, V, proj, fc, mlp_proj (all weight-only, no bias)
    per_block = (
        model_dim * model_dim  # c_q
        + model_dim * kv_dim   # c_k
        + model_dim * kv_dim   # c_v
        + model_dim * model_dim  # proj
        + model_dim * hidden   # fc
        + hidden * model_dim   # mlp proj
        + num_heads            # q_gain
        + model_dim * 3        # attn_scale, mlp_scale, resid_mix (2*dim)
    )

    total = per_block * num_layers
    total += vocab_size * model_dim  # tok_emb
    if not tie_embeddings:
        total += model_dim * vocab_size  # lm_head
    total += min(num_layers // 2, num_layers - num_layers // 2) * model_dim  # skip_weights

    # Int8 quantization: large tensors -> 1 byte/param + scales, small ones -> fp16
    # Rough estimate: ~1.05 bytes/param for quantized, zlib brings it down ~15-25%
    raw_int8_bytes = total * 1.05
    zlib_estimate_low = raw_int8_bytes * 0.70
    zlib_estimate_high = raw_int8_bytes * 0.85
    code_bytes = 48_000  # approximate train_gpt.py size

    print(f"Config: layers={num_layers} dim={model_dim} heads={num_heads} kv_heads={num_kv_heads} "
          f"mlp_mult={mlp_mult} vocab={vocab_size} tied={tie_embeddings}")
    print(f"Parameters: {total:,}")
    print(f"Raw int8 estimate: {raw_int8_bytes / 1e6:.2f} MB")
    print(f"Compressed estimate: {zlib_estimate_low / 1e6:.2f} - {zlib_estimate_high / 1e6:.2f} MB")
    print(f"With code (~{code_bytes // 1000}KB): {(zlib_estimate_low + code_bytes) / 1e6:.2f} - "
          f"{(zlib_estimate_high + code_bytes) / 1e6:.2f} MB")
    under = (zlib_estimate_high + code_bytes) < 16_000_000
    print(f"Fits under 16MB: {'YES' if under else 'NO (likely over)'}")
    print()
    return total

# Default baseline
estimate()

# The config from your message
estimate(vocab_size=16384, num_layers=12, model_dim=384, num_heads=12, num_kv_heads=1, mlp_mult=2, tie_embeddings=True)

# A few interesting configs to explore
estimate(vocab_size=1024, num_layers=12, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=2, tie_embeddings=True)
estimate(vocab_size=1024, num_layers=12, model_dim=576, num_heads=8, num_kv_heads=2, mlp_mult=2, tie_embeddings=True)
estimate(vocab_size=2048, num_layers=10, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=2, tie_embeddings=True)
