# Hydra-3: Three-Head Ensemble Transformer

**Score:** TBD
**Author:** DigitalSword99
**Date:** 2026-03-25

## Summary

Three-head ensemble on a shared 11L/512d attention backbone. Each step randomly trains one of three architecturally diverse MLP heads. At eval, all three heads contribute via logit averaging + aggressive Adam TTT for the full 10-min eval budget.

## Novel Techniques
1. **Three-Head Ensemble** — MoE + Dense + Wide MLP heads, trained via random selection, ensembled at eval
2. **MoE Head (8 PHM experts)** — Kronecker-factored experts with top-1 routing, 8x MLP capacity
3. **Aggressive Adam TTT** — Full-model Adam adaptation for 600s (vs SOTA's SGD 530s)
4. **TrigramHash(4096)** — Hash-based trigram features added to embeddings
5. **PHMLinear attention** — Kronecker-factored Q/K/V/Proj for compression

## Proven Techniques
BigramHash(2048), XSA(last 4), LeakyReLU(0.5)², EMA(0.997)+SWA, GQA(8H/4KV), Cosine Warmdown(3500), Muon WD(0.04), Learned-Frequency RoPE, Multi-Resolution Skips

## Architecture
- Shared: 11 layers, 512 dim, 8 heads, 4 KV heads, U-Net skips
- Head A: MoE (8 PHM experts, top-1, 2x expansion)
- Head B: Dense (3x expansion)
- Head C: Wide (512→640→512, 2x expansion)
- 1024 vocab, tied embeddings, 2048 seq length

## Running
```bash
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
