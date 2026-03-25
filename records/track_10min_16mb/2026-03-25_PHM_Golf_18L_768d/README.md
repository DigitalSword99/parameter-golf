# PHM-Golf: Kronecker-Factored Transformer

**Score:** TBD
**Author:** DigitalSword99
**Date:** 2026-03-25

## Summary

First use of Parameterized Hypercomplex Multiplication (PHM) in the Parameter Golf challenge. PHM factorizes every linear layer as W = Σ Aᵢ ⊗ Sᵢ (Kronecker products), achieving 4x structural parameter reduction. This enables an 18-layer, 768-dim model (vs SOTA's 11L/512d) at comparable parameter count.

## Novel Techniques
1. **PHM(n=4) Layers** — Kronecker-factored linear layers (4x structural compression)
2. **APoT Quantization** — Non-uniform powers-of-two quantization with GPTQ-lite clip search
3. **Learned-Frequency RoPE** — Learnable positional frequencies (subsumes Partial RoPE)
4. **Multi-Resolution Skips** — Decoder layers connect to multiple encoder layers
5. **Learnable Layer Scaling** — Adaptive 1/√(layer+1) normalization
6. **Score-First TTT** — Test-time training with SGD adaptation on S matrices

## Proven Techniques
BigramHash(1536), XSA(last 4), LeakyReLU(0.5)², EMA(0.997)+SWA, GQA(12H/4KV), Cosine Warmdown(3500), Muon WD(0.04)

## Architecture
18 layers, 768 dim, 12 heads, 4 KV heads, 3x MLP (2304 hidden), PHM n=4 on all linear layers, 1024 vocab, tied embeddings, 2048 sequence length

## Running
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
