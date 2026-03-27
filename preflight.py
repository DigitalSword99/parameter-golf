"""Preflight check: verify pod is ready for 8xH100 distributed training."""
import sys, os

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False

def main():
    ok = True
    print("=== Pod Preflight Check ===\n")

    # 1. Python packages
    for pkg in ["torch", "sentencepiece", "huggingface_hub"]:
        ok &= check(f"import {pkg}", lambda p=pkg: __import__(p))

    import torch
    import torch.distributed as dist

    # 2. CUDA
    ok &= check("CUDA available", lambda: None if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("No CUDA")))
    n = torch.cuda.device_count()
    ok &= check(f"GPU count = {n} (need 8)", lambda: None if n == 8 else (_ for _ in ()).throw(RuntimeError(f"Got {n}")))

    # 3. GPU-to-GPU transfer
    if n >= 2:
        def gpu_xfer():
            a = torch.ones(1, device="cuda:0")
            b = a.to("cuda:1")
            assert b.item() == 1.0
        ok &= check("GPU 0 -> GPU 1 transfer", gpu_xfer)

    # 4. NCCL (only works under torchrun)
    if os.environ.get("RANK") is not None:
        def nccl_test():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            dist.barrier()
            t = torch.ones(1, device=f"cuda:{local_rank}")
            dist.all_reduce(t)
            assert t.item() == float(dist.get_world_size())
            dist.destroy_process_group()
        ok &= check("NCCL init + barrier + all_reduce", nccl_test)
    else:
        print("\n  SKIP  NCCL test (run with torchrun to test)")
        print("         torchrun --nproc_per_node=8 preflight.py")

    # 5. Data paths
    data_path = os.environ.get("DATA_PATH", "")
    tok_path = os.environ.get("TOKENIZER_PATH", "")
    if data_path:
        import glob
        ok &= check(f"DATA_PATH has train shards", lambda: None if glob.glob(os.path.join(data_path, "fineweb_train_*.bin")) else (_ for _ in ()).throw(FileNotFoundError("no train shards")))
        ok &= check(f"DATA_PATH has val shards", lambda: None if glob.glob(os.path.join(data_path, "fineweb_val_*.bin")) else (_ for _ in ()).throw(FileNotFoundError("no val shards")))
    else:
        print("\n  SKIP  DATA_PATH not set")
    if tok_path:
        ok &= check(f"TOKENIZER_PATH exists", lambda: None if os.path.isfile(tok_path) else (_ for _ in ()).throw(FileNotFoundError(tok_path)))
    else:
        print("\n  SKIP  TOKENIZER_PATH not set")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"\n{'=== ALL CHECKS PASSED ===' if ok else '=== SOME CHECKS FAILED ==='}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
