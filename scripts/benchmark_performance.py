from alphagenome_pytorch import (
    AlphaGenome,
)
from alphagenome_pytorch.config import (
    DtypePolicy
)
import argparse
import contextlib
import gc
import os
import platform
import time
from typing import Dict

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# =============================================================================
# Hardware Information
# =============================================================================

def get_hardware_info() -> Dict[str, str]:
    """Collect hardware and software version information."""
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["cuda_version"] = torch.version.cuda or "Unknown"
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            info["driver_version"] = result.stdout.strip()
        except Exception:
            info["driver_version"] = "Unknown"
    else:
        info["gpu_name"] = "None (CPU only)"
        info["gpu_memory"] = "N/A"
        info["cuda_version"] = "N/A"
        info["driver_version"] = "N/A"

    return info


# =============================================================================
# Model Loading
# =============================================================================

def load_pytorch_model(weights_path: str, dtype_policy: DtypePolicy):
    """Load PyTorch AlphaGenome model (track means are bundled with weights)."""
    model = AlphaGenome(
        num_organisms=2,
        dtype_policy=dtype_policy,
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model

def get_one_hot_encoded_dna_sequence(seq_len):
    """Generate one-hot encoded DNA sequence of the given length."""
    rng = np.random.default_rng(42)
    seq_ints = rng.integers(0, 4, size=seq_len)
    return np.eye(4, dtype=np.float32)[seq_ints]


def main():
    parser = argparse.ArgumentParser(
        description="Profile the PyTorch model"
    )
    parser.add_argument(
        "--torch-weights",
        required=True,
        help="PyTorch checkpoint"
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of bfloat16 (default is bfloat16 for realistic benchmarking)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hardware Info")
    print("=" * 60)
    hw_info = get_hardware_info()
    for k, v in hw_info.items():
        print(f"  {k}: {v}")
    print()

    dtype_policy = DtypePolicy.full_float32() if args.full_precision else DtypePolicy.mixed_precision()
    print(f"Dtype policy: {'float32' if args.full_precision else 'bfloat16 (mixed_precision)'}")

    print("Loading PyTorch model...")
    model = load_pytorch_model(args.torch_weights, dtype_policy)
    device = next(model.parameters()).device

    os.makedirs("outputs", exist_ok=True)

    for seq_length in [2**17, 2**18, 2**19, 2**20]:
        print()
        print("=" * 60)
        print(f"Sequence length: {seq_length} ({seq_length // 1024}k bp)")
        print("=" * 60)

        human_organism_index = torch.tensor([0], dtype=torch.long, device=device)
        seq = get_one_hot_encoded_dna_sequence(seq_length)
        one_hot_with_batch_dim = torch.from_numpy(seq).unsqueeze(0).to(device)

        use_amp = not args.full_precision
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

        # Warmup (2 iterations, no profiling)
        print("Warming up...")
        model.eval()
        with torch.no_grad(), amp_context:
            for _ in range(2):
                _ = model(one_hot_with_batch_dim, human_organism_index, return_embeddings=False)
                torch.cuda.synchronize()

        # Reset memory stats before profiled run
        torch.cuda.reset_peak_memory_stats()

        # Profiled run
        print("Profiling...")
        with torch.no_grad(), amp_context:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = model(
                    one_hot_with_batch_dim,
                    human_organism_index,
                    return_embeddings=False,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

        # Report — sort by FLOPS to surface compute-heavy ops
        key_averages = prof.key_averages()
        table = key_averages.table(sort_by="flops", row_limit=20)
        print(table)

        # Compute totals across all ops (use self_ variants to avoid double-counting)
        total_flops = sum(e.flops for e in key_averages if e.flops)
        wall_clock_ms = (t1 - t0) * 1000
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

        # Save the table
        with open(f"outputs/model_inference_{seq_length}_table.txt", "w") as f:
            f.write(table)
            # Write the summary as part of the output too
            f.write(f"\n--- Summary ---")
            f.write(f"\nWall-clock time:       {wall_clock_ms:.1f} ms")
            f.write(f"\nTotal GFLOPS:          {total_flops / 1e9:.2f}")
            f.write(f"\nPeak memory allocated: {peak_allocated_gb:.2f} GB")
            f.write(f"\nPeak memory reserved:  {peak_reserved_gb:.2f} GB")
        prof.export_chrome_trace(f"outputs/model_inference_{seq_length}_trace.json")

        # Cleanup
        del preds
        del one_hot_with_batch_dim
        del human_organism_index
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
