import sys
import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from alphagenome_research.model import dna_model
from alphagenome.models import dna_output
from alphagenome_pytorch.model import AlphaGenome

def run_compare(jax_checkpoint, torch_weights='model.pth'):
    print(f"JAX Checkpoint: {jax_checkpoint}")
    print(f"Torch Weights: {torch_weights}")

    # 1. Setup Input
    print("Generating inputs...")
    B, S = 1, 131072
    seq_ints = np.random.randint(0, 4, size=(B, S))
    seq_one_hot = np.eye(4)[seq_ints].astype(np.float32)
    
    pt_input = torch.tensor(seq_one_hot)
    pt_org_index = torch.tensor([0] * B, dtype=torch.long)
    
    jax_input = jnp.array(seq_one_hot)
    jax_org_index = jnp.array([0] * B, dtype=jnp.int32)

    # 2. Load JAX
    print("Loading JAX model...")
    # Setup JAX device
    try:
        device = jax.devices('gpu')[0]
    except:
        device = jax.devices('cpu')[0]
        
    jax_wrapper = dna_model.create(jax_checkpoint, device=device)
    
    # 3. Load PyTorch (track means are bundled with weights)
    print("Loading PyTorch model...")
    pt_model = AlphaGenome(num_organisms=2)
    if os.path.exists(torch_weights):
        pt_model.load_state_dict(torch.load(torch_weights), strict=False)
    pt_model.eval()

    # 4. Run PyTorch
    print("Running PyTorch...")
    with torch.no_grad():
        pt_out = pt_model(pt_input, pt_org_index)

    # 5. Run JAX (RAW - get both 1bp and 128bp predictions)
    print("Running JAX (raw _apply_fn for both resolutions)...")
    
    # Move inputs to device
    jax_input = jax.device_put(jax_input, device)
    jax_org_index = jax.device_put(jax_org_index, device)
    
    # Call the raw apply_fn to get all predictions at all resolutions
    # This bypasses extract_predictions which filters to only 1bp for most heads
    jax_out_raw = jax_wrapper._apply_fn(
        jax_wrapper._params,
        jax_wrapper._state,
        jax_input,
        jax_org_index,
    )
    
    # Convert JAX to numpy/float32
    def to_cpu(x):
        if x is None:
            return None
        return np.array(x, dtype=np.float32)
    jax_out = jax.tree.map(to_cpu, jax_out_raw)
    
    print("\nDone! Variables 'pt_out' and 'jax_out' are available.")
    print("JAX raw output structure:")
    for k, v in jax_out.items():
        if isinstance(v, dict):
            print(f"  {k}: {list(v.keys())}")
        else:
            print(f"  {k}: {type(v)}")
    
    return pt_out, jax_out


def compare_outputs(pt_out, jax_out):
    """Compare PyTorch and JAX outputs in detail at BOTH resolutions."""
    
    # Heads that have both 1bp and 128bp predictions
    heads_with_both = ['atac', 'dnase', 'procap', 'cage', 'rna_seq']
    # Heads with only 128bp
    heads_128bp_only = ['chip_tf', 'chip_histone']
    
    print("\n" + "="*70)
    print("DETAILED COMPARISON (Both Resolutions)")
    print("="*70)
    
    def compare_arrays(jax_arr, pt_arr, name):
        """Compare two arrays and print statistics."""
        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(jax_arr) & ~np.isnan(pt_arr)
        
        if not valid_mask.any():
            print(f"    All values are NaN")
            return
        
        jax_valid = jax_arr[valid_mask]
        pt_valid = pt_arr[valid_mask]
        
        # Compute statistics
        abs_diff = np.abs(jax_valid - pt_valid)
        rel_diff = abs_diff / (np.abs(jax_valid) + 1e-8)
        
        print(f"    Shape: JAX={jax_arr.shape}, PyTorch={pt_arr.shape}")
        print(f"    Valid values: {valid_mask.sum()} / {valid_mask.size}")
        print(f"    JAX range:     [{jax_valid.min():.6f}, {jax_valid.max():.6f}]")
        print(f"    PyTorch range: [{pt_valid.min():.6f}, {pt_valid.max():.6f}]")
        print(f"    Abs diff: mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}")
        print(f"    Rel diff: mean={rel_diff.mean():.4%}, max={rel_diff.max():.4%}")
        
        # Show first few values for manual inspection
        print(f"    First 5 (JAX):     {jax_arr.flat[:5]}")
        print(f"    First 5 (PyTorch): {pt_arr.flat[:5]}")
        
        # Check if close (within bfloat16 precision ~0.8%)
        is_close = np.allclose(jax_valid, pt_valid, rtol=0.01, atol=1e-4)
        print(f"    Close (rtol=1%, atol=1e-4): {'✓ YES' if is_close else '✗ NO'}")
    
    # Compare heads with both resolutions
    for head_name in heads_with_both:
        if head_name not in jax_out:
            print(f"\n{head_name.upper()}: Not in JAX output")
            continue
        if head_name not in pt_out:
            print(f"\n{head_name.upper()}: Not in PyTorch output")
            continue
        
        jax_head = jax_out[head_name]
        pt_head = pt_out[head_name]
        
        print(f"\n{head_name.upper()}:")
        
        # Compare 1bp resolution
        jax_key_1bp = 'predictions_1bp'
        if jax_key_1bp in jax_head and 1 in pt_head:
            print(f"  Resolution 1bp:")
            compare_arrays(jax_head[jax_key_1bp], pt_head[1].cpu().numpy(), f"{head_name}_1bp")
        else:
            print(f"  Resolution 1bp: Missing (JAX has {list(jax_head.keys())}, PT has {list(pt_head.keys())})")
        
        # Compare 128bp resolution
        jax_key_128bp = 'predictions_128bp'
        if jax_key_128bp in jax_head and 128 in pt_head:
            print(f"  Resolution 128bp:")
            compare_arrays(jax_head[jax_key_128bp], pt_head[128].cpu().numpy(), f"{head_name}_128bp")
        else:
            print(f"  Resolution 128bp: Missing")
    
    # Compare 128bp-only heads
    for head_name in heads_128bp_only:
        if head_name not in jax_out:
            continue
        if head_name not in pt_out:
            print(f"\n{head_name.upper()}: Not in PyTorch output")
            continue
        
        jax_head = jax_out[head_name]
        pt_head = pt_out[head_name]
        
        print(f"\n{head_name.upper()} (128bp only):")
        
        jax_key_128bp = 'predictions_128bp'
        if jax_key_128bp in jax_head and 128 in pt_head:
            compare_arrays(jax_head[jax_key_128bp], pt_head[128].cpu().numpy(), f"{head_name}_128bp")
        else:
            print(f"  Missing predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jax_checkpoint", help="Path to JAX checkpoint")
    parser.add_argument("--torch_weights", default="model.pth")
    args = parser.parse_args()
    
    pt_out, jax_out = run_compare(args.jax_checkpoint, args.torch_weights)
    
    # Basic print
    print("\nPyTorch keys:", list(pt_out.keys()))
    print("JAX keys:", list(jax_out.keys()))
    
    # Detailed comparison
    compare_outputs(pt_out, jax_out)
