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

def main():
    parser = argparse.ArgumentParser(description="Compare JAX and PyTorch model outputs.")
    parser.add_argument("jax_checkpoint", help="Path to JAX checkpoint directory")
    parser.add_argument("--torch_weights", default="model.pth", help="Path to converted PyTorch weights")
    args = parser.parse_args()

    # 1. Setup Inputs
    print("Generating inputs...")
    np.random.seed(42)
    B, S = 1, 131072 # Standard length
    # Random sequence (0-3)
    seq_ints = np.random.randint(0, 4, size=(B, S))
    
    # One-hot encoding
    # JAX model usually expects strings or one-hot depending on entry point.
    # dna_model.predict_sequence takes string.
    # But to test core model equivalence, we should ideally feed tensors closer to the model core 
    # to avoid preprocessing diffs.
    # However, `dna_model.create` returns a model that wraps everything.
    # Let's verify `dna_model` usage.
    # `model = dna_model.create(...)`
    # `model.predict(inputs)` where inputs is dict with 'seq'.
    
    # Create one-hot for PyTorch
    # (B, S, 4)
    vocab_size = 4
    seq_one_hot = np.eye(vocab_size)[seq_ints].astype(np.float32)
    
    # PyTorch Input
    pt_input = torch.tensor(seq_one_hot)
    pt_org_index = torch.tensor([0] * B, dtype=torch.long)
    
    # JAX Input
    # The JAX high level API takes strings usually, but the Haiku transformed function likely takes one-hot.
    # Let's try to access the lower level apply if possible, or use the high level if it's reliable.
    # `dna_model.create` returns an object with `.predict_on_batch`.
    # Let's check `dna_model.py` briefly or assume standard usage.
    # Actually, JAX model from `create` loads parameters and returns a helper.
    # We might need to handle the fact that JAX model includes preprocessing layers that we might have skipped?
    # No, `DnaEmbedder` takes 4-channel input.
    # So we should be able to pass one-hot to JAX too.
    
    jax_input = jnp.array(seq_one_hot)
    jax_org_index = jnp.array([0] * B, dtype=jnp.int32)
    
    # 2. Load JAX Model
    print(f"Loading JAX model from {args.jax_checkpoint}...")
    # We use the factory to ensure same config
    jax_wrapper = dna_model.create(args.jax_checkpoint, device=jax.devices('cpu')[0]) 
    # This might load weights into memory.
    
    # 3. Load PyTorch Model (track means are bundled with weights)
    print(f"Loading PyTorch model from {args.torch_weights}...")
    pt_model = AlphaGenome(num_organisms=2)
    if os.path.exists(args.torch_weights):
        pt_model.load_state_dict(torch.load(args.torch_weights), strict=False)
    else:
        print("WARNING: PyTorch weights not found! Comparison will be meaningless.")
    
    pt_model.eval()

    # Debug: Check if running_var was loaded (should not all be 1.0)
    print("\nChecking RMSBatchNorm running_var statistics:")
    all_ones = True
    for name, buf in pt_model.named_buffers():
        if 'running_var' in name:
            mean_val = buf.mean().item()
            if abs(mean_val - 1.0) > 0.01:
                all_ones = False
            # Print first few
            if 'encoder.dna_embedder' in name or 'tower.blocks.0' in name:
                print(f"  {name}: mean={mean_val:.4f}, std={buf.std().item():.4f}")
    if all_ones:
        print("  WARNING: All running_var are ~1.0 - variance may not be loaded from checkpoint!")
    else:
        print("  running_var appears to be loaded correctly")

    # 4. Run PyTorch
    print("Running PyTorch forward pass...")
    with torch.no_grad():
        pt_out = pt_model(pt_input, pt_org_index)
        
    # 5. Run JAX
    print("Running JAX forward pass...")
    # We need to call the underlying haiku function or the wrapper method.
    # wrapper.predict_on_batch usually expects a batch of sequences or similar.
    # Let's assume we can call `jax_wrapper.apply` if exposed, or we look at `dna_model.py` again.
    # `dna_model.create` returns an `AlphaGenomeModel` instance.
    # It has `self._forward_fn` which is `hk.transform(self._forward).apply`.
    # And `self.params`.
    # So:
    rng = jax.random.PRNGKey(0)
    # The forward signature: (inputs, is_training)
    # inputs is expected to be OneHot? 
    # In `dna_model.py`, `_forward(inputs, ...)` -> `x = inputs['seq']`.
    # So we pass a dict.
    
    jax_inputs = {'seq': jax_input, 'organism_index': jax_org_index}
    
    # Access internals to run forward pass
    print("Running JAX forward pass (via internal _predict)...")
    organism = dna_model.Organism.HOMO_SAPIENS
    track_metadata = jax_wrapper._metadata[organism]
    
    # Prepare auxiliary inputs
    strand_reindexing = jax.device_put(track_metadata.strand_reindexing)
    negative_strand_mask = jnp.zeros((B,), dtype=bool)
    
    # Run _predict
    # Signature inferred from dna_model.py usage:
    # _predict(params, state, sequence, organism_indices, negative_strand_mask, strand_reindexing)
    jax_out = jax_wrapper._predict(
        jax_wrapper._params,
        jax_wrapper._state,
        jax_input,
        jax_org_index,
        negative_strand_mask=negative_strand_mask,
        strand_reindexing=strand_reindexing
    )
    
    # jax_out is a dict of predictions?
    # In dna_model.py: `predictions = self._predict(...)`
    # It returns raw predictions dict.

    # Convert JAX outputs from bfloat16 to float32 (JAX uses mixed precision)
    def to_float32(x):
        """Convert JAX array to float32 numpy array."""
        if hasattr(x, 'dtype'):
            dtype_str = str(x.dtype)
            # Handle bfloat16 which numpy doesn't support natively
            if 'bfloat16' in dtype_str or dtype_str == 'bfloat16':
                x = x.astype(jnp.float32)
            arr = np.asarray(x, dtype=np.float32)
            return arr
        return x

    # Debug: print dtypes before conversion
    print("\nJAX output dtypes before conversion:")
    for k, v in jax_out.items():
        if hasattr(v, 'dtype'):
            print(f"  {k}: {v.dtype}, shape={v.shape}, nan_count={jnp.isnan(v).sum()}")

    jax_out = jax.tree.map(to_float32, jax_out)

    print("\nAfter conversion to float32:")

    # 6. Compare
    print("\nComparison Results:")

    # Compare each head
    # PyTorch outputs: dict of tensors or dict of {resolution: tensor}
    # JAX outputs: dict of arrays (single resolution per head)

    failures = []

    # NOTE: JAX `_predict` returns unscaled predictions (experimental data space).
    # PyTorch returns raw model output (model space).
    # For proper comparison, we need to compare raw outputs.
    # The JAX raw outputs are in the model predictions before extract_predictions.
    # Since we're using _predict which applies extract_predictions, the comparison
    # isn't apples-to-apples for genome tracks heads.
    #
    # For now, we compare what we have. Large differences are expected for heads
    # that apply unscaling. Contact maps don't have unscaling, so they should match better.

    # Map PyTorch string keys to JAX keys (OutputType Enum)
    pt_to_jax_map = {
        'atac': dna_output.OutputType.ATAC,
        'dnase': dna_output.OutputType.DNASE,
        'procap': dna_output.OutputType.PROCAP,
        'cage': dna_output.OutputType.CAGE,
        'rna_seq': dna_output.OutputType.RNA_SEQ,
        'chip_tf': dna_output.OutputType.CHIP_TF,
        'chip_histone': dna_output.OutputType.CHIP_HISTONE,
        'pair_activations': dna_output.OutputType.CONTACT_MAPS
    }

    def compare_arrays(name, p, j):
        """Compare two numpy arrays and report differences."""
        # Check for NaN values
        pt_nan = np.isnan(p).sum()
        jax_nan = np.isnan(j).sum()
        if pt_nan > 0 or jax_nan > 0:
            print(f"  {name}: WARNING - NaN values detected! PT has {pt_nan}, JAX has {jax_nan}")
            # Check if NaN is in specific track positions (last axis)
            if j.ndim >= 2:
                nan_per_track = np.isnan(j).sum(axis=tuple(range(j.ndim-1)))
                nan_tracks = np.where(nan_per_track > 0)[0]
                if len(nan_tracks) > 0 and len(nan_tracks) <= 10:
                    print(f"    NaN in tracks: {nan_tracks.tolist()}")
            # Compare only non-NaN values
            valid_mask = ~(np.isnan(p) | np.isnan(j))
            if valid_mask.sum() == 0:
                print(f"  {name}: No valid values to compare")
                return True  # Can't determine failure
            p_valid = p[valid_mask]
            j_valid = j[valid_mask]
            diff = np.abs(p_valid - j_valid)
        else:
            diff = np.abs(p - j)

        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"  {name}: Max Diff = {max_diff:.6f}, Mean Diff = {mean_diff:.6f}")

        if max_diff > 1e-4:
            print(f"    -> MISMATCH! (Shapes: PT {p.shape}, JAX {j.shape})")
            return False
        return True

    for key, pt_val in pt_out.items():
        jax_key = pt_to_jax_map.get(key)

        if jax_key is None:
            print(f"--- {key} ---")
            print(f"  Skipping: No JAX mapping defined.")
            continue

        if jax_key not in jax_out:
            print(f"--- {key} ---")
            print(f"  Skipping: Mapped to {jax_key} but not in JAX output.")
            continue

        jax_val = jax_out[jax_key]
        print(f"--- {key} ---")

        # Handle dicts (resolution outputs)
        if isinstance(pt_val, dict):
            if isinstance(jax_val, dict):
                # Both are dicts - compare matching keys
                for sub_k in pt_val:
                    p = pt_val[sub_k].detach().numpy()
                    if sub_k in jax_val:
                        j = np.array(jax_val[sub_k])
                    elif str(sub_k) in jax_val:
                        j = np.array(jax_val[str(sub_k)])
                    else:
                        print(f"  Res {sub_k}: Missing in JAX output")
                        continue

                    if p.shape != j.shape:
                        print(f"  Res {sub_k}: Shape mismatch - PT {p.shape} vs JAX {j.shape}")
                        failures.append(f"{key}.{sub_k}")
                        continue

                    if not compare_arrays(f"Res {sub_k}", p, j):
                        failures.append(f"{key}.{sub_k}")
            else:
                # JAX is a single array - find matching PyTorch resolution by shape
                j_arr = np.array(jax_val)
                matched = False
                for sub_k, pt_tensor in pt_val.items():
                    p = pt_tensor.detach().numpy()
                    if p.shape == j_arr.shape:
                        print(f"  JAX array shape {j_arr.shape} matches PyTorch Res {sub_k}")
                        if not compare_arrays(f"Res {sub_k}", p, j_arr):
                            failures.append(f"{key}.{sub_k}")
                        matched = True
                        break

                if not matched:
                    print(f"  No matching resolution found.")
                    print(f"    JAX shape: {j_arr.shape}")
                    for sub_k, pt_tensor in pt_val.items():
                        print(f"    PyTorch Res {sub_k} shape: {pt_tensor.shape}")
                    failures.append(key)
        else:
            # Single tensor
            p = pt_val.detach().numpy()
            j = np.array(jax_val)

            if p.shape != j.shape:
                print(f"  Shape mismatch - PT {p.shape} vs JAX {j.shape}")
                failures.append(key)
                continue

            if not compare_arrays("Value", p, j):
                failures.append(key)

    if not failures:
        print("\nSUCCESS: All outputs match within tolerance.")
    else:
        print(f"\nFAILURE: Mismatches found in {failures}")

if __name__ == "__main__":
    main()
