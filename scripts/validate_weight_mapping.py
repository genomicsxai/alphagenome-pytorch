"""Validate weight mappings between JAX and PyTorch AlphaGenome models.

This script audits all parameter mappings to ensure:
1. Every PyTorch parameter has a corresponding JAX key
2. The JAX key exists in the checkpoint
3. The transform produces the correct shape

Usage:
    python scripts/validate_weight_mapping.py \\
        --jax-checkpoint /path/to/checkpoint

    # For verbose output showing all mappings:
    python scripts/validate_weight_mapping.py \\
        --jax-checkpoint /path/to/checkpoint --verbose
"""

import argparse
import os
import sys
import numpy as np

# Helper to find alphagenome_research - set ALPHAGENOME_RESEARCH_PATH env var if needed
ALPHAGENOME_RESEARCH_PATH = os.environ.get('ALPHAGENOME_RESEARCH_PATH')
if ALPHAGENOME_RESEARCH_PATH and ALPHAGENOME_RESEARCH_PATH not in sys.path:
    sys.path.append(ALPHAGENOME_RESEARCH_PATH)


def validate_mappings(jax_params: dict, pt_state_dict: dict, verbose: bool = False):
    """Audit all weight mappings for correctness.

    Args:
        jax_params: Flattened JAX parameter dictionary
        pt_state_dict: PyTorch state dictionary
        verbose: If True, print all mappings (not just failures)

    Returns:
        Dictionary with validation results:
        - matched: List of successfully mapped parameters
        - shape_mismatch: List of (pt_name, jax_name, error_msg) tuples
        - jax_key_missing: List of (pt_name, jax_name) tuples
        - unmapped: List of unmapped PyTorch parameters
    """
    from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax
    from alphagenome_pytorch.jax_compat.transforms import apply_transform

    results = {
        "matched": [],
        "shape_mismatch": [],
        "jax_key_missing": [],
        "unmapped": [],
    }

    for pt_name, pt_param in pt_state_dict.items():
        pt_shape = tuple(pt_param.shape)
        jax_name = map_pytorch_to_jax(pt_name)

        if jax_name is None:
            results["unmapped"].append(pt_name)
            if verbose:
                print(f"  UNMAPPED: {pt_name}")
            continue

        if jax_name not in jax_params:
            results["jax_key_missing"].append((pt_name, jax_name))
            if verbose:
                print(f"  JAX KEY MISSING: {pt_name} -> {jax_name}")
            continue

        jax_param = np.array(jax_params[jax_name])
        jax_shape = tuple(jax_param.shape)

        try:
            transformed = apply_transform(pt_name, jax_param, pt_shape)
            results["matched"].append(pt_name)
            if verbose:
                print(f"  OK: {pt_name} | JAX {jax_shape} -> PT {pt_shape}")
        except ValueError as e:
            results["shape_mismatch"].append((pt_name, jax_name, str(e)))
            if verbose:
                print(f"  SHAPE MISMATCH: {pt_name} | {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate weight mappings between JAX and PyTorch"
    )
    parser.add_argument(
        "--jax-checkpoint",
        required=True,
        help="Path to JAX checkpoint directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all mappings, not just failures"
    )
    args = parser.parse_args()

    print(f"Loading JAX checkpoint from {args.jax_checkpoint}...")
    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    params, state = checkpointer.restore(args.jax_checkpoint)
    print("JAX params loaded.")

    # Flatten JAX params
    flat_jax = {}

    def flatten(d, prefix=''):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, key)
            else:
                flat_jax[key] = v

    flatten(params)
    if state:
        flatten(state)

    print(f"Found {len(flat_jax)} JAX parameters")

    # Load PyTorch model
    print("Initializing PyTorch model...")
    import torch
    from alphagenome_pytorch.model import AlphaGenome
    pt_model = AlphaGenome(num_organisms=2)
    pt_state_dict = pt_model.state_dict()
    print(f"Found {len(pt_state_dict)} PyTorch parameters")

    # Validate
    print("\nValidating mappings...")
    results = validate_mappings(flat_jax, pt_state_dict, verbose=args.verbose)

    # Report
    print(f"\n{'=' * 60}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Matched:         {len(results['matched']):4d} / {len(pt_state_dict)}")
    print(f"Shape mismatch:  {len(results['shape_mismatch']):4d}")
    print(f"JAX key missing: {len(results['jax_key_missing']):4d}")
    print(f"Unmapped:        {len(results['unmapped']):4d}")

    if results["shape_mismatch"]:
        print(f"\n{'=' * 60}")
        print("SHAPE MISMATCHES (need to fix transforms.py)")
        print(f"{'=' * 60}")
        for pt, jax, err in results["shape_mismatch"]:
            print(f"\n  {pt}")
            print(f"    JAX key: {jax}")
            print(f"    Error: {err}")

    if results["jax_key_missing"]:
        print(f"\n{'=' * 60}")
        print("JAX KEY MISSING (need to fix weight_mapping.py)")
        print(f"{'=' * 60}")
        for pt, jax in results["jax_key_missing"][:20]:
            print(f"  {pt} -> {jax}")
        if len(results["jax_key_missing"]) > 20:
            print(f"  ... and {len(results['jax_key_missing']) - 20} more")

    if results["unmapped"]:
        print(f"\n{'=' * 60}")
        print("UNMAPPED PYTORCH PARAMS (need to add to weight_mapping.py)")
        print(f"{'=' * 60}")
        for pt in results["unmapped"][:20]:
            print(f"  {pt}")
        if len(results["unmapped"]) > 20:
            print(f"  ... and {len(results['unmapped']) - 20} more")

    # Summary
    total = len(pt_state_dict)
    matched = len(results["matched"])
    pass_rate = matched / total * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {pass_rate:.1f}% of parameters validated successfully")
    print(f"{'=' * 60}")

    # Exit with error code if not 100% matched
    if pass_rate < 100:
        sys.exit(1)


if __name__ == "__main__":
    main()
