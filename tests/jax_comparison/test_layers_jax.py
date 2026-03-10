
"""Architectural Logic Verification: JAX vs PyTorch Layer Comparison.

This test suite verifies that the PyTorch implementation matches the JAX
reference implementation at every layer. All comparisons are done in float32
to isolate architectural bugs from quantization noise.

Key Metrics:
- Pearson Correlation: Must be > 0.9999 (shape match)
- Cosine Similarity: Vector-space check, robust to scaling
- Scale Ratio: std(pytorch) / std(jax) - detects normalization bugs

If a layer fails, subsequent layers are marked as dependent/skipped to avoid
log spam and identify the first point of divergence.
"""

import pytest
import numpy as np
import torch
from typing import Optional
from tests.layer_utils import compute_metrics, print_layer_result


# =============================================================================
# FLOAT32 JAX EXTRACTION (cast params to float32 for precision testing)
# =============================================================================


def cast_params_to_float32(params):
    """Recursively cast all JAX params to float32."""
    import jax

    def cast_leaf(x):
        if hasattr(x, 'dtype'):
            import jax.numpy as jnp
            if 'bfloat16' in str(x.dtype) or 'float16' in str(x.dtype):
                return x.astype(jnp.float32)
        return x

    return jax.tree.map(cast_leaf, params)


def extract_jax_intermediates_float32(params, state, dna_sequence, organism_index):
    """Extract intermediate outputs from JAX model in FLOAT32.

    Forces all computation in float32 to isolate architectural bugs
    from quantization noise.

    Args:
        params: JAX model parameters (will be cast to float32)
        state: JAX model state
        dna_sequence: Input DNA sequence (B, S, 4)
        organism_index: Organism index array (B,)

    Returns:
        Dict mapping layer names to numpy arrays (float32)
    """
    import jax
    import jax.numpy as jnp
    import haiku as hk
    from alphagenome_research.model import model as jax_model_module

    # Cast params to float32
    params_f32 = cast_params_to_float32(params)
    state_f32 = cast_params_to_float32(state)

    intermediates = {}

    def to_np(x):
        """Convert JAX array to numpy float32."""
        if x is None:
            return None
        arr = np.array(x)
        return arr.astype(np.float32)

    def forward_with_intermediates(dna_seq, org_idx):
        """Run forward capturing intermediates with correct Haiku naming.

        The checkpoint uses this naming structure:
        - alphagenome/transformer_tower/pair_update_block (first)
        - alphagenome/transformer_tower/pair_update_block_1 (second)
        - etc.

        Haiku auto-deduplicates module names within a scope.
        """
        from alphagenome_research.model import embeddings as embeddings_module
        from alphagenome_research.model import attention as attention_module

        captured = {}

        with hk.name_scope('alphagenome'):
            # Run encoder
            trunk, enc_intermediates = jax_model_module.SequenceEncoder()(dna_seq)
            captured['enc_intermediates'] = enc_intermediates
            captured['encoder_output'] = trunk

            # Add organism embedding
            organism_embedding = hk.Embed(2, trunk.shape[-1])(org_idx)
            trunk = trunk + organism_embedding[:, None, :]

            # Run tower block by block to capture intermediates
            # Use transformer_tower scope to match checkpoint naming
            num_blocks = 9
            pair_acts = None
            block_intermediates = {}

            with hk.name_scope('transformer_tower'):
                for block_idx in range(num_blocks):
                    has_pair_update = (block_idx % 2 == 0)

                    # Pair update (on even blocks: 0, 2, 4, 6, 8)
                    # Haiku auto-names: pair_update_block, pair_update_block_1, etc.
                    if has_pair_update:
                        pair_acts = attention_module.PairUpdateBlock()(trunk, pair_acts)

                    # Attention bias from pair activations
                    # Haiku auto-names: attention_bias_block, attention_bias_block_1, etc.
                    if pair_acts is not None:
                        attn_bias = attention_module.AttentionBiasBlock()(pair_acts)
                    else:
                        attn_bias = None

                    # Multi-head attention (with residual)
                    # Haiku auto-names: mha_block, mha_block_1, etc.
                    trunk = trunk + attention_module.MHABlock()(trunk, attn_bias)

                    # MLP (with residual)
                    # Haiku auto-names: mlp_block, mlp_block_1, etc.
                    trunk = trunk + attention_module.MLPBlock()(trunk)

                    # Capture block output (trunk after residuals)
                    block_intermediates[f'block_{block_idx}'] = trunk

            captured['block_intermediates'] = block_intermediates
            captured['tower_output'] = trunk
            captured['pair_activations'] = pair_acts

            # Run decoder
            decoder_out = jax_model_module.SequenceDecoder()(trunk, enc_intermediates)
            captured['decoder_output'] = decoder_out

            # Run output embedders
            embeddings_128bp = embeddings_module.OutputEmbedder(2)(
                trunk, org_idx
            )
            captured['embeddings_128bp'] = embeddings_128bp

            embeddings_1bp = embeddings_module.OutputEmbedder(2)(
                decoder_out, org_idx, embeddings_128bp
            )
            captured['embeddings_1bp'] = embeddings_1bp

            if pair_acts is not None:
                embeddings_pair = embeddings_module.OutputPair(2)(
                    pair_acts, org_idx
                )
                captured['embeddings_pair'] = embeddings_pair

        return captured

    forward_fn = hk.transform_with_state(forward_with_intermediates)
    rng = jax.random.PRNGKey(0)

    # Run in FLOAT32 - key change from original
    captured, _ = forward_fn.apply(
        params_f32, state_f32, rng,
        jnp.array(dna_sequence, dtype=jnp.float32),  # float32, not bfloat16!
        jnp.array(organism_index)
    )

    # Map captured intermediates to standard names
    enc_int = captured.get('enc_intermediates', {})

    bin_size_to_layer = {
        'bin_size_1': 'encoder/dna_embedder',
        'bin_size_2': 'encoder/down_block_0',
        'bin_size_4': 'encoder/down_block_1',
        'bin_size_8': 'encoder/down_block_2',
        'bin_size_16': 'encoder/down_block_3',
        'bin_size_32': 'encoder/down_block_4',
        'bin_size_64': 'encoder/down_block_5',
    }

    for jax_name, pt_name in bin_size_to_layer.items():
        if jax_name in enc_int:
            intermediates[pt_name] = to_np(enc_int[jax_name])

    intermediates['encoder/output'] = to_np(captured.get('encoder_output'))

    # Map individual transformer block outputs
    block_ints = captured.get('block_intermediates', {})
    for block_name, block_output in block_ints.items():
        intermediates[f'tower/{block_name}'] = to_np(block_output)

    intermediates['tower/output'] = to_np(captured.get('tower_output'))
    intermediates['decoder/output'] = to_np(captured.get('decoder_output'))

    if captured.get('pair_activations') is not None:
        intermediates['tower/pair_activations'] = to_np(captured['pair_activations'])

    # Output embedders
    if captured.get('embeddings_128bp') is not None:
        intermediates['embedder_128bp/output'] = to_np(captured['embeddings_128bp'])
    if captured.get('embeddings_1bp') is not None:
        intermediates['embedder_1bp/output'] = to_np(captured['embeddings_1bp'])
    if captured.get('embeddings_pair') is not None:
        intermediates['embedder_pair/output'] = to_np(captured['embeddings_pair'])

    return intermediates


# =============================================================================
# PYTORCH INTERMEDIATE EXTRACTION (comprehensive hooks)
# =============================================================================


def extract_pytorch_intermediates(model, dna_sequence):
    """Extract intermediate outputs from PyTorch model with comprehensive hooks.

    Registers hooks on ALL sub-modules including individual transformer blocks.
    For transformer blocks, we capture the full block output (after residuals).

    Args:
        model: PyTorch AlphaGenome model
        dna_sequence: Input DNA sequence (B, S, 4) as numpy array

    Returns:
        Dict mapping layer names to numpy arrays (float32)
    """
    intermediates = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            if isinstance(out, torch.Tensor):
                arr = out.detach().cpu().numpy().astype(np.float32)
                
                # Transpose NCL (B, C, S) to NLC (B, S, C) for comparison with JAX
                # These layers are natively NCL in PyTorch but NLC in JAX reference
                ncl_layers = [
                    "encoder/dna_embedder", "encoder/pool", "encoder/output",
                    "decoder/output", "embedder_128bp/output", "embedder_1bp/output"
                ]
                if name in ncl_layers or name.startswith("encoder/down_block_") or name.startswith("decoder/up_block_"):
                    if arr.ndim == 3:
                        arr = arr.transpose(0, 2, 1)
                
                intermediates[name] = arr
        return hook

    # === Encoder hooks ===
    if hasattr(model, "encoder"):
        enc = model.encoder
        if hasattr(enc, "dna_embedder"):
            hooks.append(enc.dna_embedder.register_forward_hook(make_hook("encoder/dna_embedder")))
        if hasattr(enc, "pool"):
            hooks.append(enc.pool.register_forward_hook(make_hook("encoder/pool")))
        if hasattr(enc, "down_blocks"):
            for i, block in enumerate(enc.down_blocks):
                hooks.append(block.register_forward_hook(make_hook(f"encoder/down_block_{i}")))
        hooks.append(enc.register_forward_hook(make_hook("encoder/output")))

    # === Transformer tower - capture block outputs manually ===
    # We need to capture x AFTER residual adds, not from individual MHA/MLP hooks
    # Store original forward and replace with instrumented version
    if hasattr(model, "tower"):
        tower = model.tower
        original_forward = tower.forward

        def instrumented_forward(x, compute_dtype=None):
            pair_x = None
            for i, block in enumerate(tower.blocks):
                if block['pair_update'] is not None:
                    pair_x = block['pair_update'](x, pair_x, compute_dtype=compute_dtype)
                mha_bias = block['attn_bias'](pair_x)
                x = x + block['mha'](x, mha_bias, compute_dtype=compute_dtype)
                x = x + block['mlp'](x)
                # Capture full block output (after residuals)
                intermediates[f'tower/block_{i}'] = x.detach().cpu().numpy().astype(np.float32)
            return x, pair_x

        tower.forward = instrumented_forward

    # === Decoder hooks ===
    if hasattr(model, "decoder"):
        dec = model.decoder
        if hasattr(dec, "up_blocks"):
            for i, block in enumerate(dec.up_blocks):
                hooks.append(block.register_forward_hook(make_hook(f"decoder/up_block_{i}")))
        hooks.append(dec.register_forward_hook(make_hook("decoder/output")))

    # === Output Embedder hooks ===
    if hasattr(model, "embedder_128bp"):
        hooks.append(model.embedder_128bp.register_forward_hook(make_hook("embedder_128bp/output")))
    if hasattr(model, "embedder_1bp"):
        hooks.append(model.embedder_1bp.register_forward_hook(make_hook("embedder_1bp/output")))
    if hasattr(model, "embedder_pair"):
        hooks.append(model.embedder_pair.register_forward_hook(make_hook("embedder_pair/output")))

    # Run forward pass in float32
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pt_input = torch.tensor(dna_sequence, dtype=torch.float32, device=device)
        pt_org = torch.tensor([0], dtype=torch.long, device=device)  # Human
        outputs = model(pt_input, pt_org)

    # Capture tower output and restore original forward
    if hasattr(model, "tower"):
        tower.forward = original_forward
        # Tower output is the last block output
        if 'tower/block_8' in intermediates:
            intermediates['tower/output'] = intermediates['tower/block_8']

    # Clean up hooks
    for h in hooks:
        h.remove()

    return intermediates


# =============================================================================
# INTERMEDIATE LAYER TESTS (Main Architectural Verification)
# =============================================================================


@pytest.mark.integration
@pytest.mark.jax
class TestIntermediateLayers:
    """Layer-by-layer architectural verification in float32.

    This is the main test class. It compares JAX and PyTorch at each layer
    to identify exactly where divergence occurs.

    Failure at layer N causes layers N+1, N+2, ... to be marked as dependent.
    """

    @pytest.fixture(scope="class")
    def jax_intermediates(self, jax_checkpoint_path, random_dna_sequence):
        """Extract JAX intermediates in float32."""
        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        params, state = checkpointer.restore(str(jax_checkpoint_path))

        organism_index = np.array([0], dtype=np.int32)  # Human
        return extract_jax_intermediates_float32(
            params, state, random_dna_sequence, organism_index
        )

    @pytest.fixture(scope="class")
    def pytorch_intermediates(self, pytorch_model, random_dna_sequence):
        """Extract PyTorch intermediates in float32."""
        # Ensure model is in float32
        pytorch_model = pytorch_model.float()
        return extract_pytorch_intermediates(pytorch_model, random_dna_sequence)

    def test_architectural_verification(self, jax_intermediates, pytorch_intermediates):
        """Comprehensive layer-by-layer verification.

        Stops at first failure to identify root cause.
        """
        print("\n" + "=" * 100)
        print("ARCHITECTURAL VERIFICATION: JAX vs PyTorch (Float32)")
        print("=" * 100)
        print(f"{'Layer':<35} {'Corr':>10} {'Cosine':>10} {'Scale':>10} {'MaxDiff':>12} {'Status':>8}")
        print("-" * 100)

        # Layers in data flow order
        layers_to_compare = [
            "encoder/dna_embedder",
            "encoder/down_block_0",
            "encoder/down_block_1",
            "encoder/down_block_2",
            "encoder/down_block_3",
            "encoder/down_block_4",
            "encoder/down_block_5",
            "encoder/output",
            # All 9 transformer blocks
            "tower/block_0",
            "tower/block_1",
            "tower/block_2",
            "tower/block_3",
            "tower/block_4",
            "tower/block_5",
            "tower/block_6",
            "tower/block_7",
            "tower/block_8",
            "tower/output",
            "decoder/output",
            "embedder_128bp/output",
            "embedder_1bp/output",
            "embedder_pair/output",
        ]

        results = []
        first_failure = None

        for layer_name in layers_to_compare:
            # Skip if either model doesn't have this layer
            if layer_name not in jax_intermediates:
                print(f"{layer_name:<35} {'N/A (JAX)':<10} {'-':>10} {'-':>10} {'-':>12} {'SKIP':>8}")
                continue
            if layer_name not in pytorch_intermediates:
                print(f"{layer_name:<35} {'N/A (PT)':<10} {'-':>10} {'-':>10} {'-':>12} {'SKIP':>8}")
                continue

            jax_arr = jax_intermediates[layer_name]
            pt_arr = pytorch_intermediates[layer_name]

            # Shape mismatch is a hard failure
            if jax_arr.shape != pt_arr.shape:
                print(f"{layer_name:<35} SHAPE MISMATCH: {jax_arr.shape} vs {pt_arr.shape}")
                if first_failure is None:
                    first_failure = layer_name
                results.append((layer_name, False, 0.0, "SHAPE"))
                continue

            # Compute metrics
            result = compute_metrics(layer_name, pt_arr, jax_arr, corr_threshold=0.9999)
            status = "PASS" if result.passed else "FAIL"

            print(f"{result.name:<35} {result.pearson_corr:>10.6f} {result.cosine_sim:>10.6f} "
                  f"{result.scale_ratio:>10.4f} {result.max_diff:>12.4f} {status:>8}")

            results.append((layer_name, result.passed, result.pearson_corr, result))

            if not result.passed and first_failure is None:
                first_failure = layer_name
                # Print detailed info for first failure
                print(f"\n*** FIRST FAILURE: {layer_name} ***")
                print_layer_result(result)

        print("=" * 100)

        # Summary
        passed_count = sum(1 for _, p, _, _ in results if p)
        total_count = len(results)
        print(f"\nSUMMARY: {passed_count}/{total_count} layers passed (correlation >= 0.9999)")

        if first_failure:
            print(f"FIRST FAILURE: {first_failure}")
            print("Layers after first failure may have cascading errors.")

        print("=" * 100 + "\n")

        # Assert all layers pass
        for layer_name, passed, corr, _ in results:
            if isinstance(corr, str):  # Shape mismatch
                pytest.fail(f"{layer_name}: {corr}")
            assert passed, f"{layer_name}: correlation {corr:.6f} < 0.9999"

    def test_scale_ratio_analysis(self, jax_intermediates, pytorch_intermediates):
        """Analyze scale ratios to detect normalization bugs.

        If scale_ratio drifts from 1.0, it indicates a normalization
        constant mismatch (e.g., different eps in LayerNorm).
        """
        print("\n" + "=" * 80)
        print("SCALE RATIO ANALYSIS (std(PyTorch) / std(JAX))")
        print("=" * 80)
        print(f"{'Layer':<35} {'Scale Ratio':>15} {'Status':>10}")
        print("-" * 80)

        layers = [
            "encoder/dna_embedder",
            "encoder/down_block_0",
            "encoder/down_block_1",
            "encoder/down_block_2",
            "encoder/down_block_3",
            "encoder/down_block_4",
            "encoder/down_block_5",
            "encoder/output",
            # All 9 transformer blocks
            "tower/block_0",
            "tower/block_1",
            "tower/block_2",
            "tower/block_3",
            "tower/block_4",
            "tower/block_5",
            "tower/block_6",
            "tower/block_7",
            "tower/block_8",
            "tower/output",
            "decoder/output",
            "embedder_128bp/output",
            "embedder_1bp/output",
            "embedder_pair/output",
        ]

        scale_issues = []

        for layer_name in layers:
            if layer_name not in jax_intermediates or layer_name not in pytorch_intermediates:
                continue

            jax_arr = jax_intermediates[layer_name]
            pt_arr = pytorch_intermediates[layer_name]

            if jax_arr.shape != pt_arr.shape:
                continue

            pt_std = float(np.std(pt_arr))
            jax_std = float(np.std(jax_arr))
            scale_ratio = pt_std / jax_std if jax_std > 1e-10 else 1.0

            # Flag if scale ratio is off by more than 1%
            if not (0.99 <= scale_ratio <= 1.01):
                status = "WARN"
                scale_issues.append((layer_name, scale_ratio))
            else:
                status = "OK"

            print(f"{layer_name:<35} {scale_ratio:>15.6f} {status:>10}")

        print("=" * 80)

        if scale_issues:
            print("\nWARNING: Scale ratio issues detected (possible normalization bugs):")
            for layer, ratio in scale_issues:
                print(f"  - {layer}: ratio={ratio:.4f}")

        print("=" * 80 + "\n")
