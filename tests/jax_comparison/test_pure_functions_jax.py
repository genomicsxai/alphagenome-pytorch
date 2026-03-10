
import pytest
import torch
import numpy as np
from tests.layer_utils import compute_metrics

# =============================================================================
# PURE FUNCTION TESTS (float32 comparison)
# =============================================================================



class TestPureFunctions:
    """Test pure functions in float32 - these should match to ~1e-6.
    
    Pure functions have no learnable parameters, so differences indicate
    actual implementation bugs rather than weight mismatches.
    """

    def test_gelu(self):
        """Compare GELU activation: JAX vs PyTorch."""
        import jax.numpy as jnp
        from alphagenome_research.model import layers as jax_layers
        from alphagenome_pytorch import layers as pt_layers

        np.random.seed(42)
        x_np = np.random.randn(2, 64, 768).astype(np.float32)

        jax_out = np.array(jax_layers.gelu(jnp.array(x_np)))
        pt_out = pt_layers.gelu(torch.tensor(x_np)).numpy()

        result = compute_metrics("gelu", pt_out, jax_out, corr_threshold=0.999999)

        assert result.passed, f"GELU correlation {result.pearson_corr:.6f} < 0.999999"
        assert result.max_diff < 1e-5, f"GELU max_diff {result.max_diff:.2e} >= 1e-5"

    def test_apply_rope(self):
        """Compare Rotary Position Embeddings: JAX vs PyTorch."""
        import jax.numpy as jnp
        from alphagenome_research.model import attention as jax_attention
        from alphagenome_pytorch import attention as pt_attention

        np.random.seed(42)
        x_np = np.random.randn(2, 64, 8, 128).astype(np.float32)

        jax_out = np.array(jax_attention.apply_rope(jnp.array(x_np), None, max_position=8192))
        pt_out = pt_attention.apply_rope(torch.tensor(x_np), positions=None, max_position=8192).numpy()

        result = compute_metrics("apply_rope", pt_out, jax_out, corr_threshold=0.9999)

        assert result.passed, f"RoPE correlation {result.pearson_corr:.6f} < 0.9999"

    def test_shift(self):
        """Compare _shift operation (exact rearrangement)."""
        import jax.numpy as jnp
        from alphagenome_research.model import attention as jax_attention
        from alphagenome_pytorch import attention as pt_attention

        np.random.seed(42)
        x_np = np.random.randn(2, 8, 64, 128).astype(np.float32)

        jax_out = np.array(jax_attention._shift(jnp.array(x_np), 64, 64))
        pt_out = pt_attention._shift(torch.tensor(x_np), 64, 64).numpy()

        result = compute_metrics("_shift", pt_out, jax_out, corr_threshold=0.999999)

        # Shift is pure rearrangement - must be EXACT
        assert result.max_diff < 1e-7, f"_shift should be exact, got max_diff={result.max_diff:.2e}"

    def test_central_mask_features(self):
        """Compare _central_mask_features."""
        import jax.numpy as jnp
        from alphagenome_research.model import attention as jax_attention
        from alphagenome_pytorch import attention as pt_attention

        distances_np = np.abs(np.arange(-64, 64)).astype(np.float32)

        jax_out = np.array(jax_attention._central_mask_features(
            distances=jnp.array(distances_np), feature_size=32, seq_length=512
        ))
        pt_out = pt_attention._central_mask_features(
            torch.tensor(distances_np), 32, 512
        ).numpy()

        result = compute_metrics("_central_mask_features", pt_out, jax_out, corr_threshold=0.9999)

        assert result.passed, f"central_mask_features correlation {result.pearson_corr:.6f} < 0.9999"

    def test_pool_operations(self):
        """Compare pooling operations: max, avg, stride-16."""
        import jax.numpy as jnp
        import haiku as hk
        from alphagenome_pytorch import layers as pt_layers

        np.random.seed(42)

        test_cases = [
            ("max_pool", "max", 2, 2, (2, 128, 64)),
            ("avg_pool", "avg", 2, 2, (2, 128, 64)),
            ("stride16_pool", "mean", 16, 16, (1, 1024, 768)),
        ]

        for name, method, kernel, stride, shape in test_cases:
            x_np = np.random.randn(*shape).astype(np.float32)

            # JAX pooling
            if method in ("max",):
                def jax_pool_fn(x):
                    return hk.MaxPool(window_shape=(kernel, 1), strides=(stride, 1), padding="SAME")(x)
            else:
                def jax_pool_fn(x):
                    return hk.AvgPool(window_shape=(kernel, 1), strides=(stride, 1), padding="SAME")(x)

            jax_pool = hk.without_apply_rng(hk.transform(jax_pool_fn))
            params = jax_pool.init(None, jnp.array(x_np))
            jax_out = np.array(jax_pool.apply(params, jnp.array(x_np)))

            # PyTorch pooling (Pool1d expects NCL, so transpose NLC -> NCL -> NLC)
            pt_method = "max" if method == "max" else "avg" if method == "avg" else "mean"
            pool_pt = pt_layers.Pool1d(kernel_size=kernel, stride=stride, method=pt_method)
            x_ncl = torch.tensor(x_np).transpose(1, 2)  # NLC -> NCL
            pt_out = pool_pt(x_ncl).transpose(1, 2).numpy()  # NCL -> NLC

            result = compute_metrics(f"pool_{name}", pt_out, jax_out, corr_threshold=0.9999)

            assert result.passed, f"{name} correlation {result.pearson_corr:.6f} < 0.9999"


# =============================================================================
# NORMALIZATION LAYER TESTS
# =============================================================================



class TestNormalizationLayers:
    """Test normalization layers - critical for scale matching."""

    def test_layer_norm_standard(self):
        """Test standard LayerNorm (with centering)."""
        import jax.numpy as jnp
        import haiku as hk
        from alphagenome_pytorch import layers as pt_layers

        np.random.seed(42)
        x_np = np.random.randn(2, 64, 128).astype(np.float32)

        def jax_ln_fn(x):
            return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        jax_ln = hk.without_apply_rng(hk.transform(jax_ln_fn))
        params = jax_ln.init(None, jnp.array(x_np))
        jax_out = np.array(jax_ln.apply(params, jnp.array(x_np)))

        ln_pt = pt_layers.LayerNorm(128, rms_norm=False)
        pt_out = ln_pt(torch.tensor(x_np)).detach().numpy()

        result = compute_metrics("layer_norm_standard", pt_out, jax_out, corr_threshold=0.9999)

        assert result.passed, f"LayerNorm correlation {result.pearson_corr:.6f} < 0.9999"
        # Check scale ratio - should be very close to 1.0 for normalization
        assert 0.99 <= result.scale_ratio <= 1.01, f"LayerNorm scale_ratio {result.scale_ratio:.4f} not ~1.0"

    def test_layer_norm_rms(self):
        """Test RMS LayerNorm (without centering)."""
        import jax.numpy as jnp
        import haiku as hk
        from alphagenome_research.model import layers as jax_layers
        from alphagenome_pytorch import layers as pt_layers

        np.random.seed(42)
        x_np = np.random.randn(2, 64, 128).astype(np.float32)

        def jax_rms_ln_fn(x):
            return jax_layers.LayerNorm(rms_norm=True)(x)

        jax_rms_ln = hk.without_apply_rng(hk.transform(jax_rms_ln_fn))
        params = jax_rms_ln.init(None, jnp.array(x_np))
        jax_out = np.array(jax_rms_ln.apply(params, jnp.array(x_np)))

        rms_ln_pt = pt_layers.LayerNorm(128, rms_norm=True)
        pt_out = rms_ln_pt(torch.tensor(x_np)).detach().numpy()

        result = compute_metrics("layer_norm_rms", pt_out, jax_out, corr_threshold=0.9999)

        assert result.passed, f"RMS LayerNorm correlation {result.pearson_corr:.6f} < 0.9999"
    
# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestUtilityFunctions:

    def test_generate_splice_site_positions(self):
        """Test the splice site filtering method."""
        from alphagenome_research.model.splicing import generate_splice_site_positions as jax_generate_splice_site_positions
        from alphagenome_pytorch.utils.splicing import generate_splice_site_positions as pt_generate_splice_site_positions
        import jax.numpy as jnp
        import numpy as np

        np.random.seed(42)
        B, S = 10, 128
        ref_pt = torch.randn(B, S, 5)
        alt_pt = torch.randn(B, S, 5)

        ref_jax = jnp.array(ref_pt)
        alt_jax = jnp.array(alt_pt)

        jax_out = jax_generate_splice_site_positions(ref_jax, alt_jax, None, k=50, pad_to_length=50, threshold=0.1)
        pt_out = pt_generate_splice_site_positions(ref_pt, alt_pt, None, k=50, pad_to_length=50, threshold=0.1)

        result = compute_metrics("generate_splice_site_positions", pt_out.numpy(), np.array(jax_out), corr_threshold=0.9999)
        
        # KEY CHECK: Ensure output is integer type (LongTensor)
        # This prevents the "tensors used as indices must be long" error
        assert pt_out.dtype in (torch.long, torch.int64), f"generate_splice_site_positions must return LongTensor, got {pt_out.dtype}"

        assert result.passed, f"generate_splice_site_positions correlation {result.pearson_corr:.6f} < 0.9999"
        

