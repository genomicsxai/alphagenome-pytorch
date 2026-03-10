"""Tests for attention module, focusing on RoPE implementation."""

import pytest
import torch

from alphagenome_pytorch.attention import apply_rope


class TestApplyRope:
    """Tests for Rotary Position Embeddings (RoPE) implementation."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("num_heads", [1, 8])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_output_shape(self, batch_size, seq_len, num_heads, head_dim):
        """Verify that output shape matches input shape."""
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        result = apply_rope(x.clone())
        assert result.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("num_heads", [1, 8])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_inplace_matches_standard(self, batch_size, seq_len, num_heads, head_dim):
        """Verify that inplace=True produces identical results to inplace=False."""
        torch.manual_seed(42)
        
        # Create identical inputs (need separate tensors since one is modified)
        x_standard = torch.randn(batch_size, seq_len, num_heads, head_dim)
        x_inplace = x_standard.clone()
        
        # Apply RoPE with both methods
        result_standard = apply_rope(x_standard, inplace=False)
        result_inplace = apply_rope(x_inplace, inplace=True)
        
        # Results should be numerically identical
        torch.testing.assert_close(
            result_inplace, result_standard,
            rtol=1e-5, atol=1e-5,
            msg="Inplace RoPE should match standard RoPE"
        )

    def test_inplace_modifies_input(self):
        """Verify that inplace=True actually modifies the input tensor."""
        torch.manual_seed(42)
        x = torch.randn(1, 16, 4, 64)
        x_original = x.clone()
        
        result = apply_rope(x, inplace=True)
        
        # Result should be the same tensor object
        assert result.data_ptr() == x.data_ptr(), "Inplace should return the same tensor"
        
        # Input should be modified
        assert not torch.allclose(x, x_original), "Inplace should modify input tensor"

    def test_standard_preserves_input(self):
        """Verify that inplace=False does not modify the input tensor."""
        torch.manual_seed(42)
        x = torch.randn(1, 16, 4, 64)
        x_original = x.clone()
        
        _ = apply_rope(x, inplace=False)
        
        # Input should be unchanged
        torch.testing.assert_close(
            x, x_original,
            msg="Standard RoPE should not modify input"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_dtype_preservation(self, dtype, inplace):
        """Verify that output dtype matches input dtype for both modes."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 test skipped on CPU")
        
        x = torch.randn(1, 16, 4, 64, dtype=dtype)
        result = apply_rope(x.clone(), inplace=inplace)
        assert result.dtype == dtype

    @pytest.mark.parametrize("inplace", [False, True])
    def test_with_custom_positions(self, inplace):
        """Verify that custom positions work correctly."""
        torch.manual_seed(42)
        
        x = torch.randn(2, 16, 4, 64)
        positions = torch.arange(16).unsqueeze(0).expand(2, -1).float()
        
        result = apply_rope(x.clone(), positions=positions, inplace=inplace)
        assert result.shape == x.shape

    @pytest.mark.parametrize("inplace", [False, True])
    def test_with_custom_max_position(self, inplace):
        """Verify that custom max_position works correctly."""
        torch.manual_seed(42)
        
        x = torch.randn(1, 16, 4, 64)
        result = apply_rope(x.clone(), max_position=4096, inplace=inplace)
        assert result.shape == (1, 16, 4, 64)

    def test_different_max_positions_give_different_results(self):
        """Verify that different max_position values produce different outputs."""
        torch.manual_seed(42)
        
        x1 = torch.randn(1, 16, 4, 64)
        x2 = x1.clone()
        
        result1 = apply_rope(x1, max_position=4096)
        result2 = apply_rope(x2, max_position=8192)
        
        assert not torch.allclose(result1, result2)

    def test_values_change_after_rope(self):
        """Verify that RoPE actually transforms the values."""
        torch.manual_seed(42)
        x = torch.randn(1, 16, 4, 64)
        x_before = x.clone()
        
        result = apply_rope(x)
        
        # Values should have changed
        assert not torch.allclose(result, x_before)


class TestApplyRopeGradients:
    """Tests for gradient computation through RoPE."""

    def test_gradient_flow_standard(self):
        """Verify that gradients flow through standard RoPE."""
        x = torch.randn(1, 16, 4, 64, requires_grad=True)
        
        result = apply_rope(x, inplace=False)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_inplace(self):
        """Verify that gradients flow through inplace RoPE."""
        x = torch.randn(1, 16, 4, 64, requires_grad=True)
        
        # Need to clone for inplace since we can't do inplace on leaf tensors
        x_clone = x.clone()
        result = apply_rope(x_clone, inplace=True)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_equivalence(self):
        """Verify that gradients are equivalent between standard and inplace."""
        torch.manual_seed(42)
        
        x_standard = torch.randn(1, 16, 4, 64, requires_grad=True)
        x_inplace = x_standard.detach().clone().requires_grad_(True)
        
        # Forward pass
        result_standard = apply_rope(x_standard, inplace=False)
        result_inplace = apply_rope(x_inplace.clone(), inplace=True)
        
        # Backward pass with same upstream gradients
        grad_output = torch.ones_like(result_standard)
        result_standard.backward(grad_output)
        result_inplace.backward(grad_output.clone())
        
        torch.testing.assert_close(
            x_inplace.grad, x_standard.grad,
            rtol=1e-5, atol=1e-5,
            msg="Gradients should match between standard and inplace RoPE"
        )

    def test_gradient_shapes(self):
        """Verify that gradient shapes match input shapes."""
        x = torch.randn(2, 32, 8, 128, requires_grad=True)
        
        result = apply_rope(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad.shape == x.shape


class TestApplyRopeMemory:
    """Tests for memory efficiency of RoPE implementations."""

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil required for memory test"),
        reason="psutil not available"
    )
    def test_inplace_uses_less_memory(self):
        """Verify that inplace=True uses less peak memory than inplace=False.

        The standard implementation creates ~2x memory overhead (stack + flatten),
        while inplace uses ~0.5x overhead (only clones even indices).

        Note: Uses psutil for process-level memory tracking since tracemalloc
        doesn't track PyTorch's C++ tensor allocations.
        """
        import gc

        import psutil

        process = psutil.Process()

        def get_memory_mb():
            """Get current process memory in MB."""
            gc.collect()
            return process.memory_info().rss / 1e6

        # Use a large tensor to get meaningful memory differences
        batch_size, seq_len, num_heads, head_dim = 8, 2048, 16, 128

        # === Measure standard mode ===
        gc.collect()
        baseline = get_memory_mb()

        x_standard = torch.randn(batch_size, seq_len, num_heads, head_dim)
        _ = apply_rope(x_standard, inplace=False)

        peak_standard = get_memory_mb() - baseline

        # === Clear and measure inplace mode ===
        del x_standard
        gc.collect()
        baseline = get_memory_mb()

        x_inplace = torch.randn(batch_size, seq_len, num_heads, head_dim)
        _ = apply_rope(x_inplace, inplace=True)

        peak_inplace = get_memory_mb() - baseline

        # Clean up
        del x_inplace
        gc.collect()

        # Inplace should use less memory (allowing some noise margin)
        # We expect significant savings but process memory can be noisy
        assert peak_inplace < peak_standard * 1.1, (
            f"Inplace should use less memory. "
            f"Standard: {peak_standard:.1f}MB, Inplace: {peak_inplace:.1f}MB"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_inplace_uses_less_memory_cuda(self):
        """Verify that inplace=True uses less peak CUDA memory than inplace=False.

        Uses torch.cuda.max_memory_allocated() for accurate CUDA memory tracking.
        This test is skipped on CPU-only systems but can be run locally on GPU.
        """
        import gc

        device = torch.device("cuda")

        def get_peak_memory_mb():
            """Get peak CUDA memory allocated in MB, then reset."""
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)
            return peak

        # Use a large tensor to get meaningful memory differences
        batch_size, seq_len, num_heads, head_dim = 8, 2048, 16, 128

        # === Measure standard mode ===
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        x_standard = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device
        )
        _ = apply_rope(x_standard, inplace=False)

        peak_standard = get_peak_memory_mb()

        # === Clear and measure inplace mode ===
        del x_standard
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        x_inplace = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device
        )
        _ = apply_rope(x_inplace, inplace=True)

        peak_inplace = get_peak_memory_mb()

        # Clean up
        del x_inplace
        gc.collect()
        torch.cuda.empty_cache()

        # Inplace should use significantly less memory
        # Standard: ~2x overhead, Inplace: ~0.5x overhead
        # We expect at least 20% memory savings
        assert peak_inplace < peak_standard * 0.9, (
            f"Inplace should use at least 10% less CUDA memory. "
            f"Standard: {peak_standard:.1f}MB, Inplace: {peak_inplace:.1f}MB"
        )


class TestMHABlock:
    """Tests for the Multi-Head Attention block."""

    def test_output_shape(self):
        """MHABlock output shape should match input."""
        from alphagenome_pytorch.attention import MHABlock

        d_model = 1536
        mha = MHABlock(d_model=d_model)

        B, S = 1, 32
        x = torch.randn(B, S, d_model)
        attention_bias = torch.zeros(B, 8, S, S)  # 8 heads

        out = mha(x, attention_bias)
        assert out.shape == (B, S, d_model), (
            f"Expected ({B}, {S}, {d_model}), got {out.shape}"
        )

    def test_gradient_flow(self):
        """Gradients should flow through MHABlock to all parameters."""
        from alphagenome_pytorch.attention import MHABlock

        d_model = 1536
        mha = MHABlock(d_model=d_model)

        B, S = 1, 16
        x = torch.randn(B, S, d_model, requires_grad=True)
        attention_bias = torch.zeros(B, 8, S, S)

        out = mha(x, attention_bias)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"
        for name, param in mha.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.all(torch.isfinite(param.grad)), (
                    f"Parameter {name} has non-finite gradient"
                )

    def test_rope_translation_invariance(self):
        """For RoPE, same-offset query-key pairs should yield same rotation.

        If we shift all positions by a constant, the relative rotation
        between same-offset pairs should be identical (translation invariance).
        """
        B, S, H, C = 1, 16, 1, 64

        torch.manual_seed(42)
        x = torch.randn(B, S, H, C)

        # Apply RoPE with positions [0, 1, ..., S-1]
        pos1 = torch.arange(S).unsqueeze(0)
        out1 = apply_rope(x.clone(), positions=pos1)

        # Apply RoPE with positions [10, 11, ..., S+9]
        pos2 = torch.arange(10, S + 10).unsqueeze(0)
        out2 = apply_rope(x.clone(), positions=pos2)

        # The outputs should be DIFFERENT (position-dependent)
        assert not torch.allclose(out1, out2, atol=1e-5), (
            "RoPE with different positions should produce different outputs"
        )

        # But the inner products between adjacent pairs should be the same
        # (translation invariance of relative positions)
        dots1 = (out1[:, :-1, :, :] * out1[:, 1:, :, :]).sum(dim=-1)
        dots2 = (out2[:, :-1, :, :] * out2[:, 1:, :, :]).sum(dim=-1)

        torch.testing.assert_close(
            dots1, dots2, atol=1e-4, rtol=1e-4,
            msg="RoPE relative position inner products should be translation-invariant"
        )
