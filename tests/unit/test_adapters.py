"""
Unit tests for PEFT adapters.

Tests LoRA, Locon, IA3, and Houlsby adapters.
"""

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.extensions.finetuning.adapters import (
    LoRA,
    Locon,
    IA3,
    IA3_FF,
    AdapterHoulsby,
    HoulsbyWrapper,
    apply_lora,
    apply_locon,
    apply_ia3,
    apply_houlsby,
    merge_adapters,
    get_adapter_params,
)



@pytest.mark.unit
class TestLoRA:
    """Tests for LoRA adapter."""
    
    def test_initialization(self):
        """Test LoRA initializes correctly."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8, alpha=16)
        
        assert lora.rank == 8
        assert lora.alpha == 16
        assert lora.scale == 2.0
        assert lora.in_features == 64
        assert lora.out_features == 32
    
    def test_forward_shape(self):
        """Test output shape matches original layer."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8)
        
        x = torch.randn(2, 10, 64)
        output = lora(x)
        
        assert output.shape == (2, 10, 32)
    
    def test_original_frozen(self):
        """Test original layer is frozen."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8)
        
        assert not lora.original_layer.weight.requires_grad
        assert not lora.original_layer.bias.requires_grad
    
    def test_lora_trainable(self):
        """Test LoRA matrices are trainable."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8)
        
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad
    
    def test_initial_output_equals_original(self):
        """Test at initialization, output equals original (B initialized to 0)."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8)
        
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            original_output = linear(x)
            lora_output = lora(x)
        
        torch.testing.assert_close(lora_output, original_output)
    
    def test_merge_weights(self):
        """Test weight merging produces equivalent output."""
        linear = nn.Linear(64, 32)
        lora = LoRA(linear, rank=8)
        
        # Train LoRA a bit
        with torch.no_grad():
            lora.lora_A.weight.fill_(0.1)
            lora.lora_B.weight.fill_(0.2)
        
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            lora_output = lora(x)
            merged = lora.merge_weights()
            merged_output = merged(x)
        
        torch.testing.assert_close(merged_output, lora_output, rtol=1e-5, atol=1e-5)
    
    def test_trainable_param_count(self):
        """Test significantly fewer trainable params than full layer."""
        linear = nn.Linear(768, 768)
        lora = LoRA(linear, rank=8)
        
        full_params = sum(p.numel() for p in linear.parameters())
        lora_trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        
        # LoRA should have << 10% of original params
        assert lora_trainable < full_params * 0.1


@pytest.mark.unit
class TestLocon:
    """Tests for Locon adapter (Conv1D LoRA)."""
    
    def test_forward_shape(self):
        """Test output shape matches original layer."""
        conv = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        locon = Locon(conv, rank=4)
        
        x = torch.randn(2, 64, 100)  # (batch, channels, length)
        output = locon(x)
        
        assert output.shape == (2, 32, 100)
    
    def test_original_frozen(self):
        """Test original layer is frozen."""
        conv = nn.Conv1d(64, 32, kernel_size=5)
        locon = Locon(conv, rank=4)
        
        assert not locon.original_layer.weight.requires_grad
    
    def test_initial_output_equals_original(self):
        """Test at initialization, output equals original."""
        conv = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        locon = Locon(conv, rank=4)
        
        x = torch.randn(2, 64, 100)
        
        with torch.no_grad():
            original_output = conv(x)
            locon_output = locon(x)
        
        torch.testing.assert_close(locon_output, original_output)


@pytest.mark.unit
class TestIA3:
    """Tests for IA3 adapter."""
    
    def test_forward_shape(self):
        """Test output shape matches original layer."""
        linear = nn.Linear(64, 32)
        ia3 = IA3(linear)
        
        x = torch.randn(2, 10, 64)
        output = ia3(x)
        
        assert output.shape == (2, 10, 32)
    
    def test_param_count(self):
        """Test only output_dim trainable params."""
        linear = nn.Linear(64, 32)
        ia3 = IA3(linear)
        
        trainable = sum(p.numel() for p in ia3.parameters() if p.requires_grad)
        assert trainable == 32  # Just the scaling vector
    
    def test_initial_output_equals_original(self):
        """Test at initialization, output equals original (scale=1)."""
        linear = nn.Linear(64, 32)
        ia3 = IA3(linear)
        
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            original_output = linear(x)
            ia3_output = ia3(x)
        
        torch.testing.assert_close(ia3_output, original_output)


@pytest.mark.unit
class TestIA3FF:
    """Tests for IA3_FF adapter (input scaling)."""
    
    def test_forward_shape(self):
        """Test output shape matches original layer."""
        linear = nn.Linear(64, 32)
        ia3_ff = IA3_FF(linear)
        
        x = torch.randn(2, 10, 64)
        output = ia3_ff(x)
        
        assert output.shape == (2, 10, 32)
    
    def test_param_count(self):
        """Test only input_dim trainable params."""
        linear = nn.Linear(64, 32)
        ia3_ff = IA3_FF(linear)
        
        trainable = sum(p.numel() for p in ia3_ff.parameters() if p.requires_grad)
        assert trainable == 64  # Just the input scaling vector


@pytest.mark.unit
class TestAdapterHoulsby:
    """Tests for Houlsby bottleneck adapter."""
    
    def test_forward_shape(self):
        """Test output shape equals input shape (residual)."""
        adapter = AdapterHoulsby(input_dim=64, latent_dim=8)
        
        x = torch.randn(2, 10, 64)
        output = adapter(x)
        
        assert output.shape == x.shape
    
    def test_residual_connection(self):
        """Test output includes residual from input."""
        adapter = AdapterHoulsby(input_dim=64, latent_dim=8)
        
        # With zero weights, output should be input (just residual)
        with torch.no_grad():
            adapter.down_project.weight.zero_()
            adapter.up_project.weight.zero_()
            adapter.down_project.bias.zero_()
            adapter.up_project.bias.zero_()
        
        x = torch.randn(2, 10, 64)
        output = adapter(x)
        
        torch.testing.assert_close(output, x)
    
    def test_param_count(self):
        """Test param count is bottleneck size."""
        adapter = AdapterHoulsby(input_dim=768, latent_dim=8)
        
        params = sum(p.numel() for p in adapter.parameters())
        # down: 768*8 + 8, up: 8*768 + 768
        expected = 768 * 8 + 8 + 8 * 768 + 768
        assert params == expected


# =============================================================================
# Tests for API functions
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing apply functions."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.linear2(self.linear1(x))


@pytest.mark.unit
class TestApplyLora:
    """Tests for apply_lora function."""
    
    def test_applies_to_matching_modules(self):
        """Test LoRA is applied to modules matching target."""
        model = SimpleModel()
        model = apply_lora(model, ['linear1'], rank=4)
        
        assert isinstance(model.linear1, LoRA)
        assert isinstance(model.linear2, nn.Linear)  # Not modified
    
    def test_applies_to_multiple_modules(self):
        """Test LoRA is applied to multiple matching modules."""
        model = SimpleModel()
        model = apply_lora(model, ['linear'], rank=4)
        
        assert isinstance(model.linear1, LoRA)
        assert isinstance(model.linear2, LoRA)


@pytest.mark.unit
class TestApplyLocon:
    """Tests for apply_locon function."""
    
    def test_applies_to_conv_modules(self):
        """Test Locon is applied to Conv1d modules."""
        model = SimpleModel()
        model = apply_locon(model, ['conv1'], rank=2)
        
        assert isinstance(model.conv1, Locon)


@pytest.mark.unit  
class TestApplyIA3:
    """Tests for apply_ia3 function."""
    
    def test_applies_ia3_to_modules(self):
        """Test IA3 is applied to matching modules."""
        model = SimpleModel()
        model = apply_ia3(model, ['linear1'])
        
        assert isinstance(model.linear1, IA3)
        assert isinstance(model.linear2, nn.Linear)


@pytest.mark.unit
class TestMergeAdapters:
    """Tests for merge_adapters function."""
    
    def test_merges_lora(self):
        """Test LoRA adapters are merged."""
        model = SimpleModel()
        model = apply_lora(model, ['linear1'], rank=4)
        
        assert isinstance(model.linear1, LoRA)
        
        model = merge_adapters(model)
        
        assert isinstance(model.linear1, nn.Linear)
        assert not isinstance(model.linear1, LoRA)
    
    def test_merged_output_matches(self):
        """Test merged model produces same output."""
        model = SimpleModel()
        model = apply_lora(model, ['linear1'], rank=4)
        
        # Set some LoRA weights
        with torch.no_grad():
            model.linear1.lora_A.weight.fill_(0.1)
            model.linear1.lora_B.weight.fill_(0.2)
        
        x = torch.randn(2, 64)
        
        with torch.no_grad():
            output_before = model(x)
            model = merge_adapters(model)
            output_after = model(x)
        
        torch.testing.assert_close(output_after, output_before, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
class TestGetAdapterParams:
    """Tests for get_adapter_params function."""
    
    def test_returns_only_adapter_params(self):
        """Test only adapter params are returned."""
        model = SimpleModel()
        original_params = sum(p.numel() for p in model.parameters())
        
        model = apply_lora(model, ['linear1'], rank=4)
        
        adapter_params = get_adapter_params(model)
        adapter_param_count = sum(p.numel() for p in adapter_params)
        
        # Should be much less than original
        assert adapter_param_count < original_params
        # Should be LoRA A + B: 64*4 + 4*32 = 384
        assert adapter_param_count == 64 * 4 + 4 * 32
    
    def test_all_params_trainable(self):
        """Test all returned params are trainable."""
        model = SimpleModel()
        model = apply_lora(model, ['linear1'], rank=4)

        for param in get_adapter_params(model):
            assert param.requires_grad


@pytest.mark.unit
class TestAdapterComposition:
    """Tests for composing multiple adapter types safely."""

    def test_lora_then_houlsby_same_target_no_double_wrap(self):
        """Test Houlsby doesn't wrap LoRA's internal original_layer."""
        model = SimpleModel()

        # Apply LoRA first
        model = apply_lora(model, ['linear1'], rank=4)
        assert isinstance(model.linear1, LoRA)

        # Apply Houlsby to same target - should NOT wrap original_layer
        model = apply_houlsby(model, ['linear1'], latent_dim=4)

        # The LoRA wrapper should still be there (unchanged)
        assert isinstance(model.linear1, LoRA)
        # The original_layer inside LoRA should still be plain Linear
        assert isinstance(model.linear1.original_layer, nn.Linear)
        assert not isinstance(model.linear1.original_layer, HoulsbyWrapper)

    def test_houlsby_then_lora_same_target_no_double_wrap(self):
        """Test LoRA doesn't wrap HoulsbyWrapper's internal original_layer."""
        model = SimpleModel()

        # Apply Houlsby first
        model = apply_houlsby(model, ['linear1'], latent_dim=4)
        assert isinstance(model.linear1, HoulsbyWrapper)

        # Apply LoRA to same target - should NOT wrap original_layer
        model = apply_lora(model, ['linear1'], rank=4)

        # The Houlsby wrapper should still be there (unchanged)
        assert isinstance(model.linear1, HoulsbyWrapper)
        # The original_layer inside Houlsby should still be plain Linear
        assert isinstance(model.linear1.original_layer, nn.Linear)
        assert not isinstance(model.linear1.original_layer, LoRA)

    def test_lora_merge_works_after_houlsby_attempt(self):
        """Test merge_weights still works after Houlsby is applied to same target."""
        model = SimpleModel()

        # Apply LoRA
        model = apply_lora(model, ['linear1'], rank=4)

        # Train LoRA a bit
        with torch.no_grad():
            model.linear1.lora_A.weight.fill_(0.1)
            model.linear1.lora_B.weight.fill_(0.2)

        # Apply Houlsby to same target (should be no-op for this layer)
        model = apply_houlsby(model, ['linear1'], latent_dim=4)

        # merge_weights should still work
        x = torch.randn(2, 64)
        with torch.no_grad():
            output_before = model.linear1(x)
            merged = model.linear1.merge_weights()
            output_after = merged(x)

        torch.testing.assert_close(output_after, output_before, rtol=1e-5, atol=1e-5)

    def test_different_targets_both_applied(self):
        """Test adapters on different targets are both applied correctly."""
        model = SimpleModel()

        # Apply LoRA to linear1
        model = apply_lora(model, ['linear1'], rank=4)

        # Apply Houlsby to linear2
        model = apply_houlsby(model, ['linear2'], latent_dim=4)

        # Both should be wrapped
        assert isinstance(model.linear1, LoRA)
        assert isinstance(model.linear2, HoulsbyWrapper)

