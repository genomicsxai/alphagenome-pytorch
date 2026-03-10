"""
Unit tests for transfer learning module.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    load_trunk,
    prepare_for_transfer,
    count_trainable_params,
)


# Mock model for testing (simpler than full AlphaGenome)
class MockAlphaGenome(nn.Module):
    """Simplified AlphaGenome-like model for testing."""
    
    def __init__(self, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        
        # Trunk components
        self.encoder = nn.Linear(64, 128)
        self.tower = nn.ModuleDict({
            'block0': nn.ModuleDict({
                'to_q': nn.Linear(128, 128),
                'to_v': nn.Linear(128, 128),
                'mlp': nn.Linear(128, 128),
            })
        })
        # Conv tower for Locon testing
        self.conv_tower = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.decoder = nn.Linear(128, 64)
        
        # Heads (using simple Linear as mock)
        self.heads = nn.ModuleDict({
            'atac': nn.Linear(64, 256),
            'dnase': nn.Linear(64, 384),
        })
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.tower['block0']['to_q'](x)
        x = self.decoder(x)
        return {name: head(x) for name, head in self.heads.items()}


@pytest.mark.unit
class TestTransferConfig:
    """Tests for TransferConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TransferConfig()
        
        assert config.mode == 'linear'
        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.lora_targets == ['q_proj', 'v_proj']
        assert config.locon_rank == 4
        assert config.locon_alpha == 1
        assert config.locon_targets == ['conv_tower']
        assert config.ia3_targets == ['to_k', 'to_v']
        assert config.ia3_ff_targets == []
        assert config.new_heads == {}
        assert config.remove_heads == []
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = TransferConfig(
            mode='lora',
            lora_rank=4,
            new_heads={'custom': {'num_tracks': 10}},
        )
        
        assert config.mode == 'lora'
        assert config.lora_rank == 4
        assert 'custom' in config.new_heads
    
    def test_multi_mode_list(self):
        """Test mode accepts a list."""
        config = TransferConfig(mode=['lora', 'locon'])
        assert config.mode == ['lora', 'locon']


@pytest.mark.unit
class TestLoadTrunk:
    """Tests for load_trunk function."""
    
    def test_load_excludes_heads(self):
        """Test that head weights are excluded when requested."""
        model = MockAlphaGenome()
        
        # Save full model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            weights_path = f.name
        
        try:
            # Create new model with different head sizes (won't match)
            new_model = MockAlphaGenome()
            new_model.heads = nn.ModuleDict({
                'custom': nn.Linear(64, 100),  # Different head
            })
            
            # Load should work because heads are excluded
            load_trunk(new_model, weights_path, exclude_heads=True)
            
            # Trunk weights should be loaded
            # (here just check no error w/o comparing weights)
            assert True
        finally:
            os.unlink(weights_path)
    
    def test_load_full_weights(self):
        """Test loading full weights with exclude_heads=False."""
        model = MockAlphaGenome()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            weights_path = f.name
        
        try:
            new_model = MockAlphaGenome()
            new_model = load_trunk(new_model, weights_path, exclude_heads=False)
            
            # All weights should match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), 
                                           new_model.named_parameters()):
                torch.testing.assert_close(p1, p2)
        finally:
            os.unlink(weights_path)


@pytest.mark.unit
class TestPrepareForTransfer:
    """Tests for prepare_for_transfer function."""
    
    def test_linear_mode_freezes_trunk(self):
        """Test linear probing freezes trunk."""
        model = MockAlphaGenome()
        config = TransferConfig(mode='linear')
        
        model = prepare_for_transfer(model, config)
        
        # Check trunk is frozen
        assert not model.encoder.weight.requires_grad
        assert not model.tower['block0']['to_q'].weight.requires_grad
        
        # Check heads are trainable
        for head in model.heads.values():
            assert head.weight.requires_grad
    
    def test_remove_heads(self):
        """Test head removal."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='linear',
            remove_heads=['atac'],
        )
        
        model = prepare_for_transfer(model, config)
        
        assert 'atac' not in model.heads
        assert 'dnase' in model.heads
    
    def test_keep_heads(self):
        """Test keep_heads option."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='linear',
            keep_heads=['atac'],
        )
        
        model = prepare_for_transfer(model, config)
        
        assert 'atac' in model.heads
        assert 'dnase' not in model.heads
    
    def test_lora_mode_applies_adapters(self):
        """Test LoRA mode applies adapters."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='lora',
            lora_targets=['to_q'],
            lora_rank=4,
        )
        
        model = prepare_for_transfer(model, config)
        
        # Check LoRA was applied
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA
        assert isinstance(model.tower['block0']['to_q'], LoRA)
        
        # Check original to_v is frozen (not wrapped)
        assert not model.tower['block0']['to_v'].weight.requires_grad


@pytest.mark.unit
class TestCountTrainableParams:
    """Tests for count_trainable_params function."""
    
    def test_counts_after_linear(self):
        """Test param counting after linear probing setup."""
        model = MockAlphaGenome()
        config = TransferConfig(mode='linear')
        model = prepare_for_transfer(model, config)
        
        counts = count_trainable_params(model)
        
        assert counts['total'] > 0
        assert counts['heads'] > 0
        assert counts['other'] == 0  # Trunk frozen
    
    def test_counts_after_lora(self):
        """Test param counting after LoRA setup."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='lora',
            lora_targets=['to_q'],
        )
        model = prepare_for_transfer(model, config)
        
        counts = count_trainable_params(model)
        
        assert counts['total'] > 0
        assert counts['heads'] > 0
        assert counts['adapters'] > 0


@pytest.mark.unit
class TestTransferEdgeCases:
    """Edge case tests for transfer learning module."""

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            config = TransferConfig(mode='invalid_mode')
            model = MockAlphaGenome()
            prepare_for_transfer(model, config)

    def test_linear_mode_preserves_output_shape(self):
        """Linear probing should preserve model output shape."""
        model = MockAlphaGenome()
        x = torch.randn(2, 64)
        original_out = model(x)

        config = TransferConfig(mode='linear')
        model = prepare_for_transfer(model, config)

        new_out = model(x)
        for key in original_out:
            assert original_out[key].shape == new_out[key].shape

    def test_lora_rank_correct(self):
        """LoRA adapter should have the requested rank dimension."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='lora',
            lora_targets=['to_q'],
            lora_rank=4,
        )

        model = prepare_for_transfer(model, config)

        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA
        lora_module = model.tower['block0']['to_q']
        assert isinstance(lora_module, LoRA)
        # Check the rank is correct: lora_A is nn.Linear(in_features, rank)
        assert lora_module.lora_A.weight.shape[0] == 4

    def test_remove_all_heads(self):
        """Removing all heads should leave empty heads dict."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode='linear',
            remove_heads=['atac', 'dnase'],
        )

        model = prepare_for_transfer(model, config)
        assert len(model.heads) == 0

    def test_multi_mode_lora_locon(self):
        """Combining LoRA + Locon applies both adapter types."""
        model = MockAlphaGenome()
        config = TransferConfig(
            mode=['lora', 'locon'],
            lora_targets=['to_q'],
            lora_rank=4,
            locon_targets=['conv_tower'],
            locon_rank=2,
        )

        model = prepare_for_transfer(model, config)

        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA, Locon
        assert isinstance(model.tower['block0']['to_q'], LoRA)
        assert isinstance(model.conv_tower, Locon)

        # Trunk should be frozen except for adapters
        assert not model.encoder.weight.requires_grad
        # Heads should be trainable
        for head in model.heads.values():
            assert head.weight.requires_grad

    def test_full_mode_cannot_combine(self):
        """'full' cannot be combined with other modes."""
        config = TransferConfig(mode=['full', 'lora'])
        model = MockAlphaGenome()
        with pytest.raises(ValueError, match="cannot be combined"):
            prepare_for_transfer(model, config)

    def test_full_mode_single(self):
        """'full' mode alone trains everything."""
        model = MockAlphaGenome()
        config = TransferConfig(mode='full')
        model = prepare_for_transfer(model, config)

        # All params should be trainable
        for p in model.parameters():
            assert p.requires_grad
