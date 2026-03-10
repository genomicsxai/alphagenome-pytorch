"""
Unit tests for AlphaGenome training utilities.

Tests training config, loss, optimizer, scheduler, and metrics.
"""

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.training import (
    AlphaGenomeTrainingConfig,
    AlphaGenomeLoss,
    create_optimizer,
    create_scheduler,
)
from alphagenome_pytorch.metrics import (
    pearson_r,
    spearman_r,
    AlphaGenomeMetrics,
)


@pytest.mark.unit
class TestAlphaGenomeTrainingConfig:
    """Tests for AlphaGenomeTrainingConfig."""
    
    def test_default_values(self):
        """Test default config matches paper parameters."""
        config = AlphaGenomeTrainingConfig()
        
        assert config.learning_rate == 0.004
        assert config.weight_decay == 0.4
        assert config.warmup_steps == 5000
        assert config.total_steps == 15000
        assert config.batch_size == 64
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
    
    def test_custom_values(self):
        """Test custom config values."""
        config = AlphaGenomeTrainingConfig(
            learning_rate=0.001,
            warmup_steps=1000,
        )
        
        assert config.learning_rate == 0.001
        assert config.warmup_steps == 1000


@pytest.mark.unit
class TestAlphaGenomeLoss:
    """Tests for AlphaGenomeLoss."""
    
    def test_initialization(self):
        """Test loss module initializes correctly."""
        loss_fn = AlphaGenomeLoss()
        
        assert len(loss_fn.heads) == 8
        assert 'atac' in loss_fn.heads
        assert 'contact_maps' in loss_fn.heads
    
    def test_custom_heads(self):
        """Test loss with subset of heads."""
        loss_fn = AlphaGenomeLoss(heads=['atac', 'dnase'])
        
        assert len(loss_fn.heads) == 2
        assert 'atac' in loss_fn.heads
        assert 'contact_maps' not in loss_fn.heads
    
    def test_forward_basic(self):
        """Test forward pass computes loss."""
        loss_fn = AlphaGenomeLoss(heads=['atac'], multinomial_resolution=2)

        # Create simple outputs/targets in NLC format (batch=1, seq=4, channels=3)
        outputs = {'atac': torch.rand(1, 4, 3) + 0.1}
        targets = {'atac': torch.rand(1, 4, 3) + 0.1}
        masks = {'atac': torch.ones(1, 1, 3, dtype=torch.bool)}  # (B, 1, C) for NLC
        organism_index = torch.tensor([0])  # Human organism

        result = loss_fn(outputs, targets, organism_index, masks)

        assert 'loss' in result
        assert 'atac_loss' in result
        assert torch.isfinite(result['loss'])
        assert result['loss'].item() >= 0

    def test_forward_multiple_heads(self):
        """Test forward with multiple heads."""
        loss_fn = AlphaGenomeLoss(
            heads=['atac', 'dnase'],
            multinomial_resolution=2,
        )

        # NLC format: (B, S, C) = (1, 4, 3)
        outputs = {
            'atac': torch.rand(1, 4, 3) + 0.1,
            'dnase': torch.rand(1, 4, 3) + 0.1,
        }
        targets = {
            'atac': torch.rand(1, 4, 3) + 0.1,
            'dnase': torch.rand(1, 4, 3) + 0.1,
        }
        organism_index = torch.tensor([0])  # Human organism

        result = loss_fn(outputs, targets, organism_index)

        assert 'loss' in result
        assert 'atac_loss' in result
        assert 'dnase_loss' in result


@pytest.mark.unit
class TestCreateOptimizer:
    """Tests for create_optimizer."""
    
    def test_creates_adamw(self):
        """Test creates AdamW optimizer with correct params."""
        model = nn.Linear(10, 5)
        config = AlphaGenomeTrainingConfig(
            learning_rate=0.001,
            weight_decay=0.1,
        )
        
        optimizer = create_optimizer(model, config)
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['weight_decay'] == 0.1
    
    def test_uses_paper_defaults(self):
        """Test uses paper default parameters."""
        model = nn.Linear(10, 5)
        config = AlphaGenomeTrainingConfig()
        
        optimizer = create_optimizer(model, config)
        
        assert optimizer.defaults['lr'] == 0.004
        assert optimizer.defaults['betas'] == (0.9, 0.999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.4


@pytest.mark.unit
class TestCreateScheduler:
    """Tests for create_scheduler."""
    
    def test_warmup_phase(self):
        """Test linear warmup behavior."""
        model = nn.Linear(10, 5)
        config = AlphaGenomeTrainingConfig(
            learning_rate=0.004,
            warmup_steps=100,
            total_steps=200,
        )
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        # At step 0
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)
        
        # At step 50 (half warmup)
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(0.002, rel=0.01)
        
        # At step 100 (end warmup)
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(0.004, rel=0.01)
    
    def test_cosine_decay(self):
        """Test cosine decay after warmup."""
        model = nn.Linear(10, 5)
        config = AlphaGenomeTrainingConfig(
            learning_rate=0.004,
            warmup_steps=0,  # No warmup
            total_steps=100,
        )
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        # At step 0 (start of decay)
        assert scheduler.get_last_lr()[0] == pytest.approx(0.004, rel=0.01)
        
        # At step 50 (middle)
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(0.002, rel=0.1)
        
        # At step 100 (end)
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
class TestPearsonR:
    """Tests for pearson_r."""
    
    def test_perfect_correlation(self):
        """Test r=1 for identical vectors."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        r = pearson_r(x, x)
        assert r.item() == pytest.approx(1.0, rel=1e-5)
    
    def test_negative_correlation(self):
        """Test r=-1 for perfect negative correlation."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        r = pearson_r(x, y)
        assert r.item() == pytest.approx(-1.0, rel=1e-5)
    
    def test_no_correlation(self):
        """Test r≈0 for uncorrelated random vectors."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        y = torch.randn(1000)
        r = pearson_r(x, y)
        assert abs(r.item()) < 0.1


@pytest.mark.unit
class TestAlphaGenomeMetrics:
    """Tests for AlphaGenomeMetrics."""
    
    def test_computes_pearson_r(self):
        """Test computes Pearson R for each head."""
        metrics = AlphaGenomeMetrics(heads=['atac', 'dnase'])
        
        outputs = {
            'atac': torch.randn(10, 100),
            'dnase': torch.randn(10, 100),
        }
        targets = {
            'atac': outputs['atac'] + 0.1 * torch.randn(10, 100),  # High correlation
            'dnase': torch.randn(10, 100),  # Low correlation
        }
        
        result = metrics(outputs, targets)
        
        assert 'atac_pearson_r' in result
        assert 'dnase_pearson_r' in result
        assert 'avg_pearson_r' in result
        
        # ATAC should have high correlation
        assert result['atac_pearson_r'] > 0.5
    
    def test_handles_dict_outputs(self):
        """Test handles resolution dict outputs."""
        metrics = AlphaGenomeMetrics(heads=['atac'])
        
        outputs = {
            'atac': {1: torch.randn(10, 100), 128: torch.randn(10, 10)},
        }
        targets = {
            'atac': {1: torch.randn(10, 100)},
        }
        
        result = metrics(outputs, targets)
        
        assert 'atac_pearson_r' in result
