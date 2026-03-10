"""
Unit tests for fine-tuning training modes (linear-probe, LoRA).

Tests that forward/backward passes work correctly and that gradients
flow to the expected parameters for each mode.
"""

import gc

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    prepare_for_transfer,
    remove_all_heads,
    add_head,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params


# Use small sequence for fast tests
SEQ_LENGTH = 4096  # 2^12


def _create_model_and_head(mode: str, lora_rank: int = 8, lora_alpha: int = 16):
    """Create a model configured for the specified training mode.

    Matches the approach in finetune.py: freeze first, then create heads.
    This way heads have requires_grad=True by default.

    Args:
        mode: 'linear-probe', 'lora', or 'full'
        lora_rank: LoRA rank (only used if mode='lora')
        lora_alpha: LoRA alpha (only used if mode='lora')

    Returns:
        Tuple of (model, head, trainable_params)
    """
    # Create model
    model = AlphaGenome(
        num_organisms=1,
        dtype_policy=DtypePolicy.full_float32(),
    )

    # Freeze base model first (for non-full modes)
    if mode != "full":
        for param in model.parameters():
            param.requires_grad = False

    # Remove original heads
    model = remove_all_heads(model)

    # Create head AFTER freeze (so it has requires_grad=True by default)
    n_tracks = 4
    head = create_finetuning_head(
        assay_type="atac",
        n_tracks=n_tracks,
        resolutions=(128,),  # 128bp only for speed
        num_organisms=1,
    )
    add_head(model, "atac", head)

    trainable_params = []

    if mode == "linear-probe":
        # Head already has requires_grad=True
        trainable_params = list(head.parameters())

    elif mode == "lora":
        if lora_rank > 0:
            config = TransferConfig(
                mode="lora",
                lora_targets=["q_proj", "v_proj"],
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            model = prepare_for_transfer(model, config)
            # LoRA adapters + head (head already has requires_grad=True)
            trainable_params = get_adapter_params(model)
            trainable_params.extend(list(head.parameters()))
        else:
            # LoRA rank 0 = linear probe (head already has requires_grad=True)
            trainable_params = list(head.parameters())

    elif mode == "full":
        # All parameters trainable (model was not frozen)
        trainable_params = list(model.parameters())

    return model, head, trainable_params


def _get_frozen_backbone_flag(mode: str, lora_rank: int = 8) -> bool:
    """Get the frozen_backbone flag as used in finetune.py."""
    return mode == "linear-probe" or (mode == "lora" and lora_rank == 0)


@pytest.mark.integration
class TestLinearProbeMode:
    """Tests for linear-probe training mode."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_forward_pass(self):
        """Test forward pass works in linear-probe mode."""
        model, head, _ = _create_model_and_head("linear-probe")
        model.eval()

        # Create dummy input
        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        with torch.no_grad():
            outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        assert "embeddings_128bp" in outputs

        # Pass through head
        embeddings = {128: outputs["embeddings_128bp"]}
        preds = head(embeddings, organism_idx, return_scaled=True)

        assert 128 in preds
        expected_out_len = SEQ_LENGTH // 128
        assert preds[128].shape == (batch_size, expected_out_len, 4)  # 4 tracks

    def test_backward_pass(self):
        """Test backward pass computes gradients for head only."""
        model, head, trainable_params = _create_model_and_head("linear-probe")
        model.train()

        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        # Forward with no_grad for backbone (as in training loop)
        frozen_backbone = _get_frozen_backbone_flag("linear-probe")
        assert frozen_backbone is True

        with torch.no_grad():
            outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        embeddings = {128: outputs["embeddings_128bp"].detach()}
        preds = head(embeddings, organism_idx, return_scaled=True)

        # Compute loss and backward
        loss = preds[128].sum()
        loss.backward()

        # Check head params have gradients
        head_params_with_grad = [p for p in head.parameters() if p.grad is not None]
        assert len(head_params_with_grad) > 0, "Head params should have gradients"

        # Check backbone params do NOT have gradients
        model_without_head = model.module if hasattr(model, 'module') else model
        backbone_params = [p for name, p in model_without_head.named_parameters()
                         if 'heads' not in name]
        backbone_with_grad = [p for p in backbone_params if p.grad is not None]
        assert len(backbone_with_grad) == 0, "Backbone params should NOT have gradients in linear-probe"

    def test_only_head_params_trainable(self):
        """Test only head parameters are trainable."""
        model, head, trainable_params = _create_model_and_head("linear-probe")

        # All trainable params should be from the head
        head_param_ids = {id(p) for p in head.parameters()}
        trainable_param_ids = {id(p) for p in trainable_params}

        assert trainable_param_ids == head_param_ids


@pytest.mark.integration
class TestLoRAMode:
    """Tests for LoRA training mode."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_forward_pass(self):
        """Test forward pass works in LoRA mode."""
        model, head, _ = _create_model_and_head("lora", lora_rank=8)
        model.eval()

        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        with torch.no_grad():
            outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        assert "embeddings_128bp" in outputs

        embeddings = {128: outputs["embeddings_128bp"]}
        preds = head(embeddings, organism_idx, return_scaled=True)

        assert 128 in preds
        expected_out_len = SEQ_LENGTH // 128
        assert preds[128].shape == (batch_size, expected_out_len, 4)

    def test_backward_pass_with_lora(self):
        """Test backward pass computes gradients for LoRA + head params."""
        model, head, trainable_params = _create_model_and_head("lora", lora_rank=8)
        model.train()

        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        # LoRA mode should NOT freeze backbone (gradients flow through LoRA)
        frozen_backbone = _get_frozen_backbone_flag("lora", lora_rank=8)
        assert frozen_backbone is False, "LoRA with rank>0 should not freeze backbone"

        # Forward WITHOUT no_grad (LoRA needs gradients)
        outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        embeddings = {128: outputs["embeddings_128bp"]}
        preds = head(embeddings, organism_idx, return_scaled=True)

        # Compute loss and backward
        loss = preds[128].sum()
        loss.backward()

        # Check head params have gradients
        head_params_with_grad = [p for p in head.parameters() if p.grad is not None]
        assert len(head_params_with_grad) > 0, "Head params should have gradients"

        # Check LoRA params have gradients
        lora_params = get_adapter_params(model)
        lora_with_grad = [p for p in lora_params if p.grad is not None]
        assert len(lora_with_grad) > 0, "LoRA params should have gradients"

        # Check original backbone params do NOT have gradients
        # (they have requires_grad=False)
        # Note: norm layers are unfrozen by default (unfreeze_norm=True)
        model_module = model.module if hasattr(model, 'module') else model
        for name, param in model_module.named_parameters():
            if 'lora_' not in name and 'heads' not in name and 'norm' not in name:
                if param.requires_grad:
                    assert False, f"Original backbone param {name} should be frozen"

    def test_lora_rank_zero_equals_linear_probe(self):
        """Test LoRA with rank=0 behaves like linear-probe."""
        model, head, trainable_params = _create_model_and_head("lora", lora_rank=0)

        # Should have frozen_backbone=True
        frozen_backbone = _get_frozen_backbone_flag("lora", lora_rank=0)
        assert frozen_backbone is True, "LoRA rank=0 should freeze backbone"

        # All trainable params should be from head only
        head_param_ids = {id(p) for p in head.parameters()}
        trainable_param_ids = {id(p) for p in trainable_params}

        assert trainable_param_ids == head_param_ids

    def test_lora_params_are_trainable(self):
        """Test LoRA adapter params are marked trainable."""
        model, head, trainable_params = _create_model_and_head("lora", lora_rank=8)

        lora_params = get_adapter_params(model)

        # LoRA params should be in trainable_params
        lora_param_ids = {id(p) for p in lora_params}
        trainable_param_ids = {id(p) for p in trainable_params}

        assert lora_param_ids.issubset(trainable_param_ids), \
            "LoRA params should be subset of trainable params"

        # All LoRA params should have requires_grad=True
        for param in lora_params:
            assert param.requires_grad, "LoRA params should be trainable"


@pytest.mark.integration
class TestFullFinetuningMode:
    """Tests for full finetuning mode."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_forward_pass(self):
        """Test forward pass works in full mode."""
        model, head, _ = _create_model_and_head("full")
        model.eval()

        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        with torch.no_grad():
            outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        assert "embeddings_128bp" in outputs

    def test_backward_pass_all_params(self):
        """Test backward pass computes gradients for all params."""
        model, head, trainable_params = _create_model_and_head("full")
        model.train()

        batch_size = 1
        seq = torch.randn(batch_size, SEQ_LENGTH, 4)
        organism_idx = torch.zeros(batch_size, dtype=torch.long)

        # Full mode should NOT freeze backbone
        frozen_backbone = _get_frozen_backbone_flag("full")
        assert frozen_backbone is False

        # Forward
        outputs = model(seq, organism_idx, return_embeddings=True, resolutions=(128,), channels_last=False)

        embeddings = {128: outputs["embeddings_128bp"]}
        preds = head(embeddings, organism_idx, return_scaled=True)

        # Backward
        loss = preds[128].sum()
        loss.backward()

        # Check backbone params have gradients
        model_module = model.module if hasattr(model, 'module') else model
        backbone_with_grad = 0
        for name, param in model_module.named_parameters():
            if 'heads' not in name and param.grad is not None:
                backbone_with_grad += 1

        assert backbone_with_grad > 0, "Backbone params should have gradients in full mode"

    def test_all_params_trainable(self):
        """Test all model parameters are trainable in full mode."""
        model, head, trainable_params = _create_model_and_head("full")

        all_params = list(model.parameters())

        # All params should be trainable
        for param in all_params:
            assert param.requires_grad, "All params should be trainable in full mode"


@pytest.mark.integration
class TestFrozenBackboneFlag:
    """Tests for the frozen_backbone flag logic."""

    def test_linear_probe_freezes(self):
        """Test linear-probe mode sets frozen_backbone=True."""
        assert _get_frozen_backbone_flag("linear-probe") is True

    def test_lora_with_rank_does_not_freeze(self):
        """Test LoRA with rank>0 sets frozen_backbone=False."""
        assert _get_frozen_backbone_flag("lora", lora_rank=8) is False
        assert _get_frozen_backbone_flag("lora", lora_rank=4) is False
        assert _get_frozen_backbone_flag("lora", lora_rank=1) is False

    def test_lora_rank_zero_freezes(self):
        """Test LoRA with rank=0 sets frozen_backbone=True."""
        assert _get_frozen_backbone_flag("lora", lora_rank=0) is True

    def test_full_does_not_freeze(self):
        """Test full mode sets frozen_backbone=False."""
        assert _get_frozen_backbone_flag("full") is False
