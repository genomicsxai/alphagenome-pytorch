"""Tests for delta checkpoint functionality."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    transfer_config_from_dict,
    transfer_config_to_dict,
)
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    compute_base_model_hash,
    export_delta_weights,
    get_adapter_state_dict,
    get_new_head_state_dict,
    get_norm_state_dict,
    get_trunk_state_dict,
    load_delta_checkpoint,
    load_delta_config,
    load_delta_weights,
    save_delta_checkpoint,
    DELTA_CHECKPOINT_VERSION,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    prepare_for_transfer,
)


class TestTransferConfigSerialization:
    """Test TransferConfig to/from dict conversion."""

    def test_roundtrip_default_config(self):
        """Default config should roundtrip exactly."""
        config = TransferConfig()
        data = transfer_config_to_dict(config)
        restored = transfer_config_from_dict(data)

        assert restored.mode == config.mode
        assert restored.lora_rank == config.lora_rank
        assert restored.lora_alpha == config.lora_alpha
        assert restored.lora_targets == config.lora_targets

    def test_roundtrip_custom_config(self):
        """Custom config should roundtrip exactly."""
        config = TransferConfig(
            mode="lora",
            lora_rank=16,
            lora_alpha=32,
            lora_targets=["q_proj", "k_proj", "v_proj"],
            new_heads={"my_head": {"modality": "atac", "num_tracks": 5}},
        )
        data = transfer_config_to_dict(config)
        restored = transfer_config_from_dict(data)

        assert restored.mode == "lora"
        assert restored.lora_rank == 16
        assert restored.lora_alpha == 32
        assert restored.lora_targets == ["q_proj", "k_proj", "v_proj"]
        assert restored.new_heads == {"my_head": {"modality": "atac", "num_tracks": 5}}

    def test_tensor_track_means_is_json_safe(self):
        """track_means tensor in new_heads must not break json.dumps."""
        import json

        config = TransferConfig(
            mode="lora",
            new_heads={
                "my_head": {
                    "modality": "atac",
                    "num_tracks": 4,
                    "track_means": torch.ones(1, 4),
                }
            },
        )
        data = transfer_config_to_dict(config)

        # Must not raise TypeError
        json.dumps(data)

        # track_means should be stripped, other fields preserved
        assert "track_means" not in data["new_heads"]["my_head"]
        assert data["new_heads"]["my_head"]["modality"] == "atac"

    def test_forward_compatibility_ignores_unknown_fields(self):
        """Unknown fields from future versions should be ignored."""
        data = {
            "mode": "lora",
            "lora_rank": 8,
            "future_field": "some_value",
            "another_future_field": 123,
        }
        config = transfer_config_from_dict(data)

        assert config.mode == "lora"
        assert config.lora_rank == 8
        # Should not have future fields
        assert not hasattr(config, "future_field")


class TestAdapterParamIdentification:
    """Test identifying adapter parameters in models."""

    def test_identifies_lora_params(self):
        """LoRA adapter params should be identified."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        original = nn.Linear(64, 64)
        lora = LoRA(original, rank=4)

        # Create a simple model
        model = nn.Module()
        model.layer = lora
        model = model  # Register module

        adapter_state = get_adapter_state_dict(model)

        assert "layer.lora_A.weight" in adapter_state
        assert "layer.lora_B.weight" in adapter_state
        assert len(adapter_state) == 2

    def test_identifies_locon_params(self):
        """Locon adapter params should be identified."""
        from alphagenome_pytorch.extensions.finetuning.adapters import Locon

        original = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        locon = Locon(original, rank=4)

        model = nn.Module()
        model.conv = locon

        adapter_state = get_adapter_state_dict(model)

        assert "conv.locon_down.weight" in adapter_state
        assert "conv.locon_up.weight" in adapter_state
        assert len(adapter_state) == 2

    def test_identifies_ia3_params(self):
        """IA3 adapter params should be identified."""
        from alphagenome_pytorch.extensions.finetuning.adapters import IA3

        original = nn.Linear(64, 64)
        ia3 = IA3(original)

        model = nn.Module()
        model.layer = ia3

        adapter_state = get_adapter_state_dict(model)

        assert "layer.scale" in adapter_state
        assert len(adapter_state) == 1

    def test_identifies_houlsby_params(self):
        """Houlsby adapter params should be identified."""
        from alphagenome_pytorch.extensions.finetuning.adapters import HoulsbyWrapper

        original = nn.Linear(64, 64)
        houlsby = HoulsbyWrapper(original, latent_dim=8)

        model = nn.Module()
        model.layer = houlsby

        adapter_state = get_adapter_state_dict(model)

        assert "layer.adapter.down_project.weight" in adapter_state
        assert "layer.adapter.down_project.bias" in adapter_state
        assert "layer.adapter.up_project.weight" in adapter_state
        assert "layer.adapter.up_project.bias" in adapter_state
        assert len(adapter_state) == 4

    def test_no_adapters_returns_empty(self):
        """Model without adapters should return empty dict."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

        adapter_state = get_adapter_state_dict(model)
        assert adapter_state == {}


class TestBaseModelHash:
    """Test base model hash computation."""

    def test_hash_deterministic(self):
        """Same model should produce same hash."""
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

        hash1 = compute_base_model_hash(model)
        hash2 = compute_base_model_hash(model)

        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_hash_changes_with_different_model(self):
        """Different models should produce different hashes."""
        model1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        model2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())

        hash1 = compute_base_model_hash(model1)
        hash2 = compute_base_model_hash(model2)

        assert hash1 != hash2


class TestNewHeadStateDict:
    """Test extracting new head state dict."""

    def test_extracts_head_params(self):
        """Should extract parameters for specified heads."""
        model = nn.Module()
        model.heads = nn.ModuleDict({
            "head_a": nn.Linear(64, 10),
            "head_b": nn.Linear(64, 20),
        })

        head_state = get_new_head_state_dict(model, ["head_a"])

        assert "heads.head_a.weight" in head_state
        assert "heads.head_a.bias" in head_state
        assert "heads.head_b.weight" not in head_state
        assert len(head_state) == 2

    def test_raises_for_missing_head(self):
        """Should raise if head not found."""
        model = nn.Module()
        model.heads = nn.ModuleDict({
            "head_a": nn.Linear(64, 10),
        })

        with pytest.raises(ValueError, match="not found in model.heads"):
            get_new_head_state_dict(model, ["missing_head"])


class TestDeltaCheckpointSaveLoad:
    """Test save/load delta checkpoint roundtrip."""

    def test_save_requires_adapters_when_expected(self):
        """Should raise if adapters expected but not found."""
        model = nn.Sequential(nn.Linear(64, 64))
        model.heads = nn.ModuleDict()
        config = TransferConfig(mode="lora", new_heads={})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            with pytest.raises(ValueError, match="No adapter parameters found"):
                save_delta_checkpoint(path, model, config)

    def test_save_succeeds_for_linear_mode(self):
        """Linear mode (no adapters) should save successfully."""
        model = nn.Sequential(nn.Linear(64, 64))
        model.heads = nn.ModuleDict()
        config = TransferConfig(mode="linear", new_heads={})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config, epoch=5, val_loss=0.1)

            assert path.exists()
            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["delta_checkpoint_version"] == DELTA_CHECKPOINT_VERSION
            assert checkpoint["metadata"]["epoch"] == 5
            assert checkpoint["metadata"]["val_loss"] == 0.1

    def test_roundtrip_with_lora(self):
        """LoRA adapters should roundtrip correctly."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Create model with LoRA - use name that matches lora_targets
        original = nn.Linear(64, 64)
        lora = LoRA(original, rank=4)
        model = nn.Module()
        model.q_proj = lora  # Name matches default lora_targets
        model.heads = nn.ModuleDict()

        config = TransferConfig(mode="lora", lora_rank=4, lora_targets=["q_proj"], new_heads={})

        # Modify LoRA weights to non-default values
        with torch.no_grad():
            model.q_proj.lora_A.weight.fill_(1.0)
            model.q_proj.lora_B.weight.fill_(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"

            # Save
            save_delta_checkpoint(path, model, config)

            # Create fresh model for loading
            original2 = nn.Linear(64, 64)
            model2 = nn.Module()
            model2.q_proj = original2  # Same structure
            model2.heads = nn.ModuleDict()

            # Load (this will apply LoRA via prepare_for_transfer)
            loaded_config, metadata = load_delta_checkpoint(
                path, model2, verify_hash=False
            )

            # Check config matches
            assert loaded_config.mode == "lora"

            # Check weights were loaded
            assert torch.allclose(
                model2.q_proj.lora_A.weight,
                torch.ones_like(model2.q_proj.lora_A.weight),
            )
            assert torch.allclose(
                model2.q_proj.lora_B.weight,
                torch.full_like(model2.q_proj.lora_B.weight, 2.0),
            )

    def test_checkpoint_much_smaller_than_full(self):
        """Delta checkpoint should be much smaller than full model save."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Create larger model
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        )
        # Wrap first layer with LoRA
        model[0] = LoRA(model[0], rank=8)
        model.heads = nn.ModuleDict()

        config = TransferConfig(mode="lora", new_heads={})

        with tempfile.TemporaryDirectory() as tmpdir:
            delta_path = Path(tmpdir) / "test.delta.pth"
            full_path = Path(tmpdir) / "test.full.pth"

            # Save delta
            save_delta_checkpoint(delta_path, model, config)

            # Save full model
            torch.save(model.state_dict(), full_path)

            delta_size = delta_path.stat().st_size
            full_size = full_path.stat().st_size

            # Delta should be much smaller (at least 10x)
            assert delta_size < full_size / 10, (
                f"Delta size {delta_size} should be much smaller than "
                f"full size {full_size}"
            )


class TestTrunkStateDict:
    """Test get_trunk_state_dict key normalization."""

    def test_bare_model_returns_all_non_head_params(self):
        """Bare model trunk dict should contain all params except heads."""
        model = nn.Module()
        model.encoder = nn.Linear(64, 64)
        model.heads = nn.ModuleDict({"h": nn.Linear(64, 10)})

        trunk = get_trunk_state_dict(model)
        assert "encoder.weight" in trunk
        assert "encoder.bias" in trunk
        assert not any(k.startswith("heads.") for k in trunk)

    def test_adapted_model_normalizes_keys(self):
        """Trunk dict from adapted model should have same keys as bare model."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        model_bare = nn.Module()
        model_bare.q_proj = nn.Linear(64, 64)
        model_bare.heads = nn.ModuleDict()

        model_adapted = nn.Module()
        model_adapted.q_proj = LoRA(nn.Linear(64, 64), rank=4)
        model_adapted.heads = nn.ModuleDict({"h": nn.Linear(64, 10)})

        trunk_bare = get_trunk_state_dict(model_bare)
        trunk_adapted = get_trunk_state_dict(model_adapted)

        assert set(trunk_bare.keys()) == set(trunk_adapted.keys())


class TestHashVerification:
    """Test that trunk hash is stable across adapted/bare models (bug fix #1)."""

    def test_trunk_hash_ignores_adapters_and_heads(self):
        """Hash should be the same on a bare model and an adapted model."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Bare model
        model_bare = nn.Module()
        model_bare.trunk_linear = nn.Linear(64, 64)
        model_bare.heads = nn.ModuleDict()
        hash_bare = compute_base_model_hash(model_bare)

        # Same trunk, but with LoRA adapter and a head added
        model_adapted = nn.Module()
        model_adapted.trunk_linear = LoRA(nn.Linear(64, 64), rank=4)
        model_adapted.heads = nn.ModuleDict({"my_head": nn.Linear(64, 10)})
        hash_adapted = compute_base_model_hash(model_adapted)

        assert hash_bare == hash_adapted

    def test_save_adapted_load_bare_hash_matches(self):
        """Delta saved from adapted model should load onto bare model with verify_hash=True."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Save from an adapted model (simulates CLI save path)
        model = nn.Module()
        model.q_proj = LoRA(nn.Linear(64, 64), rank=4)
        model.heads = nn.ModuleDict()

        config = TransferConfig(
            mode="lora", lora_rank=4, lora_targets=["q_proj"], new_heads={}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config)

            # Fresh base model without adapters — verify_hash=True (default)
            model2 = nn.Module()
            model2.q_proj = nn.Linear(64, 64)
            model2.heads = nn.ModuleDict()

            loaded_config, _ = load_delta_checkpoint(path, model2)
            assert loaded_config.mode == "lora"

    def test_hash_stable_despite_requires_grad_differences(self):
        """Hash must not depend on requires_grad state (norm freeze/unfreeze)."""
        # Fresh model: all params trainable (requires_grad=True)
        model_fresh = nn.Module()
        model_fresh.linear = nn.Linear(64, 64)
        model_fresh.norm = nn.LayerNorm(64)
        model_fresh.heads = nn.ModuleDict()

        # Frozen model: trunk frozen, simulating post-prepare_for_transfer
        model_frozen = nn.Module()
        model_frozen.linear = nn.Linear(64, 64)
        model_frozen.norm = nn.LayerNorm(64)
        model_frozen.heads = nn.ModuleDict()
        for p in model_frozen.parameters():
            p.requires_grad = False

        assert compute_base_model_hash(model_fresh) == compute_base_model_hash(model_frozen)

    def test_save_frozen_load_fresh_roundtrip(self):
        """Save from frozen adapted model, load onto fresh model with verify_hash=True."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Build adapted model with frozen trunk + norms
        model = nn.Module()
        model.norm = nn.LayerNorm(64)
        model.q_proj = LoRA(nn.Linear(64, 64), rank=4)
        model.heads = nn.ModuleDict()
        for p in model.norm.parameters():
            p.requires_grad = False

        config = TransferConfig(
            mode="lora", lora_rank=4, lora_targets=["q_proj"], new_heads={}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config)

            # Fresh base model — all params have requires_grad=True by default
            model2 = nn.Module()
            model2.norm = nn.LayerNorm(64)
            model2.q_proj = nn.Linear(64, 64)
            model2.heads = nn.ModuleDict()

            # This must NOT raise
            loaded_config, _ = load_delta_checkpoint(path, model2)
            assert loaded_config.mode == "lora"

    def test_hash_mismatch_raises(self):
        """Mismatched trunk should raise ValueError."""
        model = nn.Module()
        model.layer = nn.Linear(64, 64)
        model.heads = nn.ModuleDict()
        config = TransferConfig(mode="linear", new_heads={})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config)

            # Different trunk structure
            model2 = nn.Module()
            model2.layer = nn.Linear(64, 128)  # Different shape
            model2.heads = nn.ModuleDict()

            with pytest.raises(ValueError, match="Base model structure mismatch"):
                load_delta_checkpoint(path, model2)


class TestEncoderOnlyDeltaCheckpoint:
    """Test encoder-only head config survives delta checkpoint roundtrip (bug fix #2)."""

    def test_encoder_only_flag_roundtrips_in_config(self):
        """encoder_only flag in new_heads should survive serialization."""
        new_heads_config = {
            "atac": {
                "modality": "atac",
                "num_tracks": 4,
                "resolutions": [128],
                "encoder_only": True,
            }
        }
        config = TransferConfig(mode="encoder-only", new_heads=new_heads_config)
        data = transfer_config_to_dict(config)
        restored = transfer_config_from_dict(data)

        assert restored.mode == "encoder-only"
        assert restored.new_heads["atac"]["encoder_only"] is True
        assert restored.new_heads["atac"]["resolutions"] == [128]

    def test_prepare_for_transfer_creates_encoder_only_head(self):
        """prepare_for_transfer should create head with in_channels=1536 for encoder_only."""
        new_heads_config = {
            "atac": {
                "modality": "atac",
                "num_tracks": 4,
                "resolutions": [128],
                "encoder_only": True,
            }
        }
        config = TransferConfig(mode="encoder-only", new_heads=new_heads_config)

        model = nn.Module()
        model.heads = nn.ModuleDict()
        # Provide named_parameters for _freeze_trunk
        model.named_parameters = lambda: iter([])

        model = prepare_for_transfer(model, config)

        head = model.heads["atac"]
        # encoder_only heads use in_channels=1536 (raw encoder embedding dim)
        # Check that the first conv/linear layer has 1536 input channels
        first_param = next(head.parameters())
        assert first_param.shape[1] == 1536 or first_param.shape[-1] == 1536


class TestExportAndLoadDeltaWeights:
    """Test export_delta_weights / load_delta_config / load_delta_weights roundtrip."""

    def _make_lora_model_and_config(self):
        """Helper: create a simple model with LoRA adapters and a head."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        model = nn.Module()
        model.q_proj = LoRA(nn.Linear(64, 64), rank=4)
        model.heads = nn.ModuleDict({
            "my_head": nn.Linear(64, 10),
        })
        config = TransferConfig(
            mode="lora",
            lora_rank=4,
            lora_targets=["q_proj"],
            new_heads={"my_head": {"modality": "atac", "num_tracks": 10}},
        )
        # Set recognizable weights
        with torch.no_grad():
            model.q_proj.lora_A.weight.fill_(1.5)
            model.q_proj.lora_B.weight.fill_(2.5)
        return model, config

    def test_load_delta_config_pth(self):
        """load_delta_config reads TransferConfig from .pth delta weights."""
        model, config = self._make_lora_model_and_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "delta.pth"
            export_delta_weights(model, config, path, format="pth")

            loaded_config = load_delta_config(path)
            assert loaded_config.mode == "lora"
            assert loaded_config.lora_rank == 4
            assert "my_head" in loaded_config.new_heads

    def test_load_delta_config_safetensors(self):
        """load_delta_config reads TransferConfig from .safetensors delta weights."""
        model, config = self._make_lora_model_and_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "delta.safetensors"
            export_delta_weights(model, config, path, format="safetensors")

            loaded_config = load_delta_config(path)
            assert loaded_config.mode == "lora"
            assert loaded_config.lora_rank == 4
            assert loaded_config.lora_targets == ["q_proj"]

    def test_load_delta_weights_roundtrip_pth(self):
        """Exported delta weights can be loaded back into a prepared model (.pth)."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        model, config = self._make_lora_model_and_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "delta.pth"
            export_delta_weights(model, config, path, format="pth")

            # Create fresh model and prepare it
            model2 = nn.Module()
            model2.q_proj = LoRA(nn.Linear(64, 64), rank=4)
            model2.heads = nn.ModuleDict({
                "my_head": nn.Linear(64, 10),
            })

            loaded_config = load_delta_weights(model2, path)
            assert loaded_config.mode == "lora"

            # Check weights match
            assert torch.allclose(
                model2.q_proj.lora_A.weight,
                torch.full_like(model2.q_proj.lora_A.weight, 1.5),
            )
            assert torch.allclose(
                model2.q_proj.lora_B.weight,
                torch.full_like(model2.q_proj.lora_B.weight, 2.5),
            )

    def test_load_delta_config_raises_on_invalid_file(self):
        """load_delta_config should raise ValueError for files without config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pth"
            torch.save({"just_weights": {}}, path)

            with pytest.raises(ValueError, match="missing transfer_config"):
                load_delta_config(path)


class TestNormLayerDeltaCheckpoint:
    """Test that unfrozen norm layers are saved/loaded in delta checkpoints (bug fix #3)."""

    def test_get_norm_state_dict_finds_trainable_norms(self):
        """get_norm_state_dict should return trainable norm params only."""
        model = nn.Module()
        model.norm = nn.LayerNorm(64)
        model.linear = nn.Linear(64, 64)

        # Freeze everything, then unfreeze norm
        for p in model.parameters():
            p.requires_grad = False
        for p in model.norm.parameters():
            p.requires_grad = True

        norm_state = get_norm_state_dict(model)
        assert "norm.weight" in norm_state
        assert "norm.bias" in norm_state
        assert len(norm_state) == 2

    def test_get_norm_state_dict_skips_frozen_norms(self):
        """Frozen norm layers should not appear in norm state dict."""
        model = nn.Module()
        model.norm = nn.LayerNorm(64)

        for p in model.parameters():
            p.requires_grad = False

        norm_state = get_norm_state_dict(model)
        assert norm_state == {}

    def test_norm_params_saved_in_delta_checkpoint(self):
        """Delta checkpoint should contain norm_state_dict."""
        model = nn.Module()
        model.norm = nn.LayerNorm(64)
        model.linear = nn.Linear(64, 64)
        model.heads = nn.ModuleDict()

        # Freeze everything, then unfreeze norm
        for p in model.parameters():
            p.requires_grad = False
        for p in model.norm.parameters():
            p.requires_grad = True

        # Modify norm weights to non-default values
        with torch.no_grad():
            model.norm.weight.fill_(2.0)
            model.norm.bias.fill_(0.5)

        config = TransferConfig(mode="linear", unfreeze_norm=True, new_heads={})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config)

            checkpoint = torch.load(path, weights_only=False)
            assert "norm_state_dict" in checkpoint
            assert "norm.weight" in checkpoint["norm_state_dict"]
            assert "norm.bias" in checkpoint["norm_state_dict"]
            assert torch.allclose(checkpoint["norm_state_dict"]["norm.weight"],
                                  torch.full((64,), 2.0))

    def test_norm_params_restored_from_delta_checkpoint(self):
        """Norm param values should be restored when loading delta checkpoint."""
        from alphagenome_pytorch.extensions.finetuning.adapters import LoRA

        # Save model with modified norms
        model = nn.Module()
        model.norm = nn.LayerNorm(64)
        model.q_proj = LoRA(nn.Linear(64, 64), rank=4)
        model.heads = nn.ModuleDict()

        for p in model.norm.parameters():
            p.requires_grad = True

        with torch.no_grad():
            model.norm.weight.fill_(3.0)
            model.norm.bias.fill_(0.7)

        config = TransferConfig(
            mode="lora", lora_rank=4, lora_targets=["q_proj"],
            unfreeze_norm=True, new_heads={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.delta.pth"
            save_delta_checkpoint(path, model, config)

            # Load onto fresh model (norm at default init)
            model2 = nn.Module()
            model2.norm = nn.LayerNorm(64)
            model2.q_proj = nn.Linear(64, 64)
            model2.heads = nn.ModuleDict()

            load_delta_checkpoint(path, model2, verify_hash=False)

            # Norm values should match saved values
            assert torch.allclose(model2.norm.weight,
                                  torch.full((64,), 3.0))
            assert torch.allclose(model2.norm.bias,
                                  torch.full((64,), 0.7))
