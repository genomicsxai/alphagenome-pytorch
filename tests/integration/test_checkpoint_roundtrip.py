"""
Tests for checkpoint save/load roundtrips.

Verifies that model serialization preserves parameters exactly,
and that partial loading (missing heads) works correctly.
"""

import gc
import os
import tempfile
from pathlib import Path

import pytest
import torch

from alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy


def _make_model(**kwargs):
    """Create a model for testing."""
    model = AlphaGenome(
        num_organisms=2,
        dtype_policy=DtypePolicy.full_float32(),
        **kwargs,
    )
    model.eval()
    return model


@pytest.mark.integration
class TestStateDictRoundtrip:
    """Tests for state_dict save/load cycle."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_save_load_preserves_all_parameters(self):
        """state_dict → save → load → all parameters should match exactly."""
        torch.manual_seed(42)
        model = _make_model()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name

        try:
            torch.manual_seed(999)  # Different init
            model2 = _make_model()
            model2.load_state_dict(torch.load(path, weights_only=True))

            # Compare every parameter
            for name, param in model.named_parameters():
                param2 = dict(model2.named_parameters())[name]
                torch.testing.assert_close(
                    param, param2, atol=0, rtol=0,
                    msg=f"Parameter {name} differs after load",
                )

            # Compare every buffer (e.g., track means)
            for name, buf in model.named_buffers():
                buf2 = dict(model2.named_buffers())[name]
                torch.testing.assert_close(
                    buf, buf2, atol=0, rtol=0,
                    msg=f"Buffer {name} differs after load",
                )
        finally:
            os.unlink(path)

    def test_output_matches_after_reload(self):
        """Model outputs should be identical before and after save/load."""
        torch.manual_seed(42)
        model = _make_model()

        # Create input
        torch.manual_seed(0)
        seq_len = 16384
        indices = torch.randint(0, 4, (1, seq_len))
        x = torch.zeros(1, seq_len, 4)
        x.scatter_(2, indices.unsqueeze(-1), 1.0)
        org = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            out_before = model(x, org)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name

        try:
            model2 = _make_model()
            model2.load_state_dict(torch.load(path, weights_only=True))
            model2.eval()

            with torch.no_grad():
                out_after = model2(x, org)

            for head_name in out_before:
                if isinstance(out_before[head_name], dict):
                    for res in out_before[head_name]:
                        torch.testing.assert_close(
                            out_before[head_name][res],
                            out_after[head_name][res],
                            atol=0,
                            rtol=0,
                        )
                else:
                    torch.testing.assert_close(
                        out_before[head_name], out_after[head_name], atol=0, rtol=0
                    )
        finally:
            os.unlink(path)


@pytest.mark.integration
class TestPartialLoading:
    """Tests for loading weights with mismatched heads."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_load_trunk_with_missing_head_keys(self):
        """Loading state_dict after removing head keys should not error."""
        torch.manual_seed(42)
        model = _make_model()
        state = model.state_dict()

        # Remove all head-related keys
        trunk_state = {
            k: v for k, v in state.items()
            if not k.startswith("heads.")
        }

        # Create a new model with potentially different heads
        model2 = _make_model()
        # Load trunk only (strict=False to allow missing head keys)
        missing, unexpected = model2.load_state_dict(trunk_state, strict=False)

        # Missing keys should only be head-related
        assert all("heads." in k for k in missing), (
            f"Non-head keys missing: {[k for k in missing if 'heads.' not in k]}"
        )
        assert len(unexpected) == 0

    def test_trunk_parameters_match_after_partial_load(self):
        """After partial load, trunk parameters should match the saved model."""
        torch.manual_seed(42)
        model = _make_model()
        state = model.state_dict()

        trunk_state = {
            k: v for k, v in state.items()
            if not k.startswith("heads.")
        }

        torch.manual_seed(999)
        model2 = _make_model()
        model2.load_state_dict(trunk_state, strict=False)

        # Trunk params must match
        for name, param in model.named_parameters():
            if not name.startswith("heads."):
                param2 = dict(model2.named_parameters())[name]
                torch.testing.assert_close(
                    param, param2, atol=0, rtol=0,
                    msg=f"Trunk param {name} differs",
                )


@pytest.mark.integration
class TestStateDictKeys:
    """Tests for state_dict key structure."""

    @pytest.fixture(scope="class")
    def model(self):
        """Shared model instance for key inspection tests."""
        model = _make_model()
        yield model
        del model
        gc.collect()

    def test_state_dict_contains_expected_prefixes(self, model):
        """State dict should contain encoder, tower, decoder, and heads keys."""
        keys = set(model.state_dict().keys())

        prefixes = ["encoder.", "tower.", "decoder.", "heads."]
        for prefix in prefixes:
            matching = [k for k in keys if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys with prefix '{prefix}'"

    def test_state_dict_key_count_is_stable(self, model):
        """Total number of state dict keys should be consistent across runs."""
        keys1 = sorted(model.state_dict().keys())
        keys2 = sorted(model.state_dict().keys())

        assert keys1 == keys2, "State dict keys should be stable"


@pytest.mark.integration
class TestFromDelta:
    """Tests for AlphaGenome.from_delta() classmethod."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Force garbage collection after each test."""
        yield
        gc.collect()

    def test_from_delta_reconstructs_model(self):
        """from_delta should produce a model with correct adapters and heads."""
        from alphagenome_pytorch.extensions.finetuning.transfer import (
            TransferConfig,
            prepare_for_transfer,
        )
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            export_delta_weights,
        )

        torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pth"
            delta_path = Path(tmpdir) / "delta.pth"

            # 1. Create base model and save its weights
            base_model = AlphaGenome(
                num_organisms=1,
                dtype_policy=DtypePolicy.full_float32(),
            )
            torch.save(base_model.state_dict(), base_path)

            # 2. Set up adapters and a new head on a copy
            config = TransferConfig(
                mode="lora",
                lora_rank=4,
                lora_alpha=8,
                lora_targets=["q_proj", "v_proj"],
                remove_heads=list(base_model.heads.keys()),
                new_heads={
                    "my_atac": {
                        "modality": "atac",
                        "num_tracks": 4,
                        "resolutions": [128],
                    },
                },
            )
            adapted = AlphaGenome(
                num_organisms=1,
                dtype_policy=DtypePolicy.full_float32(),
            )
            adapted.load_state_dict(torch.load(base_path, weights_only=True), strict=False)
            adapted = prepare_for_transfer(adapted, config)

            # Set recognizable adapter weights
            with torch.no_grad():
                for name, param in adapted.named_parameters():
                    if "lora_A" in name:
                        param.fill_(1.0)
                    elif "lora_B" in name:
                        param.fill_(2.0)

            # 3. Export delta weights
            export_delta_weights(adapted, config, delta_path, format="pth")

            # 4. Reconstruct via from_delta
            loaded = AlphaGenome.from_delta(
                delta_path,
                base_path,
                dtype_policy=DtypePolicy.full_float32(),
                num_organisms=1,
            )

            # 5. Verify structure
            assert "my_atac" in loaded.heads
            # Original heads should have been removed
            assert "atac" not in loaded.heads

            # 6. Verify LoRA adapters are present
            lora_params = [
                n for n, _ in loaded.named_parameters()
                if "lora_A" in n or "lora_B" in n
            ]
            assert len(lora_params) > 0, "LoRA adapters should be present"

            # 7. Verify adapter weights were loaded correctly
            for name, param in loaded.named_parameters():
                if "lora_A" in name:
                    assert torch.allclose(param, torch.ones_like(param))
                elif "lora_B" in name:
                    assert torch.allclose(param, torch.full_like(param, 2.0))

            # 8. Verify trunk weights match the base model
            #    LoRA wraps layers as .original_layer.weight, so normalize
            #    back to the base model key format for comparison.
            head_prefixes = (
                "heads.", "contact_maps_head.",
                "splice_sites_classification_head.",
                "splice_sites_usage_head.",
                "splice_sites_junction_head.",
            )
            base_state = base_model.state_dict()
            matched = 0
            for name, param in loaded.named_parameters():
                if name.startswith(head_prefixes) or "lora_" in name:
                    continue
                base_key = name.replace(".original_layer.", ".")
                if base_key in base_state:
                    torch.testing.assert_close(
                        param, base_state[base_key], atol=0, rtol=0,
                        msg=f"Trunk param {name} should match base model",
                    )
                    matched += 1
            assert matched > 0, "Should have matched at least some trunk params"
