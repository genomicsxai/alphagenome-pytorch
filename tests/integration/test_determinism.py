"""
Tests for model determinism.

Verifies that identical inputs + seeds produce identical outputs,
and that eval mode produces bitwise-reproducible results.
"""

import gc

import pytest
import torch

from alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy


def _make_small_model():
    """Create a small AlphaGenome model for fast testing."""
    model = AlphaGenome(
        num_organisms=2,
        dtype_policy=DtypePolicy.full_float32(),
    )
    model.eval()
    return model


def _make_input(batch_size=1, seq_len=16384, seed=42):
    """Create deterministic one-hot DNA input."""
    torch.manual_seed(seed)
    indices = torch.randint(0, 4, (batch_size, seq_len))
    onehot = torch.zeros(batch_size, seq_len, 4)
    onehot.scatter_(2, indices.unsqueeze(-1), 1.0)
    organism_index = torch.zeros(batch_size, dtype=torch.long)
    return onehot, organism_index


@pytest.mark.integration
class TestModelDeterminism:
    """Tests for reproducible model behavior."""

    @pytest.fixture(scope="class")
    def model(self):
        """Shared model instance for determinism tests."""
        torch.manual_seed(0)
        model = _make_small_model()
        yield model
        del model
        gc.collect()

    def test_eval_mode_deterministic(self, model):
        """model.eval() should produce identical outputs across repeated calls."""
        x, org = _make_input(seed=42)

        with torch.no_grad():
            out1 = model(x, org)
            out2 = model(x, org)

        # Every output tensor should be bitwise identical
        for head_name in out1:
            if isinstance(out1[head_name], dict):
                for res in out1[head_name]:
                    torch.testing.assert_close(
                        out1[head_name][res],
                        out2[head_name][res],
                        atol=0,
                        rtol=0,
                        msg=f"{head_name}@{res} not deterministic",
                    )
            else:
                torch.testing.assert_close(
                    out1[head_name],
                    out2[head_name],
                    atol=0,
                    rtol=0,
                    msg=f"{head_name} not deterministic",
                )

    def test_same_seed_same_output(self):
        """Two models initialized with the same seed should produce identical outputs."""
        # Model 1
        torch.manual_seed(123)
        model1 = _make_small_model()

        # Model 2 (same seed)
        torch.manual_seed(123)
        model2 = _make_small_model()

        x, org = _make_input(seed=99)

        with torch.no_grad():
            out1 = model1(x, org)
            out2 = model2(x, org)

        for head_name in out1:
            if isinstance(out1[head_name], dict):
                for res in out1[head_name]:
                    torch.testing.assert_close(
                        out1[head_name][res],
                        out2[head_name][res],
                        atol=0,
                        rtol=0,
                    )
            else:
                torch.testing.assert_close(
                    out1[head_name], out2[head_name], atol=0, rtol=0
                )

    def test_different_seed_different_output(self):
        """Different weight seeds should produce different outputs (sanity check)."""
        torch.manual_seed(1)
        model1 = _make_small_model()

        torch.manual_seed(2)
        model2 = _make_small_model()

        x, org = _make_input(seed=42)

        with torch.no_grad():
            out1 = model1(x, org)
            out2 = model2(x, org)

        # At least one head output should differ
        any_different = False
        for head_name in out1:
            if isinstance(out1[head_name], dict):
                for res in out1[head_name]:
                    if not torch.allclose(out1[head_name][res], out2[head_name][res]):
                        any_different = True
                        break
            else:
                if not torch.allclose(out1[head_name], out2[head_name]):
                    any_different = True
            if any_different:
                break

        assert any_different, "Different seeds should produce different outputs"

    def test_different_input_different_output(self, model):
        """Different inputs should produce different outputs (sanity check)."""
        x1, org1 = _make_input(seed=1)
        x2, org2 = _make_input(seed=2)

        with torch.no_grad():
            out1 = model(x1, org1)
            out2 = model(x2, org2)

        any_different = False
        for head_name in out1:
            if isinstance(out1[head_name], dict):
                for res in out1[head_name]:
                    if not torch.allclose(out1[head_name][res], out2[head_name][res]):
                        any_different = True
                        break
            else:
                if not torch.allclose(out1[head_name], out2[head_name]):
                    any_different = True
            if any_different:
                break

        assert any_different, "Different inputs should produce different outputs"
