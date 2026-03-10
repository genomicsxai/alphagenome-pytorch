"""
Unit tests for AlphaGenome.predict() and _upcast_outputs().
"""

import gc

import pytest
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy


@pytest.mark.integration
class TestUpcastOutputs:
    """Tests for _upcast_outputs static method."""

    def test_upcasts_bfloat16(self):
        """bfloat16 tensors should be upcast to float32."""
        x = torch.randn(2, 3, dtype=torch.bfloat16)
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.float32

    def test_upcasts_float16(self):
        """float16 tensors should be upcast to float32."""
        x = torch.randn(2, 3, dtype=torch.float16)
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.float32

    def test_preserves_float32(self):
        """float32 tensors should be unchanged."""
        x = torch.randn(2, 3, dtype=torch.float32)
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.float32

    def test_preserves_int(self):
        """Integer tensors should be unchanged."""
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.int32

    def test_preserves_int64(self):
        """int64 tensors should be unchanged."""
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.int64

    def test_preserves_bool(self):
        """Bool tensors should be unchanged."""
        x = torch.tensor([True, False])
        result = AlphaGenome._upcast_outputs(x)
        assert result.dtype == torch.bool

    def test_recursive_dict(self):
        """Should recursively upcast nested dicts."""
        outputs = {
            'atac': {
                1: torch.randn(2, 3, dtype=torch.bfloat16),
                128: torch.randn(2, 3, dtype=torch.bfloat16),
            },
            'indices': torch.tensor([0, 1], dtype=torch.int32),
        }
        result = AlphaGenome._upcast_outputs(outputs)
        assert result['atac'][1].dtype == torch.float32
        assert result['atac'][128].dtype == torch.float32
        assert result['indices'].dtype == torch.int32

    def test_recursive_list(self):
        """Should recursively upcast lists."""
        outputs = [torch.randn(2, dtype=torch.bfloat16), torch.tensor([1])]
        result = AlphaGenome._upcast_outputs(outputs)
        assert isinstance(result, list)
        assert result[0].dtype == torch.float32
        assert result[1].dtype == torch.int64

    def test_non_tensor_passthrough(self):
        """Non-tensor values should pass through unchanged."""
        assert AlphaGenome._upcast_outputs(42) == 42
        assert AlphaGenome._upcast_outputs("hello") == "hello"
        assert AlphaGenome._upcast_outputs(None) is None


@pytest.mark.integration
class TestPredict:
    """Tests for AlphaGenome.predict() method."""

    @pytest.fixture(scope="class")
    def model_fp32(self):
        """Shared full_float32 model instance for the test class."""
        model = AlphaGenome(dtype_policy=DtypePolicy.full_float32())
        yield model
        del model
        gc.collect()

    @pytest.fixture(scope="class")
    def model_mixed(self):
        """Shared mixed_precision model instance for the test class."""
        model = AlphaGenome(dtype_policy=DtypePolicy.mixed_precision())
        yield model
        del model
        gc.collect()

    @pytest.fixture(params=["full_float32", "mixed_precision"])
    def model_any_precision(self, request, model_fp32, model_mixed):
        """Parametrized fixture providing both precision models."""
        if request.param == "full_float32":
            return model_fp32
        return model_mixed

    def test_all_outputs_float32(self, model_any_precision):
        """predict() should always return float32 outputs regardless of policy."""
        x = torch.randn(1, 2048, 4)
        org = torch.tensor([0])

        outputs = model_any_precision.predict(x, org)

        def assert_float32(d, path=""):
            for k, v in d.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert v.dtype == torch.float32, (
                        f"{path}{k} has dtype {v.dtype}, expected float32"
                    )
                elif isinstance(v, dict):
                    assert_float32(v, f"{path}{k}/")

        assert_float32(outputs)

    def test_no_grad(self, model_fp32):
        """predict() should not track gradients."""
        x = torch.randn(1, 2048, 4)
        org = torch.tensor([0])

        outputs = model_fp32.predict(x, org)

        def assert_no_grad(d):
            for v in d.values():
                if torch.is_tensor(v):
                    assert not v.requires_grad
                elif isinstance(v, dict):
                    assert_no_grad(v)

        assert_no_grad(outputs)

    def test_accepts_float32_input(self, model_mixed):
        """predict() should handle float32 input (common case)."""
        x = torch.randn(1, 2048, 4, dtype=torch.float32)
        org = torch.tensor([0])

        # Should not raise
        outputs = model_mixed.predict(x, org)
        assert isinstance(outputs, dict)

    def test_accepts_int_organism_index(self, model_fp32):
        """predict() should handle integer organism_index by broadcasting."""
        x = torch.randn(2, 2048, 4)
        org = 0  # int instead of tensor

        # Should not raise type error during embedding lookup
        outputs = model_fp32.predict(x, org)
        assert isinstance(outputs, dict)

    def test_return_embeddings(self, model_mixed):
        """predict() should forward kwargs like return_embeddings."""
        x = torch.randn(1, 2048, 4)
        org = torch.tensor([0])

        outputs = model_mixed.predict(x, org, return_embeddings=True)

        assert 'embeddings_128bp' in outputs
        assert 'embeddings_1bp' in outputs
        assert outputs['embeddings_128bp'].dtype == torch.float32
        assert outputs['embeddings_1bp'].dtype == torch.float32

    def test_predict_matches_forward_values(self, model_fp32):
        """predict() should produce same values as manual forward + upcast."""
        x = torch.randn(1, 2048, 4)
        org = torch.tensor([0])

        # Manual inference
        with torch.no_grad():
            manual_outputs = model_fp32(x, org)

        # Using predict()
        pred_outputs = model_fp32.predict(x, org)

        for key in manual_outputs:
            if isinstance(manual_outputs[key], dict):
                for res in manual_outputs[key]:
                    torch.testing.assert_close(
                        pred_outputs[key][res], manual_outputs[key][res]
                    )
            elif torch.is_tensor(manual_outputs[key]):
                torch.testing.assert_close(
                    pred_outputs[key], manual_outputs[key]
                )

