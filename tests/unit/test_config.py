"""Tests for DtypePolicy configuration."""

import pytest
import torch

from alphagenome_pytorch.config import DtypePolicy


class TestDtypePolicyDefaults:
    """Test default values and construction."""

    def test_default_values(self):
        """Default construction uses bfloat16 for compute/output, float32 for params."""
        policy = DtypePolicy()
        assert policy.params_dtype == torch.float32
        assert policy.compute_dtype == torch.bfloat16
        assert policy.output_dtype == torch.bfloat16

    def test_explicit_construction(self):
        """Explicit dtype specification works."""
        policy = DtypePolicy(
            params_dtype=torch.float16,
            compute_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        assert policy.params_dtype == torch.float16
        assert policy.compute_dtype == torch.float32
        assert policy.output_dtype == torch.bfloat16


class TestDtypePolicyFactoryMethods:
    """Test factory class methods."""

    def test_default_factory(self):
        """default() returns full_float32 policy (single source of truth)."""
        policy = DtypePolicy.default()
        expected = DtypePolicy.full_float32()
        assert policy.params_dtype == expected.params_dtype
        assert policy.compute_dtype == expected.compute_dtype
        assert policy.output_dtype == expected.output_dtype

    def test_mixed_precision_factory(self):
        """mixed_precision() returns JAX-matching mixed precision policy."""
        policy = DtypePolicy.mixed_precision()
        assert policy.params_dtype == torch.float32
        assert policy.compute_dtype == torch.bfloat16
        assert policy.output_dtype == torch.bfloat16

    def test_full_float32_factory(self):
        """full_float32() returns all-float32 policy."""
        policy = DtypePolicy.full_float32()
        assert policy.params_dtype == torch.float32
        assert policy.compute_dtype == torch.float32
        assert policy.output_dtype == torch.float32


class TestDtypePolicyFromString:
    """Test from_string() parsing."""

    def test_full_string(self):
        """Parse complete policy string."""
        policy = DtypePolicy.from_string("params=float32,compute=bfloat16,output=bfloat16")
        assert policy.params_dtype == torch.float32
        assert policy.compute_dtype == torch.bfloat16
        assert policy.output_dtype == torch.bfloat16

    def test_partial_string_uses_defaults(self):
        """Omitted keys use default values."""
        policy = DtypePolicy.from_string("compute=float32")
        assert policy.params_dtype == torch.float32  # default
        assert policy.compute_dtype == torch.float32  # specified
        assert policy.output_dtype == torch.bfloat16  # default

    def test_all_dtypes(self):
        """All supported dtypes can be parsed."""
        policy = DtypePolicy.from_string("params=float16,compute=float32,output=bfloat16")
        assert policy.params_dtype == torch.float16
        assert policy.compute_dtype == torch.float32
        assert policy.output_dtype == torch.bfloat16

    def test_whitespace_handling(self):
        """Whitespace is stripped correctly."""
        policy = DtypePolicy.from_string(" params = float32 , compute = bfloat16 ")
        assert policy.params_dtype == torch.float32
        assert policy.compute_dtype == torch.bfloat16

    def test_unknown_key_raises(self):
        """Unknown key raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy key"):
            DtypePolicy.from_string("invalid=float32")

    def test_unknown_dtype_raises(self):
        """Unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dtype"):
            DtypePolicy.from_string("params=float64")

    def test_malformed_string_raises(self):
        """Malformed string without '=' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid policy part"):
            DtypePolicy.from_string("params:float32")

    def test_duplicate_key_raises(self):
        """Duplicate keys raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate policy key"):
            DtypePolicy.from_string("params=float32,params=bfloat16")


class TestDtypePolicyCastMethods:
    """Test cast_to_* methods."""

    def test_cast_to_compute(self):
        """cast_to_compute converts to compute_dtype."""
        policy = DtypePolicy(compute_dtype=torch.bfloat16)
        x = torch.tensor([1.0], dtype=torch.float32)
        result = policy.cast_to_compute(x)
        assert result.dtype == torch.bfloat16

    def test_cast_to_compute_no_op(self):
        """cast_to_compute is no-op if already correct dtype."""
        policy = DtypePolicy(compute_dtype=torch.float32)
        x = torch.tensor([1.0], dtype=torch.float32)
        result = policy.cast_to_compute(x)
        assert result is x  # Same object, not a copy

    def test_cast_to_output(self):
        """cast_to_output converts to output_dtype."""
        policy = DtypePolicy(output_dtype=torch.float32)
        x = torch.tensor([1.0], dtype=torch.bfloat16)
        result = policy.cast_to_output(x)
        assert result.dtype == torch.float32

    def test_cast_to_params(self):
        """cast_to_params converts to params_dtype."""
        policy = DtypePolicy(params_dtype=torch.float32)
        x = torch.tensor([1.0], dtype=torch.bfloat16)
        result = policy.cast_to_params(x)
        assert result.dtype == torch.float32


class TestDtypePolicyRepr:
    """Test string representation."""

    def test_repr(self):
        """repr shows all dtypes clearly."""
        policy = DtypePolicy.mixed_precision()
        repr_str = repr(policy)
        assert "params=float32" in repr_str
        assert "compute=bfloat16" in repr_str
        assert "output=bfloat16" in repr_str
