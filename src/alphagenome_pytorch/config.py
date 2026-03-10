"""Precision and dtype configuration for AlphaGenome.

This module defines the DtypePolicy class for mixed-precision training,
matching JAX's jmp.Policy behavior.

Example:
    # Full float32 (default, works everywhere):
    policy = DtypePolicy.full_float32()
    model = AlphaGenome(dtype_policy=policy, ...)

    # JAX-matching mixed precision (params=f32, compute/output=bf16):
    policy = DtypePolicy.mixed_precision()
    model = AlphaGenome(dtype_policy=policy, ...)

    # Parse from string (JAX-style):
    policy = DtypePolicy.from_string('params=float32,compute=bfloat16,output=bfloat16')
"""

import torch
from dataclasses import dataclass


@dataclass
class DtypePolicy:
    """Mixed precision policy matching JAX's jmp.Policy.

    Controls dtype for params, compute, and output separately, exactly
    matching JAX's JMP (JAX Mixed Precision) behavior.

    JAX default policy: 'params=float32,compute=bfloat16,output=bfloat16'
    - params: Model weights stored in float32 for stability
    - compute: Intermediate computations in bfloat16 for speed/memory
    - output: Model outputs in bfloat16

    Attributes:
        params_dtype: Dtype for model parameters/weights.
        compute_dtype: Dtype for intermediate computations.
        output_dtype: Dtype for model outputs.

    Examples:
        # Full float32 (default, works everywhere)
        policy = DtypePolicy.full_float32()

        # JAX-matching mixed precision
        policy = DtypePolicy.mixed_precision()

        # From JAX-style string
        policy = DtypePolicy.from_string('params=float32,compute=bfloat16,output=bfloat16')

        # Explicit construction (defaults to bfloat16 compute/output)
        policy = DtypePolicy(
            params_dtype=torch.float32,
            compute_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
    """

    params_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_string(cls, policy_str: str) -> "DtypePolicy":
        """Parse JAX-style policy string.

        Args:
            policy_str: Policy string, e.g., 'params=float32,compute=bfloat16,output=bfloat16'.
                Keys are optional; any omitted keys use defaults:
                params=float32, compute=bfloat16, output=bfloat16.

        Returns:
            DtypePolicy instance.

        Raises:
            ValueError: If policy string is malformed, contains unknown dtype,
                or has duplicate keys.
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        kwargs = {}
        parts = policy_str.split(",")
        for part in parts:
            part = part.strip()
            if "=" not in part:
                raise ValueError(f"Invalid policy part: {part!r}, expected 'key=value'")
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in ("params", "compute", "output"):
                raise ValueError(f"Unknown policy key: {key!r}")
            if value not in dtype_map:
                raise ValueError(f"Unknown dtype: {value!r}")

            kwarg_key = f"{key}_dtype"
            if kwarg_key in kwargs:
                raise ValueError(f"Duplicate policy key: {key!r}")
            kwargs[kwarg_key] = dtype_map[value]

        return cls(**kwargs)

    @classmethod
    def default(cls) -> "DtypePolicy":
        """Default policy for AlphaGenome.

        Currently returns full_float32() for maximum compatibility.
        Change this single method to update the default across the codebase.
        """
        return cls.full_float32()

    @classmethod
    def full_float32(cls) -> "DtypePolicy":
        """Full float32 policy.

        All operations use float32 for maximum numerical precision.
        Works on all hardware (CPU, older GPUs, etc.).
        """
        return cls(
            params_dtype=torch.float32,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )

    @classmethod
    def mixed_precision(cls) -> "DtypePolicy":
        """Mixed-precision policy matching JAX JMP.

        Uses: params=float32, compute=bfloat16, output=bfloat16

        - Parameters stored in float32 for numerical stability
        - Computation in bfloat16 for speed and memory efficiency
        - Outputs in bfloat16

        Note: Requires GPU with bfloat16 support (Ampere+). Use full_float32()
        for CPU or older GPUs.
        """
        return cls(
            params_dtype=torch.float32,
            compute_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )

    def cast_to_compute(self, x: torch.Tensor) -> torch.Tensor:
        """Cast tensor to compute dtype."""
        if x.dtype != self.compute_dtype:
            return x.to(self.compute_dtype)
        return x

    def cast_to_output(self, x: torch.Tensor) -> torch.Tensor:
        """Cast tensor to output dtype."""
        if x.dtype != self.output_dtype:
            return x.to(self.output_dtype)
        return x

    def cast_to_params(self, x: torch.Tensor) -> torch.Tensor:
        """Cast tensor to params dtype."""
        if x.dtype != self.params_dtype:
            return x.to(self.params_dtype)
        return x

    def __repr__(self) -> str:
        def dtype_name(dt: torch.dtype) -> str:
            return str(dt).replace("torch.", "")

        return (
            f"DtypePolicy(params={dtype_name(self.params_dtype)}, "
            f"compute={dtype_name(self.compute_dtype)}, "
            f"output={dtype_name(self.output_dtype)})"
        )


# Legacy constants for backwards compatibility
DEFAULT_COMPUTE_DTYPE = torch.float32
ACCUMULATE_DTYPE = torch.float32
