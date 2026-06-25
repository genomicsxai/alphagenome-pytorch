"""Registry of attribution methods.

Adding a new method = one row here. Clients need not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .gradient import gradient_x_input
from .ism import saturation_ism
from .types import AttributionResult


@dataclass(frozen=True)
class MethodSpec:
    func: Callable[..., AttributionResult]
    kind: str  # discriminator carried into JSON, e.g. "base_matrix"
    supports_raw_gradient: bool


class UnsupportedMethodError(KeyError):
    """Raised when the requested attribution method is not in the registry."""


METHODS: dict[str, MethodSpec] = {
    "input_x_gradient": MethodSpec(
        func=gradient_x_input, kind="base_matrix", supports_raw_gradient=True,
    ),
    "saturation_ism": MethodSpec(
        func=saturation_ism, kind="base_matrix", supports_raw_gradient=False,
    ),
}


def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise UnsupportedMethodError(
            f"Unknown attribution method {name!r}. "
            f"Supported: {sorted(METHODS)}."
        )
    return METHODS[name]
