"""RoPE inverse-frequency table: correctly rounded float32 geomspace (JAX semantics)."""

import math

import numpy as np
import pytest
import torch

from alphagenome_pytorch.attention import _rope_base_freqs


@pytest.mark.parametrize("num_freq,max_position", [(384, 2**20), (64, 8192), (768, 8192)])
def test_base_freqs_are_correctly_rounded(num_freq, max_position):
    exact = (10.0 ** np.linspace(0.0, math.log10(max_position - num_freq + 1), num_freq)).astype(np.float32)
    got = _rope_base_freqs(num_freq, max_position).numpy()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, exact)


@pytest.mark.parametrize("num_freq,max_position", [(384, 2**20), (64, 8192)])
def test_base_freqs_close_to_jax_geomspace(num_freq, max_position):
    jnp = pytest.importorskip("jax.numpy")
    ref = np.asarray(jnp.geomspace(1, max_position - num_freq + 1, num_freq))
    got = _rope_base_freqs(num_freq, max_position).numpy()
    # JAX's float32 geomspace is itself backend-dependent in the last bit.
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=0)


def test_table_moves_to_input_device():
    x = torch.zeros(1, 4, 1, 8)
    from alphagenome_pytorch.attention import apply_rope
    out = apply_rope(x.clone(), max_position=64)
    assert out.device == x.device and torch.isfinite(out).all()
