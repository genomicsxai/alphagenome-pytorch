"""RoPE inverse-frequency table: correctly rounded float32 geomspace (JAX semantics)."""

import math

import numpy as np
import pytest
import torch

from alphagenome_pytorch.attention import _rope_base_freqs

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_freq,max_position", [(384, 2**20), (64, 8192), (768, 8192)])
def test_base_freqs_are_correctly_rounded(num_freq, max_position, device):
    exact = (10.0 ** np.linspace(0.0, math.log10(max_position - num_freq + 1), num_freq)).astype(np.float32)
    table = _rope_base_freqs(num_freq, max_position, torch.device(device))
    assert table.device.type == device
    got = table.cpu().numpy()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, exact)


@pytest.mark.parametrize("num_freq,max_position", [(384, 2**20), (64, 8192)])
def test_base_freqs_close_to_jax_geomspace(num_freq, max_position):
    jnp = pytest.importorskip("jax.numpy")
    ref = np.asarray(jnp.geomspace(1, max_position - num_freq + 1, num_freq))
    got = _rope_base_freqs(num_freq, max_position, torch.device("cpu")).numpy()
    # JAX's float32 geomspace is itself backend-dependent in the last bit.
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=0)
