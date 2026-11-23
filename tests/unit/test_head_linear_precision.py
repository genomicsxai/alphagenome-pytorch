"""Head projections follow JAX _MultiOrganismLinear: bf16 operands, fp32 accumulate and output."""

import pytest
import torch

from alphagenome_pytorch.heads import MultiOrganismConv1d, MultiOrganismLinear


def _reference(x, w, b):
    xd = x.to(torch.float64)
    wd = w.to(x.dtype).to(torch.float64)
    bd = b.to(x.dtype).to(torch.float64)
    return xd, wd, bd


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_linear_bf16_input_gives_fp32_exact_product(device):
    torch.manual_seed(0)
    layer = MultiOrganismLinear(16, 5, 2).to(device)
    x = torch.randn(2, 7, 16).to(device=device, dtype=torch.bfloat16)
    org = torch.tensor([0, 1], device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = layer(x, org)
    assert out.dtype == torch.float32
    xd, wd, bd = _reference(x, layer.weight[org], layer.bias[org])
    ref = torch.bmm(xd, wd) + bd.unsqueeze(1)
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_conv1d_bf16_input_gives_fp32_exact_product(device):
    torch.manual_seed(0)
    layer = MultiOrganismConv1d(16, 5, 2).to(device)
    x = torch.randn(2, 16, 9).to(device=device, dtype=torch.bfloat16)
    org = torch.tensor([1, 0], device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = layer(x, org)
    assert out.dtype == torch.float32
    xd, wd, bd = _reference(x, layer.weight[org], layer.bias[org])
    ref = torch.einsum("bcs,boc->bos", xd, wd) + bd.unsqueeze(2)
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)


def test_fp32_input_stays_fp32():
    layer = MultiOrganismLinear(8, 3, 1)
    out = layer(torch.randn(1, 4, 8), torch.tensor([0]))
    assert out.dtype == torch.float32
