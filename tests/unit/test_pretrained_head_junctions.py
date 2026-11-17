"""--pretrained-head-samples splice_junctions:<k> copies the conv whole."""

import torch
from torch import nn

from alphagenome_pytorch.extensions.finetuning.transfer import load_pretrained_head_weights

H, IN, T_PT, T = 6, 4, 5, 3
ROPE = ("pos_donor", "pos_acceptor", "neg_donor", "neg_acceptor")


class _Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, H, IN))
        self.bias = nn.Parameter(torch.zeros(1, H))


class _JunctionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _Conv()
        self.rope_params = nn.ParameterDict({k: nn.Parameter(torch.zeros(1, 2, T, H)) for k in ROPE})


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.splice_sites_junction_head = _JunctionHead()


def _pretrained(tmp_path):
    g = torch.Generator().manual_seed(0)
    p = "splice_sites_junction_head"
    sd = {f"{p}.conv.weight": torch.randn(2, H, IN, generator=g),
          f"{p}.conv.bias": torch.randn(2, H, generator=g)}
    for k in ROPE:
        sd[f"{p}.rope_params.{k}"] = torch.randn(2, 2, T_PT, H, generator=g)
    path = tmp_path / "pt.pth"
    torch.save(sd, path)
    return sd, str(path)


def test_int_index_copies_conv_whole_and_broadcasts_rope(tmp_path):
    sd, path = _pretrained(tmp_path)
    model = _Model()
    load_pretrained_head_weights(model, path, {"splice_junctions": 2}, organism_idx=0)
    head = model.splice_sites_junction_head
    p = "splice_sites_junction_head"
    torch.testing.assert_close(head.conv.weight.data, sd[f"{p}.conv.weight"][0:1])
    torch.testing.assert_close(head.conv.bias.data, sd[f"{p}.conv.bias"][0:1])
    for k in ROPE:
        expected = sd[f"{p}.rope_params.{k}"][0:1, :, 2:3, :].expand(1, 2, T, H)
        torch.testing.assert_close(head.rope_params[k].data, expected)


def test_list_index_gathers_rope_tissues(tmp_path):
    sd, path = _pretrained(tmp_path)
    model = _Model()
    load_pretrained_head_weights(model, path, {"splice_junctions": [4, None, 0]}, organism_idx=1)
    head = model.splice_sites_junction_head
    p = "splice_sites_junction_head"
    torch.testing.assert_close(head.conv.weight.data, sd[f"{p}.conv.weight"][1:2])
    for k in ROPE:
        src = sd[f"{p}.rope_params.{k}"][1:2]
        torch.testing.assert_close(head.rope_params[k].data[:, :, 0], src[:, :, 4])
        assert torch.all(head.rope_params[k].data[:, :, 1] == 0)
        torch.testing.assert_close(head.rope_params[k].data[:, :, 2], src[:, :, 0])
