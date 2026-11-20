"""Defaults/behaviour aligned with alphagenome_research and alphagenome_ft."""

import torch

from alphagenome_pytorch.extensions.finetuning import training



def test_top_k_uses_probabilities_and_threshold():
    # 1 batch, 5 classes, 6 positions. Donor+ (channel 0) logit is highest at
    # position 4 but softmax-probability is highest at position 1.
    logits = torch.full((1, 5, 6), -10.0)
    logits[0, 4] = 0.0                     # background
    logits[0, 0, 1] = 3.0                  # p ~ 0.95
    logits[0, 0, 4] = 5.0
    logits[0, 4, 4] = 8.0                  # p(donor+) ~ 0.05 at pos 4
    logits[0, 0, 2] = 0.0                  # p = 0.5
    pos = training._top_k_positions_from_logits(logits, top_k=4, threshold=0.1)
    assert pos.shape == (1, 4, 4)
    assert pos[0, 0].tolist() == [1, 2, -1, -1]
    # No sites pass for other roles -> all padding.
    assert (pos[0, 1:] == -1).all()
    # threshold=0 keeps the plain top-k, sorted ascending.
    pos0 = training._top_k_positions_from_logits(logits, top_k=3, threshold=0.0)
    assert sorted(pos0[0, 0].tolist()) == pos0[0, 0].tolist()
    assert set(pos0[0, 0].tolist()) >= {1, 2}


def test_top_k_matches_alphagenome_research():
    import importlib.util
    import pytest
    if importlib.util.find_spec("alphagenome_research") is None:
        pytest.skip("alphagenome_research not installed")
    import jax.numpy as jnp
    import numpy as np
    from alphagenome_research.model import splicing

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 300, generator=g) * 3
    ours = training._top_k_positions_from_logits(logits, top_k=64, threshold=0.1)
    probs = torch.softmax(logits, dim=1).permute(0, 2, 1).numpy()
    ref = splicing.generate_splice_site_positions(
        jnp.asarray(probs), alt=None, splice_sites=None, k=64, pad_to_length=64, threshold=0.1,
    )
    np.testing.assert_array_equal(ours.numpy(), np.asarray(ref))


def test_cli_defaults():
    from alphagenome_pytorch.extensions.finetuning.args import build_parser
    ns = build_parser().parse_args([])
    assert ns.min_alpha_juncs == 0
    assert ns.splice_num_segments == 1
    assert ns.num_segments is None  # alphagenome_research 2**17 bp segments
    assert ns.max_splice_sites == 256
    assert ns.junction_top_k == 512
    assert ns.splice_site_threshold == 0.1


def test_multinomial_segments_follow_alphagenome_research():
    f = training._compute_multinomial_resolution
    assert f(131072, None, res=1) == 131072        # 1 segment at 131 kb
    assert f(1048576, None, res=1) == 131072       # 8 segments at 1 Mb
    assert f(2097152, None, res=1) == 131072       # 16 segments at 2 Mb
    assert f(1024, None, res=128) == 1024          # 128 bp track at 131 kb
    assert f(8192, None, res=128) == 1024
    assert f(65536, None, res=1) == 65536          # shorter than one segment
    assert f(131072, 8) == 16384                   # explicit --num-segments still honoured


def test_cosine_schedule_clamps_past_total_steps():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = training.create_lr_scheduler(opt, warmup_steps=0, total_steps=10, schedule="cosine")
    lam = sched.lr_lambdas[0]
    assert lam(10) == 0.0
    assert lam(15) == 0.0


def test_aligned_sampler_matches_alphagenome_ft_order():
    import numpy as np
    from alphagenome_pytorch.extensions.finetuning.runner import AlignedShuffleSampler

    n, seed, epoch, bs, world = 23, 1234, 3, 2, 4
    order = np.arange(n)
    np.random.default_rng(seed + epoch).shuffle(order)
    # alphagenome_ft: global batches of bs*world, sharded contiguously, drop_last.
    global_batches = [order[i:i + bs * world] for i in range(0, n - n % (bs * world), bs * world)]

    per_rank = []
    for rank in range(world):
        s = AlignedShuffleSampler(range(n), seed=seed, batch_size=bs, world_size=world, rank=rank)
        s.set_epoch(epoch)
        idx = list(s)
        assert len(idx) == len(s)
        per_rank.append([idx[i:i + bs] for i in range(0, len(idx), bs)])
    for step, gb in enumerate(global_batches):
        got = np.concatenate([per_rank[r][step] for r in range(world)])
        np.testing.assert_array_equal(got, gb)

    single = AlignedShuffleSampler(range(n), seed=seed, batch_size=bs)
    single.set_epoch(epoch)
    np.testing.assert_array_equal(list(single), order)


def test_call_splice_head_masks_padded_junction_pairs():
    from alphagenome_pytorch.heads import SpliceSitesJunctionHead

    head = SpliceSitesJunctionHead(in_channels=8, num_tissues=2, num_organisms=1, hidden_dim=4)
    emb = torch.randn(1, 8, 20)
    positions = torch.tensor([[[3, 7, -1], [5, -1, -1], [2, -1, -1], [9, 11, -1]]])
    out = training._call_splice_head(
        head, {1: emb}, torch.zeros(1, dtype=torch.long), positions, False,
    )
    pos, neg = out["pos_counts"], out["neg_counts"]
    assert torch.all(pos[0, 2] == 0) and torch.all(pos[0, :, 1:] == 0)
    assert torch.all(neg[0, 1:] == 0) and torch.all(neg[0, :, 2] == 0)
    assert torch.all(pos[0, :2, 0] > 0)


def test_junction_einsum_is_bf16_inputs_fp32_output():
    from alphagenome_pytorch.heads import _einsum_bf16_inputs_f32_output

    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 5, 3, 8, generator=g)
    b = torch.randn(2, 4, 3, 8, generator=g)
    out = _einsum_bf16_inputs_f32_output("bdth,bath->bdat", a, b)
    ref = torch.einsum("bdth,bath->bdat", a.bfloat16().double(), b.bfloat16().double())
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), ref, rtol=1e-6, atol=1e-6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _einsum_bf16_inputs_f32_output("bdth,bath->bdat", a, b).dtype == torch.float32


def test_top_k_ties_resolve_to_lower_index():
    logits = torch.zeros(1, 5, 10)          # all positions tie (p = 0.2 each)
    pos = training._top_k_positions_from_logits(logits, top_k=3, threshold=0.0)
    assert pos[0, 0].tolist() == [0, 1, 2]


def _parse_cli(*extra):
    from alphagenome_pytorch.extensions.finetuning.args import parse_args
    base = ["--genome", "hg38.fa", "--train-bed", "train.bed", "--val-bed", "val.bed",
            "--pretrained-weights", "model.pth"]
    return parse_args(base + list(extra))


def test_shared_training_defaults():
    ns = _parse_cli("--modality", "atac", "--bigwig", "a.bw")
    assert ns.weight_decay == 0.0
    assert ns.warmup_steps == 0
    assert ns.lr_schedule == "constant"
    assert ns.mode == "linear-probe"
    assert ns.sequence_length == 1048576
    assert ns.rope_init == "truncated_normal"
    assert ns.resume == "auto"
    # No splice_site head -> no implicit pretrained-head init.
    assert ns.pretrained_head_sample_dict == {}


def test_splice_site_pretrained_init_default_and_opt_out():
    splice = ["--modality", "splice_site,splice_usage,splice_junctions",
              "--star-junctions", "s.SJ.out.tab", "--ssu", "s.parquet"]
    assert _parse_cli(*splice).pretrained_head_sample_dict == {"splice_site": 0}
    opt_out = _parse_cli(*splice, "--pretrained-head-samples", "splice_site:NA")
    assert opt_out.pretrained_head_sample_dict == {"splice_site": None}
    other = _parse_cli(*splice, "--pretrained-head-samples", "splice_usage:3")
    assert other.pretrained_head_sample_dict == {"splice_usage": 3}
