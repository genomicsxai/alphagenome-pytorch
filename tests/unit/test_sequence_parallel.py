"""Unit tests for SequenceParallelism.

Tests sharding, gathering, and position-extraction logic across three modes:

- Mocked distributed (no GPU required) — the ``@pytest.mark.unit`` classes below.
- Single GPU with real dist initialisation — ``TestSequenceParallelismGPU``
  (skipped when CUDA is unavailable).
- Multi-GPU with real NCCL via ``torch.multiprocessing.spawn`` —
  ``TestSequenceParallelismMultiGPU`` (skipped when fewer than 2 CUDA devices
  are present; marked ``slow``).
"""

from __future__ import annotations

import os
import socket
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from alphagenome_pytorch.sequence_parallel import (
    SequenceParallelism,
    create_sequence_parallel_strategy,
)


# ---------------------------------------------------------------------------
# Helpers shared across all test modes
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _patch_dist_single_rank(monkeypatch):
    """Patch torch.distributed to appear as world_size=1, rank=0."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def _all_gather_single(tensor_list, tensor, group=None, async_op=False):  # noqa: ARG001
        tensor_list[0].copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", _all_gather_single)


def _patch_dist_two_ranks(monkeypatch, rank, shard_r0, shard_r1):
    """Patch torch.distributed to appear as world_size=2 at a given rank.

    all_gather is mocked to return shard_r0 and shard_r1 regardless of caller.
    """
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)

    def _all_gather_two(tensor_list, _tensor, group=None, async_op=False):  # noqa: ARG001
        tensor_list[0].copy_(shard_r0)
        tensor_list[1].copy_(shard_r1)

    monkeypatch.setattr(dist, "all_gather", _all_gather_two)


def _make_mock_model(B: int = 1, S: int = 512) -> MagicMock:
    """Return a minimal MagicMock of AlphaGenome for forward() tests.

    Shapes follow the real model for B=1, S=512 (L_trunk = S // 128 = 4).
    """
    L_trunk = S // 128
    C_trunk, C_128bp, C_decoder = 1536, 3072, 768

    model = MagicMock()
    model.dtype_policy.cast_to_compute.side_effect = lambda x: x
    model.dtype_policy.compute_dtype = None  # float32 default

    # encoder: NLC (B,S,4) -> (NCL trunk, intermediates)
    model.encoder.return_value = (
        torch.zeros(B, C_trunk, L_trunk),
        {},
    )

    # organism_embed: (B,) -> (B, C_trunk)
    model.organism_embed.return_value = torch.zeros(B, C_trunk)

    # tower: NLC -> (NLC, pair_activations)
    model.tower.return_value = (
        torch.zeros(B, L_trunk, C_trunk),
        None,
    )

    # embedder_128bp: NCL -> NCL
    model.embedder_128bp.return_value = torch.zeros(B, C_128bp, L_trunk)

    # decoder: NCL -> NCL
    model.decoder.return_value = torch.zeros(B, C_decoder, S)

    # embedder_1bp: NCL -> NCL
    model.embedder_1bp.return_value = torch.zeros(B, C_trunk, S)

    return model


# ---------------------------------------------------------------------------
# Module-level worker functions for mp.spawn (must be top-level / picklable)
# ---------------------------------------------------------------------------

def _init_process_group(rank: int, world_size: int, port: int) -> None:
    """Initialise the default process group for a spawned worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _worker_shard_gather_roundtrip(
    rank: int, world_size: int, port: int, L: int, overlap: int
) -> None:
    """Verify shard → gather is identity on every rank (NCCL worker)."""
    _init_process_group(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    x_full = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, L)
    sp = SequenceParallelism(overlap_highres=overlap, overlap_lowres=overlap)

    shard = sp.shard_sequence(x_full, overlap=overlap)
    result = sp.gather_full(shard, overlap=overlap, expected_len=L)

    assert result.shape[-1] == L, f"Shape mismatch on rank {rank}"
    assert torch.allclose(result, x_full), f"Values mismatch on rank {rank}"

    dist.destroy_process_group()


def _worker_gather_positions(
    rank: int, world_size: int, port: int, L: int, overlap: int
) -> None:
    """Verify gather_positions collects all global positions (NCCL worker)."""
    _init_process_group(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    x_full = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, L)
    sp = SequenceParallelism(overlap_highres=overlap, overlap_lowres=overlap)

    shard = sp.shard_sequence(x_full, overlap=overlap)
    indices = torch.arange(L, device=device)
    gathered = sp.gather_positions(
        shard, overlap=overlap, global_length=L, global_indices=indices
    )

    assert gathered.shape[-1] == L, (
        f"Expected {L} positions on rank {rank}, got {gathered.shape[-1]}"
    )

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Mocked-distributed tests (CPU, no GPU required)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShardSequence:
    """Tests for SequenceParallelism.shard_sequence."""

    def test_single_rank_returns_full_tensor(self):
        """With world_size=1, sharding is identity."""
        sp = SequenceParallelism(overlap_highres=4, overlap_lowres=2)
        x = torch.arange(32).float().view(1, 1, 32)
        shard = sp.shard_sequence(x, overlap=4)
        assert shard.shape == x.shape
        assert torch.equal(shard, x)

    def test_return_bounds_flag(self):
        """return_bounds=True yields (tensor, (start, end)) tuple."""
        sp = SequenceParallelism()
        x = torch.zeros(1, 1, 64)
        result = sp.shard_sequence(x, overlap=4, return_bounds=True)
        assert isinstance(result, tuple) and len(result) == 2
        _, (start, end) = result
        assert start == 0 and end == 64

    def test_overlap_clipped_at_boundaries(self, monkeypatch):
        """Rank 0 start is clamped to 0; last rank end is clamped to L."""
        sp = SequenceParallelism()
        x = torch.arange(16).float().view(1, 1, 16)

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)

        _, (s0, e0) = sp.shard_sequence(x, overlap=4, return_bounds=True)
        assert s0 == 0
        assert e0 == 8 + 4  # base=8, end=min(16, 8+4)=12

    def test_two_rank_shards_cover_full_sequence(self, monkeypatch):
        """Union of rank 0 and rank 1 non-overlap regions covers [0, L)."""
        L, overlap = 16, 2
        x = torch.arange(L).float().view(1, 1, L)
        sp = SequenceParallelism()

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)

        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        shard_r0 = sp.shard_sequence(x, overlap)  # [0 .. 8+2)

        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        shard_r1 = sp.shard_sequence(x, overlap)  # [8-2 .. 16)

        # Non-overlapping cores: rank 0 owns [0, 8), rank 1 owns [8, 16)
        core_r0 = shard_r0[..., :L // 2]    # first 8
        core_r1 = shard_r1[..., overlap:]   # trim left overlap → last 8

        combined = torch.cat([core_r0, core_r1], dim=-1)
        assert combined.shape[-1] == L
        assert torch.equal(combined, x)


@pytest.mark.unit
class TestGatherFull:
    """Tests for SequenceParallelism.gather_full."""

    def test_single_rank_is_identity(self, monkeypatch):
        """With world_size=1, gather_full returns the input unchanged."""
        _patch_dist_single_rank(monkeypatch)
        sp = SequenceParallelism()
        x = torch.randn(1, 8, 10)
        result = sp.gather_full(x, overlap=2, expected_len=10)
        assert result.shape == x.shape
        assert torch.allclose(result, x)

    def test_roundtrip_two_ranks(self, monkeypatch):
        """shard + gather round-trips the original sequence for world_size=2."""
        L, overlap = 16, 2
        x = torch.arange(L).float().view(1, 1, L)
        sp = SequenceParallelism()

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)

        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        shard_r0 = sp.shard_sequence(x, overlap)  # shape (1,1,10)

        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        shard_r1 = sp.shard_sequence(x, overlap)  # shape (1,1,10)

        _patch_dist_two_ranks(monkeypatch, rank=0, shard_r0=shard_r0, shard_r1=shard_r1)
        result = sp.gather_full(shard_r0, overlap=overlap, expected_len=L)

        assert result.shape[-1] == L
        assert torch.equal(result, x)

    def test_expected_len_assertion(self, monkeypatch):
        """gather_full raises when result length differs from expected_len."""
        _patch_dist_single_rank(monkeypatch)
        sp = SequenceParallelism()
        x = torch.zeros(1, 4, 10)
        with pytest.raises(AssertionError):
            sp.gather_full(x, overlap=0, expected_len=99)


@pytest.mark.unit
class TestSubsetGlobalPositionsLocally:
    """Tests for SequenceParallelism.subset_global_positions_locally."""

    def test_extracts_positions_in_range(self):
        """Positions that fall in this rank's region are returned."""
        sp = SequenceParallelism()
        # world=1, rank=0 → owns the entire sequence [0, L)
        x = torch.arange(32).float().view(1, 1, 32)
        indices = torch.tensor([0, 5, 10, 20])
        result = sp.subset_global_positions_locally(
            x, overlap=0, global_length=32, global_indices=indices
        )
        assert result.shape[-1] == len(indices)
        assert torch.equal(result, x[..., indices])

    def test_empty_when_no_positions_in_range(self, monkeypatch):
        """Returns zero-length tensor when no indices fall in this rank's region."""
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)  # owns [0, 8)

        sp = SequenceParallelism()
        x = torch.zeros(1, 4, 8)
        indices = torch.tensor([8, 10, 14])  # all in rank 1's region [8, 16)
        result = sp.subset_global_positions_locally(
            x, overlap=0, global_length=16, global_indices=indices
        )
        assert result.shape[-1] == 0

    def test_global_to_local_index_conversion(self, monkeypatch):
        """Global indices are correctly remapped to local shard offsets."""
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)  # owns [8, 16)

        sp = SequenceParallelism()
        x = torch.arange(8).float().view(1, 1, 8)  # local positions 0-7
        indices = torch.tensor([8, 12])             # global positions 8 and 12
        result = sp.subset_global_positions_locally(
            x, overlap=0, global_length=16, global_indices=indices
        )
        assert result.shape[-1] == 2
        # global 8 → local 0 → value 0.0; global 12 → local 4 → value 4.0
        assert result[0, 0, 0].item() == 0.0
        assert result[0, 0, 1].item() == 4.0


@pytest.mark.unit
class TestSequenceParallelismForward:
    """Tests for SequenceParallelism.forward with a mocked AlphaGenome model."""

    def test_output_keys_both_resolutions(self, monkeypatch):
        """forward(resolutions=(1, 128)) returns embeddings for both resolutions."""
        _patch_dist_single_rank(monkeypatch)
        model = _make_mock_model()
        sp = SequenceParallelism()

        embeddings, _ = sp.forward(
            model,
            torch.zeros(1, 512, 4),
            torch.zeros(1, dtype=torch.long),
            resolutions=(1, 128),
        )

        assert 1 in embeddings
        assert 128 in embeddings

    def test_output_keys_128bp_only_skips_decoder(self, monkeypatch):
        """forward(resolutions=(128,)) omits 1bp key and does not call decoder."""
        _patch_dist_single_rank(monkeypatch)
        model = _make_mock_model()
        sp = SequenceParallelism()

        embeddings, _ = sp.forward(
            model,
            torch.zeros(1, 512, 4),
            torch.zeros(1, dtype=torch.long),
            resolutions=(128,),
        )

        assert 128 in embeddings
        assert 1 not in embeddings
        model.decoder.assert_not_called()
        model.embedder_1bp.assert_not_called()

    def test_output_shapes(self, monkeypatch):
        """Embedding shapes match expected (B, C, S_local) NCL format."""
        _patch_dist_single_rank(monkeypatch)
        B, S = 1, 512
        L_trunk = S // 128
        model = _make_mock_model(B=B, S=S)
        sp = SequenceParallelism()

        embeddings, _ = sp.forward(
            model,
            torch.zeros(B, S, 4),
            torch.zeros(B, dtype=torch.long),
            resolutions=(1, 128),
        )

        assert embeddings[128].shape == (B, 3072, L_trunk)
        assert embeddings[1].shape == (B, 1536, S)

    def test_encoder_called_with_nlc(self, monkeypatch):
        """Encoder must receive input in NLC format (B, S, 4), not NCL."""
        _patch_dist_single_rank(monkeypatch)
        model = _make_mock_model()
        sp = SequenceParallelism()

        sp.forward(model, torch.zeros(1, 512, 4), torch.zeros(1, dtype=torch.long))

        encoder_input = model.encoder.call_args[0][0]
        assert encoder_input.shape[-1] == 4, (
            f"Encoder should receive NLC (B,S,4) but got shape {encoder_input.shape}"
        )

    def test_pair_activations_passed_through(self, monkeypatch):
        """pair_activations from the transformer tower are returned as-is."""
        _patch_dist_single_rank(monkeypatch)
        model = _make_mock_model()
        sentinel = object()
        model.tower.return_value = (torch.zeros(1, 4, 1536), sentinel)
        sp = SequenceParallelism()

        _, pair_acts = sp.forward(
            model, torch.zeros(1, 512, 4), torch.zeros(1, dtype=torch.long)
        )

        assert pair_acts is sentinel


@pytest.mark.unit
class TestCreateSequenceParallelStrategy:
    """Tests for create_sequence_parallel_strategy factory function."""

    def test_returns_sequence_parallelism_instance(self):
        """Factory returns a SequenceParallelism with the given overlaps."""
        sp = create_sequence_parallel_strategy(overlap_highres=512, overlap_lowres=16)
        assert isinstance(sp, SequenceParallelism)
        assert sp.overlap_highres == 512
        assert sp.overlap_lowres == 16

    def test_default_overlaps(self):
        """Default overlap values are 1024 (highres) and 32 (lowres)."""
        sp = create_sequence_parallel_strategy()
        assert sp.overlap_highres == 1024
        assert sp.overlap_lowres == 32


# ---------------------------------------------------------------------------
# Single-GPU tests (real tensors on GPU, no inter-process communication)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSequenceParallelismGPU:
    """Verify GPU tensor placement for operations that do not need dist."""

    def test_shard_sequence_preserves_cuda_device(self):
        """shard_sequence output stays on the same CUDA device as input."""
        sp = SequenceParallelism(overlap_highres=4)
        x = torch.zeros(1, 4, 32, device="cuda")
        shard = sp.shard_sequence(x, overlap=4)
        assert shard.is_cuda

    def test_subset_positions_preserves_cuda_device(self):
        """subset_global_positions_locally keeps tensors on CUDA."""
        sp = SequenceParallelism()
        x = torch.zeros(1, 4, 32, device="cuda")
        indices = torch.tensor([0, 5, 10], device="cuda")
        result = sp.subset_global_positions_locally(
            x, overlap=0, global_length=32, global_indices=indices
        )
        assert result.is_cuda

    def test_forward_output_on_cuda_with_mocked_model(self, monkeypatch):
        """forward() returns CUDA embeddings when mock model outputs CUDA tensors."""
        _patch_dist_single_rank(monkeypatch)
        B, S = 1, 512
        L_trunk = S // 128
        C_trunk, C_128bp, C_decoder = 1536, 3072, 768

        model = MagicMock()
        model.dtype_policy.cast_to_compute.side_effect = lambda x: x
        model.dtype_policy.compute_dtype = None
        model.encoder.return_value = (
            torch.zeros(B, C_trunk, L_trunk, device="cuda"), {}
        )
        model.organism_embed.return_value = torch.zeros(B, C_trunk, device="cuda")
        model.tower.return_value = (
            torch.zeros(B, L_trunk, C_trunk, device="cuda"), None
        )
        model.embedder_128bp.return_value = torch.zeros(B, C_128bp, L_trunk, device="cuda")
        model.decoder.return_value = torch.zeros(B, C_decoder, S, device="cuda")
        model.embedder_1bp.return_value = torch.zeros(B, C_trunk, S, device="cuda")

        sp = SequenceParallelism()
        embeddings, _ = sp.forward(
            model,
            torch.zeros(B, S, 4, device="cuda"),
            torch.zeros(B, dtype=torch.long, device="cuda"),
        )

        assert embeddings[128].is_cuda
        assert embeddings[1].is_cuda


# ---------------------------------------------------------------------------
# Multi-GPU tests using real NCCL (requires >= 2 CUDA devices)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs",
)
class TestSequenceParallelismMultiGPU:
    """End-to-end distributed tests using real NCCL across two GPUs.

    Each test spawns two worker processes via ``torch.multiprocessing.spawn``
    and checks that the distributed gather operations produce correct results.
    """

    def test_shard_gather_roundtrip_two_gpus(self):
        """shard + gather_full reconstructs the original tensor on both ranks."""
        mp.spawn(
            _worker_shard_gather_roundtrip,
            args=(2, _find_free_port(), 256, 8),
            nprocs=2,
            join=True,
        )

    def test_gather_positions_two_gpus(self):
        """gather_positions collects all global indices from both ranks."""
        mp.spawn(
            _worker_gather_positions,
            args=(2, _find_free_port(), 256, 8),
            nprocs=2,
            join=True,
        )
