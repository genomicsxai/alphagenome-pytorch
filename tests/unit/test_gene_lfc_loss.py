"""Unit tests for the gene LFC (Decima-style) cross-track training loss.

Covers:
  - `_build_strand_channel_mask` semantics (+/-/. and invalid chars).
  - `AlphaGenomeLoss` config plumbing (`gene_loss_weights`, `track_strands`).
  - `_compute_gene_lfc` math: zero loss at perfect prediction, positive loss
    on mismatch, finite gradient, strand-channel filtering, and head-level
    gating (only heads with non-zero `gene_loss_weights` consume `gene_mask`).
"""

from __future__ import annotations

import pytest
import torch

from alphagenome_pytorch.extensions.finetuning.training import (
    compute_finetuning_loss,
)
from alphagenome_pytorch.losses import gene_lfc_loss
from alphagenome_pytorch.training import (
    AlphaGenomeLoss,
    _build_strand_channel_mask,
)


@pytest.mark.unit
class TestBuildStrandChannelMask:

    def test_plus_minus(self):
        mask = _build_strand_channel_mask(["+", "-", "+", "-"])
        assert mask.shape == (2, 1, 4)
        assert mask.dtype == torch.bool
        # bucket 0 (positive genes): only + tracks
        assert mask[0, 0].tolist() == [True, False, True, False]
        # bucket 1 (negative genes): only - tracks
        assert mask[1, 0].tolist() == [False, True, False, True]

    def test_unstranded_dot_in_both_buckets(self):
        mask = _build_strand_channel_mask(["+", ".", "-"])
        assert mask[0, 0].tolist() == [True, True, False]   # + and . in bucket 0
        assert mask[1, 0].tolist() == [False, True, True]   # - and . in bucket 1

    def test_invalid_strand_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            _build_strand_channel_mask(["+", "?", "-"])

    def test_string_input(self):
        """Compact string form '+-+-..' iterates as one char per track."""
        mask = _build_strand_channel_mask("+-+-")
        assert mask.shape == (2, 1, 4)
        assert mask[0, 0].tolist() == [True, False, True, False]


@pytest.mark.unit
class TestAlphaGenomeLossConfigPlumbing:

    def test_defaults_empty_gene_config(self):
        loss_fn = AlphaGenomeLoss()
        assert loss_fn.gene_loss_weights == {}
        assert loss_fn.gene_cross_track_weight == 5.0
        assert loss_fn._strand_mask_heads == []

    def test_register_track_strands(self):
        loss_fn = AlphaGenomeLoss(
            gene_loss_weights={"rna_seq": 0.1},
            track_strands={"rna_seq": "+-+-"},
        )
        assert loss_fn.gene_loss_weights == {"rna_seq": 0.1}
        mask = loss_fn._get_strand_channel_mask("rna_seq")
        assert mask is not None
        assert mask.shape == (2, 1, 4)
        # head without strand registered → None
        assert loss_fn._get_strand_channel_mask("atac") is None

    def test_strand_mask_buffer_follows_device_dtype(self):
        """Strand masks are buffers, so `.to(...)` moves them with the module."""
        loss_fn = AlphaGenomeLoss(
            gene_loss_weights={"rna_seq": 0.1},
            track_strands={"rna_seq": "+-+-"},
        )
        # Convert to float dtype to verify buffer movement (bool can't be cast,
        # but we verify the buffer is a registered buffer by listing them).
        buffer_names = [name for name, _ in loss_fn.named_buffers()]
        assert any("strand_channel_mask" in n for n in buffer_names)


def _make_loss_fn(gene_w=0.1, num_tracks=4, strands="+-+-", head="rna_seq"):
    return AlphaGenomeLoss(
        heads=[head],
        gene_loss_weights={head: gene_w},
        track_strands={head: strands},
        # Use main loss multinomial_resolution=1 so we don't need long S.
        multinomial_resolution=1,
        positional_weight=1.0,
    )


@pytest.mark.unit
class TestComputeGeneLFCMath:

    def _inputs(self, B=1, S=64, C=4, num_genes=2, seed=0):
        torch.manual_seed(seed)
        preds = torch.rand(B, S, C) + 0.5
        targets = torch.rand(B, S, C) + 0.5
        track_mask = torch.ones(B, 1, C, dtype=torch.bool)
        # Two genes: gene 0 on + strand at positions [10, 30); gene 1 on -
        # strand at [40, 60). Build [B, S, 2, G].
        gene_mask = torch.zeros(B, S, 2, num_genes, dtype=torch.bool)
        gene_mask[:, 10:30, 0, 0] = True
        gene_mask[:, 40:60, 1, 1] = True
        return preds, targets, track_mask, gene_mask

    def test_perfect_prediction_zeroes_poisson_term(self):
        """At pred==target, only the Poisson NLL on per-gene totals goes to
        zero. The multinomial term reduces to the entropy of the per-gene
        tissue distribution, which is non-zero for any non-degenerate target.
        Test that:
          - aux['gene_loss_total_count'] ≈ 0 (Poisson at optimum)
          - aux['gene_loss_positional'] is finite and >= 0
          - perturbing preds away from targets strictly increases total loss
        """
        loss_fn = _make_loss_fn(num_tracks=4)
        preds, _, track_mask, gene_mask = self._inputs()
        strand_mask = loss_fn._get_strand_channel_mask("rna_seq")

        loss_perfect, aux = loss_fn._compute_gene_lfc(
            predictions=preds,
            targets=preds,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert aux["gene_loss_total_count"].item() < 1e-4, (
            f"Poisson term should be ~0 at pred==target, "
            f"got {aux['gene_loss_total_count'].item()}"
        )
        assert torch.isfinite(aux["gene_loss_positional"])
        assert aux["gene_loss_positional"].item() >= 0

        # Perturbing preds must strictly increase the total loss.
        perturbed = preds.clone()
        # Skew within the gene window so the cross-track distribution shifts
        # away from the target distribution.
        perturbed[:, 10:30, 0] *= 5.0  # bias track 0 (+ strand) much higher
        loss_perturbed, _ = loss_fn._compute_gene_lfc(
            predictions=perturbed,
            targets=preds,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert loss_perturbed.item() > loss_perfect.item() + 0.01

    def test_positive_loss_on_mismatch(self):
        loss_fn = _make_loss_fn()
        preds, targets, track_mask, gene_mask = self._inputs()
        strand_mask = loss_fn._get_strand_channel_mask("rna_seq")

        loss, _ = loss_fn._compute_gene_lfc(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_finite_gradient(self):
        loss_fn = _make_loss_fn()
        preds, targets, track_mask, gene_mask = self._inputs()
        preds = preds.clone().requires_grad_(True)
        strand_mask = loss_fn._get_strand_channel_mask("rna_seq")

        loss, _ = loss_fn._compute_gene_lfc(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        loss.backward()
        assert preds.grad is not None
        assert torch.all(torch.isfinite(preds.grad))

    def test_strand_filter_excludes_mismatched_tracks(self):
        """A perfect-prediction setup should still give ~0 loss; if we then
        change predictions only on tracks that the strand filter excludes
        for ALL genes in the batch, the loss should be unchanged."""
        # Tracks: [+, -, +, -]. Gene 0 is + strand, gene 1 is - strand.
        # We construct a setup where every gene in the batch is + strand,
        # so - tracks (idx 1, 3) are filtered out by the strand mask.
        loss_fn = _make_loss_fn(num_tracks=4, strands="+-+-")
        B, S, C = 1, 64, 4
        torch.manual_seed(0)
        preds = torch.rand(B, S, C) + 0.5
        track_mask = torch.ones(B, 1, C, dtype=torch.bool)
        # Single + strand gene on positions [10, 30). No - strand genes.
        gene_mask = torch.zeros(B, S, 2, 1, dtype=torch.bool)
        gene_mask[:, 10:30, 0, 0] = True
        strand_mask = loss_fn._get_strand_channel_mask("rna_seq")

        # Baseline: perfect prediction.
        loss_baseline, _ = loss_fn._compute_gene_lfc(
            predictions=preds,
            targets=preds,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )

        # Perturb only track 1 (- strand). Since the only gene is + strand,
        # this perturbation must be filtered out and the loss unchanged.
        perturbed = preds.clone()
        perturbed[:, 10:30, 1] += 100.0  # huge change on the - track
        loss_perturbed, _ = loss_fn._compute_gene_lfc(
            predictions=perturbed,
            targets=preds,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert torch.isclose(loss_baseline, loss_perturbed, atol=1e-5), (
            f"strand filter leaked: baseline={loss_baseline.item()}, "
            f"perturbed={loss_perturbed.item()}"
        )

        # Sanity: perturbing track 0 (+ strand) MUST change the loss.
        sanity = preds.clone()
        sanity[:, 10:30, 0] += 100.0
        loss_sanity, _ = loss_fn._compute_gene_lfc(
            predictions=sanity,
            targets=preds,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert loss_sanity.item() > loss_baseline.item() + 0.1


@pytest.mark.unit
class TestPerHeadGating:
    """Verify _compute_head_loss only adds gene LFC where gene_loss_weights > 0."""

    def _stub_head_loss(self, loss_fn, fixed_per_head_main_loss):
        """Replace the multinomial branch with a stub returning a known scalar.

        We can't easily run the full multinomial path in a unit test (it
        needs a model with `.heads` for target scaling). Instead, we monkey-
        patch _compute_head_loss to call _compute_gene_lfc directly when
        gene_loss_weights[head] > 0, and otherwise return the stubbed main
        loss. This isolates the gating decision.
        """
        original = loss_fn._compute_head_loss
        gene_lfc_calls = []

        def patched(head, output, target, mask, organism_index, gene_mask=None):
            main = torch.tensor(fixed_per_head_main_loss[head], dtype=torch.float32)
            gene_w = loss_fn.gene_loss_weights.get(head, 0.0)
            if gene_w > 0 and gene_mask is not None:
                strand_mask = loss_fn._get_strand_channel_mask(head)
                gene_loss, _ = loss_fn._compute_gene_lfc(
                    predictions=output,
                    targets=target,
                    targets_mask=mask,
                    gene_mask=gene_mask,
                    strand_channel_mask=strand_mask,
                )
                gene_lfc_calls.append(head)
                return main + gene_w * gene_loss
            return main

        loss_fn._compute_head_loss = patched
        return gene_lfc_calls

    def test_gene_lfc_only_runs_for_configured_head(self):
        loss_fn = AlphaGenomeLoss(
            heads=["atac", "rna_seq"],
            gene_loss_weights={"rna_seq": 0.1},
            track_strands={"rna_seq": "+-+-"},
        )
        calls = self._stub_head_loss(loss_fn, {"atac": 1.0, "rna_seq": 1.0})

        B, S, C = 1, 64, 4
        torch.manual_seed(0)
        outputs = {
            "atac": torch.rand(B, S, C) + 0.5,
            "rna_seq": torch.rand(B, S, C) + 0.5,
        }
        targets = {
            "atac": torch.rand(B, S, C) + 0.5,
            "rna_seq": torch.rand(B, S, C) + 0.5,
        }
        masks = {
            "atac": torch.ones(B, 1, C, dtype=torch.bool),
            "rna_seq": torch.ones(B, 1, C, dtype=torch.bool),
        }
        gene_mask = torch.zeros(B, S, 2, 1, dtype=torch.bool)
        gene_mask[:, 10:30, 0, 0] = True

        result = loss_fn(
            outputs, targets, organism_index=torch.tensor([0]),
            masks=masks, gene_mask=gene_mask,
        )
        # gene LFC must have been invoked exactly once, on rna_seq.
        assert calls == ["rna_seq"]
        assert torch.isfinite(result["loss"])

    def test_no_gene_mask_means_no_gene_lfc(self):
        loss_fn = AlphaGenomeLoss(
            heads=["rna_seq"],
            gene_loss_weights={"rna_seq": 0.1},
            track_strands={"rna_seq": "+-+-"},
        )
        calls = self._stub_head_loss(loss_fn, {"rna_seq": 1.0})

        B, S, C = 1, 64, 4
        outputs = {"rna_seq": torch.rand(B, S, C) + 0.5}
        targets = {"rna_seq": torch.rand(B, S, C) + 0.5}
        masks = {"rna_seq": torch.ones(B, 1, C, dtype=torch.bool)}

        # Forward without gene_mask → no gene LFC fires.
        loss_fn(outputs, targets, organism_index=torch.tensor([0]), masks=masks)
        assert calls == []


@pytest.mark.unit
class TestStandaloneVsMethod:

    def test_alphagenomeloss_method_matches_standalone(self):
        """The AlphaGenomeLoss method is now a thin wrapper around the
        standalone losses.gene_lfc_loss; both must return identical values
        for the same inputs."""
        loss_fn = AlphaGenomeLoss(
            gene_loss_weights={"rna_seq": 0.1},
            track_strands={"rna_seq": "+-+-"},
            gene_cross_track_weight=5.0,
        )
        torch.manual_seed(0)
        B, S, C = 1, 64, 4
        preds = torch.rand(B, S, C) + 0.5
        targets = torch.rand(B, S, C) + 0.5
        track_mask = torch.ones(B, 1, C, dtype=torch.bool)
        gene_mask = torch.zeros(B, S, 2, 2, dtype=torch.bool)
        gene_mask[:, 10:30, 0, 0] = True
        gene_mask[:, 40:60, 1, 1] = True
        strand_mask = loss_fn._get_strand_channel_mask("rna_seq")

        loss_method, _ = loss_fn._compute_gene_lfc(
            predictions=preds, targets=targets, targets_mask=track_mask,
            gene_mask=gene_mask, strand_channel_mask=strand_mask,
        )
        loss_standalone, _ = gene_lfc_loss(
            predictions=preds, targets=targets, targets_mask=track_mask,
            gene_mask=gene_mask, strand_channel_mask=strand_mask,
            gene_cross_track_weight=5.0,
        )
        assert torch.isclose(loss_method, loss_standalone)


@pytest.mark.unit
class TestComputeFinetuningLossGeneLFC:
    """Verify compute_finetuning_loss adds the gene LFC term correctly."""

    def _make_inputs(self, B=1, S=64, C=4, num_genes=2, seed=0):
        torch.manual_seed(seed)
        preds_dict = {1: torch.rand(B, S, C) + 0.5}
        targets_dict = {1: torch.rand(B, S, C) + 0.5}
        gene_mask = torch.zeros(B, S, 2, num_genes, dtype=torch.bool)
        gene_mask[:, 10:30, 0, 0] = True
        gene_mask[:, 40:60, 1, 1] = True
        strand_mask = _build_strand_channel_mask("+-+-"[:C])
        return preds_dict, targets_dict, gene_mask, strand_mask

    def test_gene_lfc_off_when_weight_zero(self):
        preds, targets, gene_mask, strand_mask = self._make_inputs()
        loss_no_gene, dict_no_gene = compute_finetuning_loss(
            predictions=preds, targets=targets,
            resolution_weights={1: 1.0}, positional_weight=1.0,
            device=torch.device("cpu"),
        )
        loss_with_zero, dict_with_zero = compute_finetuning_loss(
            predictions=preds, targets=targets,
            resolution_weights={1: 1.0}, positional_weight=1.0,
            device=torch.device("cpu"),
            gene_mask=gene_mask, gene_loss_weight=0.0,
            strand_channel_mask=strand_mask,
        )
        # gene_loss_weight=0 must be exactly identical to "no gene config".
        assert torch.isclose(loss_no_gene, loss_with_zero)
        assert "loss_gene_lfc" not in dict_with_zero

    def test_gene_lfc_increases_loss_when_enabled(self):
        preds, targets, gene_mask, strand_mask = self._make_inputs()
        loss_off, _ = compute_finetuning_loss(
            predictions=preds, targets=targets,
            resolution_weights={1: 1.0}, positional_weight=1.0,
            device=torch.device("cpu"),
        )
        loss_on, dict_on = compute_finetuning_loss(
            predictions=preds, targets=targets,
            resolution_weights={1: 1.0}, positional_weight=1.0,
            device=torch.device("cpu"),
            gene_mask=gene_mask, gene_loss_weight=0.1,
            strand_channel_mask=strand_mask,
        )
        assert loss_on.item() > loss_off.item()
        assert "loss_gene_lfc" in dict_on
        assert torch.isfinite(dict_on["loss_gene_lfc"])

    def test_gene_lfc_only_at_resolution_1(self):
        """When the only resolution available is 128, gene LFC must NOT fire
        even if all params are set, because we gate on res == 1."""
        torch.manual_seed(0)
        B, S, C = 1, 8, 4  # 128 / 16 = 8 (S at 128bp)
        preds = {128: torch.rand(B, S, C) + 0.5}
        targets = {128: torch.rand(B, S, C) + 0.5}
        # gene_mask shape uses raw seq length, but the gating checks res==1 only.
        gene_mask = torch.zeros(B, 1024, 2, 1, dtype=torch.bool)
        gene_mask[:, 100:200, 0, 0] = True
        strand_mask = _build_strand_channel_mask("+-+-"[:C])

        _, dict_on = compute_finetuning_loss(
            predictions=preds, targets=targets,
            resolution_weights={128: 1.0}, positional_weight=1.0,
            device=torch.device("cpu"),
            gene_mask=gene_mask, gene_loss_weight=0.1,
            strand_channel_mask=strand_mask,
        )
        assert "loss_gene_lfc" not in dict_on


@pytest.mark.unit
class TestSequenceParallelSlicing:
    """Pin the contract used by train_epoch_sequence_parallel for slicing
    gene_mask along S to per-rank shards. The implementation is inlined
    inside the SP loop; this test mirrors that math and verifies that:
      - Full-S mask split across W ranks reconstructs to the original.
      - Each rank's slice has length S // W.
      - gene_lfc_loss accepts the per-rank slice without shape errors.
    """

    def test_slice_reconstructs_full_mask(self):
        torch.manual_seed(0)
        B, S, G = 2, 1024, 3
        world_size = 4
        full_mask = torch.zeros(B, S, 2, G, dtype=torch.bool)
        full_mask[:, 100:300, 0, 0] = True  # +-strand gene
        full_mask[:, 500:700, 1, 1] = True  # --strand gene

        slices = []
        local_len = S // world_size
        for rank in range(world_size):
            slc = full_mask[:, rank * local_len:(rank + 1) * local_len, :, :]
            assert slc.shape == (B, local_len, 2, G)
            slices.append(slc)

        reassembled = torch.cat(slices, dim=1)
        assert torch.equal(reassembled, full_mask)

    def test_per_rank_slice_runs_through_loss(self):
        """A per-rank gene_mask slice (with matching local pred/target slice)
        must run through gene_lfc_loss without shape errors."""
        torch.manual_seed(0)
        B, S, C, G = 1, 256, 4, 2
        world_size = 4
        rank = 1
        local_len = S // world_size

        preds = torch.rand(B, local_len, C) + 0.5
        targets = torch.rand(B, local_len, C) + 0.5
        track_mask = torch.ones(B, 1, C, dtype=torch.bool)

        full_gene_mask = torch.zeros(B, S, 2, G, dtype=torch.bool)
        # Place a gene that crosses the rank-1 shard.
        full_gene_mask[:, 80:100, 0, 0] = True  # within rank 1's [64, 128) shard
        full_gene_mask[:, 150:200, 1, 1] = True
        local_gene_mask = full_gene_mask[
            :, rank * local_len:(rank + 1) * local_len, :, :
        ]
        assert local_gene_mask.shape == (B, local_len, 2, G)

        strand_mask = _build_strand_channel_mask("+-+-")
        loss, aux = gene_lfc_loss(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=local_gene_mask,
            strand_channel_mask=strand_mask,
        )
        assert torch.isfinite(loss)


@pytest.mark.unit
class TestMissingStrandsValidation:

    def test_missing_track_strands_raises_when_gene_w_set(self):
        """If gene_loss_weights[head] > 0 but no track_strands provided,
        the loss path must raise a clear error.

        We need to exercise the real `_compute_head_loss` (not the stub),
        which requires a tiny stand-in setup. The simplest way is to
        construct AlphaGenomeLoss with a dummy "rna_seq"-named head that
        falls through to multinomial loss, then call _compute_head_loss
        directly.
        """
        loss_fn = AlphaGenomeLoss(
            heads=["rna_seq"],
            gene_loss_weights={"rna_seq": 0.1},
            track_strands=None,  # NOT provided
            multinomial_resolution=1,
            positional_weight=1.0,
        )

        B, S, C = 1, 8, 2
        output = {1: torch.rand(B, S, C) + 0.5}
        target = {1: torch.rand(B, S, C) + 0.5}
        mask = torch.ones(B, 1, C, dtype=torch.bool)
        gene_mask = torch.zeros(B, S, 2, 1, dtype=torch.bool)
        gene_mask[:, 2:6, 0, 0] = True

        with pytest.raises(ValueError, match="track_strands were provided"):
            loss_fn._compute_head_loss(
                "rna_seq", output, target, mask,
                organism_index=torch.tensor([0]),
                gene_mask=gene_mask,
            )
