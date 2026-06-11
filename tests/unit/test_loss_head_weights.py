"""Tests for AlphaGenomeLoss head-weight aggregation.

Pins the contract for `DEFAULT_HEAD_WEIGHTS` and the divisor used in
`AlphaGenomeLoss.forward()`. Mirrors upstream JAX `HeadConfig.loss_weight`
semantics: a weight of 0.2 down-weights that head's contribution to the
batch-mean loss by exactly 0.2x relative to a weight-1.0 head with the
same residual.
"""

import pytest
import torch

from alphagenome_pytorch.training import (
    AlphaGenomeLoss,
    DEFAULT_HEAD_WEIGHTS,
)


@pytest.mark.unit
class TestDefaultHeadWeights:
    """Pin upstream parity for default head weights."""

    def test_count_heads_weighted_one(self):
        for name in (
            'atac', 'dnase', 'procap', 'cage', 'rna_seq',
            'chip_tf', 'chip_histone', 'contact_maps',
        ):
            assert DEFAULT_HEAD_WEIGHTS[name] == 1.0

    def test_splice_site_heads_weighted_one(self):
        assert DEFAULT_HEAD_WEIGHTS['splice_sites'] == 1.0
        assert DEFAULT_HEAD_WEIGHTS['splice_site_usage'] == 1.0

    def test_splice_junctions_weighted_zero_point_two(self):
        """Upstream weights splice_junctions at 0.2."""
        assert DEFAULT_HEAD_WEIGHTS['splice_junctions'] == 0.2


@pytest.mark.unit
class TestAggregationDivisor:
    """Verify the `sum(w_i * L_i) / num_heads` aggregation contract.

    We bypass `_compute_head_loss` by monkey-patching it so we can pin
    arbitrary per-head losses and verify the aggregator math directly.
    """

    @staticmethod
    def _make_loss_with_fixed_per_head(per_head_losses, head_weights=None):
        loss_fn = AlphaGenomeLoss(
            heads=list(per_head_losses.keys()),
            head_weights=head_weights,
        )

        # Replace _compute_head_loss with a stub that returns a known loss
        # per head, irrespective of input tensors.
        def _stub(self, head, output, target, mask, organism_index):
            return torch.tensor(per_head_losses[head], dtype=torch.float32)

        loss_fn._compute_head_loss = _stub.__get__(loss_fn, AlphaGenomeLoss)
        return loss_fn

    def _run(self, loss_fn, head_names):
        # Build minimal outputs/targets so forward() iterates the heads.
        outputs = {h: torch.zeros(1, 2, 1) for h in head_names}
        targets = {h: torch.zeros(1, 2, 1) for h in head_names}
        organism_index = torch.tensor([0])
        return loss_fn(outputs, targets, organism_index)

    def test_junction_enters_at_0p2_with_default_weights(self):
        """Equal residual on atac vs splice_junctions: junction contributes
        exactly 0.2 / num_heads of the total, atac contributes 1.0 / num_heads.
        """
        per_head = {'atac': 1.0, 'splice_junctions': 1.0}
        loss_fn = self._make_loss_with_fixed_per_head(per_head)
        result = self._run(loss_fn, list(per_head.keys()))

        # total = (1.0 * 1.0 + 0.2 * 1.0) / 2 = 0.6
        assert torch.isclose(result['loss'], torch.tensor(0.6))
        assert torch.isclose(result['atac_loss'], torch.tensor(1.0))
        assert torch.isclose(result['splice_junctions_loss'], torch.tensor(1.0))

    def test_override_restores_full_weight(self):
        """User-provided head_weights={'splice_junctions': 1.0} overrides default."""
        per_head = {'atac': 1.0, 'splice_junctions': 1.0}
        loss_fn = self._make_loss_with_fixed_per_head(
            per_head,
            head_weights={'atac': 1.0, 'splice_junctions': 1.0},
        )
        result = self._run(loss_fn, list(per_head.keys()))

        # total = (1.0 + 1.0) / 2 = 1.0
        assert torch.isclose(result['loss'], torch.tensor(1.0))

    def test_user_can_zero_head(self):
        """Setting weight to 0 drops the head's contribution but the divisor
        still counts it (we average the weighted losses)."""
        per_head = {'atac': 4.0, 'splice_junctions': 4.0}
        loss_fn = self._make_loss_with_fixed_per_head(
            per_head,
            head_weights={'atac': 1.0, 'splice_junctions': 0.0},
        )
        result = self._run(loss_fn, list(per_head.keys()))

        # total = (1.0 * 4 + 0.0 * 4) / 2 = 2.0
        assert torch.isclose(result['loss'], torch.tensor(2.0))

    def test_all_default_heads_aggregate(self):
        """All 11 default heads with unit per-head loss: total = sum(weights)/11."""
        head_names = list(DEFAULT_HEAD_WEIGHTS.keys())
        per_head = {h: 1.0 for h in head_names}
        loss_fn = self._make_loss_with_fixed_per_head(per_head)
        result = self._run(loss_fn, head_names)

        expected = sum(DEFAULT_HEAD_WEIGHTS.values()) / len(head_names)
        assert torch.isclose(result['loss'], torch.tensor(expected))

    def test_partial_override_preserves_other_defaults(self):
        """A partial head_weights dict overrides only the named head; the rest
        keep their defaults (splice_junctions stays at 0.2, not 1.0)."""
        per_head = {'atac': 1.0, 'splice_junctions': 1.0}
        loss_fn = self._make_loss_with_fixed_per_head(
            per_head,
            head_weights={'atac': 2.0},  # only atac specified
        )
        result = self._run(loss_fn, list(per_head.keys()))

        # atac override applied, splice_junctions still at its 0.2 default.
        assert loss_fn.head_weights['splice_junctions'] == 0.2
        # total = (2.0 * 1.0 + 0.2 * 1.0) / 2 = 1.1
        assert torch.isclose(result['loss'], torch.tensor(1.1))


@pytest.mark.unit
class TestNoConstantMutation:
    """The module-level DEFAULT_HEAD_WEIGHTS must never be aliased or mutated."""

    def test_instance_dict_is_a_copy(self):
        before = dict(DEFAULT_HEAD_WEIGHTS)
        loss_fn = AlphaGenomeLoss()  # head_weights=None path

        assert loss_fn.head_weights is not DEFAULT_HEAD_WEIGHTS
        loss_fn.head_weights['atac'] = 99.0  # mutate the instance copy
        assert DEFAULT_HEAD_WEIGHTS == before  # constant untouched
