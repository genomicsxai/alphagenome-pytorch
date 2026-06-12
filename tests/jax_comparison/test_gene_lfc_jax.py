"""JAX-PyTorch equivalence tests for gene LFC (cross-track) loss.

Verifies that `alphagenome_pytorch.losses.gene_lfc_loss` is numerically
equivalent to the upstream JAX `GenomeTracksHead._compute_cross_track_loss`
(google-deepmind/alphagenome_research, commit fd44ed0).

Pattern follows tests/jax_comparison/test_losses_jax.py: inline the JAX
math as a free helper (the upstream version is a method on a Haiku module
that pulls strand-channel mask and gene_cross_track_weight from `self`,
both of which we pass in here as plain arrays/floats), then compare
JAX-vs-PyTorch results to `decimal=5`.

Run: pytest tests/jax_comparison/test_gene_lfc_jax.py -v
"""

import numpy as np
import pytest

# Skip module if JAX is unavailable.
pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

import jax.numpy as jnp


@pytest.fixture
def jax_losses():
    from alphagenome_research.model import losses as jax_losses
    return jax_losses


@pytest.fixture
def torch_losses():
    from alphagenome_pytorch import losses as torch_losses
    return torch_losses


@pytest.fixture
def torch():
    import torch
    return torch


def _jax_gene_lfc(
    *,
    jax_losses,
    organism_index,
    predictions,
    targets,
    targets_mask,
    gene_mask,
    strand_channel_mask,
    gene_cross_track_weight,
):
    """Verbatim port of upstream `_compute_cross_track_loss` body.

    The only differences vs. the upstream method:
      - `strand_channel_mask` is passed in instead of read from
        `self._strand_channel_mask`. Shape `[num_organisms, 2, 1, C]`.
      - `gene_cross_track_weight` is a parameter instead of `self.`.
      - The chex rank check is omitted; not needed for a numerical test.

    Mirrors upstream's `_get_param_for_index` via simple advanced indexing.
    """
    gene_length = jnp.sum(gene_mask.astype(jnp.float32), axis=-3)
    safe_gene_length = jnp.maximum(gene_length[..., None], 1.0)

    y_true = (
        jnp.einsum('bsc,bs2g->b2gc', targets.astype(jnp.float32), gene_mask)
        / safe_gene_length
    )
    y_pred = (
        jnp.einsum('bsc,bs2g->b2gc', predictions.astype(jnp.float32), gene_mask)
        / safe_gene_length
    )

    batch_size, *_, num_channels = predictions.shape
    if targets_mask is not None:
        targets_mask = jnp.max(
            targets_mask.astype(jnp.float32), axis=-2, keepdims=True
        )
    else:
        targets_mask = jnp.ones((batch_size, 1, num_channels))
    combined_mask = (gene_length > 0)[..., None] * targets_mask.reshape(
        batch_size, 1, 1, num_channels
    )

    # Per-organism strand mask, broadcast to [B, 2, 1, C].
    strand_mask = jnp.asarray(strand_channel_mask)[(organism_index,)]
    combined_mask = (combined_mask * strand_mask).astype(bool)

    total_pred = jnp.einsum('b2gc,b2gc->b2g', y_pred, combined_mask)[..., None]
    total_true = jnp.einsum('b2gc,b2gc->b2g', y_true, combined_mask)[..., None]

    loss_total_count = jax_losses.poisson_loss(
        y_true=total_true,
        y_pred=total_pred,
        mask=combined_mask.any(axis=-1, keepdims=True),
    )
    num_active = jnp.sum(
        combined_mask.astype(jnp.float32), axis=-1, keepdims=True
    )
    loss_total_count = loss_total_count / jnp.maximum(jnp.max(num_active), 1.0)

    prob_predictions = y_pred.astype(jnp.float32) / (total_pred + 1e-7)
    loss_positional = -y_true * jnp.log(prob_predictions + 1e-7)
    # Upstream renamed `_safe_masked_mean` → `safe_masked_mean` in fd44ed0;
    # fall back to the private name for older checkouts.
    smm = getattr(
        jax_losses, "safe_masked_mean",
        getattr(jax_losses, "_safe_masked_mean", None),
    )
    loss_positional = smm(loss_positional, combined_mask)

    return loss_total_count + gene_cross_track_weight * loss_positional


def _make_inputs(B=2, S=64, C=4, G=3, num_organisms=1, seed=0):
    """Synthetic inputs for both implementations. Returns numpy arrays
    so each implementation does its own jnp / torch.tensor conversion.
    """
    rng = np.random.default_rng(seed)
    predictions = (rng.random((B, S, C)).astype(np.float32) + 0.5)
    targets = (rng.random((B, S, C)).astype(np.float32) + 0.5)
    targets_mask = np.ones((B, 1, C), dtype=bool)

    # Place a few genes inside the window. + strand on bucket 0,
    # - strand on bucket 1. G slots, some zero-padded.
    gene_mask = np.zeros((B, S, 2, G), dtype=bool)
    gene_mask[:, 8:24, 0, 0] = True   # + strand gene
    gene_mask[:, 30:50, 1, 1] = True  # - strand gene
    # gene index 2 left as zero-padding to exercise the gene_length=0 path.

    # `[num_organisms, 2, 1, C]` strand-channel mask. Tracks alternate +/-
    # for the first half, '.' for the rest.
    strands_per_org = []
    for _ in range(num_organisms):
        plus = np.array([(c % 2 == 0) or c >= C // 2 for c in range(C)])
        minus = np.array([(c % 2 == 1) or c >= C // 2 for c in range(C)])
        strands_per_org.append(np.stack([plus, minus])[:, None, :])
    strand_channel_mask = np.stack(strands_per_org)  # [num_orgs, 2, 1, C]

    organism_index = np.zeros((B,), dtype=np.int32)
    return predictions, targets, targets_mask, gene_mask, strand_channel_mask, organism_index


@pytest.mark.jax
class TestGeneLFCLossEquivalence:
    """Numerical parity between PyTorch `gene_lfc_loss` and a verbatim
    JAX port of upstream's `_compute_cross_track_loss`."""

    def test_random_inputs(self, jax_losses, torch_losses, torch):
        preds, targets, tmask, gmask, scm, oid = _make_inputs(seed=42)
        gctw = 5.0

        jax_loss = _jax_gene_lfc(
            jax_losses=jax_losses,
            organism_index=jnp.array(oid),
            predictions=jnp.array(preds),
            targets=jnp.array(targets),
            targets_mask=jnp.array(tmask),
            gene_mask=jnp.array(gmask),
            strand_channel_mask=jnp.array(scm),
            gene_cross_track_weight=gctw,
        )

        # PyTorch: strand_channel_mask is [2, 1, C] (no organism dim) since
        # all organism_index values point at the same row.
        torch_loss, _ = torch_losses.gene_lfc_loss(
            predictions=torch.tensor(preds),
            targets=torch.tensor(targets),
            targets_mask=torch.tensor(tmask),
            gene_mask=torch.tensor(gmask),
            strand_channel_mask=torch.tensor(scm[0]),
            gene_cross_track_weight=gctw,
        )
        np.testing.assert_almost_equal(
            float(jax_loss), torch_loss.item(), decimal=5
        )

    def test_with_partial_track_mask(self, jax_losses, torch_losses, torch):
        """Some tracks unavailable → both implementations must drop them
        identically when computing per-gene totals and tissue distribution."""
        preds, targets, _, gmask, scm, oid = _make_inputs(seed=7)
        # Mask out 2nd and 4th tracks (out of 4).
        tmask = np.array([[[True, False, True, False]]] * preds.shape[0], dtype=bool)
        gctw = 5.0

        jax_loss = _jax_gene_lfc(
            jax_losses=jax_losses,
            organism_index=jnp.array(oid),
            predictions=jnp.array(preds),
            targets=jnp.array(targets),
            targets_mask=jnp.array(tmask),
            gene_mask=jnp.array(gmask),
            strand_channel_mask=jnp.array(scm),
            gene_cross_track_weight=gctw,
        )
        torch_loss, _ = torch_losses.gene_lfc_loss(
            predictions=torch.tensor(preds),
            targets=torch.tensor(targets),
            targets_mask=torch.tensor(tmask),
            gene_mask=torch.tensor(gmask),
            strand_channel_mask=torch.tensor(scm[0]),
            gene_cross_track_weight=gctw,
        )
        np.testing.assert_almost_equal(
            float(jax_loss), torch_loss.item(), decimal=5
        )

    def test_no_track_mask_defaults_to_all_available(
        self, jax_losses, torch_losses, torch
    ):
        """JAX takes None and replaces with ones. Our PyTorch impl does the
        same. Match must hold."""
        preds, targets, _, gmask, scm, oid = _make_inputs(seed=11)
        gctw = 5.0

        jax_loss = _jax_gene_lfc(
            jax_losses=jax_losses,
            organism_index=jnp.array(oid),
            predictions=jnp.array(preds),
            targets=jnp.array(targets),
            targets_mask=None,
            gene_mask=jnp.array(gmask),
            strand_channel_mask=jnp.array(scm),
            gene_cross_track_weight=gctw,
        )
        torch_loss, _ = torch_losses.gene_lfc_loss(
            predictions=torch.tensor(preds),
            targets=torch.tensor(targets),
            targets_mask=None,
            gene_mask=torch.tensor(gmask),
            strand_channel_mask=torch.tensor(scm[0]),
            gene_cross_track_weight=gctw,
        )
        np.testing.assert_almost_equal(
            float(jax_loss), torch_loss.item(), decimal=5
        )

    def test_zero_padded_gene_columns_are_inert(
        self, jax_losses, torch_losses, torch
    ):
        """The trailing G_max - num_genes columns of gene_mask are all zero
        (i.e. gene_length=0 for those slots). Both implementations must
        produce the same loss whether G is sized tightly or generously."""
        preds, targets, tmask, gmask_tight, scm, oid = _make_inputs(G=2, seed=3)
        # Pad the gene axis with extra zero columns (G=2 → G=5).
        B, S = gmask_tight.shape[:2]
        gmask_padded = np.concatenate(
            [gmask_tight, np.zeros((B, S, 2, 3), dtype=bool)], axis=-1
        )
        assert gmask_padded.shape[-1] == 5
        gctw = 5.0

        for gmask_variant in (gmask_tight, gmask_padded):
            jax_loss = _jax_gene_lfc(
                jax_losses=jax_losses,
                organism_index=jnp.array(oid),
                predictions=jnp.array(preds),
                targets=jnp.array(targets),
                targets_mask=jnp.array(tmask),
                gene_mask=jnp.array(gmask_variant),
                strand_channel_mask=jnp.array(scm),
                gene_cross_track_weight=gctw,
            )
            torch_loss, _ = torch_losses.gene_lfc_loss(
                predictions=torch.tensor(preds),
                targets=torch.tensor(targets),
                targets_mask=torch.tensor(tmask),
                gene_mask=torch.tensor(gmask_variant),
                strand_channel_mask=torch.tensor(scm[0]),
                gene_cross_track_weight=gctw,
            )
            np.testing.assert_almost_equal(
                float(jax_loss), torch_loss.item(), decimal=5
            )

    def test_varying_gene_cross_track_weight(
        self, jax_losses, torch_losses, torch
    ):
        """Inner multinomial weight: paper uses 5.0 but the parity must hold
        for any value. Pin a few."""
        preds, targets, tmask, gmask, scm, oid = _make_inputs(seed=99)
        for gctw in (0.0, 1.0, 5.0, 10.0):
            jax_loss = _jax_gene_lfc(
                jax_losses=jax_losses,
                organism_index=jnp.array(oid),
                predictions=jnp.array(preds),
                targets=jnp.array(targets),
                targets_mask=jnp.array(tmask),
                gene_mask=jnp.array(gmask),
                strand_channel_mask=jnp.array(scm),
                gene_cross_track_weight=gctw,
            )
            torch_loss, _ = torch_losses.gene_lfc_loss(
                predictions=torch.tensor(preds),
                targets=torch.tensor(targets),
                targets_mask=torch.tensor(tmask),
                gene_mask=torch.tensor(gmask),
                strand_channel_mask=torch.tensor(scm[0]),
                gene_cross_track_weight=gctw,
            )
            np.testing.assert_almost_equal(
                float(jax_loss), torch_loss.item(), decimal=5,
                err_msg=f"mismatch at gene_cross_track_weight={gctw}",
            )
