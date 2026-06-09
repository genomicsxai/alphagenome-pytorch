"""Unit tests for variant-score quantile calibration."""

import numpy as np
import pandas as pd
import pytest
import torch

from alphagenome_pytorch.variant_scoring import (
    AggregationType,
    Calibration,
    CenterMaskScorer,
    ContactMapScorer,
    GeneMaskActiveScorer,
    GeneMaskLFCScorer,
    GeneMaskSplicingScorer,
    OutputType,
    PolyadenylationScorer,
    ScorerCalibration,
    SpliceJunctionScorer,
    VariantScore,
    VariantScoringModel,
    scores_to_anndata,
    scores_to_dataframe,
    tidy_scores,
)
from alphagenome_pytorch.variant_scoring.types import Interval, Variant

_KEY = "FakeScorer()"


def _make_calibration(seed: int = 42) -> Calibration:
    """Two-track calibration: track 0 unique breakpoints, track 1 with a tie plateau."""
    probabilities = np.linspace(-1.0, 1.0, 5).astype(np.float32)  # [-1,-.5,0,.5,1]
    quantiles = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],  # strictly increasing
            [0.0, 5.0, 5.0, 5.0, 9.0],  # duplicated plateau at 5.0
        ],
        dtype=np.float32,
    )
    dup_mask = np.any(np.diff(quantiles, axis=1) == 0, axis=1)
    sc = ScorerCalibration(
        quantiles=quantiles,
        probabilities=probabilities,
        dup_mask=dup_mask,
        is_signed=True,
        output_type="atac",
    )
    return Calibration({_KEY: sc}, seed=seed)


# --------------------------------------------------------------------------- core

@pytest.mark.unit
def test_has_scorer_and_keys():
    cal = _make_calibration()
    assert cal.has_scorer(_KEY)
    assert not cal.has_scorer("missing")
    assert not cal.has_scorer(None)
    assert cal.scorer_keys() == [_KEY]
    assert len(cal) == 1


@pytest.mark.unit
def test_quantile_transform_monotone_and_bounds():
    cal = _make_calibration()
    # Track 0: searchsorted(left) maps value -> probability bin.
    raw = np.array([-100.0, 0.0, 1.5, 4.0, 100.0, 0.0], dtype=np.float32)[:2]
    # Use a 2-track input (only track 0 and 1 calibrated).
    q = cal.quantile_scores(_KEY, np.array([1.5, 0.0], dtype=np.float32))
    # track0: 1.5 -> searchsorted([0,1,2,3,4],1.5,'left')=2 -> prob[2]=0.0
    assert q[0] == pytest.approx(0.0)

    # Below min -> idx 0 -> prob[0] = -1; above max -> clamp -> prob[-1] = +1
    lo = cal.quantile_scores(_KEY, np.array([-100.0, -100.0], dtype=np.float32))
    hi = cal.quantile_scores(_KEY, np.array([100.0, 100.0], dtype=np.float32))
    assert lo[0] == pytest.approx(-1.0)
    assert hi[0] == pytest.approx(1.0)

    # Monotone nondecreasing on track 0.
    vals = np.linspace(-1, 5, 40, dtype=np.float32)
    qs = [float(cal.quantile_scores(_KEY, np.array([v, 0.0], np.float32))[0]) for v in vals]
    assert all(b >= a - 1e-6 for a, b in zip(qs, qs[1:]))


@pytest.mark.unit
def test_uncalibrated_tracks_and_nan_passthrough():
    cal = _make_calibration()
    # Input longer than the 2 calibrated tracks: extras must be NaN.
    raw = np.array([1.5, 0.0, 3.3, 7.7], dtype=np.float32)
    q = cal.quantile_scores(_KEY, raw)
    assert not np.isnan(q[0]) and not np.isnan(q[1])
    assert np.isnan(q[2]) and np.isnan(q[3])

    # NaN input -> NaN output, calibrated neighbours unaffected.
    q2 = cal.quantile_scores(_KEY, np.array([np.nan, 0.0], dtype=np.float32))
    assert np.isnan(q2[0])
    assert not np.isnan(q2[1])


@pytest.mark.unit
def test_tie_break_is_seeded_and_within_plateau():
    # Track 1 has a plateau [5,5,5]; value 5.0 -> left=1, right=4 -> idx in {1,2,3,4}.
    cal_a = _make_calibration(seed=7)
    cal_b = _make_calibration(seed=7)
    cal_c = _make_calibration(seed=8)
    val = np.array([0.0, 5.0], dtype=np.float32)
    qa = float(cal_a.quantile_scores(_KEY, val)[1])
    qb = float(cal_b.quantile_scores(_KEY, val)[1])
    assert qa == qb  # same seed -> deterministic
    probs = cal_a._scorers[_KEY].probabilities
    assert qa in set(float(p) for p in probs[1:5])  # within the tied run
    # Per-call seed override also works and is reproducible.
    q_seeded = float(cal_c.quantile_scores(_KEY, val, seed=7)[1])
    assert q_seeded == qa


@pytest.mark.unit
def test_break_quantile_ties_false_is_deterministic():
    # Track 1 plateau [5,5,5]; value 5.0. With ties off -> always the left edge
    # (searchsorted side='left' -> idx 1 -> prob[1]), regardless of seed.
    val = np.array([0.0, 5.0], dtype=np.float32)
    cal = _make_calibration(seed=7)
    left_prob = float(cal._scorers[_KEY].probabilities[1])
    q1 = float(cal.quantile_scores(_KEY, val, break_quantile_ties=False, seed=1)[1])
    q2 = float(cal.quantile_scores(_KEY, val, break_quantile_ties=False, seed=999)[1])
    assert q1 == q2 == left_prob
    # Non-tied track 0 is unaffected by the flag.
    on = float(cal.quantile_scores(_KEY, val, break_quantile_ties=True, seed=1)[0])
    off = float(cal.quantile_scores(_KEY, val, break_quantile_ties=False)[0])
    assert on == off


@pytest.mark.unit
def test_input_type_preserved():
    cal = _make_calibration()
    raw_np = np.array([1.5, 0.0], dtype=np.float32)
    out_np = cal.quantile_scores(_KEY, raw_np)
    assert isinstance(out_np, np.ndarray) and out_np.shape == (2,)

    out_t = cal.quantile_scores(_KEY, torch.tensor([1.5, 0.0]))
    assert torch.is_tensor(out_t) and out_t.shape == (2,)


@pytest.mark.unit
def test_unknown_scorer_raises():
    cal = _make_calibration()
    with pytest.raises(KeyError):
        cal.quantile_scores("nope", np.zeros(2, dtype=np.float32))


# ----------------------------------------------------------------------- parquet

@pytest.mark.unit
def test_parquet_round_trip(tmp_path):
    cal = _make_calibration()
    sc = cal._scorers[_KEY]
    rows = [{
        "scorer_key": _KEY, "output_type": "atac", "row_type": "probabilities",
        "track_index": -1, "is_signed": True, "values": sc.probabilities,
    }]
    for i in range(sc.num_tracks):
        rows.append({
            "scorer_key": _KEY, "output_type": "atac", "row_type": "quantiles",
            "track_index": i, "is_signed": True, "values": sc.quantiles[i],
        })
    path = tmp_path / "calib.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    loaded = Calibration.load(path)
    assert loaded.has_scorer(_KEY)
    lsc = loaded._scorers[_KEY]
    np.testing.assert_array_equal(lsc.quantiles, sc.quantiles)
    np.testing.assert_array_equal(lsc.probabilities, sc.probabilities)
    np.testing.assert_array_equal(lsc.dup_mask, sc.dup_mask)
    assert lsc.is_signed and lsc.output_type == "atac"


# --------------------------------------------------------------- bundled package

# All recommended scorer configurations and the output types they cover.
_RECOMMENDED = [
    CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
    CenterMaskScorer(OutputType.ATAC, 501, AggregationType.ACTIVE_SUM),
    CenterMaskScorer(OutputType.DNASE, 501, AggregationType.DIFF_LOG2_SUM),
    CenterMaskScorer(OutputType.CAGE, 501, AggregationType.DIFF_LOG2_SUM),
    CenterMaskScorer(OutputType.PROCAP, 501, AggregationType.DIFF_LOG2_SUM),
    CenterMaskScorer(OutputType.CHIP_TF, 501, AggregationType.DIFF_LOG2_SUM),
    CenterMaskScorer(OutputType.CHIP_HISTONE, 2001, AggregationType.DIFF_LOG2_SUM),
    GeneMaskLFCScorer(OutputType.RNA_SEQ),
    GeneMaskActiveScorer(OutputType.RNA_SEQ),
    GeneMaskSplicingScorer(OutputType.SPLICE_SITES, None),
    GeneMaskSplicingScorer(OutputType.SPLICE_SITE_USAGE, None),
    SpliceJunctionScorer(),
    ContactMapScorer(),
    PolyadenylationScorer(),
]


@pytest.mark.unit
def test_bundled_calibration_loads():
    cal = Calibration.from_package("human")
    assert len(cal) == 19
    atac = cal._scorers[
        "CenterMaskScorer(requested_output=ATAC, width=501, aggregation_type=DIFF_LOG2_SUM)"
    ]
    assert atac.quantiles.shape == (167, 999)
    assert atac.probabilities.shape == (999,)
    assert atac.is_signed


@pytest.mark.unit
def test_recommended_scorer_keys_present_in_bundle():
    cal = Calibration.from_package("human")
    bundled = set(cal.scorer_keys())
    for scorer in _RECOMMENDED:
        assert scorer.calibration_key in bundled, scorer.calibration_key


@pytest.mark.unit
def test_bundled_quantiles_in_expected_range():
    cal = Calibration.from_package("human")
    key = "GeneMaskLFCScorer(requested_output=RNA_SEQ)"
    sc = cal._scorers[key]
    # rna_seq head has 768 tracks; only the first 667 are calibrated.
    raw = torch.zeros(768)
    raw[0] = 1e6
    raw[1] = -1e6
    q = cal.quantile_scores(key, raw)
    # Extremes map to the largest/smallest quantile probabilities (~+/-1).
    assert float(q[0]) == pytest.approx(float(sc.probabilities.max()))
    assert float(q[1]) == pytest.approx(float(sc.probabilities.min()))
    assert float(q[0]) > 0.99 and float(q[1]) < -0.99
    assert torch.isnan(q[700])  # uncalibrated track


# ------------------------------------------------------------------- wire-through

def _variant_score(scores, quantile_scores=None):
    scorer = CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)
    return VariantScore(
        variant=Variant("chr22", 100, "A", "C"),
        interval=Interval("chr22", 0, 200),
        scorer=scorer,
        scores=torch.as_tensor(scores, dtype=torch.float32),
        quantile_scores=(
            None if quantile_scores is None
            else torch.as_tensor(quantile_scores, dtype=torch.float32)
        ),
    )


@pytest.mark.unit
def test_tidy_scores_emits_quantile_column_only_when_present():
    with_q = tidy_scores([_variant_score([1.0, 2.0], [0.3, -0.7])])
    assert "quantile_score" in with_q.columns
    assert with_q["quantile_score"].tolist() == pytest.approx([0.3, -0.7])
    # quantile_score column sits next to raw_score.
    cols = list(with_q.columns)
    assert cols.index("quantile_score") == cols.index("raw_score") + 1

    without_q = tidy_scores([_variant_score([1.0, 2.0])])
    assert "quantile_score" not in without_q.columns


@pytest.mark.unit
def test_scores_to_dataframe_emits_quantile_column():
    df = scores_to_dataframe([_variant_score([1.0, 2.0], [0.3, -0.7])])
    assert df["quantile_score"].tolist() == pytest.approx([0.3, -0.7])
    df_none = scores_to_dataframe([_variant_score([1.0, 2.0])])
    assert "quantile_score" not in df_none.columns


@pytest.mark.unit
def test_scores_to_anndata_quantile_layer():
    adata = scores_to_anndata([_variant_score([1.0, 2.0], [0.3, -0.7])])
    assert "quantiles" in adata.layers
    np.testing.assert_allclose(adata.layers["quantiles"][0], [0.3, -0.7], atol=1e-6)
    adata_none = scores_to_anndata([_variant_score([1.0, 2.0])])
    assert "quantiles" not in adata_none.layers


# --------------------------------------------------------------- model integration

class _DummyModel(torch.nn.Module):
    num_organisms = 2


class _FakeScorer:
    """Duck-typed scorer whose calibration_key matches a stub calibration."""

    name = "FakeScorer()"
    calibration_key = _KEY
    required_heads = frozenset({"atac"})

    def score(self, ref_outputs, alt_outputs, variant, interval, organism_index, **kwargs):
        return VariantScore(
            variant=variant, interval=interval, scorer=self,
            scores=alt_outputs["atac"] - ref_outputs["atac"],
        )


@pytest.mark.unit
def test_score_variant_attaches_quantiles(monkeypatch):
    cal = _make_calibration()
    model = VariantScoringModel(_DummyModel(), calibration=cal)
    assert model.calibration is cal

    interval = Interval("chr22", 0, 8)
    variant = Variant("chr22", 4, "A", "C")

    def fake_predict(self, *a, **k):
        ref = {"atac": torch.tensor([0.0, 0.0])}
        alt = {"atac": torch.tensor([1.5, 5.0])}  # -> track0 bin, track1 plateau
        return ref, alt

    monkeypatch.setattr(VariantScoringModel, "predict_variant", fake_predict)

    [result] = model.score_variant(interval, variant, scorers=[_FakeScorer()])
    assert result.quantile_scores is not None
    assert float(result.quantile_scores[0]) == pytest.approx(0.0)  # 1.5 -> prob bin 2


@pytest.mark.unit
def test_calibration_none_leaves_quantiles_unset(monkeypatch):
    model = VariantScoringModel(_DummyModel())  # no calibration
    assert model.calibration is None

    def fake_predict(self, *a, **k):
        return {"atac": torch.tensor([0.0, 0.0])}, {"atac": torch.tensor([1.5, 5.0])}

    monkeypatch.setattr(VariantScoringModel, "predict_variant", fake_predict)
    [result] = model.score_variant(
        Interval("chr22", 0, 8), Variant("chr22", 4, "A", "C"), scorers=[_FakeScorer()]
    )
    assert result.quantile_scores is None
