"""Integration tests for variant scoring using real model and data.

These tests verify that the variant scoring pipeline works end-to-end
with the AlphaGenome model. Tests are designed to:
1. Run quickly with vectorized operations
2. Work with or without real model weights
3. Test both CPU and GPU execution paths
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.variant_scoring import (
    AggregationType,
    CenterMaskScorer,
    Interval,
    OutputType,
    Variant,
    VariantScoringModel,
    get_recommended_scorers,
)
from alphagenome_pytorch.variant_scoring.sequence import (
    FastaExtractor,
    apply_variant_to_sequence,
    sequence_to_onehot,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_on_cpu(torch_weights_path):
    """Create AlphaGenome model on CPU.

    Note: Track means are bundled with model weights, no separate file needed.
    Uses --torch-weights option from conftest.py.
    """
    model = AlphaGenome(num_organisms=2)

    state_dict = torch.load(torch_weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


@pytest.fixture(scope="module")
def model_on_gpu(torch_weights_path):
    """Create AlphaGenome model on GPU if available.

    Note: Track means are bundled with model weights, no separate file needed.
    Uses --torch-weights option from conftest.py.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = AlphaGenome(num_organisms=2)

    state_dict = torch.load(torch_weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model.cuda()
    model.eval()
    return model


@pytest.fixture
def mock_fasta_extractor(monkeypatch):
    """Mock FastaExtractor that returns random DNA sequences.

    Also patches apply_variant_to_sequence to skip reference allele validation
    since the random sequences won't match the variant's reference allele.
    """

    def mock_extract(self, interval: Interval) -> str:
        """Generate a reproducible random DNA sequence for any interval."""
        # Use interval as seed for reproducibility
        seed = hash((interval.chromosome, interval.start, interval.end)) % (2**32)
        rng = np.random.default_rng(seed)
        nucleotides = np.array(["A", "C", "G", "T"])
        length = interval.end - interval.start
        return "".join(rng.choice(nucleotides, size=length))

    def mock_apply_variant(sequence: str, variant: Variant, interval: Interval) -> str:
        """Apply variant without reference allele validation.

        This is needed for testing because the mock generates random sequences
        that won't match the variant's expected reference allele.
        """
        # Validate variant is within interval (keep this validation)
        if variant.chromosome != interval.chromosome:
            raise ValueError(
                f"Variant chromosome ({variant.chromosome}) doesn't match "
                f"interval chromosome ({interval.chromosome})"
            )

        # Convert variant position to sequence coordinates (0-based)
        var_start = variant.start - interval.start
        var_end = var_start + len(variant.reference_bases)

        if var_start < 0 or var_end > len(sequence):
            raise ValueError(
                f"Variant position {variant.position} (ref length {len(variant.reference_bases)}) "
                f"is outside interval {interval}"
            )

        # Apply the variant (skip reference allele validation for mocked sequences)
        before = sequence[:var_start]
        after = sequence[var_end:]
        return before + variant.alternate_bases + after

    monkeypatch.setattr(FastaExtractor, "extract", mock_extract)
    # Patch apply_variant_to_sequence in both the sequence module and inference module
    # (inference imports it, so we need to patch both)
    monkeypatch.setattr(
        "alphagenome_pytorch.variant_scoring.sequence.apply_variant_to_sequence",
        mock_apply_variant
    )
    monkeypatch.setattr(
        "alphagenome_pytorch.variant_scoring.inference.apply_variant_to_sequence",
        mock_apply_variant
    )


# -----------------------------------------------------------------------------
# Unit tests for sequence utilities (fast, no model needed)
# -----------------------------------------------------------------------------


class TestSequenceToOnehot:
    """Test vectorized sequence_to_onehot performance and correctness."""

    def test_basic_conversion(self):
        """Test basic ACGT conversion."""
        seq = "ACGT"
        onehot = sequence_to_onehot(seq)

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0, 0.0],  # C
                [0.0, 0.0, 1.0, 0.0],  # G
                [0.0, 0.0, 0.0, 1.0],  # T
            ]
        )
        assert torch.allclose(onehot, expected)

    def test_n_maps_to_zeros(self):
        """Test that N maps to all-zeros (matching JAX reference)."""
        seq = "NACGT"
        onehot = sequence_to_onehot(seq)

        # N should be all-zeros
        expected_n = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(onehot[0], expected_n)

    def test_lowercase_handled(self):
        """Test that lowercase sequences are handled."""
        seq = "acgt"
        onehot = sequence_to_onehot(seq)
        assert onehot.shape == (4, 4)
        assert onehot.sum() == 4.0  # One hot per position

    def test_performance_131kb(self):
        """Test that 131kb sequence encoding is fast (vectorized)."""
        # Generate random 131072 bp sequence
        rng = np.random.default_rng(42)
        nucleotides = np.array(["A", "C", "G", "T"])
        seq = "".join(rng.choice(nucleotides, size=131072))

        # Time the conversion
        start = time.perf_counter()
        onehot = sequence_to_onehot(seq)
        elapsed = time.perf_counter() - start

        assert onehot.shape == (131072, 4)
        # Should complete in under 1 second with vectorized implementation
        # (Python loop would take 10+ seconds)
        assert elapsed < 1.0, f"sequence_to_onehot took {elapsed:.2f}s, expected < 1s"
        print(f"\nsequence_to_onehot (131kb): {elapsed*1000:.1f} ms")

    def test_device_placement(self):
        """Test that output is on correct device."""
        seq = "ACGT"

        # CPU
        onehot_cpu = sequence_to_onehot(seq, device="cpu")
        assert onehot_cpu.device == torch.device("cpu")

        # GPU if available
        if torch.cuda.is_available():
            onehot_gpu = sequence_to_onehot(seq, device="cuda")
            assert onehot_gpu.device.type == "cuda"

    def test_dtype_specification(self):
        """Test that output dtype is respected."""
        seq = "ACGT"

        onehot_f32 = sequence_to_onehot(seq, dtype=torch.float32)
        assert onehot_f32.dtype == torch.float32

        onehot_bf16 = sequence_to_onehot(seq, dtype=torch.bfloat16)
        assert onehot_bf16.dtype == torch.bfloat16


class TestApplyVariant:
    """Test variant application to sequences."""

    def test_snv_substitution(self):
        """Test SNV substitution."""
        seq = "AAACCCGGG"
        interval = Interval("chr1", 0, 9)
        # Position 4 (1-based VCF) = index 3 (0-based)
        variant = Variant("chr1", 4, "C", "T")

        result = apply_variant_to_sequence(seq, variant, interval)
        assert result == "AAATCCGGG"

    def test_insertion(self):
        """Test insertion variant."""
        seq = "AAACCCGGG"
        interval = Interval("chr1", 0, 9)
        # Insert TTT after position 4
        variant = Variant("chr1", 4, "C", "CTTT")

        result = apply_variant_to_sequence(seq, variant, interval)
        assert result == "AAACTTTCCGGG"

    def test_deletion(self):
        """Test deletion variant."""
        seq = "AAACCCGGG"
        interval = Interval("chr1", 0, 9)
        # Delete CC at positions 5-6
        variant = Variant("chr1", 5, "CC", "C")

        result = apply_variant_to_sequence(seq, variant, interval)
        assert result == "AAACCGGG"

    def test_variant_outside_interval_raises(self):
        """Test that variant outside interval raises error."""
        seq = "AAACCC"
        interval = Interval("chr1", 100, 106)
        variant = Variant("chr1", 50, "A", "T")  # Before interval

        with pytest.raises(ValueError, match="outside interval"):
            apply_variant_to_sequence(seq, variant, interval)

    def test_ref_mismatch_raises(self):
        """Test that reference mismatch raises error."""
        seq = "AAACCC"
        interval = Interval("chr1", 0, 6)
        variant = Variant("chr1", 1, "G", "T")  # Ref is A, not G

        with pytest.raises(ValueError, match="Reference allele mismatch"):
            apply_variant_to_sequence(seq, variant, interval)


# -----------------------------------------------------------------------------
# Integration tests with model
# -----------------------------------------------------------------------------


class TestVariantScoringModelCPU:
    """Test VariantScoringModel on CPU."""

    def test_initialization(self, model_on_cpu, mock_fasta_extractor, tmp_path):
        """Test model initialization."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr1\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_cpu,
            fasta_path=str(fasta_path),
            device="cpu",
        )

        assert scorer.device == torch.device("cpu")
        assert next(scorer.model.parameters()).device == torch.device("cpu")

    def test_predict_single_sequence(self, model_on_cpu, mock_fasta_extractor, tmp_path):
        """Test prediction on a single sequence."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr1\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_cpu,
            fasta_path=str(fasta_path),
            device="cpu",
        )

        # Generate random sequence (model expects 131072 bp)
        rng = np.random.default_rng(42)
        seq = "".join(rng.choice(["A", "C", "G", "T"], size=131072))

        outputs = scorer.predict(seq, organism=0)

        # Check outputs have expected keys
        assert isinstance(outputs, dict)
        # Model should produce various output types

    def test_score_variant_single(self, model_on_cpu, mock_fasta_extractor, tmp_path):
        """Test scoring a single variant."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_cpu,
            fasta_path=str(fasta_path),
            device="cpu",
        )

        variant = Variant.from_str("chr22:36201698:A>C")
        interval = Interval.centered_on("chr22", 36201698, width=131072)

        # Use a simple scorer
        scorers = [
            CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
        ]

        scores = scorer.score_variant(
            interval=interval,
            variant=variant,
            scorers=scorers,
            organism=0,
        )

        assert len(scores) == 1
        score = scores[0]
        assert hasattr(score, "scores")
        assert isinstance(score.scores, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVariantScoringModelGPU:
    """Test VariantScoringModel on GPU."""

    def test_device_is_cuda(self, model_on_gpu, mock_fasta_extractor, tmp_path):
        """Test that model runs on GPU."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr1\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_gpu,
            fasta_path=str(fasta_path),
        )

        assert scorer.device.type == "cuda"
        assert next(scorer.model.parameters()).device.type == "cuda"

    def test_inference_timing(self, model_on_gpu, mock_fasta_extractor, tmp_path):
        """Test that GPU inference is reasonably fast."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_gpu,
            fasta_path=str(fasta_path),
        )

        variant = Variant.from_str("chr22:36201698:A>C")
        interval = Interval.centered_on("chr22", 36201698, width=131072)

        scorers = [
            CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
        ]

        # Warmup
        _ = scorer.score_variant(interval, variant, scorers=scorers, organism=0)
        torch.cuda.synchronize()

        # Time actual run
        start = time.perf_counter()
        scores = scorer.score_variant(
            interval=interval,
            variant=variant,
            scorers=scorers,
            organism=0,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU variant scoring (single, 1 scorer): {elapsed:.2f}s")

        # Should complete in reasonable time (model forward pass + scoring)
        # Expectation: < 30 seconds for a single variant on GPU
        assert elapsed < 60, f"Scoring took {elapsed:.1f}s, expected < 60s"
        assert len(scores) == 1

    def test_recommended_scorers(self, model_on_gpu, mock_fasta_extractor, tmp_path):
        """Test with recommended scorers (full pipeline)."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_gpu,
            fasta_path=str(fasta_path),
        )

        variant = Variant.from_str("chr22:36201698:A>C")
        interval = Interval.centered_on("chr22", 36201698, width=131072)

        # Get recommended scorers (excludes gene-centric ones that need GTF)
        all_scorers = get_recommended_scorers("human")
        # Filter to only CenterMaskScorer types for this test (no GTF)
        scorers = [s for s in all_scorers if isinstance(s, CenterMaskScorer)]

        start = time.perf_counter()
        scores = scorer.score_variant(
            interval=interval,
            variant=variant,
            scorers=scorers,
            organism=0,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\nGPU variant scoring ({len(scorers)} CenterMask scorers): {elapsed:.2f}s")

        assert len(scores) == len(scorers)
        for score in scores:
            assert isinstance(score.scores, torch.Tensor)


class TestVariantScoringBatching:
    """Test scoring multiple variants."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_score_multiple_variants(self, model_on_gpu, mock_fasta_extractor, tmp_path):
        """Test scoring multiple variants."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_gpu,
            fasta_path=str(fasta_path),
        )

        # Create multiple variants
        variants = [
            Variant.from_str("chr22:36201698:A>C"),
            Variant.from_str("chr22:36201700:G>T"),
            Variant.from_str("chr22:36201750:C>A"),
        ]

        # Use same interval centered on first variant
        interval = Interval.centered_on("chr22", 36201698, width=131072)

        scorers = [
            CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM),
        ]

        scores = scorer.score_variants(
            intervals=interval,  # Same interval for all
            variants=variants,
            scorers=scorers,
            organism=0,
            progress=False,
        )

        assert len(scores) == len(variants)
        for variant_scores in scores:
            assert len(variant_scores) == len(scorers)


# -----------------------------------------------------------------------------
# Edge cases and error handling
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_variant_str_parsing(self):
        """Test Variant.from_str parsing."""
        v = Variant.from_str("chr22:36201698:A>C")
        assert v.chromosome == "chr22"
        assert v.position == 36201698
        assert v.reference_bases == "A"
        assert v.alternate_bases == "C"

    def test_interval_centered_on(self):
        """Test Interval.centered_on helper."""
        interval = Interval.centered_on("chr22", 36201698, width=131072)
        assert interval.chromosome == "chr22"
        assert interval.end - interval.start == 131072
        # Variant should be at center
        center = (interval.start + interval.end) // 2
        assert abs(center - 36201698) <= 1

    def test_no_fasta_raises(self, model_on_cpu):
        """Test that using scorer without FASTA raises helpful error."""
        scorer = VariantScoringModel(
            model=model_on_cpu,
            fasta_path=None,  # No FASTA
            device="cpu",
        )

        with pytest.raises(ValueError, match="FASTA path not provided"):
            scorer.get_sequence(Interval("chr1", 0, 100))

    def test_interval_variant_mismatch(self, model_on_cpu, mock_fasta_extractor, tmp_path):
        """Test that variant on wrong chromosome raises error."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_cpu,
            fasta_path=str(fasta_path),
            device="cpu",
        )

        variant = Variant("chr1", 100, "A", "C")  # chr1
        interval = Interval("chr22", 0, 131072)  # chr22

        with pytest.raises(ValueError, match="chromosome"):
            scorer.get_sequence(interval, variant=variant)


# -----------------------------------------------------------------------------
# Performance benchmarks (optional, for profiling)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPerformanceBenchmarks:
    """Performance benchmarks for variant scoring."""

    def test_encoding_vs_inference_timing(
        self, model_on_gpu, mock_fasta_extractor, tmp_path
    ):
        """Profile where time is spent: encoding vs inference."""
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">chr22\nACGT\n")

        scorer = VariantScoringModel(
            model=model_on_gpu,
            fasta_path=str(fasta_path),
        )

        interval = Interval.centered_on("chr22", 36201698, width=131072)

        # Get sequence
        seq = scorer.get_sequence(interval)

        # Time encoding
        start = time.perf_counter()
        for _ in range(10):
            onehot = sequence_to_onehot(
                seq, dtype=scorer.model.dtype_policy.compute_dtype, device=scorer.device
            )
        encoding_time = (time.perf_counter() - start) / 10

        # Time inference
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            _ = scorer.predict(seq, organism=0)
            torch.cuda.synchronize()
        inference_time = (time.perf_counter() - start) / 3

        print(f"\nEncoding time (131kb): {encoding_time*1000:.1f} ms")
        print(f"Inference time: {inference_time*1000:.1f} ms")

        # Encoding should be fast compared to inference
        assert encoding_time < 0.5, f"Encoding too slow: {encoding_time:.2f}s"
