"""Integration tests for AlphaGenome forward on varied sequence lengths."""

import math

import pytest
import torch

from alphagenome_pytorch import AlphaGenome

def _random_dna_onehot(length: int, *, batch_size: int = 1, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 4, (batch_size, length), generator=generator)
    return torch.nn.functional.one_hot(indices, num_classes=4).to(torch.float32)


def _assert_default_head_shapes(
    outputs,
    *,
    case: str,
    sequence_length: int,
    expected_128bp_tokens: int,
    resolutions,
):
    assert "atac" in outputs, f"{case}: missing default genome track head"
    assert set(outputs["atac"]) == set(resolutions), f"{case}: unexpected resolutions for atac"

    for resolution in resolutions:
        expected_tokens = sequence_length if resolution == 1 else expected_128bp_tokens
        assert outputs["atac"][resolution].shape == (1, expected_tokens, 256), (
            f"{case}: atac@{resolution} returned shape {tuple(outputs['atac'][resolution].shape)}"
        )


@pytest.mark.integration
class TestSequenceLengthForward:
    """Covers forward() across sequence lengths that stress pooling boundaries."""

    @pytest.fixture(scope="class")
    def model(self):
        model = AlphaGenome()
        model.eval()
        return model

    @pytest.mark.parametrize(
        ("sequence_length", "resolutions", "case"),
        [
            (4097, (128,), "indivisible_by_128_embed_128"),
            pytest.param(
                4097,
                (1, 128),
                "indivisible_by_128_embed_1",
                marks=pytest.mark.xfail(
                    raises=RuntimeError,
                    reason="1bp decoder path currently mismatches odd-length skip connections",
                ),
            ),
            (4098, (128,), "even_indivisible_by_128_embed_128"),
            pytest.param(
                4098,
                (1, 128),
                "even_indivisible_by_128_embed_1",
                marks=pytest.mark.xfail(
                    raises=RuntimeError,
                    reason="1bp decoder path currently mismatches skip connections for some non-128-divisible lengths",
                ),
            ),
            (4224, (128,), "128bp_tokens_not_divisible_by_16_embed_128"),
            (4224, (1, 128), "128bp_tokens_not_divisible_by_16_embed_1"),
            (4351, (128,), "indivisible_by_128_embed_128_plus_one"),
            pytest.param(
                4351,
                (1, 128),
                "indivisible_by_128_embed_1_plus_one",
                marks=pytest.mark.xfail(
                    raises=RuntimeError,
                    reason="1bp decoder path currently mismatches odd-length skip connections",
                ),
            ),
            (4350, (128,), "even_indivisible_by_128_embed_128_minus_two"),
            pytest.param(
                4350,
                (1, 128),
                "even_indivisible_by_128_embed_1_minus_two",
                marks=pytest.mark.xfail(
                    raises=RuntimeError,
                    reason="1bp decoder path currently mismatches skip connections for some non-128-divisible lengths",
                ),
            ),
            (4352, (128,), "128bp_tokens_not_divisible_by_16_embed_128_exact"),
            (4352, (1, 128), "128bp_tokens_not_divisible_by_16_embed_1_exact"),
        ],
    )
    def test_forward_supports_varied_sequence_lengths(self, model, sequence_length, resolutions, case):
        """Forward should succeed for lengths that are awkward for pooling math."""
        dna_sequence = _random_dna_onehot(sequence_length, seed=sequence_length)
        organism_index = torch.tensor([1], dtype=torch.long)
        expected_128bp_tokens = math.ceil(sequence_length / 128)
        expected_pair_tokens = math.ceil(expected_128bp_tokens / 16)
        expect_1bp_embeddings = 1 in resolutions

        with torch.inference_mode():
            outputs = model(
                dna_sequence,
                organism_index,
                resolutions=resolutions,
                return_embeddings=True,
            )

        assert outputs["embeddings_128bp"].shape == (1, expected_128bp_tokens, 3072), case

        if expect_1bp_embeddings:
            assert outputs["embeddings_1bp"].shape == (1, sequence_length, 1536), case
        else:
            assert "embeddings_1bp" not in outputs, case

        _assert_default_head_shapes(
            outputs,
            case=case,
            sequence_length=sequence_length,
            expected_128bp_tokens=expected_128bp_tokens,
            resolutions=resolutions,
        )

        assert outputs["contact_maps"].shape == (
            1,
            expected_pair_tokens,
            expected_pair_tokens,
            28,
        ), f"{case}: unexpected contact_maps shape"

        if expect_1bp_embeddings:
            classification = outputs["splice_sites"]
            assert classification["logits"].shape == (1, sequence_length, 5), case
            assert classification["probs"].shape == (1, sequence_length, 5), case

            usage = outputs["splice_site_usage"]
            assert usage["logits"].shape == (1, sequence_length, 734), case
            assert usage["predictions"].shape == (1, sequence_length, 734), case
            assert usage["track_mask"].shape == (1, 1, 734), case

            junction = outputs["splice_junctions"]
            assert junction["pred_counts"].shape == (1, 512, 512, 734), case
            assert junction["splice_site_positions"].shape == (1, 4, 512), case
            assert junction["splice_junction_mask"].shape == (1, 512, 512, 734), case
        else:
            assert "splice_sites" not in outputs, case
            assert "splice_site_usage" not in outputs, case
            assert "splice_junctions" not in outputs, case
