"""Unit tests for the shared sequence-length validator.

``validate_sequence_length`` in ``alphagenome_pytorch.model`` is the single
source of truth for which input lengths the model can process; the serving layer
and the predict CLI both delegate to it.

The binding constraint is a multiple of 2048 (not merely 128): the transformer's
pair path pools the 128 bp trunk by 16 and expands the attention bias back by 16,
so a length that is a multiple of 128 but not of 2048 crashes the forward pass.
``test_tower_forward_matches_validator`` pins that reproduction so the validator
rule and the model geometry cannot drift apart.
"""

from __future__ import annotations

import pytest

from alphagenome_pytorch.model import (
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    SEQUENCE_LENGTH_BIN,
    SEQUENCE_LENGTH_MULTIPLE,
    validate_sequence_length,
)


class TestValidateSequenceLength:
    @pytest.mark.parametrize(
        "length",
        [
            2048,  # MIN — one pair bin (the shortest structurally valid window)
            4096,  # documented CPU smoke-test window
            16384,  # standard 16KB
            131072,  # standard 128KB (default)
            524288,  # standard 512KB
            1048576,  # MAX — standard 1MB
        ],
    )
    def test_accepts_multiples_of_2048_in_range(self, length: int) -> None:
        validate_sequence_length(length)  # must not raise

    @pytest.mark.parametrize(
        "length",
        [
            128,  # a multiple of 128 but not 2048 — the crash case
            1024,  # a multiple of 128 but not 2048 — the crash case
            2000,  # not a multiple of 128 at all
            2049,  # 2048 + 1
            4095,  # just below a valid multiple
            127,  # below MIN and not a multiple
            0,  # zero
            -2048,  # negative
            MAX_SEQUENCE_LENGTH + SEQUENCE_LENGTH_MULTIPLE,  # above MAX
        ],
    )
    def test_rejects_invalid_lengths(self, length: int) -> None:
        with pytest.raises(ValueError, match="not supported"):
            validate_sequence_length(length)

    def test_error_message_names_the_rule(self) -> None:
        with pytest.raises(ValueError) as exc:
            validate_sequence_length(2000)
        msg = str(exc.value)
        assert str(SEQUENCE_LENGTH_MULTIPLE) in msg
        assert str(MIN_SEQUENCE_LENGTH) in msg
        assert str(MAX_SEQUENCE_LENGTH) in msg

    def test_constants(self) -> None:
        assert SEQUENCE_LENGTH_BIN == 128
        assert SEQUENCE_LENGTH_MULTIPLE == 2048
        assert MIN_SEQUENCE_LENGTH == 2048
        assert MAX_SEQUENCE_LENGTH == 2 ** 20


class TestServingDelegatesToCore:
    def test_serving_alias_is_core_function(self) -> None:
        from alphagenome_pytorch.extensions.serving.adapter import (
            _validate_sequence_length,
        )

        assert _validate_sequence_length is validate_sequence_length

    def test_standard_lengths_all_valid_under_new_rule(self) -> None:
        from alphagenome_pytorch.extensions.serving.adapter import (
            SUPPORTED_SEQUENCE_LENGTHS,
        )

        for value in SUPPORTED_SEQUENCE_LENGTHS.values():
            validate_sequence_length(value)  # must not raise


class TestValidatorMatchesModelGeometry:
    """The validator must accept exactly the trunk lengths the tower can run.

    This is the regression guard for the reviewer's finding: the tower crashes
    for a trunk length that is not a multiple of 16 (input not a multiple of
    2048), so the validator rejecting non-2048 multiples is not a stylistic
    choice — it prevents a hard crash in the attention forward pass.
    """

    @pytest.mark.parametrize(
        "input_length,should_run",
        [
            (128, False),  # trunk S=1  -> crashes in the value matmul
            (1024, False),  # trunk S=8  -> crashes in the attention-bias add
            (2048, True),  # trunk S=16 -> valid
            (4096, True),  # trunk S=32 -> valid
        ],
    )
    def test_tower_forward_matches_validator(
        self, input_length: int, should_run: bool
    ) -> None:
        import torch

        from alphagenome_pytorch.model import TransformerTower

        # The validator's verdict must agree with whether the tower can run.
        validator_accepts = True
        try:
            validate_sequence_length(input_length)
        except ValueError:
            validator_accepts = False
        assert validator_accepts == should_run

        d_model = 1536
        tower = TransformerTower(d_model).eval()
        trunk = torch.randn(1, input_length // SEQUENCE_LENGTH_BIN, d_model)
        if should_run:
            with torch.no_grad():
                out, _ = tower(trunk)
            assert out.shape == (1, input_length // SEQUENCE_LENGTH_BIN, d_model)
        else:
            with pytest.raises(RuntimeError):
                with torch.no_grad():
                    tower(trunk)
