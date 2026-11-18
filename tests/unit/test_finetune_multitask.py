"""Unit tests for multimodal support in the finetuning entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from alphagenome_pytorch.extensions.finetuning import runner as finetune_module
from alphagenome_pytorch.extensions.finetuning.args import parse_args
from alphagenome_pytorch.extensions.finetuning.runner import (
    MultimodalDataset,
    collate_multimodal,
    format_epoch_summary,
    load_track_metadata_for_finetune,
    split_val_metrics,
    track_names_for_metric,
    unwrap_training_model,
)
from alphagenome_pytorch.extensions.finetuning.training import (
    _compute_multinomial_resolution,
)


def _required_cli_args() -> list[str]:
    """Return the minimal required CLI args for parse_args tests."""
    return [
        "finetune.py",
        "--genome",
        "hg38.fa",
        "--train-bed",
        "train.bed",
        "--val-bed",
        "val.bed",
        "--pretrained-weights",
        "model.pth",
    ]


@pytest.mark.unit
class TestComputeMultinomialResolution:
    """Tests for _compute_multinomial_resolution utility."""

    def test_default_8_segments(self):
        assert _compute_multinomial_resolution(256) == 32
        assert _compute_multinomial_resolution(1024) == 128
        assert _compute_multinomial_resolution(64) == 8

    def test_custom_segments(self):
        assert _compute_multinomial_resolution(256, num_segments=4) == 64
        assert _compute_multinomial_resolution(256, num_segments=16) == 16

    def test_min_segment_size(self):
        assert _compute_multinomial_resolution(64, min_segment_size=16) == 16
        assert _compute_multinomial_resolution(256, min_segment_size=16) == 32

    def test_small_sequence(self):
        assert _compute_multinomial_resolution(8) == 1
        assert _compute_multinomial_resolution(4) == 1


@pytest.mark.unit
class TestFinetuneDevice:
    @pytest.mark.parametrize(
        "requested, cuda_available, mps_available, expected",
        [
            ("auto", False, False, "cpu"),
            ("auto", False, True, "mps"),
            ("auto", True, True, "cuda:0"),
            ("cpu", True, True, "cpu"),
            ("mps", True, True, "mps"),
            ("cuda:2", True, True, "cuda:2"),
        ],
    )
    def test_resolve_device(self, monkeypatch, requested, cuda_available, mps_available, expected):
        from alphagenome_pytorch.extensions.finetuning.distributed import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
        assert resolve_device(requested) == torch.device(expected)

    @pytest.mark.parametrize(
        "requested, error",
        [("mps", "MPS is unavailable"), ("cuda", "CUDA is unavailable"), ("xpu", "Invalid fine-tuning device")],
    )
    def test_reports_unavailable_or_unsupported_device(self, monkeypatch, requested, error):
        from alphagenome_pytorch.extensions.finetuning.distributed import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(ValueError, match=error):
            resolve_device(requested)

    def test_torchrun_uses_local_rank_and_rejects_explicit_index(self, monkeypatch):
        from alphagenome_pytorch.extensions.finetuning.distributed import setup_distributed

        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        events = []
        monkeypatch.setattr(torch.cuda, "set_device", events.append)
        monkeypatch.setattr(torch.distributed, "init_process_group", lambda *, backend: events.append(backend))

        for requested in ("cuda:0", "mps"):
            with pytest.raises(ValueError, match="LOCAL_RANK"):
                setup_distributed(requested)
        assert events == []

        assert setup_distributed() == (3, 4, 1, torch.device("cuda:1"))
        assert events == [torch.device("cuda:1"), "nccl"]

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        with pytest.raises(ValueError, match="CUDA is unavailable"):
            setup_distributed()
        assert events == [torch.device("cuda:1"), "nccl"]


@pytest.mark.unit
class TestFinetuneDtype:
    cli_args = _required_cli_args()[1:] + ["--modality", "atac", "--bigwig", "atac.bw"]

    @pytest.mark.parametrize(
        "device, use_amp, expected",
        [
            ("cuda", True, "bfloat16"),
            ("cuda", False, "float32"),
            ("cpu", True, "float32"),
            ("mps", True, "float32"),
        ],
    )
    def test_defaults_follow_device_and_amp(self, device, use_amp, expected):
        from alphagenome_pytorch.extensions.finetuning.args import resolve_dtype

        args = parse_args(self.cli_args + ([] if use_amp else ["--no-amp"]))
        assert args.dtype is None
        assert resolve_dtype(args.dtype, cuda=device == "cuda", use_amp=not args.no_amp) == expected

    def test_explicit_bfloat16_is_kept_with_amp(self):
        from alphagenome_pytorch.extensions.finetuning.args import resolve_dtype

        args = parse_args(self.cli_args + ["--dtype", "bfloat16"])
        assert resolve_dtype(args.dtype, cuda=False, use_amp=not args.no_amp) == "bfloat16"

    def test_bfloat16_and_no_amp_are_rejected_on_the_cli(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(self.cli_args + ["--dtype", "bfloat16", "--no-amp"])
        assert "--no-amp conflicts with --dtype bfloat16" in capsys.readouterr().err

    @pytest.mark.parametrize("amp_setting", ["use_amp: false", "no_amp: true"])
    def test_bfloat16_and_no_amp_are_rejected_in_yaml(self, tmp_path, capsys, amp_setting):
        pytest.importorskip("yaml")
        config = tmp_path / "config.yaml"
        config.write_text(f"dtype: bfloat16\n{amp_setting}\n")
        argv = self.cli_args + ["--config", str(config)]

        with pytest.raises(SystemExit):
            parse_args(argv)
        assert "--no-amp conflicts with --dtype bfloat16" in capsys.readouterr().err

        args = parse_args(argv + ["--dtype", "float32"])
        assert args.dtype == "float32"
        assert args.no_amp


@pytest.mark.unit
class TestParseArgsMultimodal:
    """Tests for multimodal/task-weight parsing in scripts/finetune.py."""

    def test_parse_args_with_two_modalities_and_weights(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--bigwig",
                "atac1.bw",
                "atac2.bw",
                "--modality",
                "rna_seq",
                "--bigwig",
                "rna1.bw",
                "--modality-weights",
                "atac:1.0,rna_seq:0.5",
                "--resolutions",
                "1,128",
            ],
        )

        args = parse_args()

        assert args.is_multimodal is True
        assert args.modalities == ["atac", "rna_seq"]
        assert args.modality_to_bigwigs["atac"] == ["atac1.bw", "atac2.bw"]
        assert args.modality_to_bigwigs["rna_seq"] == ["rna1.bw"]
        assert args.global_resolutions == (1, 128)
        assert args.modality_resolutions["atac"] == (1, 128)
        assert args.modality_resolutions["rna_seq"] == (1, 128)
        assert args.modality_weight_dict["atac"] == pytest.approx(1.0)
        assert args.modality_weight_dict["rna_seq"] == pytest.approx(0.5)

    def test_parse_args_missing_modality_weight_defaults_to_one(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--bigwig",
                "atac1.bw",
                "--modality",
                "rna_seq",
                "--bigwig",
                "rna1.bw",
                "--modality-weights",
                "atac:2.0",
            ],
        )

        args = parse_args()
        assert args.modality_weight_dict["atac"] == pytest.approx(2.0)
        assert args.modality_weight_dict["rna_seq"] == pytest.approx(1.0)

    def test_parse_args_rejects_mismatched_modality_and_bigwig_groups(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--modality",
                "rna_seq",
                "--bigwig",
                "atac1.bw",
            ],
        )

        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.unit
class TestParseArgsStrandPairs:
    """Tests for --strand-pairs / config strand_pairs parsing."""

    def _stranded_argv(self, *extra: str) -> list[str]:
        return (
            _required_cli_args()
            + [
                "--modality", "atac", "--bigwig", "atac1.bw", "atac2.bw",
                "--modality", "rna_seq", "--bigwig",
                "rp1.bw", "rm1.bw", "rp2.bw", "rm2.bw",
            ]
            + list(extra)
        )

    def test_auto_pairs_consecutive_and_leaves_others_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._stranded_argv("--strand-pairs", "rna_seq:auto"))
        args = parse_args()
        assert args.modality_strand_pairs["rna_seq"] == [(0, 1), (2, 3)]
        # Unstranded modality is untouched.
        assert args.modality_strand_pairs["atac"] is None

    def test_explicit_string_pairs(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._stranded_argv("--strand-pairs", "rna_seq:0,2;1,3"))
        args = parse_args()
        assert args.modality_strand_pairs["rna_seq"] == [(0, 2), (1, 3)]

    def test_no_strand_pairs_all_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._stranded_argv())
        args = parse_args()
        assert args.modality_strand_pairs["atac"] is None
        assert args.modality_strand_pairs["rna_seq"] is None

    def test_auto_rejects_odd_bigwig_count(self, monkeypatch):
        # atac has 2 bigwigs (even); point auto at a modality with an odd count.
        argv = (
            _required_cli_args()
            + ["--modality", "rna_seq", "--bigwig", "rp1.bw", "rm1.bw", "rp2.bw"]
            + ["--strand-pairs", "rna_seq:auto"]
        )
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            parse_args()

    def test_rejects_unknown_modality(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._stranded_argv("--strand-pairs", "cage:auto"))
        with pytest.raises(SystemExit):
            parse_args()

    @pytest.mark.parametrize("spec", ["rna_seq:", "rna_seq:;", "rna_seq: ; "])
    def test_rejects_empty_explicit_spec(self, monkeypatch, spec):
        # A blank explicit spec is almost certainly a typo: reject rather than
        # silently apply no averaging.
        monkeypatch.setattr(sys, "argv", self._stranded_argv("--strand-pairs", spec))
        with pytest.raises(SystemExit):
            parse_args()

    def test_rejects_empty_config_list(self, monkeypatch, tmp_path):
        yaml = pytest.importorskip("yaml")
        config = {
            "modalities": {
                "rna_seq": {"bigwig": ["rp1.bw", "rm1.bw"], "strand_pairs": []},
            }
        }
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys, "argv", _required_cli_args() + ["--config", str(config_path)]
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_config_list_of_lists(self, monkeypatch, tmp_path):
        yaml = pytest.importorskip("yaml")
        config = {
            "modalities": {
                "atac": {"bigwig": ["atac1.bw", "atac2.bw"]},
                "rna_seq": {
                    "bigwig": ["rp1.bw", "rm1.bw", "rp2.bw", "rm2.bw"],
                    "strand_pairs": [[0, 1], [2, 3]],
                },
            }
        }
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys, "argv", _required_cli_args() + ["--config", str(config_path)]
        )
        args = parse_args()
        assert args.modality_strand_pairs["rna_seq"] == [(0, 1), (2, 3)]
        assert args.modality_strand_pairs["atac"] is None

    def test_config_strand_accepts_string_and_list_forms(self, monkeypatch, tmp_path):
        """modalities.<head>.strand may be a compact/separated string or a YAML list."""
        yaml = pytest.importorskip("yaml")

        def _strands_for(strand_spec):
            config = {
                "modalities": {
                    "rna_seq": {
                        "bigwig": ["rp1.bw", "rm1.bw", "rp2.bw", "rm2.bw"],
                        "strand": strand_spec,
                    },
                }
            }
            config_path = tmp_path / "train.yaml"
            config_path.write_text(yaml.safe_dump(config))
            monkeypatch.setattr(
                sys, "argv", _required_cli_args() + ["--config", str(config_path)]
            )
            return parse_args().modality_strands["rna_seq"]

        assert _strands_for("+-+-") == "+-+-"                    # compact string
        assert _strands_for("+,-,+,-") == "+-+-"                 # separated string
        assert _strands_for(["+", "-", "+", "-"]) == "+-+-"      # YAML list

    def test_config_strand_list_wrong_length_rejected(self, monkeypatch, tmp_path):
        yaml = pytest.importorskip("yaml")
        config = {
            "modalities": {
                "rna_seq": {
                    "bigwig": ["rp1.bw", "rm1.bw", "rp2.bw", "rm2.bw"],
                    "strand": ["+", "-"],  # 2 chars, 4 bigwigs
                },
            }
        }
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys, "argv", _required_cli_args() + ["--config", str(config_path)]
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_cli_overrides_config_strand_pairs(self, monkeypatch, tmp_path):
        yaml = pytest.importorskip("yaml")
        config = {
            "modalities": {
                "rna_seq": {
                    "bigwig": ["rp1.bw", "rm1.bw", "rp2.bw", "rm2.bw"],
                    "strand_pairs": "auto",
                },
            }
        }
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + ["--config", str(config_path), "--strand-pairs", "rna_seq:0,2;1,3"],
        )
        args = parse_args()
        # CLI explicit pairs win over config 'auto'.
        assert args.modality_strand_pairs["rna_seq"] == [(0, 2), (1, 3)]


@pytest.mark.unit
class TestMultimodalDataset:
    """Tests for MultimodalDataset wrapper."""

    def test_length(self):
        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=100)
        mock_ds1.__getitem__ = MagicMock(return_value=(torch.randn(256, 4), {128: torch.randn(256, 5)}))

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=100)
        mock_ds2.__getitem__ = MagicMock(return_value=(torch.randn(256, 4), {128: torch.randn(256, 3)}))

        dataset = MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})
        assert len(dataset) == 100

    def test_length_mismatch_raises(self):
        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=100)

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=50)

        with pytest.raises(ValueError, match="same length"):
            MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})

    def test_getitem_returns_all_modalities(self):
        seq = torch.randn(256, 4)
        targets1 = {128: torch.randn(256, 5)}
        targets2 = {128: torch.randn(256, 3)}

        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=10)
        mock_ds1.__getitem__ = MagicMock(return_value=(seq, targets1))

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=10)
        mock_ds2.__getitem__ = MagicMock(return_value=(seq, targets2))

        dataset = MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})
        result_seq, result_targets = dataset[0]

        assert torch.equal(result_seq, seq)
        assert "atac" in result_targets
        assert "rna_seq" in result_targets
        assert 128 in result_targets["atac"]
        assert 128 in result_targets["rna_seq"]


@pytest.mark.unit
class TestCollateMultimodal:
    """Tests for collate_multimodal function."""

    def test_collate_single_modality(self):
        batch = [
            (torch.randn(256, 4), {"atac": {128: torch.randn(256, 5)}}),
            (torch.randn(256, 4), {"atac": {128: torch.randn(256, 5)}}),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert sequences.shape == (2, 256, 4)
        assert "atac" in modality_targets
        assert 128 in modality_targets["atac"]
        assert modality_targets["atac"][128].shape == (2, 256, 5)

    def test_collate_multiple_modalities(self):
        batch = [
            (
                torch.randn(256, 4),
                {
                    "atac": {128: torch.randn(256, 5)},
                    "rna_seq": {1: torch.randn(256, 3), 128: torch.randn(256, 3)},
                },
            ),
            (
                torch.randn(256, 4),
                {
                    "atac": {128: torch.randn(256, 5)},
                    "rna_seq": {1: torch.randn(256, 3), 128: torch.randn(256, 3)},
                },
            ),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert sequences.shape == (2, 256, 4)
        assert "atac" in modality_targets
        assert "rna_seq" in modality_targets
        assert modality_targets["atac"][128].shape == (2, 256, 5)
        assert modality_targets["rna_seq"][1].shape == (2, 256, 3)
        assert modality_targets["rna_seq"][128].shape == (2, 256, 3)

    def test_collate_preserves_batch_order(self):
        seq1 = torch.ones(256, 4)
        seq2 = torch.zeros(256, 4)
        targets1 = torch.ones(256, 3)
        targets2 = torch.zeros(256, 3)

        batch = [
            (seq1, {"atac": {128: targets1}}),
            (seq2, {"atac": {128: targets2}}),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert torch.equal(sequences[0], seq1)
        assert torch.equal(sequences[1], seq2)
        assert torch.equal(modality_targets["atac"][128][0], targets1)
        assert torch.equal(modality_targets["atac"][128][1], targets2)

    def test_collate_preserves_gene_mask(self):
        # Regression guard: the collate historically dropped item[2], silently
        # no-op'ing the gene-LFC training loss. It must now survive collation as
        # extras["gene_mask"], stacked with a leading batch axis.
        gm1 = torch.zeros(256, 2, 4, dtype=torch.bool)
        gm2 = torch.ones(256, 2, 4, dtype=torch.bool)
        batch = [
            (torch.randn(256, 4), {"rna_seq": {1: torch.randn(256, 3)}}, gm1),
            (torch.randn(256, 4), {"rna_seq": {1: torch.randn(256, 3)}}, gm2),
        ]
        sequences, modality_targets, extras = collate_multimodal(batch)
        assert "gene_mask" in extras
        assert extras["gene_mask"].shape == (2, 256, 2, 4)
        assert torch.equal(extras["gene_mask"][0], gm1)
        assert torch.equal(extras["gene_mask"][1], gm2)
        assert "coords" not in extras

    def test_collate_preserves_coords(self):
        batch = [
            (torch.randn(256, 4), {"rna_seq": {1: torch.randn(256, 3)}}, ("chr1", 0, 256)),
            (torch.randn(256, 4), {"rna_seq": {1: torch.randn(256, 3)}}, ("chr2", 5, 261)),
        ]
        sequences, modality_targets, extras = collate_multimodal(batch)
        assert extras["coords"] == [("chr1", 0, 256), ("chr2", 5, 261)]
        assert "gene_mask" not in extras

    def test_collate_preserves_gene_mask_and_coords(self):
        gm = torch.ones(256, 2, 4, dtype=torch.bool)
        batch = [
            (torch.randn(256, 4), {"rna_seq": {1: torch.randn(256, 3)}}, gm, ("chr1", 0, 256)),
        ]
        sequences, modality_targets, extras = collate_multimodal(batch)
        assert extras["gene_mask"].shape == (1, 256, 2, 4)
        assert extras["coords"] == [("chr1", 0, 256)]

    def test_unpack_batch_decodes_collate_contract(self):
        # _unpack_batch is the consumer-side decode of the collate output.
        from alphagenome_pytorch.extensions.finetuning.training import _unpack_batch

        seqs = torch.randn(2, 8, 4)
        targets = {"rna_seq": {1: torch.randn(2, 8, 3)}}
        # 2-tuple (no extras) -> empty extras dict
        s, t, e = _unpack_batch((seqs, targets))
        assert s is seqs and t is targets and e == {}
        # 3-tuple -> extras passed straight through
        extras = {"gene_mask": torch.ones(2, 8, 2, 4), "coords": [("chr1", 0, 8), ("chr2", 5, 13)]}
        s, t, e = _unpack_batch((seqs, targets, extras))
        assert e is extras


@pytest.mark.unit
class TestParseArgsGeneExprEval:
    """Validation of the --gene-expr-eval flag in scripts/finetune.py."""

    def _rna_cli(self, monkeypatch, extra):
        monkeypatch.setattr(
            sys, "argv",
            _required_cli_args()
            + ["--modality", "rna_seq", "--bigwig", "rna1.bw"] + extra,
        )

    def test_happy_path_sets_flags(self, monkeypatch):
        self._rna_cli(monkeypatch, [
            "--gene-expr-eval",
            "--gene-expr-annotation", "gencode.parquet",
            "--track-strands", "+",
        ])
        args = parse_args()
        assert args.gene_expr_eval is True
        assert args.gene_expr_annotation == "gencode.parquet"

    def test_falls_back_to_gtf_annotation(self, monkeypatch):
        self._rna_cli(monkeypatch, [
            "--gene-expr-eval", "--gtf", "genes.gtf", "--track-strands", "+",
        ])
        args = parse_args()  # --gtf satisfies the annotation requirement
        assert args.gene_expr_eval is True

    def test_requires_annotation(self, monkeypatch):
        self._rna_cli(monkeypatch, ["--gene-expr-eval", "--track-strands", "+"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_requires_strands(self, monkeypatch):
        self._rna_cli(monkeypatch, [
            "--gene-expr-eval", "--gene-expr-annotation", "gencode.parquet",
        ])
        with pytest.raises(SystemExit):
            parse_args()

    def test_requires_rna_seq_modality(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            _required_cli_args()
            + ["--modality", "atac", "--bigwig", "atac1.bw",
               "--gene-expr-eval", "--gene-expr-annotation", "gencode.parquet"],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_off_by_default(self, monkeypatch):
        self._rna_cli(monkeypatch, [])
        assert parse_args().gene_expr_eval is False


@pytest.mark.unit
class TestParseArgsThreePrimeRna:
    """A 3'-biased library is the rna_seq head plus a per-head reduction, not a
    modality of its own: the head name stays an output type, so the fine-tune
    still serves and scores."""

    def _cli(self, monkeypatch, extra):
        monkeypatch.setattr(
            sys, "argv",
            _required_cli_args()
            + ["--modality", "rna_seq", "--bigwig", "mic+.bw", "mic-.bw"]
            + extra,
        )

    def test_track_strands_accepted(self, monkeypatch):
        self._cli(monkeypatch, ["--track-strands", "+-"])
        assert parse_args().modality_strands["rna_seq"] == "+-"

    def test_reduce_func_flag_reaches_the_head(self, monkeypatch):
        self._cli(monkeypatch, ["--track-strands", "+-",
                                "--gene-reduce-func", "sum"])
        assert parse_args().modality_gene_reduce_funcs == {"rna_seq": "sum"}

    def test_satisfies_gene_loss_weight(self, monkeypatch):
        self._cli(monkeypatch, [
            "--track-strands", "+-", "--gtf", "genes.gtf",
            "--gene-loss-weight", "0.1", "--gene-reduce-func", "sum",
        ])
        assert parse_args().gene_loss_weight == 0.1

    def test_satisfies_gene_expr_eval_and_resolves_head(self, monkeypatch):
        self._cli(monkeypatch, [
            "--track-strands", "+-", "--gene-expr-annotation", "gencode.parquet",
            "--gene-expr-eval", "--gene-reduce-func", "sum",
        ])
        args = parse_args()
        assert args.gene_expr_eval is True
        # The metric must read the head the same way the loss does.
        assert args.modality_gene_reduce_funcs["rna_seq"] == "sum"

    def test_reduce_func_flag_defaults_to_auto(self, monkeypatch):
        self._cli(monkeypatch, ["--track-strands", "+-"])
        assert parse_args().gene_reduce_func == "auto"

    def test_reduce_func_flag_rejects_unknown_value(self, monkeypatch):
        self._cli(monkeypatch, ["--track-strands", "+-",
                                "--gene-reduce-func", "median"])
        with pytest.raises(SystemExit):
            parse_args()

    def test_three_prime_config_needs_no_new_modality(self, monkeypatch, tmp_path):
        """The whole 3' setup in a YAML config: one rna_seq head, strands, and
        the reduction that both the loss and the metric will use."""
        yaml = pytest.importorskip("yaml")
        config = {
            "modalities": {
                "rna_seq": {
                    "bigwig": ["mic+.bw", "mic-.bw"],
                    "strand": "+-",
                    "gene_reduce_func": "sum",
                    "strand_pairs": "auto",
                },
            },
        }
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys, "argv", _required_cli_args() + ["--config", str(config_path)]
        )
        args = parse_args()
        assert args.modality_gene_reduce_funcs == {"rna_seq": "sum"}
        assert args.modality_strands == {"rna_seq": "+-"}
        assert args.modality_strand_pairs == {"rna_seq": [(0, 1)]}


@pytest.mark.unit
class TestEpochReporting:
    """The epoch boundary runs after hours of compute, so everything there has
    to survive a metric it did not expect. A per-track list formatted with
    ``:.4f`` raised ``unsupported format string passed to list.__format__`` and
    killed a run that had just finished epoch 1."""

    # What validate_multihead actually returns for a stranded RNA head.
    METRICS = {
        "rna_seq_1bp_profile_pearson_r_mean": 0.42,
        "rna_seq_1bp_profile_pearson_r_std": 0.11,
        "rna_seq_1bp_profile_pearson_r_per_track": [0.4, 0.44],
        "rna_seq_1bp_profile_pearson_r_values": [0.3, 0.5, 0.4, 0.44],
        "rna_seq_1bp_count_pearson_r": 0.61,
        "rna_seq_1bp_count_pearson_r_per_track": [0.6, 0.62],
        "rna_seq_gene_log_expr_pearson_one_vs_rest_lfc": 0.25,
        "rna_seq_gene_log_expr_pearson_one_vs_rest_lfc_per_track": [0.2, 0.3],
        "rna_seq_gene_log_expr_n_genes": 812,
    }
    NAMES = {"rna_seq": ["Microglia+", "Microglia-"]}

    def test_summary_survives_per_track_lists(self):
        line = format_epoch_summary(1, 6.81, 3.25, {"rna_seq": 6.81}, self.METRICS)
        assert "train_loss=6.8100" in line and "val_loss=3.2500" in line
        assert "rna_seq_1bp_count_pearson_r=0.6100" in line
        # The lists are skipped rather than formatted.
        assert "per_track" not in line

    def test_summary_skips_any_non_scalar(self):
        """By type, so a metric shape nobody anticipated cannot end an epoch."""
        line = format_epoch_summary(
            1, 1.0, 1.0, {}, {"rna_seq_pearson_weird": {"a": 1}, "rna_seq_pearson_ok": 0.5}
        )
        assert "rna_seq_pearson_ok=0.5000" in line and "weird" not in line

    def test_per_track_lists_keep_names_outside_epoch_scalars(self):
        extra, histograms, tracks = split_val_metrics(
            self.METRICS, self.NAMES, gene_expr_sample_names=["A", "B"],
        )
        assert tracks["rna_seq_1bp_profile_pearson_r"][0] == (0, "Microglia+", 0.4)
        assert tracks["rna_seq_1bp_count_pearson_r"][1] == (1, "Microglia-", 0.62)
        assert tracks["rna_seq_gene_log_expr_pearson_one_vs_rest_lfc"][0] == (0, "A", 0.2)
        assert histograms["rna_seq_1bp_profile_pearson_r_values"] == [0.3, 0.5, 0.4, 0.44]
        assert extra["val_loss_rna_seq_gene_log_expr_n_genes"] == 812
        assert all("/" not in key for key in extra)

    def test_epoch_resume_keeps_scalar_headers_and_writes_separate_track_rows(self, tmp_path):
        import csv

        from alphagenome_pytorch.extensions.finetuning.logging import TrainingLogger

        TrainingLogger(tmp_path).log_epoch(
            1, 0.5, 0.4, 0.001, extra={"original": 0.2},
        )
        logger = TrainingLogger(tmp_path)
        logger.use_wandb = True
        logger.wandb = MagicMock()
        logger.log_epoch(
            2, 0.5, 0.4, 0.001, extra={"new": 0.8, "original": 0.3},
            track_metrics={"rna_seq_count": [(0, "A", 0.6), (1, "B", 0.7)]},
        )
        with (tmp_path / "epoch_log.csv").open() as file:
            rows = list(csv.DictReader(file))
        assert [(row["original"], row["new"]) for row in rows] == [("0.2", ""), ("0.3", "0.8")]
        assert "rna_seq_count/A" not in rows[1]
        with (tmp_path / "epoch_track_metrics.csv").open() as file:
            assert list(csv.reader(file)) == [
                ["epoch", "metric", "track_index", "track_name", "value"],
                ["2", "rna_seq_count", "0", "A", "0.6"],
                ["2", "rna_seq_count", "1", "B", "0.7"],
            ]
        assert logger.wandb.log.call_args.args[0]["epoch/rna_seq_count/A"] == 0.6

    def test_per_track_falls_back_to_histogram_when_unnamed(self):
        extra, histograms, tracks = split_val_metrics(
            {"rna_seq_1bp_count_pearson_r_per_track": [0.1, 0.2, 0.3]}, self.NAMES
        )
        assert histograms["rna_seq_1bp_count_pearson_r_per_track_values"] == [0.1, 0.2, 0.3]
        assert extra == {}

    def test_nonfinite_histograms_do_not_block_wandb_epoch_logging(self, tmp_path):
        from alphagenome_pytorch.extensions.finetuning.logging import TrainingLogger

        logger = TrainingLogger(tmp_path)
        logger.use_wandb = True
        logger.wandb = MagicMock()
        logger.log_epoch(
            1, train_loss=0.5, val_loss=0.4, lr=0.001,
            histograms={
                "mixed_values": [0.25, float("nan"), float("inf"), -0.5],
                "undefined_values": [float("nan"), -float("inf")],
            },
        )

        logger.wandb.Histogram.assert_called_once_with([0.25, -0.5])
        logger.wandb.log.assert_called_once()
        metrics = logger.wandb.log.call_args.args[0]
        assert metrics["epoch/val_loss"] == 0.4
        assert "epoch/mixed_values" in metrics
        assert "epoch/undefined_values" not in metrics

    def test_fallback_does_not_overwrite_region_distribution(self):
        """A wide head has both a per-(region, track) `_values` list and a
        per-track list; the fallback histogram must not replace the former."""
        names = {"rna_seq": [f"t{i}" for i in range(40)]}
        extra, histograms, tracks = split_val_metrics(
            {
                "rna_seq_1bp_profile_pearson_r_values": [0.9] * 80,
                "rna_seq_1bp_profile_pearson_r_per_track": [0.1] * 40,
            },
            names,
        )
        assert histograms["rna_seq_1bp_profile_pearson_r_values"] == [0.9] * 80
        assert histograms["rna_seq_1bp_profile_pearson_r_per_track_values"] == [0.1] * 40
        assert tracks["rna_seq_1bp_profile_pearson_r"][-1] == (39, "t39", 0.1)


@pytest.mark.unit
class TestTrackNamesForMetric:
    """Per-track metric keys are labeled with the owning head's track names."""

    def test_matches_head_prefix(self):
        names = {"rna_seq": ["Mic+", "Mic-"], "atac": ["a"]}
        assert track_names_for_metric(
            "rna_seq_1bp_count_pearson_r_per_track", names
        ) == ["Mic+", "Mic-"]
        assert track_names_for_metric(
            "atac_128bp_profile_pearson_r_per_track", names
        ) == ["a"]

    def test_unknown_head_returns_none(self):
        assert track_names_for_metric(
            "cage_1bp_count_pearson_r_per_track", {"atac": ["a"]}
        ) is None


@pytest.mark.unit
class TestUnwrapTrainingModel:
    """Tests for finetune.py wrapper unwrapping."""

    def test_unwraps_compile_wrapper(self):
        base = torch.nn.Linear(4, 4)

        class FakeCompiled(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self._orig_mod = module

        wrapped = FakeCompiled(base)

        assert unwrap_training_model(wrapped) is base

    def test_unwraps_compile_then_ddp(self, monkeypatch):
        base = torch.nn.Linear(4, 4)

        class FakeDDP(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        class FakeCompiled(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self._orig_mod = module

        monkeypatch.setattr(finetune_module, "DDP", FakeDDP)
        wrapped = FakeCompiled(FakeDDP(base))

        assert unwrap_training_model(wrapped) is base


@pytest.mark.unit
class TestLoadTrackMetadataForFinetune:
    """Tests for --track-metadata loading/validation/embedding."""

    @staticmethod
    def _write_csv(path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_none_path_returns_unchanged(self):
        names = {"atac": ["a", "b"]}
        out_names, rows = load_track_metadata_for_finetune(None, names, rank=0)
        assert out_names is names
        assert rows is None

    def test_happy_path_overrides_names_and_embeds_rows(self, tmp_path):
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name,biosample_name",
            "human,atac,liver,Liver",
            "human,atac,brain,Brain",
        ])
        out_names, rows = load_track_metadata_for_finetune(
            str(csv), {"atac": ["bw0", "bw1"]}, rank=0,
        )
        assert out_names == {"atac": ["liver", "brain"]}
        assert [r["track_name"] for r in rows] == ["liver", "brain"]

    def test_count_mismatch_raises(self, tmp_path):
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name",
            "human,atac,liver",
        ])
        with pytest.raises(ValueError, match="Counts must match"):
            load_track_metadata_for_finetune(str(csv), {"atac": ["bw0", "bw1"]}, rank=0)

    def test_mouse_tracks_embed_under_organism_one(self, tmp_path):
        """--organism mouse validates and embeds mouse (organism=1) tracks."""
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name",
            "mouse,atac,liver",
            "mouse,atac,brain",
        ])
        out_names, rows = load_track_metadata_for_finetune(
            str(csv), {"atac": ["bw0", "bw1"]}, rank=0, organism="mouse",
        )
        assert out_names == {"atac": ["liver", "brain"]}
        assert all(int(r["organism"]) == 1 for r in rows)

    def test_mouse_tracks_without_organism_flag_raise(self, tmp_path):
        """Mouse-tagged tracks without --organism mouse must raise (the trainer
        would otherwise forward at the human embedding)."""
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name",
            "mouse,atac,liver",
            "mouse,atac,brain",
        ])
        with pytest.raises(ValueError, match="trains organism 0"):
            load_track_metadata_for_finetune(str(csv), {"atac": ["bw0", "bw1"]}, rank=0)

    def test_organism_flag_conflicts_with_parquet_raises(self, tmp_path):
        """--organism mouse but human-tagged parquet -> clear error."""
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name",
            "human,atac,liver",
            "human,atac,brain",
        ])
        with pytest.raises(ValueError, match="trains organism 1"):
            load_track_metadata_for_finetune(
                str(csv), {"atac": ["bw0", "bw1"]}, rank=0, organism="mouse",
            )

    def test_mixed_organism_not_supported(self, tmp_path):
        """A mixed human+mouse catalog is rejected (single-organism fine-tune)."""
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "organism,output_type,track_name",
            "human,atac,liver",
            "mouse,atac,m_liver",
        ])
        with pytest.raises(ValueError, match="not supported yet"):
            load_track_metadata_for_finetune(str(csv), {"atac": ["bw0", "bw1"]}, rank=0)

    def test_organism_flag_fills_missing_column(self, tmp_path):
        """--organism mouse fills rows lacking an 'organism' value -> organism 1."""
        csv = tmp_path / "meta.csv"
        self._write_csv(csv, [
            "output_type,track_name",
            "atac,liver",
            "atac,brain",
        ])
        _names, rows = load_track_metadata_for_finetune(
            str(csv), {"atac": ["bw0", "bw1"]}, rank=0, organism="mouse",
        )
        assert all(int(r["organism"]) == 1 for r in rows)
