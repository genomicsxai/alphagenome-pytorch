"""Tests for the agt CLI skeleton: parser, dispatch, --json, error format, dep gating."""

from __future__ import annotations

import io
import json
import sys
import types
from unittest import mock

import pytest

from alphagenome_pytorch.cli._main import build_parser, main
from alphagenome_pytorch.cli._output import emit_json, emit_jsonl, emit_error, emit_text
from alphagenome_pytorch.cli._deps import MissingExtraError, require_extra
from alphagenome_pytorch.cli.preprocess import parse_target


# =============================================================================
# Output formatting
# =============================================================================

class TestEmitJson:
    def test_pretty_json(self):
        buf = io.StringIO()
        emit_json({"hello": "world"}, file=buf)
        result = json.loads(buf.getvalue())
        assert result == {"hello": "world"}
        # Pretty-printed → has newlines
        assert "\n" in buf.getvalue()

    def test_handles_non_serializable(self):
        """emit_json passes default=str for non-serializable types."""
        buf = io.StringIO()
        from pathlib import Path
        emit_json({"path": Path("/tmp/test")}, file=buf)
        result = json.loads(buf.getvalue())
        assert result["path"] == "/tmp/test"


class TestEmitJsonl:
    def test_single_line(self):
        buf = io.StringIO()
        emit_jsonl({"a": 1, "b": 2}, file=buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"a": 1, "b": 2}


class TestEmitError:
    def test_json_mode(self, capsys):
        exc = ValueError("something broke")
        emit_error(exc, json_mode=True)
        err = capsys.readouterr().err
        data = json.loads(err)
        assert data["error"] == "ValueError"
        assert data["message"] == "something broke"

    def test_text_mode(self, capsys):
        exc = FileNotFoundError("no file")
        emit_error(exc, json_mode=False)
        err = capsys.readouterr().err
        assert "no file" in err


class TestEmitText:
    def test_adds_newline(self):
        buf = io.StringIO()
        emit_text("hello", file=buf)
        assert buf.getvalue() == "hello\n"

    def test_preserves_existing_newline(self):
        buf = io.StringIO()
        emit_text("hello\n", file=buf)
        assert buf.getvalue() == "hello\n"


# =============================================================================
# Parser construction
# =============================================================================

class TestParser:
    def test_builds_without_error(self):
        parser = build_parser()
        assert parser is not None

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json", "serve", "--weights", "model.pth", "--fasta", "hg38.fa"])
        assert args.json_output is True
        assert args.command == "serve"

    def test_no_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--weights", "model.pth", "--fasta", "hg38.fa"])
        assert args.json_output is False

    def test_subcommands_registered(self):
        parser = build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

        args = parser.parse_args(["serve", "--weights", "model.pth", "--fasta", "hg38.fa"])
        assert args.command == "serve"

    def test_all_subcommand_names(self):
        """Verify all expected subcommands are present by checking --help output."""
        parser = build_parser()
        # Check the subparsers action has all expected commands
        for action in parser._subparsers._actions:
            if hasattr(action, '_parser_class'):
                choices = getattr(action, 'choices', {}) or {}
                if choices:
                    for cmd in ["info", "predict", "finetune", "score", "convert", "preprocess", "serve"]:
                        assert cmd in choices, f"Command '{cmd}' not registered"


# =============================================================================
# Dispatch
# =============================================================================

class TestDispatch:
    def test_no_command_shows_help(self):
        rc = main([])
        assert rc == 0

    def test_serve_requires_args(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["serve"])
        assert excinfo.value.code == 2

    def test_serve_dispatches_to_extension(self):
        fake_cli = types.ModuleType("alphagenome_pytorch.extensions.serving.cli")
        fake_cli.run = mock.Mock(return_value=0)
        with mock.patch("alphagenome_pytorch.cli.serve.require_extra") as require_extra, mock.patch.dict(
            sys.modules, {"alphagenome_pytorch.extensions.serving.cli": fake_cli}
        ):
            rc = main(["serve", "--weights", "model.pth", "--fasta", "hg38.fa"])

        assert rc == 0
        require_extra.assert_called_once_with("serving", "serve")
        fake_cli.run.assert_called_once()

    def test_unknown_error_emits_json(self, capsys):
        """Exceptions are caught and formatted as JSON when --json is set."""
        # Use a simulated error in a command that doesn't require extra deps
        with mock.patch(
            "alphagenome_pytorch.cli.info._run_heads",
            side_effect=RuntimeError("boom"),
        ):
            rc = main(["--json", "info"])
        assert rc == 1
        err = capsys.readouterr().err
        data = json.loads(err)
        assert data["error"] == "RuntimeError"


# =============================================================================
# Dependency gating
# =============================================================================

class TestRequireExtra:
    def test_passes_when_deps_importable(self):
        # 'os' and 'sys' are always available — use a test extra
        with mock.patch(
            "alphagenome_pytorch.cli._deps._EXTRA_PROBES",
            {"test_extra": ["os", "sys"]},
        ):
            require_extra("test_extra", "test")  # should not raise

    def test_raises_when_deps_missing(self):
        with mock.patch(
            "alphagenome_pytorch.cli._deps._EXTRA_PROBES",
            {"test_extra": ["nonexistent_module_xyz"]},
        ):
            with pytest.raises(MissingExtraError) as excinfo:
                require_extra("test_extra", "test")
            assert excinfo.value.missing == ["nonexistent_module_xyz"]
            assert "pip install alphagenome-pytorch[test_extra]" in str(excinfo.value)


# =============================================================================
# Preprocess: parse_target
# =============================================================================

class TestParseTarget:
    def test_plain_number(self):
        assert parse_target("100") == 100.0

    def test_k_suffix(self):
        assert parse_target("50k") == 50_000.0

    def test_M_suffix(self):
        assert parse_target("100M") == 100_000_000.0

    def test_G_suffix(self):
        assert parse_target("1G") == 1_000_000_000.0

    def test_decimal(self):
        assert parse_target("1.5M") == 1_500_000.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_target("abc")


# =============================================================================
# Info: heads overview (use StringIO to capture output directly)
# =============================================================================

class TestInfoHeads:
    def test_info_default(self):
        """agt info should return 0."""
        rc = main(["info"])
        assert rc == 0

    def test_info_heads_flag(self):
        """agt info --heads should return 0."""
        rc = main(["info", "--heads"])
        assert rc == 0

    def test_info_json_heads(self):
        """agt --json info --heads produces valid JSON."""
        from alphagenome_pytorch.cli import info
        from alphagenome_pytorch.cli._output import emit_json as orig_emit

        buf = io.StringIO()

        # Patch on the info module where emit_json was imported
        with mock.patch.object(info, 'emit_json', side_effect=lambda data, **kw: orig_emit(data, file=buf)):
            args = mock.MagicMock()
            args.json_output = True
            rc = info._run_heads(args)

        assert rc == 0
        data = json.loads(buf.getvalue())
        assert "heads" in data
        names = [h["name"] for h in data["heads"]]
        assert "atac" in names
        assert "splice_junctions" in names

    def test_head_info_contents(self):
        """Verify all expected heads are in the output."""
        from alphagenome_pytorch.cli.info import _HEAD_INFO
        expected = ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf",
                     "chip_histone", "contact_maps", "splice_sites",
                     "splice_junctions", "splice_site_usage"]
        for name in expected:
            assert name in _HEAD_INFO, f"Missing head: {name}"

    def test_head_dimensions(self):
        """Check dimension values match known constants."""
        from alphagenome_pytorch.cli.info import _HEAD_INFO
        assert _HEAD_INFO["atac"]["dimension"] == 256
        assert _HEAD_INFO["contact_maps"]["dimension"] == 28
        assert _HEAD_INFO["splice_sites"]["dimension"] == 5
        assert _HEAD_INFO["splice_junctions"]["dimension"] == 734


class TestInfoWeights:
    def _run_weights_json(self, path, **overrides):
        from alphagenome_pytorch.cli import info
        from alphagenome_pytorch.cli._output import emit_json as orig_emit

        args = mock.MagicMock()
        args.weights_file = str(path)
        args.json_output = True
        args.track_means = None
        args.validate = False
        args.diff = None
        args.organism = None
        args.top = None
        for key, value in overrides.items():
            setattr(args, key, value)

        buf = io.StringIO()
        with mock.patch.object(
            info,
            "emit_json",
            side_effect=lambda data, **kw: orig_emit(data, file=buf),
        ):
            rc = info._run_weights(args)
        return rc, json.loads(buf.getvalue())

    def test_weights_summary_accepts_raw_state_dict(self, tmp_path):
        import torch

        path = tmp_path / "weights.pth"
        torch.save({
            "heads.atac.weight": torch.ones(2, 2),
            "heads.atac.track_means": torch.ones(2, 3),
        }, path)

        rc, data = self._run_weights_json(path)

        assert rc == 0
        assert data["total_parameters"] == 10
        assert data["dtype"] == "torch.float32"
        assert data["has_track_means"] is True
        assert data["heads"] == ["atac"]

    def test_weights_summary_unwraps_full_training_checkpoint(self, tmp_path):
        import torch

        path = tmp_path / "best_model.pth"
        torch.save({
            "epoch": 3,
            "val_loss": 0.25,
            "model_state_dict": {
                "heads.rna_seq.weight": torch.ones(3, 4),
                "heads.rna_seq.track_means": torch.ones(2, 4),
            },
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        }, path)

        rc, data = self._run_weights_json(path)

        assert rc == 0
        assert data["total_parameters"] == 20
        assert data["has_track_means"] is True
        assert data["heads"] == ["rna_seq"]

    def test_track_means_unwraps_full_training_checkpoint(self, tmp_path):
        import torch
        from alphagenome_pytorch.cli import info
        from alphagenome_pytorch.cli._output import emit_json as orig_emit

        path = tmp_path / "best_model.pth"
        torch.save({
            "epoch": 3,
            "model_state_dict": {
                "heads.atac.track_means": torch.tensor([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]),
            },
        }, path)

        args = mock.MagicMock()
        args.weights_file = str(path)
        args.json_output = True
        args.track_means = "atac"
        args.validate = False
        args.diff = None
        args.organism = "mouse"
        args.top = None

        buf = io.StringIO()
        with mock.patch.object(
            info,
            "emit_json",
            side_effect=lambda data, **kw: orig_emit(data, file=buf),
        ):
            rc = info._run_weights(args)

        assert rc == 0
        data = json.loads(buf.getvalue())
        assert data["organism"] == "mouse"
        assert data["track_means"] == [
            {"index": 0, "mean": 3.0},
            {"index": 1, "mean": 4.0},
        ]

    def test_weights_summary_unwraps_delta_checkpoint(self, tmp_path):
        import torch

        path = tmp_path / "best_model.delta.pth"
        torch.save({
            "delta_checkpoint_version": 1,
            "adapter_state_dict": {"encoder.adapter.weight": torch.ones(1)},
            "head_state_dict": {
                "heads.custom.weight": torch.ones(2, 2),
                "heads.custom.track_means": torch.ones(2, 3),
            },
            "norm_state_dict": {},
            "metadata": {"epoch": 4},
        }, path)

        rc, data = self._run_weights_json(path)

        assert rc == 0
        assert data["total_parameters"] == 11
        assert data["has_track_means"] is True
        assert data["heads"] == ["custom"]

    def test_diff_unwraps_other_training_checkpoint(self, tmp_path):
        import torch

        raw_path = tmp_path / "raw.pth"
        full_path = tmp_path / "full.pth"
        state_dict = {"heads.atac.weight": torch.ones(2, 2)}
        torch.save(state_dict, raw_path)
        torch.save({"epoch": 1, "model_state_dict": state_dict}, full_path)

        rc, data = self._run_weights_json(raw_path, diff=str(full_path))

        assert rc == 0
        assert data == {"added": [], "removed": [], "changed": []}


# =============================================================================
# Score: scorer resolution, VCF parsing, end-to-end dispatch
# =============================================================================

class TestScoreResolveScorers:
    def test_recommended_default(self):
        from alphagenome_pytorch.cli import score as score_cli

        sentinel = [object(), object(), object()]
        with mock.patch(
            "alphagenome_pytorch.variant_scoring.get_recommended_scorers",
            return_value=sentinel,
        ) as mocked:
            result = score_cli.resolve_scorers("recommended", "human")
        mocked.assert_called_once_with("human")
        assert result is sentinel

    def test_single_named_scorer(self):
        from alphagenome_pytorch.cli import score as score_cli

        scorers = score_cli.resolve_scorers("atac", "human")
        assert len(scorers) == 1
        # CenterMaskScorer over ATAC
        assert scorers[0].requested_output.value == "atac"

    def test_multiple_named_scorers(self):
        from alphagenome_pytorch.cli import score as score_cli

        scorers = score_cli.resolve_scorers("atac,dnase", "human")
        outs = sorted(s.requested_output.value for s in scorers)
        assert outs == ["atac", "dnase"]

    def test_unknown_scorer_raises(self):
        from alphagenome_pytorch.cli import score as score_cli

        with pytest.raises(ValueError, match="Unknown scorer"):
            score_cli.resolve_scorers("not_a_real_scorer", "human")

    def test_recommended_cannot_combine(self):
        from alphagenome_pytorch.cli import score as score_cli

        with pytest.raises(ValueError, match="cannot be combined"):
            score_cli.resolve_scorers("recommended,atac", "human")

    def test_empty_spec_raises(self):
        from alphagenome_pytorch.cli import score as score_cli

        with pytest.raises(ValueError, match="cannot be empty"):
            score_cli.resolve_scorers(",,,", "human")


class TestScoreParseVcf:
    def test_parses_minimal_vcf(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli

        vcf = tmp_path / "test.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "chr22\t36201698\trs1\tA\tC\n"
            "chr1\t1000\t.\tG\tT\n"
        )
        variants = score_cli.parse_vcf(vcf)
        assert len(variants) == 2
        assert variants[0].chromosome == "chr22"
        assert variants[0].position == 36201698
        assert variants[0].reference_bases == "A"
        assert variants[0].alternate_bases == "C"
        assert variants[0].name == "rs1"
        assert variants[1].name == ""  # "." → empty

    def test_too_few_columns_raises(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli

        vcf = tmp_path / "bad.vcf"
        vcf.write_text("chr1\t100\trs1\tA\n")
        with pytest.raises(ValueError, match="expected ≥5"):
            score_cli.parse_vcf(vcf)

    def test_non_integer_pos_raises(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli

        vcf = tmp_path / "bad.vcf"
        vcf.write_text("chr1\tnotanumber\trs1\tA\tC\n")
        with pytest.raises(ValueError, match="not an integer"):
            score_cli.parse_vcf(vcf)


class TestScoreFlatten:
    def test_flat_list(self):
        from alphagenome_pytorch.cli.score import _flatten
        assert _flatten([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_nested(self):
        from alphagenome_pytorch.cli.score import _flatten
        assert _flatten([[1, 2], [3, [4, 5]]]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_scalar(self):
        from alphagenome_pytorch.cli.score import _flatten
        assert _flatten(7) == [7.0]


class TestScoreRun:
    """End-to-end: mock model + VariantScoringModel, assert score_variant
    is called with the correct shape (Interval, Variant, scorers=...)."""

    def _make_fake_score(self, variant, interval, scorer):
        """Construct a stand-in object matching VariantScore's accessed attrs."""
        import torch

        class FakeOutputType:
            value = "atac"

        class FakeScore:
            pass

        s = FakeScore()
        s.variant = variant
        s.interval = interval
        s.scorer = scorer
        s.scorer_name = "CenterMaskScorer(atac)"
        s.output_type = FakeOutputType()
        s.is_signed = True
        s.scores = torch.tensor([0.1, -0.2, 0.3])
        s.gene_id = None
        s.gene_name = None
        s.gene_type = None
        s.gene_strand = None
        s.junction_start = None
        s.junction_end = None
        return s

    def test_run_calls_score_variant_with_correct_shape(self, tmp_path, capsys):
        from alphagenome_pytorch.cli import score as score_cli
        from alphagenome_pytorch.variant_scoring import Interval, Variant

        # Need real files for the existence check.
        model_path = tmp_path / "m.pth"
        model_path.write_bytes(b"")
        fasta_path = tmp_path / "g.fa"
        fasta_path.write_text(">chr1\nA\n")

        captured: dict = {}

        class FakeScoringModel:
            def __init__(self, model, **kwargs):
                captured["init_kwargs"] = kwargs

            def score_variant(self, interval, variant, scorers, to_cpu=False):
                captured["interval"] = interval
                captured["variant"] = variant
                captured["scorers"] = scorers
                captured["to_cpu"] = to_cpu
                return [self_outer._make_fake_score(variant, interval, scorers[0])]

        self_outer = self  # noqa: F841 (used in nested class)

        fake_model = mock.MagicMock()
        fake_model.eval.return_value = fake_model

        with mock.patch(
            "alphagenome_pytorch.AlphaGenome.from_pretrained",
            return_value=fake_model,
        ), mock.patch(
            "alphagenome_pytorch.variant_scoring.VariantScoringModel",
            FakeScoringModel,
        ):
            rc = score_cli.run(mock.MagicMock(
                model=str(model_path),
                fasta=str(fasta_path),
                variant="chr22:36201698:A>C",
                vcf=None,
                scorer="atac",
                organism="human",
                width=131072,
                gtf=None,
                polya=None,
                output=None,
                device="cpu",
                json_output=False,
            ))

        assert rc == 0
        # Locks the call shape — this is exactly the bug Copilot pointed at.
        assert isinstance(captured["interval"], Interval)
        assert isinstance(captured["variant"], Variant)
        assert captured["interval"].chromosome == "chr22"
        assert captured["interval"].width == 131072
        assert captured["variant"].position == 36201698
        assert captured["to_cpu"] is True
        assert captured["init_kwargs"]["fasta_path"] == str(fasta_path)
        assert captured["init_kwargs"]["default_organism"] == "human"

    def test_run_emits_json_when_requested(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli
        from alphagenome_pytorch.cli._output import emit_json as orig_emit

        model_path = tmp_path / "m.pth"
        model_path.write_bytes(b"")
        fasta_path = tmp_path / "g.fa"
        fasta_path.write_text(">chr1\nA\n")

        class FakeScoringModel:
            def __init__(self, model, **kwargs):
                pass

            def score_variant(self_inner, interval, variant, scorers, to_cpu=False):
                return [self._make_fake_score(variant, interval, scorers[0])]

        fake_model = mock.MagicMock()
        fake_model.eval.return_value = fake_model

        buf = io.StringIO()

        with mock.patch(
            "alphagenome_pytorch.AlphaGenome.from_pretrained",
            return_value=fake_model,
        ), mock.patch(
            "alphagenome_pytorch.variant_scoring.VariantScoringModel",
            FakeScoringModel,
        ), mock.patch.object(
            score_cli, "emit_json",
            side_effect=lambda data, **kw: orig_emit(data, file=buf),
        ):
            rc = score_cli.run(mock.MagicMock(
                model=str(model_path),
                fasta=str(fasta_path),
                variant="chr22:36201698:A>C",
                vcf=None,
                scorer="atac",
                organism="human",
                width=131072,
                gtf=None,
                polya=None,
                output=None,
                device="cpu",
                json_output=True,
            ))

        assert rc == 0
        data = json.loads(buf.getvalue())
        assert "variants" in data
        assert len(data["variants"]) == 1
        rec = data["variants"][0]
        assert rec["variant"] == "chr22:36201698:A>C"
        assert rec["scores"] == pytest.approx([0.1, -0.2, 0.3])

    def test_run_writes_tsv(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli

        model_path = tmp_path / "m.pth"
        model_path.write_bytes(b"")
        fasta_path = tmp_path / "g.fa"
        fasta_path.write_text(">chr1\nA\n")
        out_path = tmp_path / "scores.tsv"

        class FakeScoringModel:
            def __init__(self, model, **kwargs):
                pass

            def score_variant(self_inner, interval, variant, scorers, to_cpu=False):
                return [self._make_fake_score(variant, interval, scorers[0])]

        fake_model = mock.MagicMock()
        fake_model.eval.return_value = fake_model

        with mock.patch(
            "alphagenome_pytorch.AlphaGenome.from_pretrained",
            return_value=fake_model,
        ), mock.patch(
            "alphagenome_pytorch.variant_scoring.VariantScoringModel",
            FakeScoringModel,
        ):
            rc = score_cli.run(mock.MagicMock(
                model=str(model_path),
                fasta=str(fasta_path),
                variant="chr22:36201698:A>C",
                vcf=None,
                scorer="atac",
                organism="human",
                width=131072,
                gtf=None,
                polya=None,
                output=str(out_path),
                device="cpu",
                json_output=False,
            ))

        assert rc == 0
        text = out_path.read_text()
        lines = text.strip().split("\n")
        # Header + 3 tracks
        assert len(lines) == 4
        assert lines[0].split("\t")[0] == "variant"
        assert lines[1].split("\t")[6] == "0"  # track_index column

    def test_run_missing_model_raises(self, tmp_path):
        from alphagenome_pytorch.cli import score as score_cli

        rc_args = mock.MagicMock(
            model=str(tmp_path / "missing.pth"),
            fasta=str(tmp_path / "missing.fa"),
            variant="chr1:100:A>C",
            vcf=None,
            scorer="atac",
            organism="human",
            width=131072,
            gtf=None,
            polya=None,
            output=None,
            device="cpu",
            json_output=False,
        )
        with pytest.raises(FileNotFoundError, match="Model"):
            score_cli.run(rc_args)
