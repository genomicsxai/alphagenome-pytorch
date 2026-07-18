"""Tests for the agt CLI skeleton: parser, dispatch, --json, error format, dep gating."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path
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


# =============================================================================
# agt finetune — flags are declared on the subparser, impl lives in the package
# =============================================================================

class TestFinetuneCommand:
    """Regressions for the two bugs that made 'agt finetune' unusable.

    Previously the subparser took its arguments as argparse.REMAINDER, which
    rejects leading flags ('unrecognized arguments: --mode'), and run() imported
    'scripts.finetune', which is not importable from an installed wheel.
    """

    def test_leading_flags_are_accepted(self):
        """--mode lora used to die with 'unrecognized arguments: --mode'."""
        parser = build_parser()
        args = parser.parse_args(["finetune", "--mode", "lora", "--lr", "0.001"])
        assert args.command == "finetune"
        assert args.mode == "lora"
        assert args.lr == 0.001

    def test_help_lists_real_flags(self):
        """--help must show the training flags, not a REMAINDER placeholder."""
        parser = build_parser()
        sub = parser._subparsers._group_actions[0].choices["finetune"]
        text = sub.format_help()
        for flag in ("--mode", "--genome", "--bigwig", "--lora-rank",
                     "--sequence-parallel", "--pretrained-weights"):
            assert flag in text, f"{flag} missing from 'agt finetune --help'"
        assert "finetune_args" not in text

    def test_does_not_import_scripts_package(self):
        """The impl must resolve from the package, not a repo-root 'scripts' dir."""
        import alphagenome_pytorch.cli.finetune as ft
        src = Path(ft.__file__).read_text()
        assert "from scripts" not in src and "import scripts" not in src

    def test_help_works_outside_repo_via_module_entry(self, tmp_path):
        """The wheel-install case, end to end.

        Run from a cwd outside the repo with PYTHONPATH cleared, so a repo-root
        'scripts' package is not importable. This covers both the module entry
        point torchrun needs and the flags being declared on the subparser.
        """
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-m", "alphagenome_pytorch.cli", "finetune", "--help"],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
        for flag in ("--mode", "--sequence-parallel", "--pretrained-weights"):
            assert flag in proc.stdout, f"{flag} missing from help"

    def test_flag_parity_with_standalone_script_parser(self):
        """agt finetune and scripts/finetune.py must accept the same options."""
        from alphagenome_pytorch.extensions.finetuning.args import build_parser as ft_build

        standalone = {o for a in ft_build()._actions for o in a.option_strings}
        parser = build_parser()
        sub = parser._subparsers._group_actions[0].choices["finetune"]
        via_agt = {o for a in sub._actions for o in a.option_strings}
        assert standalone - via_agt == set(), "flags missing from 'agt finetune'"

    def test_cli_entry_module_exists_for_torchrun(self):
        """torchrun needs a module target: python -m alphagenome_pytorch.cli."""
        import alphagenome_pytorch.cli
        main_py = Path(alphagenome_pytorch.cli.__file__).parent / "__main__.py"
        assert main_py.exists(), "cli/__main__.py missing; torchrun -m would fail"

    def test_run_forwards_parsed_args_to_runner(self):
        """run() should hand the namespace it parsed straight to the runner."""
        from alphagenome_pytorch.cli import finetune as ft_cli

        parser = build_parser()
        args = parser.parse_args([
            "finetune", "--mode", "lora", "--modality", "atac", "--bigwig", "a.bw",
            "--genome", "g.fa", "--train-bed", "t.bed", "--val-bed", "v.bed",
            "--pretrained-weights", "w.pth",
        ])
        args._argv = ["finetune", "--mode", "lora"]

        seen = {}
        fake_runner = types.ModuleType("runner")
        fake_runner.main = lambda a: seen.setdefault("args", a) and 0
        with mock.patch.dict(sys.modules, {
            "alphagenome_pytorch.extensions.finetuning.runner": fake_runner
        }):
            rc = ft_cli.run(args)
        assert rc == 0
        assert seen["args"].mode == "lora"
        # postprocess_args ran: derived fields are present
        assert seen["args"].modalities == ["atac"]


# =============================================================================
# agt predict — gene-count AnnData output
# =============================================================================

class TestPredictAnnData:
    def test_anndata_flags_registered(self):
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", "m.pth", "--output", "out", "--head", "rna_seq",
            "--chromosomes", "chr1", "--fasta", "hg38.fa",
            "--anndata", "counts.h5ad", "--annotation", "genes.parquet",
            "--aggregate-over", "gene-body", "--aggregate-func", "log-mean",
        ])
        assert args.anndata == "counts.h5ad"
        assert args.annotation == "genes.parquet"
        assert args.aggregate_over == "gene-body"
        assert args.aggregate_func == "log-mean"

    def test_defaults_match_script(self):
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", "m.pth", "--output", "out", "--head", "rna_seq",
            "--chromosomes", "chr1", "--fasta", "hg38.fa",
        ])
        assert args.anndata is None
        assert args.aggregate_over == "exons"
        assert args.aggregate_func == "sum"

    def test_anndata_requires_annotation(self, tmp_path):
        from alphagenome_pytorch.cli import predict as predict_cli

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--chromosomes", "chr1", "--fasta", str(fasta),
            "--anndata", "counts.h5ad",
        ])
        with pytest.raises(ValueError, match="requires --annotation"):
            predict_cli.run(args)

    def test_anndata_rejects_non_chromosome_modes(self, tmp_path):
        from alphagenome_pytorch.cli import predict as predict_cli

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        ann = tmp_path / "genes.parquet"; ann.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--locus", "chr1:1-100", "--fasta", str(fasta),
            "--anndata", "counts.h5ad", "--annotation", str(ann),
        ])
        with pytest.raises(ValueError, match="--anndata cannot be combined"):
            predict_cli.run(args)

    def test_missing_annotation_file_raises(self, tmp_path):
        from alphagenome_pytorch.cli import predict as predict_cli

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--chromosomes", "chr1", "--fasta", str(fasta),
            "--anndata", "counts.h5ad", "--annotation", str(tmp_path / "nope.parquet"),
        ])
        with pytest.raises(FileNotFoundError, match="Annotation not found"):
            predict_cli.run(args)


class TestPredictAnnDataWiring:
    """The --aggregate-* flags must map onto the packaged function's kwargs.

    A wrong mapping here yields silently wrong numbers rather than an error,
    so pin the translation: sum -> reduce='sum'; mean -> reduce='mean';
    log-mean -> reduce='mean' with log=True.
    """

    class _FakeModel:
        def __init__(self):
            self.heads = {"rna_seq": object()}

        def eval(self):
            return self

    def _run(self, tmp_path, *extra):
        from alphagenome_pytorch.cli import predict as predict_cli
        from alphagenome_pytorch.extensions import inference as inf

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        ann = tmp_path / "genes.parquet"; ann.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--chromosomes", "chr1,chr2", "--fasta", str(fasta),
            "--anndata", "counts.h5ad", "--annotation", str(ann), "--device", "cpu",
            *extra,
        ])
        captured = {}
        with mock.patch.object(predict_cli, "_load_model",
                               return_value=(self._FakeModel(), None, None)), \
             mock.patch.object(inf, "predict_full_chromosomes_to_anndata",
                               side_effect=lambda **kw: captured.update(kw)):
            rc = predict_cli.run(args)
        assert rc == 0
        return captured

    def test_sum_is_default(self, tmp_path):
        kw = self._run(tmp_path)
        assert kw["over"] == "exons"
        assert kw["reduce"] == "sum"
        assert kw["log"] is False

    def test_mean(self, tmp_path):
        kw = self._run(tmp_path, "--aggregate-func", "mean")
        assert kw["reduce"] == "mean" and kw["log"] is False

    def test_log_mean_maps_to_mean_plus_log(self, tmp_path):
        kw = self._run(tmp_path, "--aggregate-func", "log-mean")
        assert kw["reduce"] == "mean" and kw["log"] is True

    def test_gene_body_maps_to_underscore_form(self, tmp_path):
        """The CLI spells it 'gene-body'; the function expects 'gene_body'."""
        kw = self._run(tmp_path, "--aggregate-over", "gene-body")
        assert kw["over"] == "gene_body"

    def test_output_path_and_chromosomes(self, tmp_path):
        kw = self._run(tmp_path)
        assert kw["output_path"] == str(tmp_path / "counts.h5ad")
        assert kw["chromosomes"] == ["chr1", "chr2"]
        assert kw["head"] == "rna_seq"


# =============================================================================
# agt convert / preprocess — impl must resolve from the package, not scripts/
# =============================================================================

class TestScriptsIndependence:
    """Regression for 'No module named scripts.*' on pip/wheel installs.

    convert and preprocess imported from scripts/ lazily inside run(), so
    --help parsed fine and only execution failed. A subprocess run from a tmp
    cwd with PYTHONPATH cleared is what actually catches that.
    """

    @pytest.mark.parametrize("module", ["convert", "preprocess", "finetune", "predict"])
    def test_cli_module_does_not_import_scripts(self, module):
        import importlib
        mod = importlib.import_module(f"alphagenome_pytorch.cli.{module}")
        src = Path(mod.__file__).read_text()
        assert "from scripts" not in src, f"cli/{module}.py still imports from scripts/"
        assert "import scripts" not in src, f"cli/{module}.py still imports scripts/"

    def test_no_cli_module_imports_scripts(self):
        """Belt and braces: sweep the whole cli package."""
        import alphagenome_pytorch.cli
        cli_dir = Path(alphagenome_pytorch.cli.__file__).parent
        offenders = [
            p.name for p in cli_dir.glob("*.py")
            if "from scripts" in p.read_text() or "import scripts" in p.read_text()
        ]
        assert offenders == [], f"cli modules still importing scripts/: {offenders}"

    def _run_outside_repo(self, tmp_path, argv):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-m", "alphagenome_pytorch.cli", *argv],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
        )

    def test_convert_reaches_real_work_outside_repo(self, tmp_path):
        """Must fail on the checkpoint's contents, not on importing scripts/."""
        ckpt = tmp_path / "fake_ckpt"; ckpt.mkdir()
        proc = self._run_outside_repo(
            tmp_path, ["convert", "--input", str(ckpt), "--output", "out.pth"]
        )
        combined = proc.stdout + proc.stderr
        assert "No module named 'scripts" not in combined, combined
        # Got far enough to hand the path to the checkpoint reader.
        assert "_METADATA" in combined or "Loading JAX checkpoint" in combined, combined

    def test_preprocess_reaches_real_work_outside_repo(self, tmp_path):
        """Must fail inside the BigWig reader, not on importing scripts/."""
        bw = tmp_path / "fake.bw"; bw.write_text("not a bigwig")
        proc = self._run_outside_repo(
            tmp_path,
            ["preprocess", "bigwig-to-mmap", "--input", str(bw), "--output", str(tmp_path / "out")],
        )
        combined = proc.stdout + proc.stderr
        assert "No module named 'scripts" not in combined, combined
        assert "error during file opening" in combined or "bw is NULL" in combined, combined


class TestBigwigToMmapWiring:
    """agt preprocess must delegate to the packaged batch function."""

    def test_honours_workers_flag(self, tmp_path):
        """--workers was accepted but ignored before; it must reach the impl."""
        from alphagenome_pytorch.cli import preprocess as pre_cli
        from alphagenome_pytorch.extensions.finetuning import preprocessing

        seen = {}

        def fake(files, output_dir, **kw):
            seen["files"], seen["output_dir"] = list(files), output_dir
            seen.update(kw)
            return [{"input": f, "output": f"{output_dir}/{i}",
                     "elapsed_s": 0.1, "size_mb": 1.0} for i, f in enumerate(files)]

        parser = build_parser()
        args = parser.parse_args([
            "preprocess", "bigwig-to-mmap", "--input", "a.bw", "b.bw",
            "--output", str(tmp_path), "--workers", "7", "--dtype", "float16",
            "--chromosomes", "chr1", "chr2",
        ])
        with mock.patch.object(preprocessing, "convert_bigwigs_to_mmap", side_effect=fake), \
             mock.patch.object(pre_cli, "require_extra"):
            rc = pre_cli.run(args)

        assert rc == 0
        assert seen["workers"] == 7, "--workers must be forwarded to the impl"
        assert seen["chromosomes"] == ["chr1", "chr2"]
        assert seen["files"] == ["a.bw", "b.bw"]

    def test_json_schema_preserved(self, tmp_path):
        """Assert on the payload handed to emit_json.

        Not via capsys/capfd: emit_json takes `file=sys.stdout` as a default
        argument, bound at import time, so the write lands in whatever stdout
        object existed then — neither fixture's buffer sees it.
        """
        from alphagenome_pytorch.cli import preprocess as pre_cli
        from alphagenome_pytorch.extensions.finetuning import preprocessing

        def fake(files, output_dir, **kw):
            return [{"input": "a.bw", "output": "out/a", "elapsed_s": 0.5, "size_mb": 12.345}]

        parser = build_parser()
        args = parser.parse_args([
            "--json", "preprocess", "bigwig-to-mmap", "--input", "a.bw", "--output", str(tmp_path),
        ])
        with mock.patch.object(preprocessing, "convert_bigwigs_to_mmap", side_effect=fake), \
             mock.patch.object(pre_cli, "require_extra"), \
             mock.patch.object(pre_cli, "emit_json") as emitted:
            rc = pre_cli.run(args)

        assert rc == 0
        payload = emitted.call_args[0][0]
        assert payload["records_processed"] == 1
        assert payload["output_files"] == [{"path": "out/a", "tracks": 1, "size_mb": 12.3}]
        # Must stay JSON-serializable — the --json contract.
        assert json.loads(json.dumps(payload, default=str)) == payload


class TestConvertBigwigsToMmapBatch:
    """The extracted batch helper: layout, ordering, and worker handling."""

    def _fake_convert(self, monkeypatch, delays=None):
        """Stub convert_single_bigwig so no real BigWig I/O is needed."""
        from alphagenome_pytorch.extensions.finetuning import preprocessing

        def stub(bigwig_path, output_dir, chromosomes=None, dtype=None):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if delays:
                import time as _t
                _t.sleep(delays.get(Path(bigwig_path).name, 0))
            return Path(output_dir), 0.01, 1.0

        monkeypatch.setattr(preprocessing, "convert_single_bigwig", stub)
        return preprocessing

    def test_single_file_writes_directly_to_output_dir(self, tmp_path, monkeypatch):
        pre = self._fake_convert(monkeypatch)
        out = tmp_path / "mm"
        recs = pre.convert_bigwigs_to_mmap(["a.bw"], out)
        assert len(recs) == 1
        assert recs[0]["output"] == str(out), "single input must not create a subdir"

    def test_multiple_files_get_stem_subdirs(self, tmp_path, monkeypatch):
        pre = self._fake_convert(monkeypatch)
        out = tmp_path / "mm"
        recs = pre.convert_bigwigs_to_mmap(["x/a.bw", "y/b.bw"], out, workers=1)
        assert [r["output"] for r in recs] == [str(out / "a"), str(out / "b")]

    def test_results_stay_in_input_order_when_parallel(self, tmp_path, monkeypatch):
        """Completion order varies with threads; the returned order must not."""
        pre = self._fake_convert(monkeypatch, delays={"a.bw": 0.15, "b.bw": 0.0, "c.bw": 0.0})
        recs = pre.convert_bigwigs_to_mmap(
            ["a.bw", "b.bw", "c.bw"], tmp_path / "mm", workers=3
        )
        assert [r["input"] for r in recs] == ["a.bw", "b.bw", "c.bw"]

    def test_on_result_fires_once_per_file(self, tmp_path, monkeypatch):
        pre = self._fake_convert(monkeypatch)
        seen = []
        pre.convert_bigwigs_to_mmap(
            ["a.bw", "b.bw"], tmp_path / "mm", workers=2, on_result=seen.append
        )
        assert len(seen) == 2
        assert {Path(r["input"]).name for r in seen} == {"a.bw", "b.bw"}

    def test_exported_from_package(self):
        """datasets.py's docstring points at convert_bigwigs_to_mmap()."""
        import alphagenome_pytorch.extensions.finetuning as ft
        assert callable(ft.convert_bigwigs_to_mmap)


class TestPredictCheckpointTrackNames:
    """--checkpoint + --tracks must label the *selected* tracks.

    Checkpoint metadata names every track in the head. Selecting a subset with
    --tracks narrows the prediction array but not the name list, so writers that
    zip names against columns (write_bigwig does) index past the end. The
    standalone predict_full_chromosome.py subset the names; agt predict did not.
    """

    class _FakeModel:
        def __init__(self):
            self.heads = {"rna_seq": object()}

        def eval(self):
            return self

    def _captured_track_names(self, tmp_path, ckpt_names, extra):
        from alphagenome_pytorch.cli import predict as predict_cli
        from alphagenome_pytorch.extensions import inference as inf

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        ckpt = tmp_path / "ft.pth"; ckpt.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--chromosomes", "chr1", "--fasta", str(fasta),
            "--checkpoint", str(ckpt), "--device", "cpu", *extra,
        ])
        seen = {}
        with mock.patch.object(predict_cli, "_load_model",
                               return_value=(self._FakeModel(), ckpt_names, None)), \
             mock.patch.object(inf, "predict_full_chromosomes_to_bigwig",
                               side_effect=lambda **kw: seen.update(kw) or {}):
            rc = predict_cli.run(args)
        assert rc == 0
        return seen

    def test_names_subset_to_selected_tracks(self, tmp_path):
        names = [f"track_{i}" for i in range(768)]
        seen = self._captured_track_names(tmp_path, names, ["--tracks", "0,5,9"])
        assert seen["track_indices"] == [0, 5, 9]
        assert seen["track_names"] == ["track_0", "track_5", "track_9"], (
            "checkpoint names must be subset to --tracks, or the writer "
            "indexes past the narrowed prediction array"
        )
        assert len(seen["track_names"]) == len(seen["track_indices"])

    def test_full_names_kept_without_track_selection(self, tmp_path):
        names = [f"track_{i}" for i in range(4)]
        seen = self._captured_track_names(tmp_path, names, [])
        assert seen["track_names"] == names

    def test_explicit_track_names_win_and_are_not_resubset(self, tmp_path):
        """--track-names already describes only the selected tracks."""
        names = [f"ckpt_{i}" for i in range(768)]
        seen = self._captured_track_names(
            tmp_path, names, ["--tracks", "0,5,9", "--track-names", "a,b,c"]
        )
        assert seen["track_names"] == ["a", "b", "c"]


class TestPredictStrandFlags:
    """--gene-strand / --track-strands must reach the anndata aggregator."""

    class _FakeModel:
        def __init__(self):
            self.heads = {"rna_seq": object()}

        def eval(self):
            return self

    def _run(self, tmp_path, *extra):
        from alphagenome_pytorch.cli import predict as predict_cli
        from alphagenome_pytorch.extensions import inference as inf

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        ann = tmp_path / "genes.parquet"; ann.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "rna_seq", "--chromosomes", "chr1", "--fasta", str(fasta),
            "--anndata", "c.h5ad", "--annotation", str(ann), "--device", "cpu", *extra,
        ])
        captured = {}
        with mock.patch.object(predict_cli, "_load_model",
                               return_value=(self._FakeModel(), None, None)), \
             mock.patch.object(inf, "predict_full_chromosomes_to_anndata",
                               side_effect=lambda **kw: captured.update(kw)):
            rc = predict_cli.run(args)
        assert rc == 0
        return captured

    def test_default_is_all_strands(self, tmp_path):
        kw = self._run(tmp_path)
        assert kw["strand"] is None, "'all' means no strand filtering"
        assert kw["track_strands"] is None

    def test_match_forwards_strand_and_tracks(self, tmp_path):
        kw = self._run(tmp_path, "--gene-strand", "match", "--track-strands", "+-+.")
        assert kw["strand"] == "match"
        assert kw["track_strands"] == ["+", "-", "+", "."]

    def test_separated_strand_form_accepted(self, tmp_path):
        kw = self._run(tmp_path, "--gene-strand", "match", "--track-strands", "+,-,+,.")
        assert kw["track_strands"] == ["+", "-", "+", "."]

    def test_match_auto_resolves_strands_from_builtin(self, tmp_path):
        """No --track-strands needed for a native head — inferred from metadata."""
        kw = self._run(tmp_path, "--gene-strand", "match")
        assert kw["strand"] == "match"
        assert kw["track_strands"] is not None
        # rna_seq: 768 tracks, strands from the bundled catalog.
        assert len(kw["track_strands"]) == 768
        assert set(kw["track_strands"]) <= {"+", "-", "."}

    def test_match_auto_strands_subset_by_tracks(self, tmp_path):
        kw = self._run(tmp_path, "--gene-strand", "match", "--tracks", "0,5,9")
        assert kw["track_indices"] == [0, 5, 9]
        assert kw["track_strands"] is not None
        assert len(kw["track_strands"]) == 3

    def test_explicit_track_strands_override_metadata(self, tmp_path):
        kw = self._run(tmp_path, "--gene-strand", "match",
                       "--tracks", "0,1,2,3", "--track-strands", "+-+.")
        assert kw["track_strands"] == ["+", "-", "+", "."]

    def test_match_errors_when_no_strand_metadata(self, tmp_path):
        """A head absent from the catalog (and no --track-strands) fails fast."""
        from alphagenome_pytorch.cli import predict as predict_cli
        from alphagenome_pytorch.extensions import inference as inf

        model = tmp_path / "m.pth"; model.write_text("")
        fasta = tmp_path / "g.fa"; fasta.write_text("")
        ann = tmp_path / "genes.parquet"; ann.write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(model), "--output", str(tmp_path),
            "--head", "not_a_real_head", "--chromosomes", "chr1", "--fasta", str(fasta),
            "--anndata", "c.h5ad", "--annotation", str(ann), "--device", "cpu",
            "--gene-strand", "match",
        ])
        # Errors before the model even loads, so _load_model must not be reached.
        with mock.patch.object(predict_cli, "_load_model",
                               side_effect=AssertionError("should fail before load")):
            with pytest.raises(ValueError, match="built-in metadata has none"):
                predict_cli.run(args)

    def test_invalid_strand_chars_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="invalid characters"):
            self._run(tmp_path, "--gene-strand", "match", "--track-strands", "+-x")

    def test_owns_flags_the_script_used_to_declare(self):
        """These lived only in predict_full_chromosome.py before it became a shim."""
        parser = build_parser()
        sub = parser._subparsers._group_actions[0].choices["predict"]
        flags = {o for a in sub._actions for o in a.option_strings}
        for flag in ("--gene-strand", "--track-strands", "--anndata", "--annotation",
                     "--aggregate-over", "--aggregate-func"):
            assert flag in flags


class TestPredictFullChromosomeShim:
    """scripts/predict_full_chromosome.py delegates to agt predict.

    Omitting --chromosomes has always meant chr1-22,chrX for this script, while
    agt predict wants an explicit input selector; the shim restores that default
    rather than letting a previously-valid command start erroring.
    """

    def _shim(self):
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "scripts" / "predict_full_chromosome.py"
        spec = importlib.util.spec_from_file_location("pfc_shim", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_targets_agt_predict(self):
        assert self._shim().to_predict_argv(["--model", "m"])[0] == "predict"

    def test_injects_historical_chromosome_default(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            DEFAULT_CHROMOSOMES,
        )
        out = self._shim().to_predict_argv(["--model", "m", "--fasta", "g", "--head", "atac"])
        assert out[-2:] == ["--chromosomes", ",".join(DEFAULT_CHROMOSOMES)]

    @pytest.mark.parametrize("selector", [
        ["--chromosomes", "chr1"],
        ["--chromosomes=chr1"],
        ["--locus", "chr1:1-100"],
        ["--bed", "r.bed"],
        ["--sequences", "s.fa"],
    ])
    def test_explicit_selector_not_overridden(self, selector):
        out = self._shim().to_predict_argv(["--model", "m", *selector])
        assert out[-len(selector):] == selector
        assert out.count("--chromosomes") <= 1

    def test_default_comes_from_the_package(self):
        """The list must not be re-hardcoded here, or it drifts from the impl."""
        src = (Path(__file__).resolve().parents[2] / "scripts"
               / "predict_full_chromosome.py").read_text()
        assert "DEFAULT_CHROMOSOMES" in src
        assert "chr22" not in src, "chromosome list should come from the package"

    def test_is_thin(self):
        src = (Path(__file__).resolve().parents[2] / "scripts"
               / "predict_full_chromosome.py").read_text()
        assert "add_argument" not in src, "shim must not re-declare flags"


class TestFinetuneParserImportBoundary:
    """The flag layer must not drag in the training module (torch/tqdm).

    args.py reads MODALITY_CONFIGS from the dependency-light `modalities` module,
    and finetuning/__init__.py exposes the rest of training lazily. Building the
    finetune parser should therefore leave `finetuning.training` out of
    sys.modules. Checked in a fresh interpreter, since once any earlier test
    imports training it would show up process-wide.
    """

    TRAINING_MOD = "alphagenome_pytorch.extensions.finetuning.training"

    def _training_loaded_after(self, snippet: str) -> bool:
        code = (
            "import sys\n"
            f"{snippet}\n"
            f"print('LOADED' if {self.TRAINING_MOD!r} in sys.modules else 'ABSENT')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.strip().endswith(("LOADED", "ABSENT")), proc.stdout + proc.stderr
        return proc.stdout.strip().endswith("LOADED")

    def test_building_finetune_parser_does_not_import_training(self):
        loaded = self._training_loaded_after(
            "from alphagenome_pytorch.extensions.finetuning.args import build_parser\n"
            "build_parser()"
        )
        assert not loaded, "building the finetune parser imported finetuning.training"

    def test_building_agt_parser_does_not_import_training(self):
        loaded = self._training_loaded_after(
            "from alphagenome_pytorch.cli._main import build_parser\n"
            "build_parser()"
        )
        assert not loaded, "agt help registration imported finetuning.training"

    def test_control_accessing_a_training_symbol_does_import_it(self):
        """Sanity: the module isn't simply unimportable — using it still loads it."""
        loaded = self._training_loaded_after(
            "from alphagenome_pytorch.extensions.finetuning import train_epoch\n"
            "assert callable(train_epoch)"
        )
        assert loaded, "accessing a lazy training symbol should import the module"


class TestHelpWorksWithCoreDepsOnly:
    """Every --help path must work on a bare install (core deps only).

    A bare `pip install alphagenome-pytorch` has torch/numpy/safetensors but none
    of the optional extras (tqdm, pyBigWig, pandas, ...). Building any subparser
    must not import them, or `agt --help` breaks before it can even tell the user
    which extra to install. Two things kept this honest: the finetune flag layer
    reads MODALITY_CONFIGS from the light `modalities` module (training stays
    lazy), and `full_chromosome` imports tqdm lazily so `info` can read
    HEAD_CONFIGS without it. Runs in a subprocess with those modules blocked.
    """

    # Optional deps a bare install lacks — the union of the pyproject extras.
    OPTIONAL = [
        "tqdm", "pyBigWig", "pyfaidx", "pandas", "pyranges", "pyarrow",
        "anndata", "grpc", "jax", "jaxlib", "haiku", "orbax", "alphagenome",
        "chex", "einshape", "tensorflow", "kagglehub", "aiohttp", "requests",
        "logomaker", "yaml",
    ]

    def test_all_help_paths_work_without_optional_deps(self):
        code = (
            "import builtins, io, contextlib\n"
            f"BLOCKED = set({self.OPTIONAL!r})\n"
            "real = builtins.__import__\n"
            "def fake(name, *a, **k):\n"
            "    if name.split('.')[0] in BLOCKED:\n"
            "        raise ModuleNotFoundError(name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = fake\n"
            "from alphagenome_pytorch.cli._main import main\n"
            "cmds = ([], ['--help'], ['finetune','--help'], ['predict','--help'],\n"
            "        ['info','--help'], ['score','--help'], ['convert','--help'],\n"
            "        ['preprocess','--help'], ['serve','--help'])\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n"
            "    for argv in cmds:\n"
            "        try:\n"
            "            rc = main(argv)\n"
            "            assert rc == 0, (argv, rc)\n"
            "        except SystemExit as e:\n"
            "            assert e.code in (0, None), (argv, e.code)\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.strip().endswith("OK"), proc.stdout + proc.stderr


class TestPredictStrandFromCheckpoint:
    """--gene-strand match infers strands from a finetuned checkpoint's metadata."""

    class _Model:
        def __init__(self, head):
            self.heads = {head: object()}

        def eval(self):
            return self

    def _run(self, tmp_path, meta_rows, head="my_rna", extra=()):
        from alphagenome_pytorch.cli import predict as predict_cli
        from alphagenome_pytorch.extensions import inference as inf

        for n in ("m.pth", "g.fa", "genes.parquet", "ft.pth"):
            (tmp_path / n).write_text("")
        parser = build_parser()
        args = parser.parse_args([
            "predict", "--model", str(tmp_path / "m.pth"), "--output", str(tmp_path),
            "--head", head, "--chromosomes", "chr1", "--fasta", str(tmp_path / "g.fa"),
            "--anndata", "c.h5ad", "--annotation", str(tmp_path / "genes.parquet"),
            "--device", "cpu", "--checkpoint", str(tmp_path / "ft.pth"),
            "--gene-strand", "match", *extra,
        ])
        captured = {}
        with mock.patch.object(predict_cli, "_load_model",
                               return_value=(self._Model(head), None, meta_rows)), \
             mock.patch.object(inf, "predict_full_chromosomes_to_anndata",
                               side_effect=lambda **kw: captured.update(kw)):
            rc = predict_cli.run(args)
        assert rc == 0
        return captured

    def test_resolves_from_checkpoint_metadata(self, tmp_path):
        from alphagenome_pytorch.extensions.finetuning.runner import apply_training_strands
        rows = apply_training_strands(
            None, {"my_rna": ["+", "-", "."]}, {"my_rna": ["a", "b", "c"]}, "human"
        )
        kw = self._run(tmp_path, rows)
        assert kw["track_strands"] == ["+", "-", "."]

    def test_subset_by_tracks(self, tmp_path):
        from alphagenome_pytorch.extensions.finetuning.runner import apply_training_strands
        rows = apply_training_strands(
            None, {"my_rna": ["+", "-", ".", "+"]}, {"my_rna": ["a", "b", "c", "d"]}, "human"
        )
        kw = self._run(tmp_path, rows, extra=("--tracks", "0,3"))
        assert kw["track_strands"] == ["+", "+"]

    def test_errors_when_checkpoint_has_no_strands(self, tmp_path):
        # Custom head, no builtin fallback (never for a checkpoint), no embedded strands.
        with pytest.raises(ValueError, match="checkpoint embeds no strand metadata"):
            self._run(tmp_path, meta_rows=None)


class TestApplyTrainingStrands:
    """runner.apply_training_strands — self-describing, complete-catalog checkpoints."""

    @staticmethod
    def _fn():
        from alphagenome_pytorch.extensions.finetuning.runner import apply_training_strands
        return apply_training_strands

    def test_skeleton_covers_all_heads_not_just_strand_bearing(self):
        """Multimodal run, strands only for rna_seq: atac must still be in the catalog.

        A partial catalog (rna_seq only) is treated as authoritative by serving
        and blanks atac's metadata — the regression this guards against.
        """
        rows = self._fn()(
            None,
            {"rna_seq": ["+", "-"]},                       # strands for rna_seq only
            {"rna_seq": ["r0", "r1"], "atac": ["a0", "a1", "a2"]},
            "human",
        )
        assert {r["output_name"] for r in rows} == {"rna_seq", "atac"}
        rna = [r for r in rows if r["output_name"] == "rna_seq"]
        atac = [r for r in rows if r["output_name"] == "atac"]
        assert [r["strand"] for r in rna] == ["+", "-"]
        assert [r["track_name"] for r in atac] == ["a0", "a1", "a2"]
        assert all("strand" not in r for r in atac), "unspecified strand must stay absent"

    def test_overlays_onto_rich_metadata_preserving_fields(self):
        rich = [
            {"output_name": "rna_seq", "track_index": 0, "track_name": "x", "biosample_name": "K562"},
            {"output_name": "rna_seq", "track_index": 1, "track_name": "y", "biosample_name": "K562"},
        ]
        rows = self._fn()(rich, {"rna_seq": ["+", "-"]}, {"rna_seq": ["x", "y"]}, "human")
        assert [r["strand"] for r in rows] == ["+", "-"]
        assert all(r["biosample_name"] == "K562" for r in rows)  # rich fields kept

    def test_training_strands_override_disagreeing_metadata(self):
        rich = [{"output_name": "rna_seq", "track_index": 0, "strand": "+"}]
        rows = self._fn()(rich, {"rna_seq": ["-"]}, {"rna_seq": ["x"]}, "human")
        assert rows[0]["strand"] == "-"  # training strand wins

    def test_mouse_organism_index(self):
        rows = self._fn()(None, {"rna_seq": ["+"]}, {"rna_seq": ["t0"]}, "mouse")
        assert rows[0]["organism"] == 1

    def test_roundtrips_through_the_predict_reader(self):
        from alphagenome_pytorch.cli.predict import _strands_from_checkpoint
        rows = self._fn()(None, {"h": ["+", "-", "+"]}, {"h": ["a", "b", "c"]}, "human")
        assert _strands_from_checkpoint(rows, "h") == ["+", "-", "+"]
