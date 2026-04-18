"""Tests for the agt CLI skeleton: parser, dispatch, --json, error format, dep gating."""

from __future__ import annotations

import io
import json
import sys
from unittest import mock

import pytest

from alphagenome_pytorch.cli._main import build_parser, main
from alphagenome_pytorch.cli._output import emit_json, emit_jsonl, emit_error, emit_text
from alphagenome_pytorch.cli._deps import require_extra
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
        args = parser.parse_args(["--json", "serve"])
        assert args.json_output is True
        assert args.command == "serve"

    def test_no_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.json_output is False

    def test_subcommands_registered(self):
        parser = build_parser()
        # Verify each known command name can be parsed (only commands without required args)
        for cmd in ["info", "serve"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

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

    def test_serve_returns_1(self):
        rc = main(["serve"])
        assert rc == 1

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

    def test_exits_when_deps_missing(self):
        with mock.patch(
            "alphagenome_pytorch.cli._deps._EXTRA_PROBES",
            {"test_extra": ["nonexistent_module_xyz"]},
        ):
            with pytest.raises(SystemExit):
                require_extra("test_extra", "test")


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
