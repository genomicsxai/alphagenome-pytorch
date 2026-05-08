"""Unit tests for bundle URI parsing and resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from alphagenome_pytorch.extensions.serving.bundle import (
    BundleError,
    DEFAULT_ADAPTER_FILENAME,
    Manifest,
)
from alphagenome_pytorch.extensions.serving.uri import (
    HF_INSTALL_HINT,
    BundleURI,
    parse_bundle_uri,
    publish_bundle,
    resolve_bundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_bundle(tmp_path: Path, *, name: str = "b") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    Manifest(id=name, base_model_hash="sha256:demo").dump(d)
    (d / DEFAULT_ADAPTER_FILENAME).write_bytes(b"")
    return d


# ---------------------------------------------------------------------------
# parse_bundle_uri
# ---------------------------------------------------------------------------


class TestParseBundleUri:
    def test_bare_absolute_path(self, tmp_path: Path) -> None:
        uri = parse_bundle_uri(str(tmp_path))
        assert uri.scheme == "local"
        assert Path(uri.path) == tmp_path

    def test_bare_relative_path(self) -> None:
        uri = parse_bundle_uri("./bundles/foo")
        assert uri.scheme == "local"
        # Relative paths get resolved to absolute.
        assert Path(uri.path).is_absolute()
        assert uri.path.endswith("bundles/foo")

    def test_local_scheme_absolute(self) -> None:
        uri = parse_bundle_uri("local:/abs/path")
        assert uri.scheme == "local"
        assert uri.path == "/abs/path"

    def test_local_scheme_relative(self) -> None:
        uri = parse_bundle_uri("local:./rel")
        assert uri.scheme == "local"
        assert Path(uri.path).is_absolute()

    def test_file_scheme(self) -> None:
        uri = parse_bundle_uri("file:///abs/path")
        assert uri.scheme == "local"
        assert uri.path == "/abs/path"

    def test_hf_basic(self) -> None:
        uri = parse_bundle_uri("hf://org/repo")
        assert uri.scheme == "hf"
        assert uri.path == "org/repo"
        assert uri.subdir is None
        assert uri.revision is None
        assert uri.hf_repo_id() == "org/repo"

    def test_hf_with_subdir(self) -> None:
        uri = parse_bundle_uri("hf://org/repo/sub/dir")
        assert uri.scheme == "hf"
        assert uri.path == "org/repo"
        assert uri.subdir == "sub/dir"
        assert uri.revision is None

    def test_hf_with_revision(self) -> None:
        uri = parse_bundle_uri("hf://org/repo@v1")
        assert uri.scheme == "hf"
        assert uri.subdir is None
        assert uri.revision == "v1"

    def test_hf_with_subdir_and_revision(self) -> None:
        uri = parse_bundle_uri("hf://org/repo/sub@main")
        assert uri.path == "org/repo"
        assert uri.subdir == "sub"
        assert uri.revision == "main"

    @pytest.mark.parametrize("bad", [
        "",
        "  ",
        "hf://",
        "hf://org",
        "hf://org/",
        "hf:///repo",
        "hf://org/repo@",
        "local:",
        "file://",
        "ftp://nope/path",
    ])
    def test_invalid_uris_raise(self, bad: str) -> None:
        with pytest.raises(BundleError):
            parse_bundle_uri(bad)


# ---------------------------------------------------------------------------
# resolve_bundle (local)
# ---------------------------------------------------------------------------


class TestResolveLocal:
    def test_resolve_bare_path(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        paths = resolve_bundle(str(bundle))
        assert paths.bundle_dir == bundle
        assert paths.adapter_safetensors.name == DEFAULT_ADAPTER_FILENAME

    def test_resolve_local_uri(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        paths = resolve_bundle(f"local:{bundle}")
        assert paths.bundle_dir == bundle

    def test_resolve_file_uri(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        paths = resolve_bundle(f"file://{bundle}")
        assert paths.bundle_dir == bundle

    def test_resolve_missing_bundle(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError):
            resolve_bundle(str(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# resolve_bundle (hf://) — fully mocked
# ---------------------------------------------------------------------------


class TestResolveHf:
    def test_hf_resolution_calls_snapshot_download(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path, name="cached")

        fake_module = mock.MagicMock()
        fake_module.snapshot_download.return_value = str(bundle)

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            paths = resolve_bundle("hf://org/repo@v2", token="t", cache_dir=tmp_path)

        assert paths.bundle_dir == bundle
        fake_module.snapshot_download.assert_called_once_with(
            repo_id="org/repo",
            revision="v2",
            cache_dir=str(tmp_path),
            token="t",
            allow_patterns=None,
        )

    def test_hf_resolution_with_subdir_uses_allow_patterns(
        self, tmp_path: Path
    ) -> None:
        local_root = tmp_path / "cached"
        local_root.mkdir()
        # Bundle lives in a subdirectory inside the snapshot.
        sub_bundle = _make_local_bundle(local_root, name="run-42")

        fake_module = mock.MagicMock()
        fake_module.snapshot_download.return_value = str(local_root)

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
            paths = resolve_bundle("hf://org/repo/run-42")

        assert paths.bundle_dir == sub_bundle
        kwargs = fake_module.snapshot_download.call_args.kwargs
        assert kwargs["allow_patterns"] == ["run-42/*"]

    def test_hf_resolution_missing_extra_raises_install_hint(self) -> None:
        # Force the import to fail by removing huggingface_hub from sys.modules
        # and inserting a non-importable stub.
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="alphagenome-pytorch\\[hf\\]"):
                resolve_bundle("hf://org/repo")

    def test_install_hint_message_text(self) -> None:
        # Sanity check on the canonical hint string.
        assert "[hf]" in HF_INSTALL_HINT


# ---------------------------------------------------------------------------
# publish_bundle
# ---------------------------------------------------------------------------


class TestPublishBundle:
    def test_publish_calls_create_repo_and_upload(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        fake = mock.MagicMock()
        fake.HfApi.return_value.upload_folder.return_value = mock.MagicMock(
            commit_url="https://hf.co/org/repo/commit/abc"
        )

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            url = publish_bundle(
                str(bundle), "hf://org/repo", token="tok", private=True,
                commit_message="hi",
            )

        assert url == "https://hf.co/org/repo/commit/abc"
        fake.create_repo.assert_called_once()
        fake.HfApi.assert_called_once_with(token="tok")
        upload_kwargs = fake.HfApi.return_value.upload_folder.call_args.kwargs
        assert upload_kwargs["repo_id"] == "org/repo"
        assert upload_kwargs["folder_path"] == str(bundle)
        assert upload_kwargs["path_in_repo"] == ""
        assert upload_kwargs["commit_message"] == "hi"

    def test_publish_with_subdir(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        fake = mock.MagicMock()
        fake.HfApi.return_value.upload_folder.return_value = mock.MagicMock(
            commit_url=None
        )

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            url = publish_bundle(str(bundle), "hf://org/repo/runs/1@main")

        kwargs = fake.HfApi.return_value.upload_folder.call_args.kwargs
        assert kwargs["path_in_repo"] == "runs/1"
        assert kwargs["revision"] == "main"
        assert url == "org/repo"  # falls back to repo id when commit_url is missing

    def test_publish_rejects_local_target(self, tmp_path: Path) -> None:
        bundle = _make_local_bundle(tmp_path)
        with pytest.raises(BundleError, match="hf://"):
            publish_bundle(str(bundle), "local:/somewhere")

    def test_publish_validates_bundle_first(self, tmp_path: Path) -> None:
        # Empty dir — not a bundle.
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(BundleError):
            publish_bundle(str(empty), "hf://org/repo")


# ---------------------------------------------------------------------------
# serve._resolve_checkpoint_arg
# ---------------------------------------------------------------------------


class TestServeCheckpointResolution:
    def test_passes_through_file_path(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.extensions.serving.cli import (
            _resolve_checkpoint_arg,
        )

        f = tmp_path / "ck.delta.pth"
        f.write_bytes(b"x")
        assert _resolve_checkpoint_arg(str(f)) == str(f)

    def test_resolves_local_bundle_dir(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.extensions.serving.cli import (
            _resolve_checkpoint_arg,
        )

        bundle = _make_local_bundle(tmp_path)
        result = _resolve_checkpoint_arg(str(bundle))
        assert result == str(bundle / DEFAULT_ADAPTER_FILENAME)

    def test_resolves_local_uri(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.extensions.serving.cli import (
            _resolve_checkpoint_arg,
        )

        bundle = _make_local_bundle(tmp_path)
        result = _resolve_checkpoint_arg(f"local:{bundle}")
        assert result == str(bundle / DEFAULT_ADAPTER_FILENAME)

    def test_missing_plain_path_passes_through(self, tmp_path: Path) -> None:
        """Plain (non-URI) paths are passed through; downstream loader surfaces errors."""
        from alphagenome_pytorch.extensions.serving.cli import (
            _resolve_checkpoint_arg,
        )

        bogus = str(tmp_path / "nope.delta.pth")
        assert _resolve_checkpoint_arg(bogus) == bogus

    def test_missing_local_uri_raises(self, tmp_path: Path) -> None:
        """Explicit local: URIs that don't exist should still raise."""
        from alphagenome_pytorch.extensions.serving.cli import (
            _resolve_checkpoint_arg,
        )

        with pytest.raises(FileNotFoundError):
            _resolve_checkpoint_arg(f"local:{tmp_path / 'nope'}")


# ---------------------------------------------------------------------------
# CLI wiring: agt adapters pull (with mocked HF)
# ---------------------------------------------------------------------------


class TestPullCli:
    def test_pull_local_bundle(self, tmp_path: Path) -> None:
        import argparse, io
        from alphagenome_pytorch.cli import adapters as adapters_cli
        from alphagenome_pytorch.cli._output import emit_text as orig_emit_text

        bundle = _make_local_bundle(tmp_path)
        buf = io.StringIO()
        with mock.patch.object(
            adapters_cli, "emit_text",
            side_effect=lambda text, **kw: orig_emit_text(text, file=buf),
        ):
            rc = adapters_cli.run(argparse.Namespace(
                adapters_command="pull",
                uri=str(bundle),
                cache_dir=None,
                token=None,
                json_output=False,
            ))
        assert rc == 0
        assert str(bundle) in buf.getvalue()

    def test_pull_hf_uri_uses_snapshot_download(self, tmp_path: Path) -> None:
        import argparse, io
        from alphagenome_pytorch.cli import adapters as adapters_cli
        from alphagenome_pytorch.cli._output import emit_text as orig_emit_text

        bundle = _make_local_bundle(tmp_path, name="cached")
        fake_module = mock.MagicMock()
        fake_module.snapshot_download.return_value = str(bundle)

        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}), \
             mock.patch.object(
                 adapters_cli, "emit_text",
                 side_effect=lambda text, **kw: orig_emit_text(text, file=buf),
             ):
            rc = adapters_cli.run(argparse.Namespace(
                adapters_command="pull",
                uri="hf://org/repo",
                cache_dir=None,
                token=None,
                json_output=False,
            ))
        assert rc == 0
        assert str(bundle) in buf.getvalue()
        fake_module.snapshot_download.assert_called_once()
