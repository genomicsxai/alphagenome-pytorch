"""Bundle URI parsing and resolution.

Supports three forms:

- bare path or ``file://`` / ``local:`` URI → local directory.
- ``hf://org/repo[/subdir][@revision]`` → Hugging Face Hub repository
  (revision optional). Requires the ``hf`` optional extra
  (``pip install 'alphagenome-pytorch[hf]'``).

A resolved URI returns a :class:`BundlePaths` rooted at a local directory
containing ``alphagenome_adapter.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from alphagenome_pytorch.extensions.serving.bundle import (
    BundleError,
    BundlePaths,
)


HF_INSTALL_HINT = (
    "Hugging Face Hub support requires the 'hf' extra. "
    "Install with: pip install 'alphagenome-pytorch[hf]'"
)


@dataclass(frozen=True)
class BundleURI:
    """Parsed bundle URI."""

    scheme: Literal["local", "hf"]
    # local: absolute filesystem path
    # hf: f"{org}/{repo}", optionally followed by subdir
    path: str
    subdir: str | None = None
    revision: str | None = None
    raw: str = ""

    @property
    def is_local(self) -> bool:
        return self.scheme == "local"

    @property
    def is_hf(self) -> bool:
        return self.scheme == "hf"

    def hf_repo_id(self) -> str:
        if not self.is_hf:
            raise ValueError(f"URI {self.raw!r} is not a Hugging Face URI")
        return self.path


def parse_bundle_uri(uri: str | os.PathLike[str]) -> BundleURI:
    """Parse a bundle URI string.

    Recognized forms::

        /abs/path/to/bundle
        ./relative/path
        file:///abs/path
        local:/abs/path
        local:./relative
        hf://org/repo
        hf://org/repo/subdir
        hf://org/repo@revision
        hf://org/repo/subdir@revision
    """
    raw = os.fspath(uri)
    s = raw.strip()
    if not s:
        raise BundleError("Empty bundle URI")

    # Plain filesystem path: no scheme, or a Windows drive letter
    if "://" not in s and not s.startswith(("local:", "file:", "hf:")):
        return BundleURI(
            scheme="local",
            path=str(Path(s).expanduser().resolve(strict=False)),
            raw=raw,
        )

    # local:/abs or local:./rel — also tolerate local:// prefix
    if s.startswith("local:"):
        rest = s[len("local:") :]
        if rest.startswith("//"):
            rest = rest[2:]
        if not rest:
            raise BundleError(f"local: URI has empty path: {raw!r}")
        return BundleURI(
            scheme="local",
            path=str(Path(rest).expanduser().resolve(strict=False)),
            raw=raw,
        )

    # file:///abs/path
    if s.startswith("file:"):
        parsed = urlparse(s)
        # urlparse keeps the path component; netloc is usually empty for file://
        path = parsed.path or ""
        if not path:
            raise BundleError(f"file: URI has empty path: {raw!r}")
        return BundleURI(
            scheme="local",
            path=str(Path(path).expanduser().resolve(strict=False)),
            raw=raw,
        )

    # hf://org/repo[/subdir][@revision]
    if s.startswith("hf://"):
        spec = s[len("hf://") :]
        if not spec:
            raise BundleError(f"hf:// URI has empty repo: {raw!r}")
        revision: str | None = None
        if "@" in spec:
            spec, revision = spec.rsplit("@", 1)
            if not revision:
                raise BundleError(f"hf:// URI has empty revision: {raw!r}")
        parts = spec.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise BundleError(
                f"hf:// URI must be hf://org/repo[/subdir]: got {raw!r}"
            )
        repo_id = "/".join(parts[:2])
        subdir = "/".join(parts[2:]) if len(parts) > 2 else None
        if subdir == "":
            subdir = None
        return BundleURI(
            scheme="hf",
            path=repo_id,
            subdir=subdir,
            revision=revision,
            raw=raw,
        )

    raise BundleError(f"Unrecognized bundle URI scheme: {raw!r}")


def resolve_bundle(
    uri: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
) -> BundlePaths:
    """Resolve a bundle URI to a local :class:`BundlePaths`.

    For local URIs this is in-place. For ``hf://`` URIs this lazily imports
    ``huggingface_hub`` and downloads the repository (or just the requested
    subdirectory) into ``cache_dir`` (or the default HF cache).
    """
    parsed = uri if isinstance(uri, BundleURI) else parse_bundle_uri(uri)

    if parsed.is_local:
        local_dir = Path(parsed.path)
        return BundlePaths.resolve(local_dir)

    if parsed.is_hf:
        local_root = _snapshot_hf_repo(parsed, cache_dir=cache_dir, token=token)
        bundle_dir = (
            local_root / parsed.subdir if parsed.subdir else local_root
        )
        return BundlePaths.resolve(bundle_dir)

    raise BundleError(f"Cannot resolve URI: {parsed.raw!r}")


def _snapshot_hf_repo(
    parsed: BundleURI,
    *,
    cache_dir: str | os.PathLike[str] | None,
    token: str | None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(HF_INSTALL_HINT) from exc

    allow_patterns: list[str] | None = None
    if parsed.subdir:
        allow_patterns = [f"{parsed.subdir.rstrip('/')}/*"]

    local_path = snapshot_download(
        repo_id=parsed.hf_repo_id(),
        revision=parsed.revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
        allow_patterns=allow_patterns,
    )
    return Path(local_path)


def publish_bundle(
    bundle_dir: str | os.PathLike[str],
    repo_uri: str | os.PathLike[str],
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a local bundle directory to a Hugging Face repository.

    Returns the URL of the uploaded commit. Requires the ``hf`` extra.
    """
    parsed = repo_uri if isinstance(repo_uri, BundleURI) else parse_bundle_uri(repo_uri)
    if not parsed.is_hf:
        raise BundleError(
            f"publish target must be an hf:// URI, got {parsed.raw!r}"
        )

    # Validate the bundle locally before pushing.
    paths = BundlePaths.resolve(bundle_dir)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(HF_INSTALL_HINT) from exc

    repo_id = parsed.hf_repo_id()

    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
    )

    api = HfApi(token=token)
    msg = commit_message or f"Publish adapter bundle {paths.bundle_dir.name}"
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(paths.bundle_dir),
        path_in_repo=parsed.subdir or "",
        revision=parsed.revision,
        commit_message=msg,
    )
    # CommitInfo has .commit_url on modern huggingface_hub; fall back gracefully.
    return getattr(commit_info, "commit_url", None) or repo_id
