"""Storage abstraction layer for the ARIA Data Lake.

Current implementation: ``LocalLakeStorage`` — plain filesystem.
Future implementations can swap in S3/R2/B2 by implementing ``ILakeStorage``
and setting ``"storage_backend"`` in ``data_lake_config.json``.

All paths stored in the index are *relative* to the lake root so that
the entire lake folder is portable (rename root → update config, done).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, List, Protocol, runtime_checkable


@runtime_checkable
class ILakeStorage(Protocol):
    """Minimal storage contract used by lake commands.

    All ``rel_path`` arguments are POSIX-style relative paths from the lake root
    (e.g. ``"sessions/qual_123/inspection_frames/frame_000001.jpg"``).
    Implementations are responsible for converting to OS-native paths as needed.
    """

    def abs_path(self, rel_path: str) -> Path:
        """Return the absolute ``Path`` for *rel_path*."""
        ...

    def exists(self, rel_path: str) -> bool:
        """Return ``True`` if the object exists in the store."""
        ...

    def copy_in(self, src: Path, rel_path: str, *, overwrite: bool = False) -> None:
        """Copy a local file *src* into the store at *rel_path*."""
        ...

    def list_prefix(self, rel_prefix: str, pattern: str = "*") -> Iterator[str]:
        """Yield relative paths that start with *rel_prefix* and match *pattern*.

        ``pattern`` is a glob pattern applied to the filename only (e.g. ``"*.jpg"``).
        """
        ...

    def makedirs(self, rel_dir: str) -> None:
        """Ensure the directory at *rel_dir* exists (and any parents)."""
        ...

    def read_text(self, rel_path: str) -> str:
        """Return the text content of *rel_path*."""
        ...

    def write_text(self, rel_path: str, text: str) -> None:
        """Write *text* to *rel_path*, creating parent directories as needed."""
        ...

    def remove(self, rel_path: str) -> None:
        """Delete *rel_path* from the store."""
        ...


class LocalLakeStorage:
    """Filesystem-backed implementation of ``ILakeStorage``.

    All operations are plain ``pathlib`` / ``shutil`` calls.
    This is the only implementation for the local-first phase.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _full(self, rel_path: str) -> Path:
        # Guard against absolute paths or path traversal
        p = Path(rel_path)
        if p.is_absolute():
            raise ValueError(f"rel_path must be relative, got: {rel_path!r}")
        return (self._root / p).resolve()

    def abs_path(self, rel_path: str) -> Path:
        return self._full(rel_path)

    def exists(self, rel_path: str) -> bool:
        return self._full(rel_path).exists()

    def copy_in(self, src: Path, rel_path: str, *, overwrite: bool = False) -> None:
        dst = self._full(rel_path)
        if dst.exists() and not overwrite:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def list_prefix(self, rel_prefix: str, pattern: str = "*") -> Iterator[str]:
        base = self._full(rel_prefix)
        if not base.exists():
            return
        for p in sorted(base.rglob(pattern)):
            if p.is_file():
                yield p.relative_to(self._root).as_posix()

    def makedirs(self, rel_dir: str) -> None:
        self._full(rel_dir).mkdir(parents=True, exist_ok=True)

    def read_text(self, rel_path: str) -> str:
        return self._full(rel_path).read_text(encoding="utf-8")

    def write_text(self, rel_path: str, text: str) -> None:
        dst = self._full(rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")

    def remove(self, rel_path: str) -> None:
        p = self._full(rel_path)
        if p.exists():
            p.unlink()


def make_storage(root: Path) -> LocalLakeStorage:
    """Factory — returns the appropriate storage backend for *root*.

    For now always returns ``LocalLakeStorage``.  When S3/R2 support is added,
    this function will read ``root/data_lake_config.json`` and dispatch on
    ``"storage_backend"``.
    """
    return LocalLakeStorage(root)
