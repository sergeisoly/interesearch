from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(directory: Path, files: tuple[str, ...]) -> None:
    """SHA256SUMS in ``directory`` lists exactly ``files`` (paths relative to it), each with its sha256."""
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    if sorted(listed) != sorted(files):
        raise ValueError(f"{directory}: SHA256SUMS lists {sorted(listed)}, expected {sorted(files)}")
    for name, digest in listed.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"{directory / name}: sha256 does not match SHA256SUMS")
