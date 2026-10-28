"""Unit tests for production artifact bootstrap and security verification."""

import io
import os
import tarfile
import tempfile
import pytest
from pathlib import Path

from scripts.ensure_artifacts import (
    verify_all_artifacts,
    safe_extract_tar,
    sha256_file,
    ALL_REQUIRED_ARTIFACTS,
)


def test_01_verify_all_artifacts_current_state():
    """Verify that current workspace passes full 25-artifact integrity check."""
    is_ok, missing = verify_all_artifacts()
    assert is_ok is True, f"Artifacts missing: {missing}"
    assert len(missing) == 0


def test_02_detects_missing_file(monkeypatch, tmp_path):
    """Verify that a missing artifact triggers failure."""
    # Temporarily point ROOT to an empty tmp_path
    import scripts.ensure_artifacts as ea
    monkeypatch.setattr(ea, "ROOT", tmp_path)
    
    is_ok, missing = ea.verify_all_artifacts()
    assert is_ok is False
    assert len(missing) == len(ALL_REQUIRED_ARTIFACTS)
    assert any("popularity.pkl" in m for m in missing)


def test_03_detects_corrupted_zero_byte_file(monkeypatch, tmp_path):
    """Verify that a 0-byte file is caught as corrupted."""
    import scripts.ensure_artifacts as ea
    monkeypatch.setattr(ea, "ROOT", tmp_path)
    
    # Create all files as 0 bytes
    for _, rel in ALL_REQUIRED_ARTIFACTS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        
    is_ok, missing = ea.verify_all_artifacts()
    assert is_ok is False
    assert len(missing) == len(ALL_REQUIRED_ARTIFACTS)
    assert any("CORRUPTED / ZERO BYTES" in m for m in missing)


def test_04_safe_extract_blocks_path_traversal(tmp_path):
    """Security Test: Safe extraction rejects path traversal (../) in archive."""
    archive_path = tmp_path / "traversal.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"malicious content"
        ti = tarfile.TarInfo(name="../escaped.txt")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(ValueError, match="Path traversal"):
            safe_extract_tar(tar, dest)


def test_05_safe_extract_blocks_absolute_paths(tmp_path):
    """Security Test: Safe extraction rejects absolute paths in archive."""
    archive_path = tmp_path / "absolute.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"root hijack"
        ti = tarfile.TarInfo(name="/etc/evil.conf")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(ValueError, match="Unsafe absolute path"):
            safe_extract_tar(tar, dest)


def test_06_safe_extract_blocks_symlinks(tmp_path):
    """Security Test: Safe extraction rejects symlink entries."""
    archive_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        ti = tarfile.TarInfo(name="link_to_passwd")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tar.addfile(ti)

    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(ValueError, match="Dangerous link"):
            safe_extract_tar(tar, dest)


def test_07_safe_extract_valid_tar(tmp_path):
    """Verify that a valid archive extracts cleanly and correctly."""
    archive_path = tmp_path / "valid.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"clean model content"
        ti = tarfile.TarInfo(name="models/test_model.pkl")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract_tar(tar, dest)

    extracted_file = dest / "models" / "test_model.pkl"
    assert extracted_file.is_file()
    assert extracted_file.read_bytes() == b"clean model content"
