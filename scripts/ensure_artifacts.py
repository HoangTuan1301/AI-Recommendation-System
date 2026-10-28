"""Bootstrap script to ensure all 16 AI model checkpoints and datasets exist.

Verifies the complete inventory of:
- 10 CineAI movie recommendation checkpoints
- 6 MusicAI song recommendation checkpoints
- Core CineAI and MusicAI processed datasets

Features:
- Validates all 25 required runtime artifacts before allowing application startup.
- Safe extraction preventing path traversal (../), absolute paths, and symlink injection.
- SHA-256 archive checksum verification when downloading from ARTIFACTS_DOWNLOAD_URL.
- Never reports READY if any checkpoint or dataset is missing or corrupted (0 bytes).
- 100% compatible with Python 3.11+ on Linux (POSIX) and Windows.
"""

import os
import sys
import tarfile
import urllib.request
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Official SHA-256 for production_artifacts.tar.gz
EXPECTED_ARCHIVE_SHA256 = "649974a0b43645105ad5718271524f434cd242701bf04674e7e5aa9849a7cbd8"

# Complete inventory of 16 model checkpoints required for live inference
ALL_16_CHECKPOINTS: List[Tuple[str, str]] = [
    ("CineAI", "models/checkpoints/popularity.pkl"),
    ("CineAI", "models/checkpoints/user_cf.pkl"),
    ("CineAI", "models/checkpoints/item_cf.pkl"),
    ("CineAI", "models/checkpoints/knn_baseline.pkl"),
    ("CineAI", "models/checkpoints/svd.pkl"),
    ("CineAI", "models/checkpoints/svdpp.pkl"),
    ("CineAI", "models/checkpoints/nmf.pkl"),
    ("CineAI", "models/checkpoints/content_based.pkl"),
    ("CineAI", "models/checkpoints/neural_cf.pkl"),
    ("CineAI", "models/checkpoints/hybrid.pkl"),
    ("MusicAI", "models/checkpoints/music/popularity.pkl"),
    ("MusicAI", "models/checkpoints/music/item_cf.pkl"),
    ("MusicAI", "models/checkpoints/music/user_cf.pkl"),
    ("MusicAI", "models/checkpoints/music/ials.pkl"),
    ("MusicAI", "models/checkpoints/music/bpr_mf.pkl"),
    ("MusicAI", "models/checkpoints/music/implicit_neumf.pkl"),
]

# Essential processed dataset files required for startup and candidate generation
ALL_REQUIRED_DATASETS: List[Tuple[str, str]] = [
    ("CineAI Data", "data/processed/movies_clean.csv"),
    ("CineAI Data", "data/processed/movie_to_idx.json"),
    ("CineAI Data", "data/processed/user_to_idx.json"),
    ("CineAI Data", "data/processed/train.csv"),
    ("MusicAI Data", "data/music/processed/songs.csv"),
    ("MusicAI Data", "data/music/processed/train.csv"),
    ("MusicAI Data", "data/music/processed/song_mapping.csv"),
    ("MusicAI Data", "data/music/processed/user_mapping.csv"),
    ("MusicAI Data", "data/music/processed/dataset_manifest.json"),
]

ALL_REQUIRED_ARTIFACTS = ALL_16_CHECKPOINTS + ALL_REQUIRED_DATASETS


def sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_all_artifacts() -> Tuple[bool, List[str]]:
    """Checks every single required artifact for presence and non-zero size."""
    missing: List[str] = []
    for category, rel_path in ALL_REQUIRED_ARTIFACTS:
        p = ROOT / Path(rel_path)
        if not p.is_file():
            missing.append(f"{rel_path} (NOT FOUND)")
        elif p.stat().st_size == 0:
            missing.append(f"{rel_path} (CORRUPTED / ZERO BYTES)")
    return (len(missing) == 0), missing


def safe_extract_tar(tar: tarfile.TarFile, target_dir: Path):
    """Safely extracts a tar archive preventing path traversal, symlink injection, and absolute paths."""
    resolved_target = target_dir.resolve()
    for member in tar.getmembers():
        # Disallow absolute paths
        if member.name.startswith(("/", "\\")):
            raise ValueError(f"Security error: Unsafe absolute path in archive: '{member.name}'")

        # Resolve destination path
        member_dest = (resolved_target / member.name).resolve()

        # Enforce destination directory boundary (prevent ../ path traversal)
        try:
            member_dest.relative_to(resolved_target)
        except ValueError:
            raise ValueError(f"Security error: Path traversal detected in archive: '{member.name}'")

        # Disallow symlinks and hardlinks
        if member.issym() or member.islnk():
            raise ValueError(f"Security error: Dangerous link detected in archive: '{member.name}'")

    # Safe to extract
    tar.extractall(path=resolved_target)


def pack_artifacts(output_tar: Path):
    """Packs all required runtime artifacts not tracked by Git into production archive."""
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    files_to_pack = [
        # CineAI processed datasets
        ("data/processed/movies_clean.csv", "data/processed/movies_clean.csv"),
        ("data/processed/movie_to_idx.json", "data/processed/movie_to_idx.json"),
        ("data/processed/user_to_idx.json", "data/processed/user_to_idx.json"),
        ("data/processed/train.csv", "data/processed/train.csv"),
        ("data/processed/test.csv", "data/processed/test.csv"),
        ("data/processed/val.csv", "data/processed/val.csv"),
        # MusicAI processed datasets
        ("data/music/processed/songs.csv", "data/music/processed/songs.csv"),
        ("data/music/processed/train.csv", "data/music/processed/train.csv"),
        ("data/music/processed/validation.csv", "data/music/processed/validation.csv"),
        ("data/music/processed/test.csv", "data/music/processed/test.csv"),
        ("data/music/processed/song_mapping.csv", "data/music/processed/song_mapping.csv"),
        ("data/music/processed/user_mapping.csv", "data/music/processed/user_mapping.csv"),
        ("data/music/processed/dataset_manifest.json", "data/music/processed/dataset_manifest.json"),
        # MusicAI model checkpoints (all 6)
        ("models/checkpoints/music/popularity.pkl", "models/checkpoints/music/popularity.pkl"),
        ("models/checkpoints/music/item_cf.pkl", "models/checkpoints/music/item_cf.pkl"),
        ("models/checkpoints/music/user_cf.pkl", "models/checkpoints/music/user_cf.pkl"),
        ("models/checkpoints/music/ials.pkl", "models/checkpoints/music/ials.pkl"),
        ("models/checkpoints/music/bpr_mf.pkl", "models/checkpoints/music/bpr_mf.pkl"),
        ("models/checkpoints/music/implicit_neumf.pkl", "models/checkpoints/music/implicit_neumf.pkl"),
    ]

    print(f"Creating production artifacts archive: {output_tar}...")
    with tarfile.open(output_tar, "w:gz") as tar:
        for rel_src, arcname in files_to_pack:
            p = ROOT / Path(rel_src)
            if p.exists() and p.stat().st_size > 0:
                tar.add(p, arcname=arcname)
                print(f"  + Added: {rel_src} ({p.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  ! ERROR: Missing or empty source file: {rel_src}", file=sys.stderr)
                sys.exit(1)

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    sha = sha256_file(output_tar)
    print(f"\n[DONE] Archive size: {size_mb:.2f} MB")
    print(f"[DONE] Archive SHA-256: {sha}")


def download_and_extract(url: str):
    """Downloads production artifacts archive, verifies SHA-256, and extracts safely."""
    dest_tar = ROOT / "tmp_production_artifacts.tar.gz"
    print(f"[Artifacts] Downloading production artifacts from: {url}...")
    try:
        # Download with timeout
        req = urllib.request.Request(url, headers={"User-Agent": "CineAI-MusicAI-Bootstrap/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_tar, "wb") as out_file:
            while chunk := response.read(65536):
                out_file.write(chunk)

        downloaded_size_mb = dest_tar.stat().st_size / (1024 * 1024)
        downloaded_sha = sha256_file(dest_tar)
        print(f"[Artifacts] Download complete ({downloaded_size_mb:.2f} MB).")
        print(f"[Artifacts] Archive SHA-256: {downloaded_sha}")

        # Verify SHA-256 checksum if configured
        expected_sha = os.getenv("ARTIFACTS_SHA256", EXPECTED_ARCHIVE_SHA256)
        if expected_sha and downloaded_sha.lower() != expected_sha.lower():
            raise ValueError(
                f"Checksum mismatch! Expected SHA-256 '{expected_sha}', but got '{downloaded_sha}'."
            )
        print("[Artifacts] SHA-256 verification passed.")

        # Safe extraction
        print("[Artifacts] Safely extracting artifacts...")
        with tarfile.open(dest_tar, "r:gz") as tar:
            safe_extract_tar(tar, ROOT)

        dest_tar.unlink(missing_ok=True)
        print("[Artifacts] Extraction completed successfully.")
    except Exception as e:
        dest_tar.unlink(missing_ok=True)
        print(f"[Artifacts] FATAL ERROR during download or extraction: {e}", file=sys.stderr)
        raise


def print_status_table():
    """Prints diagnostic table of all 16 checkpoints and datasets."""
    print("=" * 80)
    print("  PRODUCTION ARTIFACTS VERIFICATION (ALL 16 MODELS & DATASETS)")
    print("=" * 80)
    all_ok = True
    for cat, rel in ALL_REQUIRED_ARTIFACTS:
        p = ROOT / Path(rel)
        if p.is_file() and p.stat().st_size > 0:
            status = "OK"
            size_kb = f"{p.stat().st_size / 1024:.1f} KB"
        else:
            status = "FAIL"
            size_kb = "0 KB"
            all_ok = False
        print(f"  [{status:<4}] {cat:<14} | {rel:<48} | {size_kb:>10}")
    print("=" * 80)
    return all_ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--pack":
        target = ROOT / "artifacts" / "production_artifacts.tar.gz"
        pack_artifacts(target)
        return

    # Check if all 16 checkpoints and required datasets are already present
    is_complete, missing_files = verify_all_artifacts()
    if is_complete:
        print("[Artifacts] ALL 16 model checkpoints and required datasets verified present. System READY.")
        return

    # If any file is missing, attempt bootstrap download
    print(f"[Artifacts] NOTICE: {len(missing_files)} / {len(ALL_REQUIRED_ARTIFACTS)} artifacts are missing.")
    for mf in missing_files:
        print(f"  - Missing: {mf}")

    url = os.getenv("ARTIFACTS_DOWNLOAD_URL")
    if not url:
        print(
            "\n[Artifacts] BLOCKER: Required model checkpoints or datasets are missing and\n"
            "ARTIFACTS_DOWNLOAD_URL environment variable is not set.\n"
            "The system cannot start safely. Please configure ARTIFACTS_DOWNLOAD_URL on Render Dashboard.\n",
            file=sys.stderr,
        )
        print_status_table()
        sys.exit(1)

    # Download and unpack
    try:
        download_and_extract(url)
    except Exception as exc:
        print(f"\n[Artifacts] BLOCKER: Artifacts provisioning failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Re-verify complete inventory after extraction
    is_complete_after, remaining_missing = verify_all_artifacts()
    if not is_complete_after:
        print(
            "\n[Artifacts] BLOCKER: Extraction completed, but the following required files are STILL missing:",
            file=sys.stderr,
        )
        for rmf in remaining_missing:
            print(f"  - {rmf}", file=sys.stderr)
        print_status_table()
        sys.exit(1)

    print("[Artifacts] SUCCESS: All 16 model checkpoints and datasets verified. System READY for inference.")


if __name__ == "__main__":
    main()
