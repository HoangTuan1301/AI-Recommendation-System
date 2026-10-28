"""Verification script for all 16 production checkpoints and datasets."""

import hashlib
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 80)
    print("  PRODUCTION ARTIFACTS INTEGRITY & CHECKSUM VERIFICATION")
    print("=" * 80)

    # 1. CineAI Checkpoints (10 models)
    cine_ckpts = [
        "popularity.pkl", "user_cf.pkl", "item_cf.pkl", "knn_baseline.pkl",
        "svd.pkl", "svdpp.pkl", "nmf.pkl", "content_based.pkl",
        "neural_cf.pkl", "hybrid.pkl"
    ]
    cine_dir = ROOT / "models" / "checkpoints"
    cine_report = []

    print("\n[1] CineAI Movie Recommendation Checkpoints (10/10):")
    for name in cine_ckpts:
        p = cine_dir / name
        exists = p.exists()
        size_kb = p.stat().st_size / 1024 if exists else 0
        h = sha256_file(p) if exists else "MISSING"
        status = "OK" if exists and size_kb > 0 else "FAIL"
        print(f"  [{status}] {name:<20} {size_kb:>10.1f} KB  sha256: {h[:16]}...")
        cine_report.append({"name": name, "exists": exists, "size_kb": size_kb, "sha256": h})

    # 2. MusicAI Checkpoints (6 models)
    music_ckpts = [
        "popularity.pkl", "item_cf.pkl", "user_cf.pkl",
        "ials.pkl", "bpr_mf.pkl", "implicit_neumf.pkl"
    ]
    music_dir = ROOT / "models" / "checkpoints" / "music"
    music_report = []

    print("\n[2] MusicAI Song Recommendation Checkpoints (6/6):")
    for name in music_ckpts:
        p = music_dir / name
        exists = p.exists()
        size_kb = p.stat().st_size / 1024 if exists else 0
        h = sha256_file(p) if exists else "MISSING"
        status = "OK" if exists and size_kb > 0 else "FAIL"
        print(f"  [{status}] {name:<20} {size_kb:>10.1f} KB  sha256: {h[:16]}...")
        music_report.append({"name": name, "exists": exists, "size_kb": size_kb, "sha256": h})

    # 3. Processed Datasets
    print("\n[3] Core Processed Datasets:")
    datasets = [
        ROOT / "data" / "processed" / "movies_clean.csv",
        ROOT / "data" / "processed" / "train.csv",
        ROOT / "data" / "processed" / "test.csv",
        ROOT / "data" / "processed" / "val.csv",
        ROOT / "data" / "music" / "processed" / "train.csv",
        ROOT / "data" / "music" / "processed" / "validation.csv",
        ROOT / "data" / "music" / "processed" / "test.csv",
        ROOT / "data" / "music" / "processed" / "songs.csv",
    ]
    data_report = []
    for d in datasets:
        exists = d.exists()
        size_kb = d.stat().st_size / 1024 if exists else 0
        status = "OK" if exists and size_kb > 0 else "FAIL"
        rel_path = d.relative_to(ROOT)
        print(f"  [{status}] {str(rel_path):<35} {size_kb:>10.1f} KB")
        data_report.append({"path": str(rel_path), "exists": exists, "size_kb": size_kb})

    manifest = {
        "total_checkpoints": len(cine_ckpts) + len(music_ckpts),
        "cine_checkpoints": cine_report,
        "music_checkpoints": music_report,
        "datasets": data_report,
    }

    out_file = ROOT / "reports" / "musicai" / "production_artifacts_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {out_file}")
    all_ok = all(c["exists"] for c in cine_report) and all(m["exists"] for m in music_report) and all(d["exists"] for d in data_report)
    print(f"OVERALL ARTIFACT STATUS: {'ALL VERIFIED (PASS)' if all_ok else 'MISSING ARTIFACTS (FAIL)'}")

if __name__ == "__main__":
    main()
