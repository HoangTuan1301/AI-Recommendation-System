"""Phase 6A - Automated API & Recommendation Correctness Audit Script (Updated).

Tests all 8 REST endpoints, 10 models x (Top 5, 10, 20) x representative users,
validates schema, candidate set hygiene, score finiteness, and performance.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import urllib.request
import urllib.error
import pandas as pd

BASE_URL = "http://localhost:8000/api/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

def api_get(endpoint: str) -> tuple[int, Any, float]:
    url = f"{BASE_URL}{endpoint}"
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "Phase6A-Auditor"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data, elapsed_ms
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            err_data = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_data = str(e)
        return e.code, err_data, elapsed_ms

def api_post(endpoint: str, payload: Dict[str, Any]) -> tuple[int, Any, float]:
    url = f"{BASE_URL}{endpoint}"
    body = json.dumps(payload).encode("utf-8")
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Phase6A-Auditor"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data, elapsed_ms
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            err_data = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_data = str(e)
        return e.code, err_data, elapsed_ms

def run_audit() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_endpoints_tested": {},
        "recommendation_correctness": {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "violations": []
        },
        "performance_benchmarks": {
            "model_latencies_ms": {},
            "endpoint_latencies_ms": {}
        },
        "error_handling": {}
    }

    print("=== TASK 2: API INTEGRATION TESTING ===")
    
    # 1. Health
    status, data, lat = api_get("/health")
    report["api_endpoints_tested"]["health"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "valid": status == 200 and data.get("status") == "healthy" and data.get("available_models") == 10
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/health"] = round(lat, 2)
    print(f"  [Health] Status: {status}, Latency: {lat:.2f}ms, Valid: {report['api_endpoints_tested']['health']['valid']}")

    # 2. Models
    status, data, lat = api_get("/models")
    models_list = data.get("models", []) if isinstance(data, dict) else []
    report["api_endpoints_tested"]["models"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "count": len(models_list),
        "valid": status == 200 and len(models_list) == 10
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/models"] = round(lat, 2)
    print(f"  [Models] Status: {status}, Models Count: {len(models_list)}, Valid: {report['api_endpoints_tested']['models']['valid']}")

    # 3. Movies Catalog
    status, data, lat = api_get("/movies?page=1&limit=12&genre=Action&sort_by=rating")
    report["api_endpoints_tested"]["movies_catalog"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "total": data.get("total") if isinstance(data, dict) else None,
        "items_count": len(data.get("items", [])) if isinstance(data, dict) else 0,
        "valid": status == 200 and data.get("total", 0) > 0 and len(data.get("items", [])) == 12
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/movies"] = round(lat, 2)
    print(f"  [Movies Catalog] Status: {status}, Total Action: {data.get('total')}, Latency: {lat:.2f}ms")

    # 4. Movie Detail
    status, data, lat = api_get("/movies/50")  # Star Wars
    report["api_endpoints_tested"]["movie_detail"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "title": data.get("title") if isinstance(data, dict) else None,
        "valid": status == 200 and "Star Wars" in data.get("title", "") and len(data.get("similar_movies", [])) > 0
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/movies/50"] = round(lat, 2)
    print(f"  [Movie Detail] Status: {status}, Title: {data.get('title')}, Latency: {lat:.2f}ms")

    # 5. User Profile (User #1)
    status, data, lat = api_get("/users/1/profile")
    report["api_endpoints_tested"]["user_profile"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "ratings_count": data.get("ratings_count") if isinstance(data, dict) else None,
        "valid": status == 200 and data.get("user_id") == 1 and data.get("ratings_count") == 218
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/users/1/profile"] = round(lat, 2)
    print(f"  [User Profile] Status: {status}, Total Rated in Train: {data.get('ratings_count')}, Latency: {lat:.2f}ms")

    # 6. Metrics Leaderboard
    status, data, lat = api_get("/metrics")
    metrics_list = data.get("metrics", []) if isinstance(data, dict) else []
    report["api_endpoints_tested"]["metrics"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "models_count": len(metrics_list),
        "best_hit_rate_model": data.get("best_hit_rate_model"),
        "valid": status == 200 and len(metrics_list) == 10 and data.get("best_hit_rate_model") == "Neural CF (NeuMF)"
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/metrics"] = round(lat, 2)
    print(f"  [Metrics] Status: {status}, Count: {len(metrics_list)}, Best HR: {data.get('best_hit_rate_model')}")

    # 7. Compare / Model Battle
    status, data, lat = api_post("/compare", {"user_id": 1, "model_a_id": "neural_cf", "model_b_id": "user_cf", "top_n": 10})
    report["api_endpoints_tested"]["compare"] = {
        "status": status,
        "latency_ms": round(lat, 2),
        "jaccard_similarity": data.get("jaccard_similarity") if isinstance(data, dict) else None,
        "overlap_count": data.get("overlap_count") if isinstance(data, dict) else None,
        "valid": status == 200 and isinstance(data.get("jaccard_similarity"), (int, float)) and 0.0 <= data.get("jaccard_similarity") <= 1.0
    }
    report["performance_benchmarks"]["endpoint_latencies_ms"]["/compare"] = round(lat, 2)
    print(f"  [Compare Battle] Status: {status}, Overlap: {data.get('overlap_count')}, Jaccard: {data.get('jaccard_similarity')}")

    # Error cases
    print("\n=== ERROR HANDLING VERIFICATION ===")
    status_404_movie, _, _ = api_get("/movies/99999")
    status_404_user, _, _ = api_get("/users/99999/profile")
    status_422_rec_user, _, _ = api_post("/recommendations", {"user_id": 99999, "model_id": "neural_cf", "top_n": 10})
    status_400_rec_model, _, _ = api_post("/recommendations", {"user_id": 1, "model_id": "invalid_model_xyz", "top_n": 10})
    status_422_bad_k, _, _ = api_post("/recommendations", {"user_id": 1, "model_id": "neural_cf", "top_n": -5})
    
    report["error_handling"] = {
        "movie_99999_status": status_404_movie,
        "user_99999_profile_status": status_404_user,
        "rec_user_99999_status": status_422_rec_user,
        "rec_invalid_model_status": status_400_rec_model,
        "rec_negative_k_status": status_422_bad_k,
        "all_handled_correctly": (
            status_404_movie == 404 and
            status_404_user == 404 and
            status_422_rec_user == 422 and
            status_400_rec_model == 400 and
            status_422_bad_k == 422
        )
    }
    print(f"  Movie 99999: {status_404_movie} (Expected 404)")
    print(f"  User 99999: {status_404_user} (Expected 404)")
    print(f"  Rec User 99999: {status_422_rec_user} (Expected 422 Pydantic Validation)")
    print(f"  Rec Invalid Model: {status_400_rec_model} (Expected 400 Bad Request)")
    print(f"  Rec Negative K: {status_422_bad_k} (Expected 422 Validation)")

    # Read user train histories directly from train.csv to verify NO seen items recommended
    train_csv_path = REPO_ROOT / "data" / "processed" / "train.csv"
    train_df = pd.read_csv(train_csv_path)
    user_seen_dict = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Track checkpoint modification times before inference
    checkpoints_dir = REPO_ROOT / "models" / "checkpoints"
    mtimes_before = {p.name: p.stat().st_mtime for p in checkpoints_dir.glob("*.pkl")}

    print("\n=== TASK 3 & 5: RECOMMENDATION CORRECTNESS & PERFORMANCE ===")
    test_users = [1, 34, 196, 943]  # Diverse user activity profiles
    test_cutoffs = [5, 10, 20]

    for model_meta in models_list:
        mid = model_meta["id"]
        mname = model_meta["name"]
        latencies_for_model = []
        
        print(f"\n>> Auditing Model: {mname} ({mid})")
        for uid in test_users:
            seen_items = user_seen_dict.get(uid, set())
            for k in test_cutoffs:
                report["recommendation_correctness"]["total_checks"] += 1
                status, res, lat = api_post("/recommendations", {"user_id": uid, "model_id": mid, "top_n": k})
                latencies_for_model.append(lat)

                if status != 200:
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": f"HTTP {status}"
                    })
                    continue

                recs = res.get("recommendations", [])
                
                # Check 1: exact count
                if len(recs) != k:
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": f"Expected {k} recs, got {len(recs)}"
                    })
                    continue

                # Check 2: no duplicates
                rec_ids = [r["movie_id"] for r in recs]
                if len(rec_ids) != len(set(rec_ids)):
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": "Duplicate movie_ids in recommendations"
                    })
                    continue

                # Check 3: no seen items from train history
                overlap_seen = set(rec_ids).intersection(seen_items)
                if overlap_seen:
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": f"Recommended seen items: {overlap_seen}"
                    })
                    continue

                # Check 4: valid movie ID range
                invalid_ids = [i for i in rec_ids if not (1 <= i <= 1682)]
                if invalid_ids:
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": f"Invalid movie IDs: {invalid_ids}"
                    })
                    continue

                # Check 5: no NaN or Inf
                scores = [r.get("raw_score") for r in recs if r.get("raw_score") is not None]
                nan_or_inf = [s for s in scores if math.isnan(s) or math.isinf(s)]
                if nan_or_inf:
                    report["recommendation_correctness"]["failed_checks"] += 1
                    report["recommendation_correctness"]["violations"].append({
                        "model": mid, "user": uid, "k": k, "error": f"NaN or Inf score found: {nan_or_inf}"
                    })
                    continue

                # If all passed:
                report["recommendation_correctness"]["passed_checks"] += 1

        avg_lat = sum(latencies_for_model) / len(latencies_for_model)
        min_lat = min(latencies_for_model)
        max_lat = max(latencies_for_model)
        report["performance_benchmarks"]["model_latencies_ms"][mid] = {
            "model_name": mname,
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2)
        }
        print(f"     Latency: avg={avg_lat:.2f}ms (min={min_lat:.2f}ms, max={max_lat:.2f}ms)")

    # Verify checkpoints untouched
    mtimes_after = {p.name: p.stat().st_mtime for p in checkpoints_dir.glob("*.pkl")}
    checkpoint_changes = [k for k in mtimes_before if mtimes_before[k] != mtimes_after.get(k)]
    report["recommendation_correctness"]["checkpoints_unmodified"] = len(checkpoint_changes) == 0
    if checkpoint_changes:
        report["recommendation_correctness"]["violations"].append({
            "error": f"Checkpoints modified during inference: {checkpoint_changes}"
        })
    print(f"\nCheckpoint Immutability: {'PASSED (Zero files changed)' if len(checkpoint_changes) == 0 else 'FAILED'}")

    return report

if __name__ == "__main__":
    rep = run_audit()
    out_path = REPO_ROOT / "results" / "phase6a_backend_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nAudit complete. Detailed report saved to {out_path}")
