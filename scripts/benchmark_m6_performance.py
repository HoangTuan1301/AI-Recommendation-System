"""Benchmarking script for Phase M6 Multi-Domain System.

Measures real endpoint latencies (P50, P95, P99), throughput, and memory.
"""

import time
import json
import urllib.request
import statistics
import psutil
import os

BASE_URL = "http://127.0.0.1:8000"

def get_backend_ram_mb():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('uvicorn' in arg for arg in cmdline) and any('backend.app.main:app' in arg for arg in cmdline):
                mem = proc.memory_info().rss / (1024 * 1024)
                return mem
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def benchmark_endpoint(name, url, method="GET", body=None, iterations=50):
    latencies = []
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode('utf-8') if body else None

    # Warm-up 2 calls
    for _ in range(2):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req) as resp:
                resp.read()
        except Exception as e:
            pass

    for _ in range(iterations):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            assert resp.status == 200
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    print(f"| {name:<45} | {p50:6.2f} ms | {p95:6.2f} ms | {p99:6.2f} ms | {min_lat:5.2f} ms | {max_lat:6.2f} ms |")
    return {
        "endpoint": name,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(avg, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
        "iterations": iterations,
    }

def main():
    print("=" * 85)
    print("  PHASE M6 REAL API LATENCY & PERFORMANCE BENCHMARK")
    print("=" * 85)
    print(f"| {'Endpoint Target':<45} | {'P50':>9} | {'P95':>9} | {'P99':>9} | {'Min':>8} | {'Max':>9} |")
    print("-" * 85)

    results = []

    # 1. Health checks
    results.append(benchmark_endpoint("GET / (Root Multi-Domain Gateway)", f"{BASE_URL}/", iterations=100))
    results.append(benchmark_endpoint("GET /api/v1/health (CineAI Health)", f"{BASE_URL}/api/v1/health", iterations=50))
    results.append(benchmark_endpoint("GET /api/music/health (MusicAI Health)", f"{BASE_URL}/api/music/health", iterations=50))

    # 2. Model listings
    results.append(benchmark_endpoint("GET /api/v1/models (CineAI 10 Models)", f"{BASE_URL}/api/v1/models", iterations=50))
    results.append(benchmark_endpoint("GET /api/music/models (MusicAI 6 Models)", f"{BASE_URL}/api/music/models", iterations=50))

    # 3. CineAI Recommendation inference
    cine_rec_body = {"user_id": 196, "model_id": "neural_cf", "top_n": 10}
    results.append(benchmark_endpoint("POST /api/v1/recommendations (Neural CF)", f"{BASE_URL}/api/v1/recommendations", method="POST", body=cine_rec_body, iterations=50))

    cine_user_body = {"user_id": 196, "model_id": "user_cf", "top_n": 10}
    results.append(benchmark_endpoint("POST /api/v1/recommendations (User-Based CF)", f"{BASE_URL}/api/v1/recommendations", method="POST", body=cine_user_body, iterations=50))

    # 4. MusicAI Recommendation inference across models
    user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
    for m in ["item_cf", "user_cf", "ials", "bpr_mf", "implicit_neural_cf", "popularity"]:
        results.append(benchmark_endpoint(f"GET /api/music/recommendations ({m})", f"{BASE_URL}/api/music/recommendations?user_id={user_id}&model={m}&top_k=10", iterations=50))

    # 5. Song Details & User History
    results.append(benchmark_endpoint("GET /api/music/songs/{id} (Song Details)", f"{BASE_URL}/api/music/songs/SOFRQTD12A81C233C0", iterations=50))
    results.append(benchmark_endpoint("GET /api/music/users/{id}/history (User History)", f"{BASE_URL}/api/music/users/{user_id}/history", iterations=50))

    print("-" * 85)

    ram = get_backend_ram_mb()
    print(f"\nBackend Process RSS RAM: {ram:.1f} MB" if ram else "\nBackend RAM: N/A")

    out_file = os.path.join(os.path.dirname(__file__), "../reports/musicai/phase_m6_performance_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"measurements": results, "backend_ram_mb": ram}, f, indent=2)
    print(f"Benchmark results saved to: {out_file}")

if __name__ == "__main__":
    main()
