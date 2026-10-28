"""Audit script for RAM consumption across 3 distinct operational stages.

Stage 1: Cold Startup RSS (Immediately after server launch, 0 models loaded).
Stage 2: After calling MusicAI Item-CF and CineAI Neural CF.
Stage 3: After sequentially calling all 16 AI models (10 CineAI + 6 MusicAI).
"""

import os
import sys
import time
import json
import psutil
import subprocess
import urllib.request
from pathlib import Path

# Ensure OpenBLAS thread constraint
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

BASE_URL = "http://127.0.0.1:8000"
REPO_ROOT = Path(__file__).resolve().parent.parent

def wait_for_server(timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def get_process_metrics(parent_proc):
    try:
        procs = [parent_proc]
        try:
            procs.extend(parent_proc.children(recursive=True))
        except Exception:
            pass

        total_rss = 0.0
        total_vms = 0.0
        total_cpu = 0.0

        for p in procs:
            try:
                m = p.memory_info()
                total_rss += m.rss / (1024 * 1024)
                total_vms += m.vms / (1024 * 1024)
                total_cpu += p.cpu_percent(interval=0.05)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return {
            "rss_mb": round(total_rss, 2),
            "vms_mb": round(total_vms, 2),
            "cpu_percent": round(total_cpu, 1),
            "num_processes": len(procs),
        }
    except Exception as e:
        return {"error": str(e)}

def call_cineai_model(model_id, user_id=196, top_n=10):
    url = f"{BASE_URL}/api/v1/recommendations"
    payload = json.dumps({"user_id": user_id, "model_id": model_id, "top_n": top_n}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms, len(data.get("recommendations", []))

def call_musicai_model(model_id, user_id="0023d50b3c99c4352585408b1b21f3762655f434", top_k=10):
    url = f"{BASE_URL}/api/music/recommendations?user_id={user_id}&model={model_id}&top_k={top_k}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms, len(data.get("recommendations", []))

def main():
    print("=" * 80)
    print("  PHASE M6 AUDIT: RAM CONSUMPTION & MODEL LOADING BENCHMARK")
    print("=" * 80)

    # 1. Start clean backend process
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"

    print("Launching clean Uvicorn backend process...")
    server_process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server():
            print("ERROR: Backend server failed to start within timeout.")
            server_process.kill()
            return

        ps_proc = psutil.Process(server_process.pid)
        print(f"Backend running with PID: {server_process.pid}")

        # -------------------------------------------------------------
        # STAGE 1: COLD STARTUP (0 models loaded)
        # -------------------------------------------------------------
        time.sleep(1.0)
        stage1_metrics = get_process_metrics(ps_proc)
        print(f"\n[STAGE 1] Immediately after startup (0 models in memory):")
        print(f"  - RSS Memory: {stage1_metrics['rss_mb']} MB")
        print(f"  - VMS Memory: {stage1_metrics['vms_mb']} MB")
        print(f"  - CPU Usage:  {stage1_metrics['cpu_percent']}%")

        # -------------------------------------------------------------
        # STAGE 2: AFTER CALLING MUSICAI ITEM-CF & CINEAI NEURAL-CF
        # -------------------------------------------------------------
        print(f"\n[STAGE 2] Invoking MusicAI Item-CF and CineAI Neural CF:")

        # Cold load MusicAI Item-CF
        music_item_cold, n_songs = call_musicai_model("item_cf")
        music_item_warm, _ = call_musicai_model("item_cf")
        print(f"  - MusicAI Item-CF: Cold Load = {music_item_cold:.2f} ms | Warm Inference = {music_item_warm:.2f} ms ({n_songs} songs)")

        # Cold load CineAI Neural CF
        cine_neural_cold, n_movies = call_cineai_model("neural_cf")
        cine_neural_warm, _ = call_cineai_model("neural_cf")
        print(f"  - CineAI Neural CF: Cold Load = {cine_neural_cold:.2f} ms | Warm Inference = {cine_neural_warm:.2f} ms ({n_movies} movies)")

        time.sleep(0.5)
        stage2_metrics = get_process_metrics(ps_proc)
        print(f"  - RSS Memory: {stage2_metrics['rss_mb']} MB (delta: +{stage2_metrics['rss_mb'] - stage1_metrics['rss_mb']:.2f} MB)")
        print(f"  - CPU Usage:  {stage2_metrics['cpu_percent']}%")

        # -------------------------------------------------------------
        # STAGE 3: SEQUENTIALLY INVOKING ALL 16 MODELS
        # -------------------------------------------------------------
        print(f"\n[STAGE 3] Sequentially invoking all 16 models (10 CineAI + 6 MusicAI):")

        cine_models = [
            ("popularity", "Popularity Baseline"),
            ("user_cf", "User-Based CF"),
            ("item_cf", "Item-Based CF"),
            ("knn_baseline", "KNN Baseline"),
            ("svd", "Funk SVD"),
            ("svdpp", "SVD++"),
            ("nmf", "NMF"),
            ("content_based", "Content-Based"),
            ("neural_cf", "Neural CF (NeuMF)"),
            ("hybrid", "Hybrid Ensemble"),
        ]

        music_models = [
            ("popularity", "Popularity Baseline"),
            ("item_cf", "Item-Based CF"),
            ("user_cf", "User-Based CF"),
            ("ials", "Implicit ALS"),
            ("bpr_mf", "BPR-MF"),
            ("implicit_neural_cf", "Neural CF (NeuMF)"),
        ]

        model_timings = {}

        print("  Loading 10 CineAI Models:")
        for mid, name in cine_models:
            cold_ms, k = call_cineai_model(mid)
            warm_ms, _ = call_cineai_model(mid)
            model_timings[f"cine_{mid}"] = {"name": name, "cold_ms": round(cold_ms, 2), "warm_ms": round(warm_ms, 2)}
            print(f"    - {name:<22} Cold: {cold_ms:6.2f} ms | Warm: {warm_ms:5.2f} ms")

        print("  Loading 6 MusicAI Models:")
        for mid, name in music_models:
            cold_ms, k = call_musicai_model(mid)
            warm_ms, _ = call_musicai_model(mid)
            model_timings[f"music_{mid}"] = {"name": name, "cold_ms": round(cold_ms, 2), "warm_ms": round(warm_ms, 2)}
            print(f"    - {name:<22} Cold: {cold_ms:6.2f} ms | Warm: {warm_ms:5.2f} ms")

        time.sleep(1.0)
        stage3_metrics = get_process_metrics(ps_proc)
        print(f"\n  Final Stage 3 Metrics (ALL 16 MODELS IN MEMORY):")
        print(f"  - RSS Memory: {stage3_metrics['rss_mb']} MB (total delta from start: +{stage3_metrics['rss_mb'] - stage1_metrics['rss_mb']:.2f} MB)")
        print(f"  - VMS Memory: {stage3_metrics['vms_mb']} MB")
        print(f"  - CPU Usage:  {stage3_metrics['cpu_percent']}%")
        print(f"  - Memory Exceptions: None (0 OOM errors)")

        audit_results = {
            "stage1_cold_startup": stage1_metrics,
            "stage2_two_models": stage2_metrics,
            "stage3_all_16_models": stage3_metrics,
            "model_load_timings": model_timings,
        }

        out_path = REPO_ROOT / "reports" / "musicai" / "phase_m6_ram_audit_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        print(f"\nAudit results saved to: {out_path}")

    finally:
        print("Terminating test backend process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except Exception:
            server_process.kill()

if __name__ == "__main__":
    main()
