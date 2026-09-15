"""
GARAJ CPU & Resource Profiling Script
Measures real-time CPU, RAM, and per-component latency breakdowns across 60+ seconds per configuration.
Configurations tested:
  A. W2V2-AASIST Baseline
  B. Acoustic-Only Classifier
  C. C3 Fusion Model
"""

import sys
import os
import time
import asyncio
import struct
import numpy as np
import psutil
import torch
from typing import Dict, List, Any

# Ensure backend root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.detection.model_loader import load_xlsr_aasist_model
from app.features.acoustic_extractor import AcousticFeatureExtractor
from app.detection.feature_fusion import W2V2AASISTFeatureFusion, AcousticOnlyClassifier
from app.detection.aasist import AASISTBackEnd


def measure_idle_cpu(duration_sec: float = 10.0):
    print(f"--- Measuring System & Process Idle Baseline ({duration_sec}s) ---")
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime
    sys_cpu_samples = []
    proc_cpu_samples = []
    ram_samples = []

    start = time.time()
    while time.time() - start < duration_sec:
        time.sleep(0.5)
        sys_cpu_samples.append(psutil.cpu_percent(interval=None))
        proc_cpu_samples.append(proc.cpu_percent(interval=None))
        ram_samples.append(proc.memory_info().rss / (1024 * 1024))

    return {
        "sys_cpu_idle_avg": round(float(np.mean(sys_cpu_samples)), 2),
        "proc_cpu_idle_avg": round(float(np.mean(proc_cpu_samples)), 2),
        "ram_baseline_mb": round(float(np.mean(ram_samples)), 2),
    }


def profile_feature_extraction(audio_window: np.ndarray, num_windows: int = 50):
    print(f"\n--- Profiling Acoustic Feature Extraction Breakdown ({num_windows} runs) ---")
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Sub-component profiling
    t_stft, t_spectral, t_lfcc, t_cqcc, t_prosodic, t_energy, t_total = [], [], [], [], [], [], []

    # Prime
    extractor.extract_features(audio_window)

    proc = psutil.Process()
    proc.cpu_percent(interval=None)

    for _ in range(num_windows):
        t0 = time.perf_counter()

        # STFT
        import scipy.signal
        frequencies, times, stft_complex = scipy.signal.stft(
            audio_window, fs=16000, nperseg=512, noverlap=512 - 160
        )
        stft_mag = np.abs(stft_complex) + 1e-8
        t1 = time.perf_counter()

        # Spectral
        spec = extractor.extract_spectral(stft_mag)
        t2 = time.perf_counter()

        # LFCC
        lfcc = extractor.extract_lfcc(stft_mag)
        t3 = time.perf_counter()

        # CQCC
        cqcc = extractor.extract_cqcc(audio_window)
        t4 = time.perf_counter()

        # Prosodic
        prosodic = extractor.extract_prosodic(audio_window)
        t5 = time.perf_counter()

        # Energy
        energy = extractor.extract_energy(audio_window, stft_mag)
        t6 = time.perf_counter()

        t_stft.append((t1 - t0) * 1000.0)
        t_spectral.append((t2 - t1) * 1000.0)
        t_lfcc.append((t3 - t2) * 1000.0)
        t_cqcc.append((t4 - t3) * 1000.0)
        t_prosodic.append((t5 - t4) * 1000.0)
        t_energy.append((t6 - t5) * 1000.0)
        t_total.append((t6 - t0) * 1000.0)

    proc_cpu = proc.cpu_percent(interval=None)
    ram_mb = proc.memory_info().rss / (1024 * 1024)

    return {
        "stft_ms_avg": round(float(np.mean(t_stft)), 3),
        "spectral_ms_avg": round(float(np.mean(t_spectral)), 3),
        "lfcc_ms_avg": round(float(np.mean(t_lfcc)), 3),
        "cqcc_ms_avg": round(float(np.mean(t_cqcc)), 3),
        "prosodic_ms_avg": round(float(np.mean(t_prosodic)), 3),
        "energy_ms_avg": round(float(np.mean(t_energy)), 3),
        "total_feat_ms_avg": round(float(np.mean(t_total)), 3),
        "total_feat_ms_p50": round(float(np.percentile(t_total, 50)), 3),
        "total_feat_ms_p95": round(float(np.percentile(t_total, 95)), 3),
        "total_feat_ms_p99": round(float(np.percentile(t_total, 99)), 3),
        "feat_cpu_percent": round(float(proc_cpu), 2),
        "ram_mb": round(float(ram_mb), 2),
    }


def profile_config_a_w2v2_aasist(audio_window: np.ndarray, duration_sec: float = 60.0):
    print(f"\n=======================================================")
    print(f"  Profiling Config A: W2V2-AASIST Baseline ({duration_sec}s)  ")
    print(f"=======================================================")

    feat_extractor, model, device, meta = load_xlsr_aasist_model()
    if model is None:
        raise RuntimeError("Model checkpoint LA_model.pth could not be loaded!")

    model.eval()
    audio_tensor = torch.from_numpy(audio_window).unsqueeze(0).to(device)

    # Warmup / Prime PyTorch
    with torch.inference_mode():
        _ = model(audio_tensor)

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    ram_baseline = proc.memory_info().rss / (1024 * 1024)

    t_prep = []
    t_w2v2 = []
    t_aasist = []
    t_total = []
    cpu_samples = []
    ram_samples = []

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        t0 = time.perf_counter()

        # 1. Audio Preprocessing
        if audio_window.dtype != np.float32:
            proc_audio = audio_window.astype(np.float32)
        else:
            proc_audio = audio_window
        rms = float(np.sqrt(np.mean(np.square(proc_audio))))
        peak = float(np.max(np.abs(proc_audio)))
        tensor_in = torch.from_numpy(proc_audio).unsqueeze(0).to(device)
        t1 = time.perf_counter()

        # 2. W2V2 / XLS-R Backbone Forward Pass
        with torch.inference_mode():
            ssl_feats = model.ssl_model(tensor_in)
            t2 = time.perf_counter()

            # 3. AASIST Graph Backend Forward Pass
            logits = AASISTBackEnd.forward(model, ssl_feats)
            t3 = time.perf_counter()

        t_prep.append((t1 - t0) * 1000.0)
        t_w2v2.append((t2 - t1) * 1000.0)
        t_aasist.append((t3 - t2) * 1000.0)
        t_total.append((t3 - t0) * 1000.0)

        iterations += 1
        if iterations % 5 == 0:
            cpu_samples.append(proc.cpu_percent(interval=None))
            ram_samples.append(proc.memory_info().rss / (1024 * 1024))

    return {
        "config": "Config A: W2V2-AASIST Baseline",
        "duration_sec": round(time.time() - start_time, 2),
        "iterations": iterations,
        "cpu_percent_avg": round(float(np.mean(cpu_samples)), 2) if cpu_samples else round(float(proc.cpu_percent(interval=None)), 2),
        "ram_baseline_mb": round(float(ram_baseline), 2),
        "ram_peak_mb": round(float(np.max(ram_samples)), 2) if ram_samples else round(float(proc.memory_info().rss / (1024 * 1024)), 2),
        "prep_ms_avg": round(float(np.mean(t_prep)), 3),
        "w2v2_ms_avg": round(float(np.mean(t_w2v2)), 3),
        "aasist_ms_avg": round(float(np.mean(t_aasist)), 3),
        "total_ms_avg": round(float(np.mean(t_total)), 3),
        "total_ms_p50": round(float(np.percentile(t_total, 50)), 3),
        "total_ms_p95": round(float(np.percentile(t_total, 95)), 3),
        "total_ms_p99": round(float(np.percentile(t_total, 99)), 3),
        "w2v2_ms_p50": round(float(np.percentile(t_w2v2, 50)), 3),
        "aasist_ms_p50": round(float(np.percentile(t_aasist, 50)), 3),
    }


def profile_config_b_acoustic_only(audio_window: np.ndarray, duration_sec: float = 60.0):
    print(f"\n=======================================================")
    print(f"  Profiling Config B: Acoustic-Only ({duration_sec}s)      ")
    print(f"=======================================================")

    extractor = AcousticFeatureExtractor(sample_rate=16000)
    classifier = AcousticOnlyClassifier(in_dim=extractor.get_dim(), hidden_dim=128)
    classifier.eval()

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    ram_baseline = proc.memory_info().rss / (1024 * 1024)

    t_prep = []
    t_feat = []
    t_classifier = []
    t_total = []
    cpu_samples = []
    ram_samples = []

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        t0 = time.perf_counter()

        # 1. Preprocessing
        proc_audio = audio_window if audio_window.dtype == np.float32 else audio_window.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(proc_audio))))
        t1 = time.perf_counter()

        # 2. Acoustic Feature Extraction
        feats = extractor.extract_features(proc_audio)
        t2 = time.perf_counter()

        # 3. Classifier Head Forward Pass
        feats_tensor = torch.from_numpy(feats).unsqueeze(0).float()
        with torch.inference_mode():
            logits = classifier(feats_tensor)
        t3 = time.perf_counter()

        t_prep.append((t1 - t0) * 1000.0)
        t_feat.append((t2 - t1) * 1000.0)
        t_classifier.append((t3 - t2) * 1000.0)
        t_total.append((t3 - t0) * 1000.0)

        iterations += 1
        if iterations % 10 == 0:
            cpu_samples.append(proc.cpu_percent(interval=None))
            ram_samples.append(proc.memory_info().rss / (1024 * 1024))

    return {
        "config": "Config B: Acoustic-Only",
        "duration_sec": round(time.time() - start_time, 2),
        "iterations": iterations,
        "cpu_percent_avg": round(float(np.mean(cpu_samples)), 2) if cpu_samples else round(float(proc.cpu_percent(interval=None)), 2),
        "ram_baseline_mb": round(float(ram_baseline), 2),
        "ram_peak_mb": round(float(np.max(ram_samples)), 2) if ram_samples else round(float(proc.memory_info().rss / (1024 * 1024)), 2),
        "prep_ms_avg": round(float(np.mean(t_prep)), 3),
        "feat_ms_avg": round(float(np.mean(t_feat)), 3),
        "classifier_ms_avg": round(float(np.mean(t_classifier)), 3),
        "total_ms_avg": round(float(np.mean(t_total)), 3),
        "total_ms_p50": round(float(np.percentile(t_total, 50)), 3),
        "total_ms_p95": round(float(np.percentile(t_total, 95)), 3),
        "total_ms_p99": round(float(np.percentile(t_total, 99)), 3),
        "feat_ms_p50": round(float(np.percentile(t_feat, 50)), 3),
    }


def profile_config_c_fusion(audio_window: np.ndarray, duration_sec: float = 60.0):
    print(f"\n=======================================================")
    print(f"  Profiling Config C: C3 Fusion Model ({duration_sec}s)   ")
    print(f"=======================================================")

    feat_extractor, base_model, device, meta = load_xlsr_aasist_model()
    if base_model is None:
        raise RuntimeError("Base model LA_model.pth could not be loaded!")

    fusion_model = W2V2AASISTFeatureFusion(base_aasist_model=base_model)
    fusion_model.to(device)
    fusion_model.eval()

    audio_tensor = torch.from_numpy(audio_window).unsqueeze(0).to(device)

    # Warmup / Prime PyTorch
    with torch.inference_mode():
        _ = fusion_model(audio_tensor)

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    ram_baseline = proc.memory_info().rss / (1024 * 1024)

    t_prep = []
    t_acoustic_feat = []
    t_w2v2_aasist = []
    t_fusion_head = []
    t_total = []
    cpu_samples = []
    ram_samples = []

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration_sec:
        t0 = time.perf_counter()

        # 1. Audio Preprocessing
        proc_audio = audio_window if audio_window.dtype == np.float32 else audio_window.astype(np.float32)
        tensor_in = torch.from_numpy(proc_audio).unsqueeze(0).to(device)
        t1 = time.perf_counter()

        # 2. Acoustic Feature Extraction
        with torch.inference_mode():
            ac_feats = fusion_model.compute_acoustic_features(tensor_in)
            t2 = time.perf_counter()

            # 3. W2V2-AASIST Embeddings
            aasist_emb = fusion_model.extract_aasist_embeddings(tensor_in)
            t3 = time.perf_counter()

            # 4. Fusion Projection & Classification
            ac_proj = fusion_model.acoustic_proj(ac_feats)
            fused = torch.cat([aasist_emb, ac_proj], dim=-1)
            logits = fusion_model.fusion_head(fused)
            t4 = time.perf_counter()

        t_prep.append((t1 - t0) * 1000.0)
        t_acoustic_feat.append((t2 - t1) * 1000.0)
        t_w2v2_aasist.append((t3 - t2) * 1000.0)
        t_fusion_head.append((t4 - t3) * 1000.0)
        t_total.append((t4 - t0) * 1000.0)

        iterations += 1
        if iterations % 5 == 0:
            cpu_samples.append(proc.cpu_percent(interval=None))
            ram_samples.append(proc.memory_info().rss / (1024 * 1024))

    return {
        "config": "Config C: C3 Fusion Model",
        "duration_sec": round(time.time() - start_time, 2),
        "iterations": iterations,
        "cpu_percent_avg": round(float(np.mean(cpu_samples)), 2) if cpu_samples else round(float(proc.cpu_percent(interval=None)), 2),
        "ram_baseline_mb": round(float(ram_baseline), 2),
        "ram_peak_mb": round(float(np.max(ram_samples)), 2) if ram_samples else round(float(proc.memory_info().rss / (1024 * 1024)), 2),
        "prep_ms_avg": round(float(np.mean(t_prep)), 3),
        "acoustic_feat_ms_avg": round(float(np.mean(t_acoustic_feat)), 3),
        "w2v2_aasist_ms_avg": round(float(np.mean(t_w2v2_aasist)), 3),
        "fusion_head_ms_avg": round(float(np.mean(t_fusion_head)), 3),
        "total_ms_avg": round(float(np.mean(t_total)), 3),
        "total_ms_p50": round(float(np.percentile(t_total, 50)), 3),
        "total_ms_p95": round(float(np.percentile(t_total, 95)), 3),
        "total_ms_p99": round(float(np.percentile(t_total, 99)), 3),
        "w2v2_aasist_ms_p50": round(float(np.percentile(t_w2v2_aasist, 50)), 3),
        "acoustic_feat_ms_p50": round(float(np.percentile(t_acoustic_feat, 50)), 3),
    }


async def profile_websocket_streaming(duration_sec: float = 60.0):
    print(f"\n=======================================================")
    print(f"  Profiling Live WebSocket Streaming Server ({duration_sec}s) ")
    print(f"=======================================================")
    import websockets

    # Find FastAPI uvicorn PID
    uvicorn_proc = None
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "uvicorn" in cmd and "app.main:app" in cmd:
                uvicorn_proc = p
                break
        except Exception:
            pass

    if uvicorn_proc is None:
        print("Uvicorn backend process not found! Skipping server process CPU tracking.")
        return {}

    # Sample 16kHz audio chunk (100ms = 1600 samples = 3200 bytes)
    sample_chunk = (np.random.randn(1600) * 1000).astype(np.int16).tobytes()
    ws_url = "ws://localhost:8000/ws/stream"
    
    sys_cpu_samples = []
    proc_cpu_samples = []
    ram_samples = []
    latencies = []

    uvicorn_proc.cpu_percent(interval=None)

    async with websockets.connect(ws_url) as ws:
        start_time = time.time()
        seq = 0
        while time.time() - start_time < duration_sec:
            t0 = time.perf_counter()
            seq += 1
            h = struct.pack("<Id", seq, time.time() * 1000.0)
            await ws.send(h + sample_chunk)
            resp = await ws.recv()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)

            if seq % 10 == 0:
                sys_cpu_samples.append(psutil.cpu_percent(interval=None))
                proc_cpu_samples.append(uvicorn_proc.cpu_percent(interval=None))
                ram_samples.append(uvicorn_proc.memory_info().rss / (1024 * 1024))

            # Simulate real-time streaming cadence (100ms per frame)
            await asyncio.sleep(0.08)

    return {
        "duration_sec": round(time.time() - start_time, 2),
        "chunks_sent": seq,
        "sys_cpu_streaming_avg": round(float(np.mean(sys_cpu_samples)), 2),
        "proc_cpu_streaming_avg": round(float(np.mean(proc_cpu_samples)), 2),
        "ram_streaming_mb": round(float(np.mean(ram_samples)), 2),
        "ram_peak_mb": round(float(np.max(ram_samples)), 2),
        "ws_rtt_latency_ms_avg": round(float(np.mean(latencies)), 3),
        "ws_rtt_latency_ms_p50": round(float(np.percentile(latencies, 50)), 3),
        "ws_rtt_latency_ms_p95": round(float(np.percentile(latencies, 95)), 3),
        "ws_rtt_latency_ms_p99": round(float(np.percentile(latencies, 99)), 3),
    }


def main():
    print("=========================================================")
    print("      GARAJ SYSTEM RESOURCE & CPU PROFILING RUNNER      ")
    print("=========================================================")

    # Generate synthetic 64,600 samples (4.04s @ 16kHz float32)
    np.random.seed(42)
    audio_window = (np.random.randn(64600) * 0.05).astype(np.float32)

    # 1. Measure Idle Baseline
    idle_res = measure_idle_cpu(duration_sec=10.0)

    # 2. Measure Acoustic Sub-Component Breakdown
    feat_breakdown = profile_feature_extraction(audio_window, num_windows=50)

    # 3. Profile Configuration A (W2V2-AASIST Baseline) for 60 seconds
    res_a = profile_config_a_w2v2_aasist(audio_window, duration_sec=60.0)

    # 4. Profile Configuration B (Acoustic-Only) for 60 seconds
    res_b = profile_config_b_acoustic_only(audio_window, duration_sec=60.0)

    # 5. Profile Configuration C (C3 Fusion) for 60 seconds
    res_c = profile_config_c_fusion(audio_window, duration_sec=60.0)

    # 6. Profile Live WebSocket Streaming for 60 seconds
    try:
        ws_res = asyncio.run(profile_websocket_streaming(duration_sec=60.0))
    except Exception as exc:
        print(f"WebSocket profiling error: {exc}")
        ws_res = {}

    print("\n\n" + "=" * 65)
    print("                      PROFILING SUMMARY                      ")
    print("=" * 65)
    import json
    summary_data = {
        "idle_metrics": idle_res,
        "feat_breakdown": feat_breakdown,
        "config_a_w2v2_aasist": res_a,
        "config_b_acoustic_only": res_b,
        "config_c_fusion": res_c,
        "websocket_streaming": ws_res,
    }
    print(json.dumps(summary_data, indent=2))
    
    # Save output to JSON artifact for structured parsing
    with open("/home/jyno/Projects/Garaj/backend/profiling_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved full profiling results to /home/jyno/Projects/Garaj/backend/profiling_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
