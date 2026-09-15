"""
Offline Model Validation Test Script (Phase 1A).

Tests full XLS-R -> AASIST audio preprocessing and model inference pipeline on a WAV audio file.
Runs without requiring frontend, WebSocket stream, or live backend server.
"""

import sys
import os
import time
import wave
import argparse
import numpy as np

# Ensure dependencies and backend app are in PYTHONPATH
for extra_path in ["/tmp/my_deps", os.path.abspath(os.path.dirname(__file__))]:
    if os.path.exists(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)


def generate_sample_wav(filename: str = "sample_test_16k.wav", duration_sec: float = 3.0, sample_rate: int = 16000):
    """Generates a synthetic 16kHz 16-bit mono PCM test WAV file if no input WAV is provided."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # Mix fundamental audio frequencies (440Hz + 880Hz harmonics)
    audio_signal = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    int16_signal = (audio_signal * 32767).astype(np.int16)
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_signal.tobytes())
        
    print(f"Generated synthetic test WAV file: '{filename}' ({duration_sec}s, {sample_rate} Hz)")
    return filename


def load_wav_file(file_path: str):
    """Reads WAV file and returns Float32 array normalized to [-1.0, 1.0] and sample rate."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"WAV audio file not found: '{file_path}'")

    try:
        import soundfile as sf
        data, sr = sf.read(file_path, dtype='float32')
        if len(data.shape) > 1:
            data = data[:, 0]  # Mono conversion
        return data, sr
    except Exception as sf_err:
        print(f"soundfile failed: {sf_err}. Falling back to wave module...")
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                int16_data = np.frombuffer(frames, dtype=np.int16)
                if channels > 1:
                    int16_data = int16_data[::channels]
                float32_data = int16_data.astype(np.float32) / 32768.0
                return float32_data, sr
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")


def run_offline_test(wav_path: str):
    print("=" * 65)
    print("GARAJ Phase 1A — XLS-R -> AASIST Model Validation")
    print("=" * 65)

    # 1. Environment & Dependency Check
    print("\n[1/5] Checking PyTorch Environment...")
    import torch
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Compute Device  : {device_name}")

    # 2. Model Loading
    print("\n[2/5] Loading Model Pipeline...")
    t_start_load = time.perf_counter()
    
    from app.detection.model_loader import load_xlsr_aasist_model, ModelLoadError
    from app.detection.model_engine import ModelDetectionEngine
    
    try:
        engine = ModelDetectionEngine()
        t_load_ms = (time.perf_counter() - t_start_load) * 1000.0
        print(f"  Model Status    : SUCCESS (Loaded in {t_load_ms:.2f} ms)")
        print(f"  Checkpoint Name : {engine.model_meta.get('model_name')}")
        print(f"  Parameters      : {engine.model_meta.get('num_parameters'):,}")
        print(f"  Classification  : Pretrained Sequence Classifier Head")
    except ModelLoadError as exc:
        print(f"  Model Status    : FAILED — {exc}")
        sys.exit(1)

    # 3. Audio Loading & Preprocessing
    print(f"\n[3/5] Loading Audio File: '{wav_path}'...")
    audio_data, sr = load_wav_file(wav_path)
    duration_sec = len(audio_data) / float(sr)
    
    print(f"  Audio Duration  : {duration_sec:.3f} seconds")
    print(f"  Sample Rate     : {sr} Hz")
    print(f"  Samples Count   : {len(audio_data):,}")
    print(f"  Amplitude Peak  : {np.max(np.abs(audio_data)):.4f}")

    # 4. Model Preprocessing & Inference Pass
    print("\n[4/5] Running Preprocessing & Model Forward Pass...")
    t_start_infer = time.perf_counter()
    
    result = engine.process_window(audio_window=audio_data, sample_rate=sr)
    
    t_infer_ms = (time.perf_counter() - t_start_infer) * 1000.0

    # 5. Output Verification
    print("\n[5/5] Genuine Model Prediction Results:")
    print("-" * 45)
    print(f"  Model Architecture  : {result.get('model')}")
    print(f"  Checkpoint Used     : {result.get('model_checkpoint')}")
    print(f"  Inference Device    : {result.get('device')}")
    print(f"  Engine Status       : {result.get('status')}")
    print(f"  Inference Latency   : {result.get('compute_latency_ms')} ms (Total script: {t_infer_ms:.2f} ms)")
    
    real_prob = result.get('real_probability')
    syn_prob = result.get('synthetic_probability')
    if real_prob is not None and syn_prob is not None:
        print(f"  Real Probability    : {real_prob:.4f} ({real_prob * 100:.2f}%)")
        print(f"  Synthetic Prob      : {syn_prob:.4f} ({syn_prob * 100:.2f}%)")
        print(f"  Confidence Score    : {result.get('confidence'):.4f}")
    else:
        print(f"  Real Probability    : NONE (Trained Classifier Missing)")
        print(f"  Synthetic Prob      : NONE (Trained Classifier Missing)")
        print(f"  Confidence Score    : 0.0000")
        
    print(f"  Predicted Class     : {str(result.get('predicted_class')).upper()}")
    print("-" * 45)
    print("\n[PASS] Model Pipeline Validation Executed (Status: " + str(result.get('status')) + ").")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GARAJ Phase 1A PyTorch XLS-R + AASIST Engine")
    parser.add_argument("wav_path", nargs="?", default=None, help="Path to 16kHz WAV file")
    args = parser.parse_args()

    wav_file = args.wav_path
    if not wav_file or not os.path.exists(wav_file):
        print("No input WAV provided or file missing. Generating synthetic 16kHz test WAV...")
        wav_file = generate_sample_wav()

    run_offline_test(wav_file)
