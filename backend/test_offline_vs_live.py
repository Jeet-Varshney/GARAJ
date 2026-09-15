import os
import hashlib
import numpy as np
import torch
from app.detection.model_engine import ModelDetectionEngine

def test_offline_vs_live_equivalence():
    print("=================================================================")
    print("RUNNING PHASE 1C.7 OFFLINE VS LIVE MODEL EQUIVALENCE TEST")
    print("=================================================================")

    # Initialize model engine
    engine = ModelDetectionEngine()

    if not engine.model_loaded or engine.model is None:
        print("[FAIL] Model checkpoint LA_model.pth is not loaded!")
        return False

    # Check if tmp_live_window.npy exists, or generate a speech-like harmonic test window
    live_window_file = "/home/jyno/Projects/Garaj/backend/tmp_live_window.npy"
    if os.path.exists(live_window_file):
        print(f"[Input Source] Loading saved live window from {live_window_file}...")
        test_window = np.load(live_window_file)
    else:
        print("[Input Source] Generating 64,600-sample test speech window...")
        sr = 16000
        t = np.linspace(0, 64600 / sr, 64600, endpoint=False, dtype=np.float32)
        # Synthetic speech harmonic (F0=180Hz, F1=750Hz)
        test_window = (0.15 * np.sin(2 * np.pi * 180 * t) + 0.08 * np.sin(2 * np.pi * 750 * t)).astype(np.float32)

    # Standardize to 64,600 samples
    test_window = test_window[:64600]

    # Calculate input window statistics
    win_shape = test_window.shape
    win_min = float(np.min(test_window))
    win_max = float(np.max(test_window))
    win_rms = float(np.sqrt(np.mean(np.square(test_window))))
    win_peak = float(np.max(np.abs(test_window)))
    first_20 = test_window[:20]
    last_20 = test_window[-20:]
    win_hash = hashlib.md5(test_window.tobytes()).hexdigest()

    print("\n--- INPUT WINDOW STATISTICS ---")
    print(f"  Shape       : {win_shape}")
    print(f"  Dtype       : {test_window.dtype}")
    print(f"  Min / Max   : {win_min:.6f} / {win_max:.6f}")
    print(f"  RMS / Peak  : {win_rms:.6f} / {win_peak:.6f}")
    print(f"  MD5 Hash    : {win_hash}")
    print(f"  First 5     : {first_20[:5]}")
    print(f"  Last 5      : {last_20[-5:]}")

    # 1. LIVE PATH INFERENCE (via ModelDetectionEngine.process_window)
    live_result = engine.process_window(test_window, sample_rate=16000)
    live_logits = live_result.get("raw_logits")
    live_synth_prob = live_result.get("synthetic_probability")
    live_real_prob = live_result.get("real_probability")

    print("\n--- LIVE PATH RESULTS ---")
    print(f"  Status        : {live_result.get('status')}")
    print(f"  Predicted     : {live_result.get('predicted_class')}")
    print(f"  Raw Logits    : {live_logits}")
    print(f"  Synthetic Prob: {live_synth_prob}")
    print(f"  Real Prob     : {live_real_prob}")

    # 2. OFFLINE PATH INFERENCE (Direct PyTorch forward pass on saved array)
    audio_tensor = torch.from_numpy(test_window).unsqueeze(0).to(engine.device)
    with torch.inference_mode():
        offline_raw_logits = engine.model(audio_tensor)
        offline_probs = torch.softmax(offline_raw_logits, dim=-1).squeeze(0).cpu().numpy()
        off_logits_list = [round(float(offline_raw_logits[0][0]), 4), round(float(offline_raw_logits[0][1]), 4)]
        off_synth_prob = round(float(offline_probs[0]), 4)
        off_real_prob = round(float(offline_probs[1]), 4)

    print("\n--- OFFLINE PATH RESULTS ---")
    print(f"  Raw Logits    : {off_logits_list}")
    print(f"  Synthetic Prob: {off_synth_prob}")
    print(f"  Real Prob     : {off_real_prob}")

    # 3. COMPARISON & NUMERICAL EQUIVALENCE CHECK
    logit_diff = [abs(live_logits[0] - off_logits_list[0]), abs(live_logits[1] - off_logits_list[1])]
    max_logit_diff = max(logit_diff)

    print("\n--- NUMERICAL COMPARISON SUMMARY ---")
    print(f"  Logit Difference: {logit_diff} (Max Diff: {max_logit_diff:.6f})")

    assert max_logit_diff < 1e-4, f"Live vs Offline logits mismatch! Max diff: {max_logit_diff}"
    assert abs(live_synth_prob - off_synth_prob) < 1e-4, "Synthetic probability mismatch!"

    print("\n=================================================================")
    print("PASS: LIVE VS OFFLINE MODEL EQUIVALENCE VERIFIED!")
    print("Live path and Offline path produced identical logits and probabilities.")
    print("=================================================================")
    return True

if __name__ == "__main__":
    test_offline_vs_live_equivalence()
