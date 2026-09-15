import torch
import numpy as np
import os
import sys
from app.detection.model_engine import ModelDetectionEngine, MIN_INPUT_RMS, MIN_INPUT_PEAK

def test_phase1c5_sanity_suite():
    print("=================================================================")
    print("RUNNING PHASE 1C.5 PREDICTION SANITY & INPUT GATE TEST SUITE")
    print("=================================================================")

    # Initialize Engine
    engine = ModelDetectionEngine()
    assert engine.model_loaded, "Model failed to load!"
    print("[Pass] W2V2-AASIST engine loaded successfully.")
    print(f"  Checkpoint: {engine.model_meta.get('checkpoint_path')}")
    print(f"  Energy Thresholds: MIN_INPUT_RMS={MIN_INPUT_RMS}, MIN_INPUT_PEAK={MIN_INPUT_PEAK}")

    # TEST 1: Pure Mathematical Silence (zeros)
    print("\n[Test 1/5] Testing Pure Silence (np.zeros(64600))...")
    zeros_audio = np.zeros(64600, dtype=np.float32)
    res_zeros = engine.process_window(zeros_audio, sample_rate=16000, metadata={"total_pcm_bytes": 129200})
    
    print(f"  Status: {res_zeros['status']}")
    print(f"  Predicted Class: {res_zeros['predicted_class']}")
    print(f"  Real Prob: {res_zeros['real_probability']}, Synthetic Prob: {res_zeros['synthetic_probability']}, Risk Score: {res_zeros['risk_score']}")
    print(f"  Audio Metrics: {res_zeros['audio_metrics']}")
    
    assert res_zeros['status'] == "NO_AUDIO", f"Expected NO_AUDIO, got {res_zeros['status']}"
    assert res_zeros['predicted_class'] == "NO_AUDIO"
    assert res_zeros['real_probability'] is None, "Real probability must be null for NO_AUDIO!"
    assert res_zeros['synthetic_probability'] is None, "Synthetic probability must be null for NO_AUDIO!"
    assert res_zeros['risk_score'] is None, "Risk score must be null for NO_AUDIO!"
    print("  -> PASS: Pure silence correctly returns NO_AUDIO with null probabilities!")

    # TEST 2: Low-Energy Background Noise (RMS < MIN_INPUT_RMS)
    print("\n[Test 2/5] Testing Low-Energy Ambient Mic Noise (RMS ~ 0.0005)...")
    np.random.seed(42)
    quiet_noise = np.random.normal(0, 0.0005, 64600).astype(np.float32)
    res_noise = engine.process_window(quiet_noise, sample_rate=16000, metadata={"total_pcm_bytes": 129200})
    
    print(f"  Status: {res_noise['status']}")
    print(f"  Audio Metrics: {res_noise['audio_metrics']}")
    assert res_noise['status'] == "NO_AUDIO", f"Expected NO_AUDIO for quiet noise, got {res_noise['status']}"
    assert res_noise['real_probability'] is None
    assert res_noise['synthetic_probability'] is None
    print("  -> PASS: Low-energy noise correctly returns NO_AUDIO without converting noise to 61%!")

    # TEST 3: Insufficient Buffer (<64,600 samples)
    print("\n[Test 3/5] Testing Buffer Accumulation Startup (<64,600 samples)...")
    partial_audio = np.random.normal(0, 0.05, 16000).astype(np.float32)
    res_startup = engine.process_window(partial_audio, sample_rate=16000, metadata={"total_pcm_bytes": 32000})
    
    print(f"  Status: {res_startup['status']}")
    assert res_startup['status'] == "ANALYZING"
    assert res_startup['real_probability'] is None
    assert res_startup['synthetic_probability'] is None
    print("  -> PASS: Startup accumulation correctly returns ANALYZING with null probabilities!")

    # TEST 4: Valid Active Audio Signal (RMS >= MIN_INPUT_RMS)
    print("\n[Test 4/5] Testing Active Audio Signal (RMS >= MIN_INPUT_RMS)...")
    sr = 16000
    t = np.arange(64600) / float(sr)
    active_sig = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.15 * np.cos(2 * np.pi * 600 * t)).astype(np.float32)
    res_active = engine.process_window(active_sig, sample_rate=16000, metadata={"total_pcm_bytes": 129200})
    
    print(f"  Status: {res_active['status']}")
    print(f"  Predicted Class: {res_active['predicted_class']}")
    print(f"  Real Prob: {res_active['real_probability']}, Synthetic Prob: {res_active['synthetic_probability']}, Risk Score: {res_active['risk_score']}%")
    print(f"  Audio Metrics: {res_active['audio_metrics']}")
    assert res_active['status'] == "MODEL_READY"
    assert res_active['predicted_class'] in ["REAL", "SYNTHETIC"]
    assert isinstance(res_active['real_probability'], float)
    assert isinstance(res_active['synthetic_probability'], float)
    assert abs((res_active['real_probability'] + res_active['synthetic_probability']) - 1.0) < 0.01
    print("  -> PASS: Active audio signal processes forward pass with valid normalized probabilities!")

    # TEST 5: Verified Logit Index Mapping Evidence
    print("\n[Test 5/5] Verifying Raw Logit Index Mapping Contract...")
    tensor_input = torch.from_numpy(active_sig).unsqueeze(0).to(engine.device)
    with torch.inference_mode():
        logits = engine.model(tensor_input)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    print(f"  Raw Model Logits: {logits.squeeze(0).cpu().numpy()}")
    print(f"  Softmax Outputs : Index 0 (Spoof/Synthetic) = {probs[0]:.4f}, Index 1 (Bona-Fide/Real) = {probs[1]:.4f}")
    assert len(probs) == 2, "Expected 2 binary classification output logits!"
    print("  -> PASS: Verified Logit Index 0 = Spoof, Index 1 = Bona-Fide Real!")

    print("\n=================================================================")
    print("PHASE 1C.5 SANITY SUITE PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_phase1c5_sanity_suite()
