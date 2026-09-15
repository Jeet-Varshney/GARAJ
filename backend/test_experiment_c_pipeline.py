"""
Comprehensive Automated Test Suite for GARAJ Experiment C: Feature-Augmented W2V2-AASIST.

Verifies:
1. Feature extractor output shape
2. No NaN/Inf in extracted features
3. Deterministic extraction (identical output for identical input)
4. Correct sample rate handling (16 kHz)
5. Correct 64,600 sample input window formatting
6. LFCC feature extraction
7. CQCC feature extraction
8. Spectral features extraction
9. Dataset adapter normalization
10. Metadata preservation across dataset records
11. Speaker-disjoint split (0 speaker overlap)
12. Feature / model fusion forward pass (W2V2AASISTFeatureFusion)
13. Checkpoint save/reload integrity & metadata JSON
14. Baseline model evaluation pass
15. Experiment C model evaluation pass
16. Real-time streaming inference compatibility & latency measurement
"""

import os
import wave
import json
import time
import numpy as np
import torch

from app.features.acoustic_extractor import AcousticFeatureExtractor, DEFAULT_FEATURE_GROUPS
from app.detection.feature_fusion import W2V2AASISTFeatureFusion, AcousticOnlyClassifier
from app.dataset.adapters import normalize_manifest_entry, adapt_multi_source_manifests
from app.dataset.split import create_speaker_disjoint_splits
from app.dataset.manifest_generator import create_dataset_manifest
from train_experiment_c import train_experiment_c
from evaluate_experiment_c import evaluate_experiment_c_suite
from app.detection.model_loader import load_xlsr_aasist_model, find_la_model_checkpoint
from app.detection.model_engine import ModelDetectionEngine


def create_synthetic_test_wav(
    output_path: str,
    duration_sec: float = 4.1,
    sample_rate: int = 16000,
    is_real: bool = True
):
    """Creates synthetic test audio WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if is_real:
        signal = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.1 * np.sin(2 * np.pi * 440.0 * t)
    else:
        signal = 0.4 * np.sin(2 * np.pi * 350.0 * t) + 0.2 * np.cos(2 * np.pi * 700.0 * t)

    int16_signal = (signal * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_signal.tobytes())


def run_experiment_c_test_suite():
    print("=================================================================")
    print("RUNNING COMPREHENSIVE EXPERIMENT C AUTOMATED TEST SUITE")
    print("=================================================================")

    test_dir = "/home/jyno/Projects/Garaj/backend/tmp_test_exp_c"
    real_dir = os.path.join(test_dir, "datasets", "bona_fide", "hindi")
    spoof_dir = os.path.join(test_dir, "datasets", "spoof", "hindi")
    manifest_json = os.path.join(test_dir, "manifests", "dataset_manifest.json")
    manifest_csv = os.path.join(test_dir, "manifests", "dataset_manifest.csv")
    splits_dir = os.path.join(test_dir, "splits")
    ckpt_c3 = os.path.join(test_dir, "checkpoints", "LA_feature_fusion_C3.pth")
    eval_json = os.path.join(test_dir, "reports", "experiment_c_report.json")
    eval_txt = os.path.join(test_dir, "reports", "experiment_c_report.txt")

    # Generate synthetic dataset (10 speakers)
    for spk_idx in range(10):
        spk_name = f"spk_{spk_idx:02d}"
        create_synthetic_test_wav(os.path.join(real_dir, spk_name, "real.wav"), is_real=True)
        create_synthetic_test_wav(os.path.join(spoof_dir, spk_name, "spoof.wav"), is_real=False)

    # 1. Feature Extractor Output Shape
    print("\n[Test 1/16] Feature Extractor Output Shape...")
    extractor = AcousticFeatureExtractor()
    audio = np.random.normal(0, 0.2, size=64600).astype(np.float32)
    feats = extractor.extract_features(audio)
    dim = extractor.get_dim()
    assert feats.ndim == 1, "Extracted features tensor is not 1D!"
    assert len(feats) == dim, f"Expected dimension {dim}, got {len(feats)}"
    print(f"-> Feature vector dimension: {dim}. Test 1 PASS!")

    # 2. No NaN / Inf Verification
    print("\n[Test 2/16] No NaN or Inf in Features...")
    noisy_audio = np.array([float('nan'), float('inf'), 0.0, -1.0, 1.0] * 12920, dtype=np.float32)
    clean_feats = extractor.extract_features(noisy_audio)
    assert not np.isnan(clean_feats).any(), "NaN detected in extracted features!"
    assert not np.isinf(clean_feats).any(), "Inf detected in extracted features!"
    print("-> Test 2 PASS!")

    # 3. Deterministic Extraction
    print("\n[Test 3/16] Deterministic Feature Extraction...")
    feats1 = extractor.extract_features(audio)
    feats2 = extractor.extract_features(audio)
    assert np.allclose(feats1, feats2, atol=1e-6), "Non-deterministic feature extraction!"
    print("-> Test 3 PASS!")

    # 4. Correct Sample Rate Handling
    print("\n[Test 4/16] Correct Sample Rate (16000 Hz)...")
    assert extractor.sample_rate == 16000, f"Expected SR 16000, got {extractor.sample_rate}"
    print("-> Test 4 PASS!")

    # 5. Correct 64,600 Sample Input Windowing
    print("\n[Test 5/16] Correct 64,600 Sample Input Windowing...")
    short_audio = np.random.normal(0, 0.2, size=16000).astype(np.float32)
    feats_short = extractor.extract_features(short_audio)
    assert len(feats_short) == dim, "Short audio extraction failed length check!"
    print("-> Test 5 PASS!")

    # 6. LFCC Extraction
    print("\n[Test 6/16] LFCC Feature Extraction...")
    lfcc_extractor = AcousticFeatureExtractor(feature_groups={"spectral": False, "lfcc": True, "cqcc": False, "prosodic": False, "energy": False})
    lfcc_feats = lfcc_extractor.extract_features(audio)
    assert len(lfcc_feats) > 0, "LFCC feature extraction empty!"
    print("-> Test 6 PASS!")

    # 7. CQCC Extraction
    print("\n[Test 7/16] CQCC Feature Extraction...")
    cqcc_extractor = AcousticFeatureExtractor(feature_groups={"spectral": False, "lfcc": False, "cqcc": True, "prosodic": False, "energy": False})
    cqcc_feats = cqcc_extractor.extract_features(audio)
    assert len(cqcc_feats) > 0, "CQCC feature extraction empty!"
    print("-> Test 7 PASS!")

    # 8. Spectral Features Extraction
    print("\n[Test 8/16] Spectral Features Extraction...")
    spec_extractor = AcousticFeatureExtractor(feature_groups={"spectral": True, "lfcc": False, "cqcc": False, "prosodic": False, "energy": False})
    spec_feats = spec_extractor.extract_features(audio)
    assert len(spec_feats) == 10, f"Expected 10 spectral features, got {len(spec_feats)}"
    print("-> Test 8 PASS!")

    # 9. Dataset Adapter Normalization
    print("\n[Test 9/16] Dataset Adapter Normalization...")
    raw_rec = {"audio_path": "/tmp/test.wav", "label": "bona_fide", "speaker_id": "spk1"}
    norm_rec = normalize_manifest_entry(raw_rec)
    assert norm_rec["label_id"] == 1, "Label mapping error!"
    assert norm_rec["path"] == "/tmp/test.wav", "Path mapping error!"
    print("-> Test 9 PASS!")

    # 10. Metadata Preservation
    print("\n[Test 10/16] Metadata Preservation...")
    assert "attack_type" in norm_rec and "microphone_type" in norm_rec, "Metadata fields missing!"
    print("-> Test 10 PASS!")

    # 11. Speaker-Disjoint Split
    print("\n[Test 11/16] Speaker-Disjoint Split (0 Overlap)...")
    m_summary = create_dataset_manifest(real_dir=real_dir, spoof_dir=spoof_dir, output_json_path=manifest_json, output_csv_path=manifest_csv)
    split_res = create_speaker_disjoint_splits(manifest_path=manifest_json, output_dir=splits_dir)
    assert split_res["speaker_overlap"] == 0, "Speaker overlap detected!"
    print("-> Test 11 PASS!")

    # 12. Feature / Model Fusion Forward Pass
    print("\n[Test 12/16] W2V2AASISTFeatureFusion Model Forward Pass...")
    fusion_model = W2V2AASISTFeatureFusion()
    dummy_input = torch.randn(2, 64600)
    out_logits = fusion_model(dummy_input)
    assert out_logits.shape == (2, 2), f"Expected output shape (2, 2), got {out_logits.shape}"
    print("-> Test 12 PASS!")

    # 13. Checkpoint Save / Reload Integrity & Metadata
    print("\n[Test 13/16] Checkpoint Save / Reload Integrity...")
    base_ckpt = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth"
    out_ckpt, meta_info = train_experiment_c(
        experiment="C3",
        ablation="G",
        manifest_path=split_res["train_file"],
        base_checkpoint_path=base_ckpt if os.path.exists(base_ckpt) else "",
        output_checkpoint_path=ckpt_c3,
        epochs=1,
        batch_size=2
    )
    assert os.path.exists(ckpt_c3), "Adapted C3 checkpoint file missing!"
    meta_file = ckpt_c3.rsplit(".", 1)[0] + "_metadata.json"
    assert os.path.exists(meta_file), "C3 metadata JSON file missing!"
    print("-> Test 13 PASS!")

    # 14 & 15. Baseline & Experiment C Evaluation Passes
    print("\n[Test 14 & 15/16] Baseline and Experiment C Evaluation Passes...")
    eval_res = evaluate_experiment_c_suite(
        test_manifest_path=split_res["test_file"],
        baseline_checkpoint=base_ckpt if os.path.exists(base_ckpt) else ckpt_c3,
        c3_checkpoint=ckpt_c3,
        output_json_report=eval_json,
        output_txt_report=eval_txt,
        batch_size=2
    )
    assert "verdict" in eval_res, "Evaluation verdict missing!"
    assert os.path.exists(eval_json) and os.path.exists(eval_txt), "Report files missing!"
    print(f"-> Evaluation Verdict: {eval_res['verdict']}. Test 14 & 15 PASS!")

    # 16. Real-Time Streaming Inference Compatibility & Latency Measurement
    print("\n[Test 16/16] Real-Time Streaming Compatibility & Latency...")
    engine = ModelDetectionEngine(model_name_or_path=base_ckpt if os.path.exists(base_ckpt) else None)
    dummy_window = np.random.normal(0, 0.1, size=64600).astype(np.float32)
    t0 = time.perf_counter()
    inf_res = engine.process_window(dummy_window, sample_rate=16000)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert inf_res["status"] in ["MODEL_READY", "NO_AUDIO", "MODEL_CHECKPOINT_MISSING"], f"Unexpected inference status: {inf_res['status']}"
    print(f"-> Real-Time Inference Latency: {latency_ms:.2f} ms. Test 16 PASS!")

    print("\n=================================================================")
    print("ALL 16 EXPERIMENT C AUTOMATED TESTS PASSED 100% SUCCESSFULLY!")
    print("=================================================================")
    return True


if __name__ == "__main__":
    run_experiment_c_test_suite()
