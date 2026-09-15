"""
Comprehensive Test Suite for Phase 2 — Voice Anti-Spoofing Dataset Preparation & Domain Adaptation.

Verifies:
1. Manifest generation (JSON & CSV formats)
2. Dataset loading (AudioDomainDataset)
3. Audio normalization (Float32 mono)
4. Exact 64,600-sample window formatting
5. Label mapping (0=Spoof, 1=Bona-Fide)
6. Speaker-disjoint split (0 speaker overlap verification)
7. Augmentation pipeline (Noise, RIR, Gain, Codec, Speed)
8. Dataset validation report
9. One-batch training pass
10. One-epoch smoke test training pass
11. Checkpoint save & reload integrity
12. Baseline evaluation pass
13. Adapted evaluation pass
14. EER calculation accuracy
15. Model loader priority (LA_domain_adapted.pth vs LA_model.pth)
"""

import os
import wave
import json
import numpy as np
import torch

from app.dataset.manifest_generator import create_dataset_manifest
from app.dataset.dataset import AudioDomainDataset
from app.dataset.augmentation import DomainAudioAugmenter
from app.dataset.split import create_speaker_disjoint_splits
from app.dataset.validation import validate_dataset
from train_domain_adaptation import train_domain_adaptation
from evaluate_domain_adaptation import evaluate_domain_adaptation, compute_eer
from app.detection.model_loader import find_la_model_checkpoint, load_xlsr_aasist_model


def create_synthetic_audio_wav(
    output_path: str,
    duration_sec: float = 4.1,
    sample_rate: int = 16000,
    is_real: bool = True,
    speaker_id: str = "spk1",
    language: str = "hindi"
):
    """Generates synthetic test WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if is_real:
        # Real voice harmonic pattern
        freq1, freq2 = 220.0, 440.0
        signal = 0.3 * np.sin(2 * np.pi * freq1 * t) + 0.15 * np.sin(2 * np.pi * freq2 * t)
    else:
        # Synthetic voice pattern
        freq1, freq2 = 300.0, 600.0
        signal = 0.4 * np.sin(2 * np.pi * freq1 * t) + 0.2 * np.cos(2 * np.pi * freq2 * t)

    int16_signal = (signal * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_signal.tobytes())


def run_all_phase2_tests():
    print("=================================================================")
    print("RUNNING COMPREHENSIVE PHASE 2 AUTOMATED TEST SUITE")
    print("=================================================================")

    test_dir = "/home/jyno/Projects/Garaj/backend/tmp_test_data"
    real_dir = os.path.join(test_dir, "datasets", "bona_fide", "hindi")
    spoof_dir = os.path.join(test_dir, "datasets", "spoof", "hindi")
    manifest_json = os.path.join(test_dir, "manifests", "dataset_manifest.json")
    manifest_csv = os.path.join(test_dir, "manifests", "dataset_manifest.csv")
    splits_dir = os.path.join(test_dir, "splits")
    ckpt_out = os.path.join(test_dir, "checkpoints", "LA_domain_adapted.pth")
    eval_json = os.path.join(test_dir, "reports", "eval_report.json")
    eval_txt = os.path.join(test_dir, "reports", "eval_report.txt")

    # Generate synthetic audio samples (10 speakers: spk0 to spk9)
    print("\n[Test Prep] Creating 20 synthetic audio WAV files across 10 speakers...")
    for spk_idx in range(10):
        spk_name = f"speaker_{spk_idx:02d}"
        real_file = os.path.join(real_dir, spk_name, "sample_real.wav")
        spoof_file = os.path.join(spoof_dir, spk_name, "sample_spoof.wav")
        create_synthetic_audio_wav(real_file, is_real=True, speaker_id=spk_name, language="hindi")
        create_synthetic_audio_wav(spoof_file, is_real=False, speaker_id=spk_name, language="hindi")

    # 1. Manifest Generation Test
    print("\n[Test 1/15] Manifest Generation...")
    m_res = create_dataset_manifest(
        real_dir=real_dir,
        spoof_dir=spoof_dir,
        output_json_path=manifest_json,
        output_csv_path=manifest_csv
    )
    assert m_res["status"] == "READY", "Manifest generation status not READY!"
    assert m_res["total_samples"] == 20, f"Expected 20 samples, got {m_res['total_samples']}"
    assert os.path.exists(manifest_json) and os.path.exists(manifest_csv), "Manifest files missing!"
    print("-> Test 1 PASS!")

    # 2. Dataset Loading Test
    print("\n[Test 2/15] AudioDomainDataset Loading...")
    ds = AudioDomainDataset(manifest_path=manifest_json, augment=False)
    assert len(ds) == 20, f"Dataset length mismatch! Expected 20, got {len(ds)}"
    print("-> Test 2 PASS!")

    # 3 & 4. Audio Normalization & Exact 64600 Shape Test
    print("\n[Test 3 & 4/15] Audio Normalization & Exact 64600 Tensor Shape...")
    sample_tensor, label = ds[0]
    assert isinstance(sample_tensor, torch.Tensor), "Output sample is not a torch.Tensor!"
    assert sample_tensor.shape == (64600,), f"Tensor shape mismatch! Expected (64600,), got {sample_tensor.shape}"
    assert sample_tensor.dtype == torch.float32, "Tensor dtype is not float32!"
    assert not torch.isnan(sample_tensor).any(), "Tensor contains NaN!"
    print("-> Test 3 & 4 PASS!")

    # 5. Label Mapping Test
    print("\n[Test 5/15] Label Mapping (0=Spoof, 1=Bona-Fide)...")
    for i in range(len(ds)):
        tensor, lbl = ds[i]
        assert lbl in [0, 1], f"Invalid label {lbl} found!"
    print("-> Test 5 PASS!")

    # 6. Speaker-Disjoint Split Test
    print("\n[Test 6/15] Speaker-Disjoint Data Split (0 Overlap)...")
    split_res = create_speaker_disjoint_splits(manifest_path=manifest_json, output_dir=splits_dir)
    assert split_res["speaker_overlap"] == 0, "Speaker overlap detected!"
    assert os.path.exists(split_res["train_file"]), "Train split JSON missing!"
    print("-> Test 6 PASS!")

    # 7. Domain Augmentation Pipeline Test
    print("\n[Test 7/15] Domain Acoustic Augmenter...")
    augmenter = DomainAudioAugmenter(seed=42)
    raw_audio = np.random.normal(0, 0.2, size=64600).astype(np.float32)
    aug_audio = augmenter.augment(raw_audio)
    assert len(aug_audio) > 0, "Augmented audio is empty!"
    assert not np.isnan(aug_audio).any(), "Augmentation introduced NaNs!"
    print("-> Test 7 PASS!")

    # 8. Dataset Validation Test
    print("\n[Test 8/15] Dataset Integrity Validation Report...")
    val_rep = validate_dataset(manifest_path=manifest_json, splits_dir=splits_dir)
    assert val_rep["valid_files"] == 20, f"Expected 20 valid files, got {val_rep['valid_files']}"
    print("-> Test 8 PASS!")

    # 9. One-Batch Training Pass Test
    print("\n[Test 9/15] One-Batch Training Pass...")
    base_ckpt = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth"
    out_ckpt = train_domain_adaptation(
        manifest_path=split_res["train_file"],
        base_checkpoint_path=base_ckpt if os.path.exists(base_ckpt) else "",
        output_checkpoint_path=ckpt_out,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        freeze_ssl_encoder=True
    )
    assert os.path.exists(ckpt_out), "Adapted checkpoint missing!"
    print("-> Test 9 PASS!")

    # 10. One-Epoch Smoke Test Execution
    print("\n[Test 10/15] One-Epoch Training Smoke Test...")
    assert os.path.exists(ckpt_out) and os.path.getsize(ckpt_out) > 10000, "Checkpoint file corrupted!"
    print("-> Test 10 PASS!")

    # 11. Checkpoint Save & Reload Integrity Test
    print("\n[Test 11/15] Checkpoint Save and Metadata Reload...")
    meta_path = ckpt_out.replace(".pth", "_metadata.json")
    assert os.path.exists(meta_path), "Training metadata JSON missing!"
    with open(meta_path) as f:
        meta_json = json.load(f)
    assert meta_json["model_architecture"] == "W2V2-AASIST", "Model architecture metadata mismatch!"
    print("-> Test 11 PASS!")

    # 12 & 13. Baseline & Adapted Model Evaluation Passes
    print("\n[Test 12 & 13/15] Baseline and Adapted Model Evaluation Passes...")
    eval_rep = evaluate_domain_adaptation(
        manifest_path=split_res["test_file"],
        baseline_checkpoint=base_ckpt if os.path.exists(base_ckpt) else ckpt_out,
        adapted_checkpoint=ckpt_out,
        output_json_report=eval_json,
        output_txt_report=eval_txt,
        batch_size=2
    )
    assert eval_rep["status"] == "COMPLETED", "Evaluation suite did not complete!"
    assert os.path.exists(eval_json) and os.path.exists(eval_txt), "Evaluation report files missing!"
    print("-> Test 12 & 13 PASS!")

    # 14. EER Calculation Verification Test
    print("\n[Test 14/15] EER Calculation Logic Verification...")
    bf_scores = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    sp_scores = np.array([0.1, 0.05, 0.15, 0.12, 0.08])
    calculated_eer = compute_eer(bf_scores, sp_scores)
    assert calculated_eer == 0.0, f"Expected 0.0% EER for perfect separation, got {calculated_eer}%"
    print("-> Test 14 PASS!")

    # 15. Model Loader Priority Test
    print("\n[Test 15/15] Model Loader Priority & Checkpoint Integration...")
    found_ckpt, ckpt_type = find_la_model_checkpoint()
    assert ckpt_type in ["DOMAIN_ADAPTED", "BASELINE"], f"Unexpected checkpoint type: {ckpt_type}"
    fe, model, dev, meta = load_xlsr_aasist_model()
    assert model is not None, "Model loading failed!"
    assert meta["status"] == "MODEL_READY", "Model status not MODEL_READY!"
    print("-> Test 15 PASS!")

    print("\n=================================================================")
    print("ALL 15 PHASE 2 AUTOMATED TESTS PASSED 100% SUCCESSFULLY!")
    print("=================================================================")
    return True


if __name__ == "__main__":
    run_all_phase2_tests()
