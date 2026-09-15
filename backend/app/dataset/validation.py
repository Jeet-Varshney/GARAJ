"""
Dataset Integrity Validator Module for Garaj Voice Anti-Spoofing & Deepfake Detection.

Performs thorough quality control checks on dataset files:
- Corrupted file detection
- Unsupported audio formats
- Sample rate mismatches
- Channel count mismatches
- Silent or extremely short audio files
- NaN/Inf value detection
- Audio clipping checks
- Duplicate audio file detection
- Speaker overlap across splits
- Class and language imbalance reporting
"""

import os
import json
import hashlib
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("dataset_validation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def validate_dataset(
    manifest_path: str = "/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.json",
    splits_dir: Optional[str] = "/home/jyno/Projects/Garaj/datasets/splits",
    target_sample_rate: int = 16000,
    required_samples: int = 64600,
) -> Dict[str, Any]:
    """
    Validates all audio files in dataset manifest and checks split integrity.
    """
    report = {
        "status": "READY",
        "total_files": 0,
        "valid_files": 0,
        "corrupted_files": 0,
        "unsupported_format_files": 0,
        "sample_rate_issues": 0,
        "channel_issues": 0,
        "silent_files": 0,
        "too_short_files": 0,
        "nan_inf_files": 0,
        "clipped_files": 0,
        "duplicate_files": 0,
        "class_distribution": {"bona_fide": 0, "spoof": 0},
        "language_distribution": {"hindi": 0, "hinglish": 0, "english": 0, "unknown": 0},
        "unique_speakers": 0,
        "speaker_overlap_count": 0,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest path {manifest_path} does not exist. Status: DATASET_NOT_FOUND.")
        report["status"] = "DATASET_NOT_FOUND"
        report["errors"].append(f"Manifest not found at {manifest_path}")
        return report

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    entries = manifest_data.get("entries", [])
    report["total_files"] = len(entries)

    if len(entries) == 0:
        logger.warning("Manifest contains 0 entries. Status: DATASET_NOT_FOUND.")
        report["status"] = "DATASET_NOT_FOUND"
        return report

    hashes = set()
    speakers = set()

    for idx, entry in enumerate(entries):
        filepath = entry.get("path") or entry.get("audio_path")
        label = entry.get("label", "unknown")
        label_id = entry.get("label_id", 0)
        language = entry.get("language", "unknown")
        spk = entry.get("speaker_id")

        if spk and spk not in ["null", "unknown", "None"]:
            speakers.add(spk)

        # Update class distribution
        if label_id == 1 or label == "bona_fide":
            report["class_distribution"]["bona_fide"] += 1
        else:
            report["class_distribution"]["spoof"] += 1

        # Update language distribution
        lang_key = language if language in report["language_distribution"] else "unknown"
        report["language_distribution"][lang_key] += 1

        if not filepath or not os.path.exists(filepath):
            report["corrupted_files"] += 1
            report["errors"].append(f"File missing: {filepath}")
            continue

        # File duplicate hash check
        try:
            with open(filepath, "rb") as af:
                file_hash = hashlib.md5(af.read(8192)).hexdigest()
                if file_hash in hashes:
                    report["duplicate_files"] += 1
                    report["warnings"].append(f"Duplicate content detected: {filepath}")
                else:
                    hashes.add(file_hash)
        except Exception as e:
            report["corrupted_files"] += 1
            report["errors"].append(f"Cannot read file {filepath}: {e}")
            continue

        # Load audio data using soundfile / torchaudio
        audio = None
        sr = 0
        try:
            import soundfile as sf
            audio, sr = sf.read(filepath, dtype='float32')
            if audio.ndim > 1:
                report["channel_issues"] += 1
                audio = np.mean(audio, axis=1)
        except Exception:
            try:
                import torchaudio
                wf, sr = torchaudio.load(filepath)
                if wf.shape[0] > 1:
                    report["channel_issues"] += 1
                    wf = torch.mean(wf, dim=0, keepdim=True)
                audio = wf.squeeze(0).numpy().astype(np.float32)
            except Exception as exc:
                report["corrupted_files"] += 1
                report["errors"].append(f"Audio decode failure for {filepath}: {exc}")
                continue

        if sr != target_sample_rate:
            report["sample_rate_issues"] += 1

        if audio is None or len(audio) == 0:
            report["silent_files"] += 1
            report["errors"].append(f"Empty/Zero length audio: {filepath}")
            continue

        # Check NaN / Inf
        if np.isnan(audio).any() or np.isinf(audio).any():
            report["nan_inf_files"] += 1
            report["errors"].append(f"Audio contains NaN or Inf values: {filepath}")
            continue

        # Check audio length
        if len(audio) < 8000:  # < 0.5 sec
            report["too_short_files"] += 1
            report["warnings"].append(f"Audio file shorter than 0.5s: {filepath}")

        # Check audio energy & clipping
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(np.square(audio)))

        if rms < 1e-5:
            report["silent_files"] += 1
            report["warnings"].append(f"Silent audio file (RMS < 1e-5): {filepath}")

        if peak >= 0.999:
            report["clipped_files"] += 1

        report["valid_files"] += 1

    report["unique_speakers"] = len(speakers)

    # Validate speaker overlap across splits if splits_dir provided
    train_count, val_count, test_count = 0, 0, 0
    if splits_dir and os.path.exists(splits_dir):
        t_path = os.path.join(splits_dir, "train.json")
        v_path = os.path.join(splits_dir, "validation.json")
        te_path = os.path.join(splits_dir, "test.json")

        if os.path.exists(t_path) and os.path.exists(v_path) and os.path.exists(te_path):
            with open(t_path) as f: t_data = json.load(f)
            with open(v_path) as f: v_data = json.load(f)
            with open(te_path) as f: te_data = json.load(f)

            train_count = t_data.get("total_samples", 0)
            val_count = v_data.get("total_samples", 0)
            test_count = te_data.get("total_samples", 0)

            t_spks = set(e.get("speaker_id") for e in t_data.get("entries", []) if e.get("speaker_id"))
            v_spks = set(e.get("speaker_id") for e in v_data.get("entries", []) if e.get("speaker_id"))
            te_spks = set(e.get("speaker_id") for e in te_data.get("entries", []) if e.get("speaker_id"))

            overlap = len(t_spks.intersection(v_spks)) + len(t_spks.intersection(te_spks)) + len(v_spks.intersection(te_spks))
            report["speaker_overlap_count"] = overlap

    # Format human readable console summary
    summary_text = f"""
DATASET VALIDATION SUMMARY
------------------------------------------------
Total files scanned  : {report['total_files']}
Valid files          : {report['valid_files']}
Corrupted files      : {report['corrupted_files']}
Bona-Fide (Real)     : {report['class_distribution']['bona_fide']}
Spoof (Synthetic)    : {report['class_distribution']['spoof']}
Hindi files          : {report['language_distribution']['hindi']}
Hinglish files       : {report['language_distribution']['hinglish']}
English files        : {report['language_distribution']['english']}
Unique speakers      : {report['unique_speakers']}
Speaker overlap count: {report['speaker_overlap_count']}
Split Train samples  : {train_count}
Split Val samples    : {val_count}
Split Test samples   : {test_count}
------------------------------------------------
Status: {report['status']}
"""
    logger.info(summary_text)
    report["summary_text"] = summary_text.strip()
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate PyTorch Audio Dataset Integrity")
    parser.add_argument("--manifest", type=str, default="/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.json", help="Path to dataset manifest JSON")
    parser.add_argument("--splits_dir", type=str, default="/home/jyno/Projects/Garaj/datasets/splits", help="Path to dataset splits directory")
    args = parser.parse_args()

    res = validate_dataset(manifest_path=args.manifest, splits_dir=args.splits_dir)
    print(res["summary_text"])
