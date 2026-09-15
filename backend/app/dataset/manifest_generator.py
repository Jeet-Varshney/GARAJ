"""
Dataset Manifest Generator Module for Garaj Voice Anti-Spoofing Engine.

Scans dataset directories for Bona-Fide (Real) and Spoof (Synthetic) audio files,
extracts audio metadata, and generates standardized JSON and CSV manifest files.

Labels:
- 0 = Spoof / Synthetic (label: "spoof", label_id: 0)
- 1 = Bona-Fide / Real (label: "bona_fide", label_id: 1)
"""

import os
import json
import csv
import glob
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("manifest_generator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]


def _get_audio_codec(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    return ext.lstrip(".") if ext else "unknown"


def create_dataset_manifest(
    base_dataset_dir: Optional[str] = None,
    real_dir: Optional[str] = None,
    spoof_dir: Optional[str] = None,
    output_json_path: str = "/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.json",
    output_csv_path: str = "/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.csv",
    source_dataset_name: str = "garaj_domain_adaptation",
) -> Dict[str, Any]:
    """
    Scans directory hierarchy or specified real/spoof directories for audio samples and builds
    standardized dataset manifest JSON and CSV files.

    If no dataset audio files are found, status is set to 'DATASET_NOT_FOUND'.
    """
    manifest_entries: List[Dict[str, Any]] = []

    # Target scan roots
    target_dirs = []
    if base_dataset_dir and os.path.exists(base_dataset_dir):
        target_dirs.append(base_dataset_dir)
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))
    if os.path.exists(default_root) and default_root not in target_dirs:
        target_dirs.append(default_root)

    scanned_files = []

    # 1. Direct real / spoof overrides if provided
    if real_dir and os.path.exists(real_dir):
        for ext in SUPPORTED_EXTENSIONS:
            scanned_files.extend([(f, "bona_fide", 1) for f in glob.glob(os.path.join(real_dir, f"**/*{ext}"), recursive=True)])
            scanned_files.extend([(f, "bona_fide", 1) for f in glob.glob(os.path.join(real_dir, f"**/*{ext.upper()}"), recursive=True)])

    if spoof_dir and os.path.exists(spoof_dir):
        for ext in SUPPORTED_EXTENSIONS:
            scanned_files.extend([(f, "spoof", 0) for f in glob.glob(os.path.join(spoof_dir, f"**/*{ext}"), recursive=True)])
            scanned_files.extend([(f, "spoof", 0) for f in glob.glob(os.path.join(spoof_dir, f"**/*{ext.upper()}"), recursive=True)])

    # 2. General tree scanning under datasets/ (bona_fide and spoof subdirectories)
    if not scanned_files:
        for root_dir in target_dirs:
            bona_fide_path = os.path.join(root_dir, "bona_fide")
            spoof_path = os.path.join(root_dir, "spoof")

            if os.path.exists(bona_fide_path):
                for ext in SUPPORTED_EXTENSIONS:
                    for f in glob.glob(os.path.join(bona_fide_path, f"**/*{ext}"), recursive=True):
                        scanned_files.append((f, "bona_fide", 1))

            if os.path.exists(spoof_path):
                for ext in SUPPORTED_EXTENSIONS:
                    for f in glob.glob(os.path.join(spoof_path, f"**/*{ext}"), recursive=True):
                        scanned_files.append((f, "spoof", 0))

    # Deduplicate files by absolute path
    unique_files = {}
    for filepath, label_str, label_id in scanned_files:
        abs_p = os.path.abspath(filepath)
        if abs_p not in unique_files:
            unique_files[abs_p] = (label_str, label_id)

    if not unique_files:
        logger.warning("No audio dataset files found! Status: DATASET_NOT_FOUND.")
        summary = {
            "status": "DATASET_NOT_FOUND",
            "total_samples": 0,
            "bona_fide_samples": 0,
            "spoof_samples": 0,
            "entries": [],
            "output_json_path": output_json_path,
            "output_csv_path": output_csv_path,
        }
        return summary

    for abs_path, (label_str, label_id) in unique_files.items():
        rel_parts = abs_path.split(os.sep)
        
        # Determine language tag from directory structure if present
        language = "unknown"
        for lang in ["hindi", "hinglish", "english"]:
            if lang in rel_parts:
                language = lang
                break

        # Speaker ID determination: do not invent speaker IDs. Use parent folder if formatted or null
        parent_dir = os.path.basename(os.path.dirname(abs_path))
        speaker_id = parent_dir if parent_dir not in ["hindi", "hinglish", "english", "bona_fide", "spoof", "datasets"] else None

        # Microphone type / domain identification
        mic_type = "consumer_mic" if "consumer" in abs_path or "mic" in abs_path else "unknown"
        env = "noisy_room" if "noise" in abs_path else "studio_or_unknown"
        codec = _get_audio_codec(abs_path)

        entry = {
            "path": abs_path,
            "audio_path": abs_path,  # backward compatibility
            "label": label_str,
            "label_id": label_id,
            "speaker_id": speaker_id,
            "language": language,
            "source_dataset": source_dataset_name,
            "microphone_type": mic_type,
            "recording_environment": env,
            "codec": codec,
        }
        manifest_entries.append(entry)

    bona_fide_cnt = sum(1 for e in manifest_entries if e["label_id"] == 1)
    spoof_cnt = sum(1 for e in manifest_entries if e["label_id"] == 0)

    summary = {
        "status": "READY",
        "total_samples": len(manifest_entries),
        "bona_fide_samples": bona_fide_cnt,
        "spoof_samples": spoof_cnt,
        "languages": {
            "hindi": sum(1 for e in manifest_entries if e["language"] == "hindi"),
            "hinglish": sum(1 for e in manifest_entries if e["language"] == "hinglish"),
            "english": sum(1 for e in manifest_entries if e["language"] == "english"),
            "unknown": sum(1 for e in manifest_entries if e["language"] == "unknown"),
        },
        "output_json_path": output_json_path,
        "output_csv_path": output_csv_path,
        "entries": manifest_entries,
    }

    # Write JSON manifest
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset manifest JSON ({len(manifest_entries)} entries) -> {output_json_path}")

    # Write CSV manifest
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        "path", "label", "label_id", "speaker_id", "language",
        "source_dataset", "microphone_type", "recording_environment", "codec"
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in manifest_entries:
            row = {k: e.get(k) for k in fieldnames}
            writer.writerow(row)
    logger.info(f"Saved dataset manifest CSV ({len(manifest_entries)} entries) -> {output_csv_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PyTorch Dataset Manifest for Voice Deepfake Detection")
    parser.add_argument("--base_dir", type=str, default=None, help="Base dataset directory path")
    parser.add_argument("--real_dir", type=str, default=None, help="Directory containing real (bona-fide) audio files")
    parser.add_argument("--spoof_dir", type=str, default=None, help="Directory containing synthetic (spoof) audio files")
    parser.add_argument("--output_json", type=str, default="/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.json", help="Output JSON manifest file path")
    parser.add_argument("--output_csv", type=str, default="/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.csv", help="Output CSV manifest file path")
    args = parser.parse_args()

    res = create_dataset_manifest(
        base_dataset_dir=args.base_dir,
        real_dir=args.real_dir,
        spoof_dir=args.spoof_dir,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
    )
    print(f"Manifest Generation Status: {res['status']} | Total Samples: {res['total_samples']}")
