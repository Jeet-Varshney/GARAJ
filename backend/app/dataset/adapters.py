"""
Dataset Adapters for Multi-Source Voice Anti-Spoofing Datasets.

Ingests & normalizes entries from:
- ASVspoof (LA / DF partitions)
- IndicSynth (Hindi / Hinglish synthetic speech)
- GARAJ Local Datasets
- Held-out Real Consumer Microphone Datasets

Ensures uniform metadata fields across all sources:
path, label, label_id, speaker_id, language, source_dataset, microphone_type, recording_environment, codec, attack_type.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("dataset_adapters")


def normalize_manifest_entry(
    raw_entry: Dict[str, Any],
    default_source: str = "garaj",
    default_lang: str = "hindi"
) -> Dict[str, Any]:
    """
    Normalizes any raw manifest record into GARAJ Experiment C unified metadata schema.
    """
    path = raw_entry.get("path") or raw_entry.get("audio_path") or ""
    abs_path = os.path.abspath(path) if path else ""

    # Label mapping (0 = SPOOF / SYNTHETIC, 1 = BONA_FIDE / REAL)
    if "label_id" in raw_entry:
        label_id = int(raw_entry["label_id"])
        label_str = "bona_fide" if label_id == 1 else "spoof"
    elif "label" in raw_entry:
        raw_lbl = str(raw_entry["label"]).lower()
        if raw_lbl in ["1", "bona_fide", "real", "bonafide"]:
            label_id = 1
            label_str = "bona_fide"
        else:
            label_id = 0
            label_str = "spoof"
    else:
        label_id = 0
        label_str = "spoof"

    speaker_id = raw_entry.get("speaker_id")
    if speaker_id in ["null", "unknown", "None", ""]:
        speaker_id = None

    language = raw_entry.get("language") or default_lang
    source_dataset = raw_entry.get("source_dataset") or default_source
    mic_type = raw_entry.get("microphone_type") or "unknown"
    rec_env = raw_entry.get("recording_environment") or "unknown"
    codec = raw_entry.get("codec") or (os.path.splitext(abs_path)[1].lstrip(".").lower() if abs_path else "wav")
    attack_type = raw_entry.get("attack_type") or ("none" if label_id == 1 else "synthetic_unknown")

    return {
        "path": abs_path,
        "audio_path": abs_path,
        "label": label_str,
        "label_id": label_id,
        "speaker_id": speaker_id,
        "language": language,
        "source_dataset": source_dataset,
        "microphone_type": mic_type,
        "recording_environment": rec_env,
        "codec": codec,
        "attack_type": attack_type,
    }


def adapt_multi_source_manifests(
    manifest_paths: List[str],
    output_unified_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Combines and normalizes entries from multiple dataset manifest JSON files.
    """
    unified_entries = []

    for path in manifest_paths:
        if not os.path.exists(path):
            logger.warning(f"Manifest path '{path}' missing. Skipping.")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        for e in entries:
            norm_e = normalize_manifest_entry(e)
            unified_entries.append(norm_e)

    summary = {
        "status": "READY" if len(unified_entries) > 0 else "DATASET_NOT_FOUND",
        "total_samples": len(unified_entries),
        "entries": unified_entries,
    }

    if output_unified_path and unified_entries:
        os.makedirs(os.path.dirname(output_unified_path), exist_ok=True)
        with open(output_unified_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved unified multi-source dataset manifest to {output_unified_path}")

    return summary
