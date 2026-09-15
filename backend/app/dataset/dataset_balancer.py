"""
Dataset-Aware Balancer & Distribution Inspector.

Prevents single large dataset dominance (e.g. ASVspoof swamping IndicSynth or GARAJ consumer mic data).
Computes sample weights for PyTorch WeightedRandomSampler and generates detailed distribution reports.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger("dataset_balancer")


def generate_dataset_distribution_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes dataset composition and prints distribution summary report.
    """
    total = len(entries)
    if total == 0:
        return {"status": "EMPTY"}

    source_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    class_counts = {"bona_fide": 0, "spoof": 0}
    speaker_counts: Dict[str, set] = {}
    attack_counts: Dict[str, int] = {}

    for e in entries:
        src = e.get("source_dataset", "unknown")
        lang = e.get("language", "unknown")
        lbl_id = e.get("label_id", 0)
        spk = e.get("speaker_id")
        atk = e.get("attack_type", "none")

        source_counts[src] = source_counts.get(src, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if lbl_id == 1:
            class_counts["bona_fide"] += 1
        else:
            class_counts["spoof"] += 1

        if src not in speaker_counts:
            speaker_counts[src] = set()
        if spk and spk not in ["null", "unknown", "None"]:
            speaker_counts[src].add(spk)

        attack_counts[atk] = attack_counts.get(atk, 0) + 1

    speakers_per_source = {src: len(spks) for src, spks in speaker_counts.items()}

    report_text = f"""
=================================================================
DATASET DISTRIBUTION & BALANCE REPORT
=================================================================
Total Samples     : {total}
Bona-Fide (Real)  : {class_counts['bona_fide']} ({100.0 * class_counts['bona_fide'] / total:.1f}%)
Spoof (Synthetic) : {class_counts['spoof']} ({100.0 * class_counts['spoof'] / total:.1f}%)

By Dataset Source:
"""
    for src, cnt in source_counts.items():
        spk_cnt = speakers_per_source.get(src, 0)
        report_text += f"  - {src:<18}: {cnt:>6} samples | {spk_cnt:>4} unique speakers ({100.0 * cnt / total:.1f}%)\n"

    report_text += "\nBy Language:\n"
    for lang, cnt in lang_counts.items():
        report_text += f"  - {lang:<18}: {cnt:>6} samples ({100.0 * cnt / total:.1f}%)\n"

    report_text += "\nBy Attack Type:\n"
    for atk, cnt in list(attack_counts.items())[:10]:
        report_text += f"  - {atk:<18}: {cnt:>6} samples ({100.0 * cnt / total:.1f}%)\n"

    report_text += "================================================================="
    logger.info(report_text)

    return {
        "status": "READY",
        "total_samples": total,
        "class_distribution": class_counts,
        "source_distribution": source_counts,
        "language_distribution": lang_counts,
        "speakers_per_source": speakers_per_source,
        "attack_distribution": attack_counts,
        "report_text": report_text,
    }


def compute_sample_weights(entries: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Computes balanced sample weights for PyTorch WeightedRandomSampler.
    Inverse frequency weighting across dataset source and class label.
    """
    total = len(entries)
    if total == 0:
        return torch.tensor([])

    # Compute source & class counts
    source_counts: Dict[str, int] = {}
    class_counts = {0: 0, 1: 0}

    for e in entries:
        src = e.get("source_dataset", "garaj")
        lbl = int(e.get("label_id", 0))
        source_counts[src] = source_counts.get(src, 0) + 1
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    weights = []
    for e in entries:
        src = e.get("source_dataset", "garaj")
        lbl = int(e.get("label_id", 0))

        src_w = 1.0 / float(source_counts.get(src, 1))
        lbl_w = 1.0 / float(class_counts.get(lbl, 1))

        # Joint dataset & class weight
        w = src_w * lbl_w
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    # Normalize weights
    weights_tensor = weights_tensor / weights_tensor.sum()
    return weights_tensor


def create_balanced_sampler(entries: List[Dict[str, Any]]) -> WeightedRandomSampler:
    """Creates PyTorch WeightedRandomSampler for balanced batch sampling."""
    weights = compute_sample_weights(entries)
    return WeightedRandomSampler(weights=weights, num_samples=len(entries), replacement=True)
