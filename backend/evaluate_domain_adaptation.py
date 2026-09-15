"""
W2V2-AASIST Domain Adaptation Evaluation Suite for Garaj.

Evaluates BOTH Baseline (LA_model.pth) and Domain-Adapted (LA_domain_adapted.pth)
checkpoints on the exact same test split.

Calculates:
- Equal Error Rate (EER)
- Accuracy, Precision, Recall, F1 Score
- False Acceptance Rate (FAR) & False Rejection Rate (FRR)
- Confusion Matrix

Generates reports:
- `reports/domain_adaptation_report.json`
- `reports/domain_adaptation_report.txt`
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from app.detection.aasist import W2V2AASIST
from app.dataset.dataset import AudioDomainDataset

logger = logging.getLogger("evaluate_domain_adaptation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) given bonafide (real) and spoof (synthetic) detection scores.
    Scores represent real probability (higher score = more likely real).
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0

    target_scores = bonafide_scores
    nontarget_scores = spoof_scores

    thresholds = np.sort(np.concatenate([target_scores, nontarget_scores]))
    if len(thresholds) == 0:
        return 0.0

    frr = np.array([np.mean(target_scores < t) for t in thresholds])
    far = np.array([np.mean(nontarget_scores >= t) for t in thresholds])

    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = (frr[min_index] + far[min_index]) / 2.0 * 100.0
    return float(eer)


def _eval_single_checkpoint(
    checkpoint_path: str,
    manifest_path: str,
    batch_size: int = 4,
    device_str: str = "cpu"
) -> Dict[str, Any]:
    device = torch.device(device_str)
    model = W2V2AASIST()

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path '{checkpoint_path}' does not exist!")
        return {"status": "CHECKPOINT_NOT_FOUND", "checkpoint": checkpoint_path}

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    from app.detection.model_loader import convert_la_model_state_dict
    converted_state_dict = convert_la_model_state_dict(state_dict)
    model.load_state_dict(converted_state_dict, strict=False)
    model.to(device)
    model.eval()

    dataset = AudioDomainDataset(manifest_path=manifest_path, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_real_probs = []
    all_synthetic_probs = []
    metadata_list = []

    with torch.no_grad():
        for i, (audio_batch, label_batch) in enumerate(dataloader):
            audio_batch = audio_batch.to(device)
            logits = model(audio_batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            synthetic_probs = probs[:, 0]
            real_probs = probs[:, 1] if probs.shape[1] > 1 else (1.0 - synthetic_probs)
            preds = (real_probs >= synthetic_probs).astype(int)

            all_labels.extend(label_batch.numpy())
            all_preds.extend(preds)
            all_real_probs.extend(real_probs)
            all_synthetic_probs.extend(synthetic_probs)

            # Metadata tracking per sample
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset.entries))
            for idx in range(start_idx, end_idx):
                metadata_list.append(dataset.entries[idx])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_real_probs = np.array(all_real_probs)
    all_synthetic_probs = np.array(all_synthetic_probs)

    if len(all_labels) == 0:
        return {"status": "EMPTY_TEST_SET", "checkpoint": checkpoint_path}

    # Primary metrics
    acc = float(accuracy_score(all_labels, all_preds) * 100.0)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = (fp / float(fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    frr = (fn / float(fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0

    bonafide_scores = all_real_probs[all_labels == 1]
    spoof_scores = all_real_probs[all_labels == 0]
    eer = compute_eer(bonafide_scores, spoof_scores)

    # Sub-breakdown metrics: Hindi, Hinglish, English, Consumer Mic
    def _sub_metrics(mask: np.ndarray) -> Dict[str, Any]:
        if not np.any(mask):
            return {"samples": 0, "eer_pct": "N/A", "acc_pct": "N/A", "f1": "N/A"}
        sub_labels = all_labels[mask]
        sub_preds = all_preds[mask]
        sub_real_probs = all_real_probs[mask]
        sub_acc = float(accuracy_score(sub_labels, sub_preds) * 100.0)
        _, _, sub_f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average='binary', zero_division=0)
        sub_bf = sub_real_probs[sub_labels == 1]
        sub_sp = sub_real_probs[sub_labels == 0]
        sub_eer = compute_eer(sub_bf, sub_sp)
        return {
            "samples": int(np.sum(mask)),
            "eer_pct": round(sub_eer, 2),
            "acc_pct": round(sub_acc, 2),
            "f1": round(float(sub_f1), 4),
        }

    hindi_mask = np.array([e.get("language") == "hindi" for e in metadata_list])
    hinglish_mask = np.array([e.get("language") == "hinglish" for e in metadata_list])
    english_mask = np.array([e.get("language") == "english" for e in metadata_list])
    consumer_mic_mask = np.array([e.get("microphone_type") == "consumer_mic" for e in metadata_list])

    return {
        "status": "SUCCESS",
        "checkpoint": checkpoint_path,
        "total_samples": len(all_labels),
        "eer_pct": round(eer, 2),
        "acc_pct": round(acc, 2),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "far_pct": round(far, 2),
        "frr_pct": round(frr, 2),
        "confusion_matrix": {
            "tn_spoof_correct": int(tn),
            "fp_spoof_as_real": int(fp),
            "fn_real_as_spoof": int(fn),
            "tp_real_correct": int(tp),
        },
        "breakdown": {
            "hindi": _sub_metrics(hindi_mask),
            "hinglish": _sub_metrics(hinglish_mask),
            "english": _sub_metrics(english_mask),
            "consumer_mic": _sub_metrics(consumer_mic_mask),
        }
    }


def evaluate_domain_adaptation(
    manifest_path: str = "/home/jyno/Projects/Garaj/datasets/splits/test.json",
    baseline_checkpoint: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth",
    adapted_checkpoint: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_domain_adapted.pth",
    output_json_report: str = "/home/jyno/Projects/Garaj/reports/domain_adaptation_report.json",
    output_txt_report: str = "/home/jyno/Projects/Garaj/reports/domain_adaptation_report.txt",
    batch_size: int = 4,
    device_str: Optional[str] = None
) -> Dict[str, Any]:

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=================================================================")
    logger.info("W2V2-AASIST DOMAIN ADAPTATION DUAL EVALUATION SUITE")
    logger.info(f"Test Manifest      : {manifest_path}")
    logger.info(f"Baseline Checkpoint: {baseline_checkpoint}")
    logger.info(f"Adapted Checkpoint : {adapted_checkpoint}")
    logger.info("=================================================================")

    if not os.path.exists(manifest_path):
        fallback_manifest = "/home/jyno/Projects/Garaj/datasets/manifests/dataset_manifest.json"
        if os.path.exists(fallback_manifest):
            manifest_path = fallback_manifest
        else:
            logger.warning(f"Test manifest {manifest_path} not found! DATASET_NOT_FOUND.")

    # 1. Evaluate Baseline
    baseline_res = _eval_single_checkpoint(baseline_checkpoint, manifest_path, batch_size, device_str)

    # 2. Evaluate Adapted Checkpoint
    adapted_res = _eval_single_checkpoint(adapted_checkpoint, manifest_path, batch_size, device_str)

    # Determine domain adaptation improvement verdict
    improvement = "INCONCLUSIVE"
    if baseline_res.get("status") == "SUCCESS" and adapted_res.get("status") == "SUCCESS":
        base_eer = baseline_res["eer_pct"]
        adap_eer = adapted_res["eer_pct"]
        if adap_eer < base_eer or adapted_res["f1"] > baseline_res["f1"]:
            improvement = "YES"
        elif adap_eer > base_eer:
            improvement = "NO"
        else:
            improvement = "INCONCLUSIVE"

    combined_report = {
        "status": "COMPLETED",
        "test_manifest": manifest_path,
        "domain_adaptation_improvement": improvement,
        "baseline": baseline_res,
        "adapted": adapted_res,
    }

    # Format text table report
    b_eer = baseline_res.get("eer_pct", "N/A")
    a_eer = adapted_res.get("eer_pct", "N/A")
    b_acc = baseline_res.get("acc_pct", "N/A")
    a_acc = adapted_res.get("acc_pct", "N/A")
    b_prec = baseline_res.get("precision", "N/A")
    a_prec = adapted_res.get("precision", "N/A")
    b_rec = baseline_res.get("recall", "N/A")
    a_rec = adapted_res.get("recall", "N/A")
    b_f1 = baseline_res.get("f1", "N/A")
    a_f1 = adapted_res.get("f1", "N/A")
    b_far = baseline_res.get("far_pct", "N/A")
    a_far = adapted_res.get("far_pct", "N/A")
    b_frr = baseline_res.get("frr_pct", "N/A")
    a_frr = adapted_res.get("frr_pct", "N/A")

    txt_content = f"""=================================================================
GARAJ W2V2-AASIST DOMAIN ADAPTATION EVALUATION REPORT
=================================================================
Test Manifest: {manifest_path}
Domain Adaptation Improvement: {improvement}

Metric              Baseline       Adapted
------------------------------------------------
EER (%)             {b_eer:<14} {a_eer:<14}
Accuracy (%)        {b_acc:<14} {a_acc:<14}
Precision           {b_prec:<14} {a_prec:<14}
Recall              {b_rec:<14} {a_rec:<14}
F1 Score            {b_f1:<14} {a_f1:<14}
FAR (%)             {b_far:<14} {a_far:<14}
FRR (%)             {b_frr:<14} {a_frr:<14}
------------------------------------------------

SUB-BREAKDOWN METRICS:
Hindi       : Baseline EER={baseline_res.get('breakdown',{}).get('hindi',{}).get('eer_pct','N/A')}% | Adapted EER={adapted_res.get('breakdown',{}).get('hindi',{}).get('eer_pct','N/A')}%
Hinglish    : Baseline EER={baseline_res.get('breakdown',{}).get('hinglish',{}).get('eer_pct','N/A')}% | Adapted EER={adapted_res.get('breakdown',{}).get('hinglish',{}).get('eer_pct','N/A')}%
English     : Baseline EER={baseline_res.get('breakdown',{}).get('english',{}).get('eer_pct','N/A')}% | Adapted EER={adapted_res.get('breakdown',{}).get('english',{}).get('eer_pct','N/A')}%
Consumer Mic: Baseline EER={baseline_res.get('breakdown',{}).get('consumer_mic',{}).get('eer_pct','N/A')}% | Adapted EER={adapted_res.get('breakdown',{}).get('consumer_mic',{}).get('eer_pct','N/A')}%
=================================================================
"""
    logger.info(txt_content)

    os.makedirs(os.path.dirname(output_json_report), exist_ok=True)
    with open(output_json_report, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)

    os.makedirs(os.path.dirname(output_txt_report), exist_ok=True)
    with open(output_txt_report, "w", encoding="utf-8") as f:
        f.write(txt_content)

    logger.info(f"Saved evaluation report JSON -> {output_json_report}")
    logger.info(f"Saved evaluation report TXT  -> {output_txt_report}")

    return combined_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W2V2-AASIST Domain Adaptation Evaluation Suite")
    parser.add_argument("--manifest", type=str, default="/home/jyno/Projects/Garaj/datasets/splits/test.json", help="Path to test dataset manifest JSON")
    parser.add_argument("--baseline", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth", help="Path to baseline checkpoint")
    parser.add_argument("--adapted", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_domain_adapted.pth", help="Path to domain adapted checkpoint")
    parser.add_argument("--output_json", type=str, default="/home/jyno/Projects/Garaj/reports/domain_adaptation_report.json", help="Output JSON report path")
    parser.add_argument("--output_txt", type=str, default="/home/jyno/Projects/Garaj/reports/domain_adaptation_report.txt", help="Output TXT report path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    evaluate_domain_adaptation(
        manifest_path=args.manifest,
        baseline_checkpoint=args.baseline,
        adapted_checkpoint=args.adapted,
        output_json_report=args.output_json,
        output_txt_report=args.output_txt,
        batch_size=args.batch_size,
        device_str=args.device,
    )
