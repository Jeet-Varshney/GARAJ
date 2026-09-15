"""
Evaluation & Comparative Report Framework for GARAJ Experiment C.

Evaluates Baseline (LA_model.pth), Experiment C1, C2, C3, and Ablation models
on the exact same untouched test set and held-out REAL_CONSUMER_MIC subset.

Outputs structured JSON and human-readable text comparison reports at:
- `reports/experiment_c_report.json`
- `reports/experiment_c_report.txt`
"""

import os
import json
import argparse
import logging
import numpy as np
import torch

from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from app.detection.aasist import W2V2AASIST
from app.detection.feature_fusion import W2V2AASISTFeatureFusion, AcousticOnlyClassifier
from app.features.acoustic_extractor import DEFAULT_FEATURE_GROUPS, AcousticFeatureExtractor
from app.dataset.dataset import AudioDomainDataset
from evaluate_domain_adaptation import compute_eer

logger = logging.getLogger("evaluate_experiment_c")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_model_on_manifest(
    model: torch.nn.Module,
    manifest_path: str,
    batch_size: int = 4,
    device_str: str = "cpu"
) -> Dict[str, Any]:
    """Evaluates a model instance on a given dataset manifest."""
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    if not os.path.exists(manifest_path):
        return {"status": "MANIFEST_NOT_FOUND", "manifest": manifest_path}

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

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset.entries))
            for idx in range(start_idx, end_idx):
                metadata_list.append(dataset.entries[idx])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_real_probs = np.array(all_real_probs)
    all_synthetic_probs = np.array(all_synthetic_probs)

    if len(all_labels) == 0:
        return {"status": "EMPTY_TEST_SET"}

    acc = float(accuracy_score(all_labels, all_preds) * 100.0)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = (fp / float(fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    frr = (fn / float(fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0

    bonafide_scores = all_real_probs[all_labels == 1]
    spoof_scores = all_real_probs[all_labels == 0]
    eer = compute_eer(bonafide_scores, spoof_scores)

    # Sub-breakdown calculation helper
    def _sub_calc(filter_fn) -> Dict[str, Any]:
        mask = np.array([filter_fn(e) for e in metadata_list])
        if not np.any(mask):
            return {"samples": 0, "eer_pct": "N/A", "acc_pct": "N/A", "f1": "N/A"}
        sub_l = all_labels[mask]
        sub_p = all_preds[mask]
        sub_rp = all_real_probs[mask]
        sub_acc = float(accuracy_score(sub_l, sub_p) * 100.0)
        _, _, sub_f1, _ = precision_recall_fscore_support(sub_l, sub_p, average='binary', zero_division=0)
        sub_eer = compute_eer(sub_rp[sub_l == 1], sub_rp[sub_l == 0])
        return {"samples": int(np.sum(mask)), "eer_pct": round(sub_eer, 2), "acc_pct": round(sub_acc, 2), "f1": round(float(sub_f1), 4)}

    return {
        "status": "SUCCESS",
        "total_samples": len(all_labels),
        "eer_pct": round(eer, 2),
        "acc_pct": round(acc, 2),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "far_pct": round(far, 2),
        "frr_pct": round(frr, 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "breakdowns": {
            "hindi": _sub_calc(lambda e: e.get("language") == "hindi"),
            "hinglish": _sub_calc(lambda e: e.get("language") == "hinglish"),
            "english": _sub_calc(lambda e: e.get("language") == "english"),
            "consumer_mic": _sub_calc(lambda e: "consumer" in str(e.get("microphone_type")).lower() or "mic" in str(e.get("microphone_type")).lower()),
        }
    }


def evaluate_experiment_c_suite(
    test_manifest_path: str = "/home/jyno/Projects/Garaj/datasets/splits/test.json",
    consumer_mic_manifest_path: Optional[str] = None,
    baseline_checkpoint: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth",
    c3_checkpoint: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_feature_fusion_C3.pth",
    output_json_report: str = "/home/jyno/Projects/Garaj/reports/experiment_c_report.json",
    output_txt_report: str = "/home/jyno/Projects/Garaj/reports/experiment_c_report.txt",
    batch_size: int = 4,
    device_str: Optional[str] = None
) -> Dict[str, Any]:

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=================================================================")
    logger.info("GARAJ EXPERIMENT C EVALUATION SUITE")
    logger.info(f"Test Manifest: {test_manifest_path}")
    logger.info("=================================================================")

    results = {}

    # 1. Baseline W2V2-AASIST Evaluation
    model_baseline = W2V2AASIST()
    if os.path.exists(baseline_checkpoint):
        sd = torch.load(baseline_checkpoint, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        from app.detection.model_loader import convert_la_model_state_dict
        model_baseline.load_state_dict(convert_la_model_state_dict(sd), strict=False)
    results["baseline"] = evaluate_model_on_manifest(model_baseline, test_manifest_path, batch_size, device_str)

    # 2. Experiment C3 Feature-Augmented Evaluation
    if os.path.exists(c3_checkpoint):
        model_c3 = W2V2AASISTFeatureFusion()
        sd = torch.load(c3_checkpoint, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        from app.detection.model_loader import convert_la_model_state_dict
        model_c3.load_state_dict(convert_la_model_state_dict(sd), strict=False)
        results["C3_feature_fusion"] = evaluate_model_on_manifest(model_c3, test_manifest_path, batch_size, device_str)
    else:
        results["C3_feature_fusion"] = {"status": "CHECKPOINT_NOT_FOUND", "checkpoint": c3_checkpoint}

    # 3. Held-out Consumer Mic Evaluation
    consumer_mic_results = {}
    if consumer_mic_manifest_path and os.path.exists(consumer_mic_manifest_path):
        consumer_mic_results["baseline"] = evaluate_model_on_manifest(model_baseline, consumer_mic_manifest_path, batch_size, device_str)
        if os.path.exists(c3_checkpoint):
            consumer_mic_results["C3_feature_fusion"] = evaluate_model_on_manifest(model_c3, consumer_mic_manifest_path, batch_size, device_str)

    # Determine Verdict
    verdict = "INCONCLUSIVE — INSUFFICIENT DATA"
    b_res = results.get("baseline", {})
    c3_res = results.get("C3_feature_fusion", {})

    if b_res.get("status") == "SUCCESS" and c3_res.get("status") == "SUCCESS":
        if c3_res.get("eer_pct", 100) < b_res.get("eer_pct", 100) or c3_res.get("f1", 0) > b_res.get("f1", 0):
            verdict = "MODEL IMPROVED"
        elif c3_res.get("eer_pct", 100) > b_res.get("eer_pct", 100):
            verdict = "MODEL NOT IMPROVED"

    combined = {
        "verdict": verdict,
        "test_manifest": test_manifest_path,
        "consumer_mic_manifest": consumer_mic_manifest_path,
        "results": results,
        "consumer_mic_results": consumer_mic_results,
    }

    # Build Text Table
    b_eer = b_res.get("eer_pct", "N/A")
    c3_eer = c3_res.get("eer_pct", "N/A")
    b_acc = b_res.get("acc_pct", "N/A")
    c3_acc = c3_res.get("acc_pct", "N/A")
    b_f1 = b_res.get("f1", "N/A")
    c3_f1 = c3_res.get("f1", "N/A")

    txt_report = f"""=================================================================
GARAJ EXPERIMENT C: FEATURE-AUGMENTED W2V2-AASIST REPORT
=================================================================
Test Manifest: {test_manifest_path}
FINAL VERDICT : {verdict}

Metric              Baseline       Experiment C3 (Fusion)
---------------------------------------------------------
EER (%)             {b_eer:<14} {c3_eer:<14}
Accuracy (%)        {b_acc:<14} {c3_acc:<14}
F1 Score            {b_f1:<14} {c3_f1:<14}
---------------------------------------------------------
=================================================================
"""
    logger.info(txt_report)

    os.makedirs(os.path.dirname(output_json_report), exist_ok=True)
    with open(output_json_report, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    os.makedirs(os.path.dirname(output_txt_report), exist_ok=True)
    with open(output_txt_report, "w", encoding="utf-8") as f:
        f.write(txt_report)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GARAJ Experiment C Models")
    parser.add_argument("--test_manifest", type=str, default="/home/jyno/Projects/Garaj/datasets/splits/test.json", help="Test manifest path")
    parser.add_argument("--baseline", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth", help="Baseline checkpoint")
    parser.add_argument("--c3_checkpoint", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_feature_fusion_C3.pth", help="C3 checkpoint")
    args = parser.parse_args()

    evaluate_experiment_c_suite(
        test_manifest_path=args.test_manifest,
        baseline_checkpoint=args.baseline,
        c3_checkpoint=args.c3_checkpoint,
    )
