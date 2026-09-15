"""
Training Framework for GARAJ Experiment C: Feature-Augmented W2V2-AASIST.

Experiments:
  C1: Baseline W2V2-AASIST on multi-source dataset
  C2: Acoustic-feature-only classifier
  C3: W2V2-AASIST + Acoustic Feature Fusion (Primary Experiment)
  C4: Feature Ablation Study (A: W2V2 only, B: Acoustic only, C: +spectral, D: +LFCC, E: +CQCC, F: +spectral+LFCC+CQCC, G: Full set)

Checkpoints are saved with full metadata and NEVER overwrite LA_model.pth.
"""

import os
import sys
import time
import json
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader

from app.detection.aasist import W2V2AASIST
from app.detection.feature_fusion import W2V2AASISTFeatureFusion, AcousticOnlyClassifier
from app.features.acoustic_extractor import DEFAULT_FEATURE_GROUPS
from app.dataset.dataset import AudioDomainDataset
from app.dataset.dataset_balancer import generate_dataset_distribution_report, create_balanced_sampler

logger = logging.getLogger("train_experiment_c")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ablation_feature_groups(ablation_code: str) -> Dict[str, bool]:
    """Returns active feature groups configuration for ablation studies."""
    code = ablation_code.upper()
    if code in ["A", "W2V2_ONLY"]:
        return {k: False for k in DEFAULT_FEATURE_GROUPS}
    elif code in ["B", "ACOUSTIC_ONLY"]:
        return {k: True for k in DEFAULT_FEATURE_GROUPS}
    elif code == "C":
        return {"spectral": True, "lfcc": False, "cqcc": False, "prosodic": False, "energy": False}
    elif code == "D":
        return {"spectral": False, "lfcc": True, "cqcc": False, "prosodic": False, "energy": False}
    elif code == "E":
        return {"spectral": False, "lfcc": False, "cqcc": True, "prosodic": False, "energy": False}
    elif code == "F":
        return {"spectral": True, "lfcc": True, "cqcc": True, "prosodic": False, "energy": False}
    elif code in ["G", "FULL"]:
        return {k: True for k in DEFAULT_FEATURE_GROUPS}
    return dict(DEFAULT_FEATURE_GROUPS)


def train_experiment_c(
    experiment: str = "C3",
    ablation: str = "G",
    manifest_path: str = "/home/jyno/Projects/Garaj/datasets/splits/train.json",
    base_checkpoint_path: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth",
    output_checkpoint_path: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    freeze_ssl_encoder: bool = True,
    seed: int = 42,
    device_str: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    set_seed(seed)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if output_checkpoint_path is None:
        if experiment == "C1":
            output_checkpoint_path = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_multi_dataset_C1.pth"
        elif experiment == "C2":
            output_checkpoint_path = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_acoustic_only_C2.pth"
        elif experiment == "C4":
            output_checkpoint_path = f"/home/jyno/Projects/Garaj/backend/checkpoints/LA_ablation_C4_{ablation}.pth"
        else:
            output_checkpoint_path = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_feature_fusion_C3.pth"

    metadata_path = output_checkpoint_path.rsplit(".", 1)[0] + "_metadata.json"

    logger.info("=================================================================")
    logger.info(f"STARTING GARAJ EXPERIMENT {experiment} (Ablation: {ablation})")
    logger.info(f"Manifest Path   : {manifest_path}")
    logger.info(f"Base Checkpoint : {base_checkpoint_path}")
    logger.info(f"Output Path     : {output_checkpoint_path}")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate} | Device: {device}")
    logger.info("=================================================================")

    # 1. Dataset Inspection & Distribution Report
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Training manifest path '{manifest_path}' not found!")

    with open(manifest_path, "r", encoding="utf-8") as f:
        train_manifest_data = json.load(f)
    train_entries = train_manifest_data.get("entries", [])
    generate_dataset_distribution_report(train_entries)

    val_manifest = manifest_path.replace("train.json", "validation.json")
    if not os.path.exists(val_manifest):
        val_manifest = manifest_path

    train_ds = AudioDomainDataset(manifest_path=manifest_path, augment=True, seed=seed)
    val_ds = AudioDomainDataset(manifest_path=val_manifest, augment=False, seed=seed)

    sampler = create_balanced_sampler(train_entries) if train_entries else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Select Architecture based on Experiment Type
    feature_groups = get_ablation_feature_groups(ablation)

    if experiment == "C1":
        model = W2V2AASIST()
        if os.path.exists(base_checkpoint_path):
            sd = torch.load(base_checkpoint_path, map_location="cpu")
            if "state_dict" in sd: sd = sd["state_dict"]
            from app.detection.model_loader import convert_la_model_state_dict
            model.load_state_dict(convert_la_model_state_dict(sd), strict=False)
    elif experiment == "C2":
        from app.features.acoustic_extractor import AcousticFeatureExtractor
        extractor = AcousticFeatureExtractor(feature_groups=feature_groups)
        in_dim = extractor.get_dim()
        model = AcousticOnlyClassifier(in_dim=in_dim)
    else:  # Experiment C3 or C4
        base_aasist = W2V2AASIST()
        if os.path.exists(base_checkpoint_path):
            sd = torch.load(base_checkpoint_path, map_location="cpu")
            if "state_dict" in sd: sd = sd["state_dict"]
            from app.detection.model_loader import convert_la_model_state_dict
            base_aasist.load_state_dict(convert_la_model_state_dict(sd), strict=False)

        model = W2V2AASISTFeatureFusion(base_aasist_model=base_aasist, feature_groups=feature_groups)

    model.to(device)

    # Configure layer freezing
    if freeze_ssl_encoder and hasattr(model, "w2v2_aasist"):
        for p in model.w2v2_aasist.ssl_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)

    # 3. Training Pass
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for audio_batch, label_batch in train_loader:
            audio_batch = audio_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(audio_batch)
            loss = criterion(logits, label_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label_batch)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == label_batch).sum().item()
            train_total += len(label_batch)

        scheduler.step()
        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for audio_batch, label_batch in val_loader:
                audio_batch = audio_batch.to(device)
                label_batch = label_batch.to(device)

                logits = model(audio_batch)
                loss = criterion(logits, label_batch)

                val_loss += loss.item() * len(label_batch)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == label_batch).sum().item()
                val_total += len(label_batch)

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        dur = time.time() - start_t

        logger.info(f"Epoch [{epoch}/{epochs}] ({dur:.2f}s) | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.1f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.1f}%")

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), output_checkpoint_path)

    # 4. Save Metadata
    training_meta = {
        "experiment_name": f"Experiment_{experiment}_Ablation_{ablation}",
        "experiment": experiment,
        "ablation": ablation,
        "feature_groups": feature_groups,
        "output_checkpoint": output_checkpoint_path,
        "base_checkpoint": base_checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc_pct": round(best_val_acc, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    logger.info("=================================================================")
    logger.info(f"EXPERIMENT {experiment} COMPLETED SUCCESSFULLY!")
    logger.info(f"Adapted Checkpoint Saved : {output_checkpoint_path}")
    logger.info(f"Training Metadata Saved  : {metadata_path}")
    logger.info("=================================================================")

    return output_checkpoint_path, training_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARAJ Experiment C Training Pipeline")
    parser.add_argument("--experiment", type=str, default="C3", choices=["C1", "C2", "C3", "C4"], help="Experiment choice")
    parser.add_argument("--ablation", type=str, default="G", choices=["A", "B", "C", "D", "E", "F", "G"], help="Ablation configuration")
    parser.add_argument("--manifest", type=str, default="/home/jyno/Projects/Garaj/datasets/splits/train.json", help="Path to train split JSON")
    parser.add_argument("--base_checkpoint", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth", help="Base checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    train_experiment_c(
        experiment=args.experiment,
        ablation=args.ablation,
        manifest_path=args.manifest,
        base_checkpoint_path=args.base_checkpoint,
        output_checkpoint_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
