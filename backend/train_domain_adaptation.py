"""
W2V2-AASIST Domain Adaptation Fine-Tuning Pipeline for Garaj.

Fine-tunes the baseline pre-trained W2V2-AASIST model on domain adaptation data.

Experiment A (Default):
- XLS-R 300M SSL Feature Encoder: FROZEN (requires_grad = False)
- AASIST Graph Attention Layers & Classifier Head: TRAINABLE (requires_grad = True, LR = 1e-4)

Experiment B (Optional):
- XLS-R 300M SSL Feature Encoder: TRAINABLE (LR = 1e-6)
- AASIST Graph Attention & Head: TRAINABLE (LR = 1e-4)

Loss Functions:
- Weighted Cross Entropy (`weighted_ce`)
- Focal Loss (`focal`)

Saves adapted checkpoint to `backend/checkpoints/LA_domain_adapted.pth`
and metadata to `backend/checkpoints/LA_domain_adapted_metadata.json`.
"""

import os
import sys
import time
import json
import random
import argparse
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.detection.aasist import W2V2AASIST
from app.dataset.dataset import AudioDomainDataset

logger = logging.getLogger("train_domain_adaptation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in anti-spoofing binary classification.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def set_seed(seed: int = 42):
    """Sets deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_domain_adaptation(
    manifest_path: str,
    base_checkpoint_path: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth",
    output_checkpoint_path: str = "/home/jyno/Projects/Garaj/backend/checkpoints/LA_domain_adapted.pth",
    metadata_output_path: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    freeze_ssl_encoder: bool = True,
    loss_type: str = "weighted_ce",
    num_workers: int = 0,
    seed: int = 42,
    device_str: Optional[str] = None
) -> str:
    set_seed(seed)

    if metadata_output_path is None:
        metadata_output_path = output_checkpoint_path.rsplit(".", 1)[0] + "_metadata.json"

    # Resource awareness & Device detection
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if device.type == "cpu":
        logger.warning("[Device Awareness] Running on CPU. Setting num_workers=0 and maintaining modest batch size.")
        num_workers = 0

    logger.info("=================================================================")
    logger.info("STARTING W2V2-AASIST DOMAIN ADAPTATION FINE-TUNING PIPELINE")
    logger.info(f"Base Checkpoint : {base_checkpoint_path}")
    logger.info(f"Manifest Path   : {manifest_path}")
    logger.info(f"Output Path     : {output_checkpoint_path}")
    logger.info(f"Epochs          : {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    logger.info(f"Freeze SSL Enc  : {freeze_ssl_encoder} | Loss: {loss_type} | Device: {device}")
    logger.info("=================================================================")

    # 1. Instantiate W2V2-AASIST Model
    model = W2V2AASIST()

    # Load baseline pre-trained weights if available
    if os.path.exists(base_checkpoint_path):
        state_dict = torch.load(base_checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Convert state dict keys if needed
        from app.detection.model_loader import convert_la_model_state_dict
        converted_state_dict = convert_la_model_state_dict(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        logger.info(f"[Model Loader] Loaded baseline checkpoint from {base_checkpoint_path} (Missing: {len(missing_keys)})")
    else:
        logger.warning(f"[Model Loader Warning] Base checkpoint {base_checkpoint_path} not found! Initializing default model weights.")

    model.to(device)

    # 2. Configure Optimizer and Parameter Grouping (Experiment A vs Experiment B)
    if freeze_ssl_encoder:
        logger.info("[Experiment A] Freezing XLS-R 300M SSL encoder. Training AASIST graph attention & classification head.")
        for param in model.ssl_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
    else:
        logger.info("[Experiment B] Unfreezing XLS-R 300M with differential LR (SSL: 1e-6, AASIST: 1e-4).")
        ssl_params = list(model.ssl_model.parameters())
        backend_params = [p for n, p in model.named_parameters() if not n.startswith("ssl_model.")]
        optimizer = torch.optim.AdamW([
            {"params": ssl_params, "lr": 1e-6},
            {"params": backend_params, "lr": learning_rate},
        ], weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Parameter Footprint] Total: {total_params:,} | Trainable: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    # 3. Load Datasets (Train and Validation splits)
    val_manifest = manifest_path.replace("train.json", "validation.json")
    if not os.path.exists(val_manifest):
        val_manifest = manifest_path

    train_dataset = AudioDomainDataset(manifest_path=manifest_path, augment=True, seed=seed)
    val_dataset = AudioDomainDataset(manifest_path=val_manifest, augment=False, seed=seed)

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset manifest at {manifest_path} contains 0 valid samples!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) >= batch_size), num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"[Dataset Loaded] Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # 4. Criterion and Scheduler
    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
    else:
        # Weighted Cross EntropyLoss
        weights = torch.tensor([1.0, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_loss = float("inf")
    best_val_acc = 0.0

    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (audio_batch, label_batch) in enumerate(train_loader):
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
        epoch_duration = time.time() - start_time

        logger.info(f"Epoch [{epoch}/{epochs}] ({epoch_duration:.2f}s) | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.1f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.1f}%")

        # Save Best Validation Checkpoint
        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), output_checkpoint_path)
            logger.info(f"  -> Best checkpoint saved to {output_checkpoint_path}")

    # 6. Save Training Metadata
    metadata = {
        "dataset_version": "garaj_domain_v1",
        "manifest_path": manifest_path,
        "base_checkpoint": base_checkpoint_path,
        "output_checkpoint": output_checkpoint_path,
        "experiment": "Experiment_A_Frozen_XLSR" if freeze_ssl_encoder else "Experiment_B_Unfrozen_XLSR",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": loss_type,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy_pct": round(best_val_acc, 2),
        "model_architecture": "W2V2-AASIST",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
    }

    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=================================================================")
    logger.info("DOMAIN ADAPTATION FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f} | Accuracy: {best_val_acc:.1f}%")
    logger.info(f"Adapted Checkpoint : {output_checkpoint_path}")
    logger.info(f"Training Metadata  : {metadata_output_path}")
    logger.info("=================================================================")

    return output_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W2V2-AASIST Domain Adaptation Fine-Tuning Pipeline")
    parser.add_argument("--manifest", type=str, required=True, help="Path to input dataset manifest JSON file")
    parser.add_argument("--base_checkpoint", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_model.pth", help="Path to base pre-trained checkpoint")
    parser.add_argument("--output", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_domain_adapted.pth", help="Path to output fine-tuned checkpoint")
    parser.add_argument("--metadata_output", type=str, default="/home/jyno/Projects/Garaj/backend/checkpoints/LA_domain_adapted_metadata.json", help="Path to output metadata JSON file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "focal"], help="Loss function choice")
    parser.add_argument("--freeze_xlsr", type=str, default="True", help="Freeze XLS-R 300M SSL encoder (True/False)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    freeze_bool = args.freeze_xlsr.lower() in ["true", "1", "yes"]

    train_domain_adaptation(
        manifest_path=args.manifest,
        base_checkpoint_path=args.base_checkpoint,
        output_checkpoint_path=args.output,
        metadata_output_path=args.metadata_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_ssl_encoder=freeze_bool,
        loss_type=args.loss,
        num_workers=args.num_workers,
        seed=args.seed,
        device_str=args.device,
    )
