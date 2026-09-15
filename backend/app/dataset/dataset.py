"""
PyTorch Dataset for Voice Anti-Spoofing & Deepfake Detection Domain Adaptation.

Ingests 16 kHz audio files specified in manifest JSON,
standardizes waveforms to Float32 mono tensors of shape (64600,),
and applies domain acoustic augmentations (noise, RIR, AGC, compression, speed).

Label Mapping:
  0 = Spoof / Synthetic
  1 = Bona-Fide / Real
"""

import os
import json
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from app.dataset.augmentation import DomainAudioAugmenter

logger = logging.getLogger("audio_dataset")


class AudioDomainDataset(Dataset):
    """
    PyTorch Dataset implementation for Garaj anti-spoofing domain adaptation.
    """

    def __init__(
        self,
        manifest_path: str,
        target_sample_rate: int = 16000,
        required_samples: int = 64600,
        augment: bool = False,
        seed: Optional[int] = 42,
    ):
        self.target_sample_rate = target_sample_rate
        self.required_samples = required_samples
        self.augment = augment
        self.augmenter = DomainAudioAugmenter(seed=seed) if augment else None

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Dataset manifest file not found at: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.entries: List[Dict[str, Any]] = manifest_data.get("entries", [])
        logger.info(f"Initialized AudioDomainDataset ({len(self.entries)} samples) from {manifest_path} (augment={augment})")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Loads audio file as 16kHz Float32 mono numpy array."""
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.target_sample_rate:
                import scipy.signal
                num_samples = int(len(audio) * self.target_sample_rate / sr)
                audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            return audio.astype(np.float32)
        except Exception:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                    waveform = resampler(waveform)
                audio = waveform.squeeze(0).numpy().astype(np.float32)
                return audio
            except Exception as exc:
                logger.error(f"Failed decoding audio file '{file_path}': {exc}")
                return np.zeros(self.required_samples, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        file_path = entry.get("path") or entry.get("audio_path")

        # Extract numeric label_id (0=Spoof, 1=Bona-Fide)
        if "label_id" in entry:
            label = int(entry["label_id"])
        elif "label" in entry:
            lbl_val = entry["label"]
            label = 1 if lbl_val in [1, "1", "bona_fide", "real"] else 0
        else:
            label = 0

        audio = self._load_audio_file(file_path)

        # Validation: NaN/Inf check
        if np.isnan(audio).any() or np.isinf(audio).any():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment and self.augmenter is not None:
            audio = self.augmenter.augment(audio, sample_rate=self.target_sample_rate)

        # Format window to exactly 64,600 samples
        if len(audio) > self.required_samples:
            max_start = len(audio) - self.required_samples
            start = np.random.randint(0, max_start + 1) if self.augment else 0
            audio = audio[start:start + self.required_samples]
        elif len(audio) < self.required_samples:
            pad_width = self.required_samples - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant', constant_values=0.0)

        audio_tensor = torch.from_numpy(audio).float()
        return audio_tensor, label
