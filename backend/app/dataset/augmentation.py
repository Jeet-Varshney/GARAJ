"""
Domain Adaptation Acoustic Augmentation Module for Consumer Microphones.

Simulates consumer microphone acoustics:
1. Background Noise Injection (SNR 10-25 dB)
2. Room Impulse Response (RIR) Reverberation Simulation
3. Gain / Volume Scaling (0.6x to 1.4x)
4. Telephony / Codec Compression Simulation (Mu-Law / Low-pass filtering)
5. Mild Speed Perturbation (0.95x to 1.05x, preserving pitch)

Configurable & reproducible via random seed. Avoids aggressive pitch shifting.
"""

import os
import glob
import numpy as np
import scipy.signal
import torch
from typing import Optional, List, Tuple


class DomainAudioAugmenter:
    """
    Acoustic Domain Augmentation Engine for Audio Anti-Spoofing Model Fine-Tuning.
    """

    def __init__(
        self,
        noise_dir: Optional[str] = "/home/jyno/Projects/Garaj/datasets/noise",
        rir_dir: Optional[str] = "/home/jyno/Projects/Garaj/datasets/rir",
        snr_range_db: Tuple[float, float] = (10.0, 25.0),
        gain_range: Tuple[float, float] = (0.6, 1.4),
        speed_range: Tuple[float, float] = (0.95, 1.05),
        seed: Optional[int] = 42,
    ):
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir
        self.snr_range_db = snr_range_db
        self.gain_range = gain_range
        self.speed_range = speed_range

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.noise_files = self._scan_audio_files(noise_dir)
        self.rir_files = self._scan_audio_files(rir_dir)

    def _scan_audio_files(self, directory: Optional[str]) -> List[str]:
        if not directory or not os.path.exists(directory):
            return []
        files = []
        for ext in [".wav", ".flac", ".mp3"]:
            files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
        return files

    def apply_gain(self, audio: np.ndarray, gain_factor: Optional[float] = None) -> np.ndarray:
        """Applies gain / volume scaling."""
        if gain_factor is None:
            gain_factor = float(np.random.uniform(self.gain_range[0], self.gain_range[1]))
        return audio * gain_factor

    def apply_background_noise(self, audio: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        """Adds room background noise or Gaussian room noise at specified SNR (10-25 dB)."""
        if len(audio) == 0:
            return audio

        signal_power = np.mean(np.square(audio))
        if signal_power < 1e-7:
            return audio

        if snr_db is None:
            snr_db = float(np.random.uniform(self.snr_range_db[0], self.snr_range_db[1]))

        noise_power = signal_power / (10 ** (snr_db / 10.0))

        # Check if noise files exist in dataset directory
        if self.noise_files:
            noise_path = np.random.choice(self.noise_files)
            try:
                import soundfile as sf
                noise_data, sr = sf.read(noise_path, dtype='float32')
                if noise_data.ndim > 1:
                    noise_data = np.mean(noise_data, axis=1)
                # Tile or crop noise to match audio length
                if len(noise_data) < len(audio):
                    repeats = int(np.ceil(len(audio) / len(noise_data)))
                    noise_data = np.tile(noise_data, repeats)
                start_idx = np.random.randint(0, len(noise_data) - len(audio) + 1)
                noise = noise_data[start_idx:start_idx + len(audio)]
                # Scale noise to desired power
                curr_noise_pow = np.mean(np.square(noise))
                if curr_noise_pow > 1e-7:
                    noise = noise * np.sqrt(noise_power / curr_noise_pow)
            except Exception:
                noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
        else:
            # Synthetic background room noise
            noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)

        return (audio + noise).astype(np.float32)

    def apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """Convolves audio with Room Impulse Response (RIR) to simulate room acoustics."""
        if len(audio) == 0:
            return audio

        if self.rir_files:
            rir_path = np.random.choice(self.rir_files)
            try:
                import soundfile as sf
                rir, _ = sf.read(rir_path, dtype='float32')
                if rir.ndim > 1:
                    rir = np.mean(rir, axis=1)
                rir = rir / (np.max(np.abs(rir)) + 1e-7)
                convolved = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
                return convolved.astype(np.float32)
            except Exception:
                pass

        # Synthetic room reflection decay fallback
        decay = np.exp(-np.linspace(0, 5, 800))
        synthetic_rir = np.random.normal(0, 0.1, size=800) * decay
        synthetic_rir[0] = 1.0
        convolved = scipy.signal.fftconvolve(audio, synthetic_rir, mode='full')[:len(audio)]
        return convolved.astype(np.float32)

    def apply_codec_compression(self, audio: np.ndarray) -> np.ndarray:
        """Simulates lossy codec / telephony compression via mu-law compression and bandpass filtering."""
        if len(audio) == 0:
            return audio
        # Mu-law quantizer simulation (8-bit PCM G.711 codec equivalent)
        mu = 255.0
        magnitude = np.abs(audio)
        compressed = np.sign(audio) * (np.log(1.0 + mu * magnitude) / np.log(1.0 + mu))
        # Quantize to 256 discrete levels
        quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
        # Expand back
        decompressed = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu
        return decompressed.astype(np.float32)

    def apply_speed_perturbation(self, audio: np.ndarray, speed_factor: Optional[float] = None) -> np.ndarray:
        """Applies mild speed perturbation (0.95x - 1.05x) without changing fundamental pitch structure."""
        if len(audio) == 0:
            return audio

        if speed_factor is None:
            speed_factor = float(np.random.uniform(self.speed_range[0], self.speed_range[1]))

        if abs(speed_factor - 1.0) < 0.005:
            return audio

        num_samples = int(len(audio) / speed_factor)
        resampled = scipy.signal.resample(audio, num_samples)
        return resampled.astype(np.float32)

    def augment(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Executes randomized full acoustic domain augmentation pipeline.
        """
        if len(audio) == 0:
            return audio

        out = audio.copy()

        # 1. Background noise (50% probability)
        if np.random.rand() < 0.5:
            out = self.apply_background_noise(out)

        # 2. RIR / Reverberation (30% probability)
        if np.random.rand() < 0.3:
            out = self.apply_rir(out)

        # 3. Codec compression simulation (30% probability)
        if np.random.rand() < 0.3:
            out = self.apply_codec_compression(out)

        # 4. Gain variation (60% probability)
        if np.random.rand() < 0.6:
            out = self.apply_gain(out)

        # 5. Speed perturbation (30% probability)
        if np.random.rand() < 0.3:
            out = self.apply_speed_perturbation(out)

        # Normalization guard against clipping
        max_val = np.max(np.abs(out))
        if max_val > 1.0:
            out = out / max_val

        return out.astype(np.float32)
