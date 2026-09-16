"""
Model Engine Module for W2V2-AASIST Real Voice Deepfake Detection.

Implements BaseDetectionEngine interface using PyTorch W2V2-AASIST architecture.
Formats incoming audio arrays to exactly 64,600 samples (4.04s @ 16kHz) and executes PyTorch inference_mode().

STRICT SPECIFICATION:
- Requires genuine trained 'LA_model.pth' checkpoint.
- If LA_model.pth is missing: returns status: "MODEL_CHECKPOINT_MISSING" with real_probability: None.
- During startup before 64,600 genuine samples accumulate: returns status: "ANALYZING".
- Zero fake probabilities, zero random heads, zero embedding norm heuristics.
"""

import time
import numpy as np
import logging
from typing import Dict, Any, Optional
from app.detection.base import BaseDetectionEngine
from app.detection.model_loader import load_xlsr_aasist_model, ModelLoadError

logger = logging.getLogger("model_engine")


# Threshold constants for input audio energy gate
# Determines if sufficient audio energy exists for W2V2-AASIST anti-spoof inference
MIN_INPUT_RMS = 0.0030    # ~100 PCM amplitude out of 32768
MIN_INPUT_PEAK = 0.0080   # ~260 PCM amplitude out of 32768


class ModelDetectionEngine(BaseDetectionEngine):
    """
    PyTorch W2V2-AASIST Voice Deepfake Detection Engine.
    
    Implements BaseDetectionEngine interface contract.
    """

    def __init__(self, model_name_or_path: Optional[str] = None):
        self.engine_name = "W2V2_AASIST_Engine"
        self.is_stub = False
        
        try:
            self.feature_extractor, self.model, self.device, self.model_meta = load_xlsr_aasist_model(
                model_name_or_path=model_name_or_path
            )
            self.model_loaded = (self.model is not None)
            self.status = self.model_meta.get("status", "MODEL_CHECKPOINT_MISSING")
            import threading
            import hashlib
            self._is_inferring = False
            self._last_result = None
            self._lock = threading.Lock()
            self.inference_count = 0
            self._prev_window = None
            self._prev_window_hash = None
            self._prev_rms = None
        except ModelLoadError as exc:
            logger.error(f"Initialization of ModelDetectionEngine failed: {exc}")
            self.model_loaded = False
            self.status = "MODEL_LOAD_ERROR"
            self.load_error = str(exc)
            raise

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent non-blocking inference result dictionary if available."""
        with self._lock:
            if self._last_result is not None:
                return dict(self._last_result)
        return None

    def process_window(
        self,
        audio_window: np.ndarray,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes ML inference on a 16 kHz Float32 audio window using W2V2-AASIST.
        Inputs are deterministically formatted to exactly 64,600 samples (4.04s @ 16kHz).
        """
        import torch

        start_time = time.perf_counter()
        required_samples = 64600
        metadata = metadata or {}
        num_samples = len(audio_window)

        # Compute complete input audio waveform statistics for telemetry
        if num_samples > 0:
            rms = float(np.sqrt(np.mean(np.square(audio_window))))
            peak_amp = float(np.max(np.abs(audio_window)))
            min_val = float(np.min(audio_window))
            max_val = float(np.max(audio_window))
            zero_crossings = np.nonzero(np.diff(audio_window >= 0))[0] if num_samples > 1 else np.array([])
            zcr = float(len(zero_crossings) / float(num_samples))
        else:
            rms = 0.0
            peak_amp = 0.0
            min_val = 0.0
            max_val = 0.0
            zcr = 0.0

        audio_metrics = {
            "rms": round(rms, 6),
            "peak_amplitude": round(peak_amp, 6),
            "min_val": round(min_val, 6),
            "max_val": round(max_val, 6),
            "zero_crossing_rate": round(zcr, 5),
            "num_samples": num_samples,
        }

        # 1. Check if trained model checkpoint is present
        if self.model is None or not self.model_meta.get("trained_classifier", False):
            compute_latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_str = self.model_meta.get("status", "MODEL_CHECKPOINT_MISSING")
            return {
                "engine": self.engine_name,
                "model": "W2V2-AASIST",
                "model_checkpoint": self.model_meta.get("checkpoint_path"),
                "is_stub": False,
                "model_loaded": False,
                "status": status_str,
                "predicted_class": "UNAVAILABLE",
                "real_probability": None,
                "synthetic_probability": None,
                "risk_score": None,
                "confidence": 0.0,
                "message": self.model_meta.get("error", "Trained W2V2-AASIST checkpoint LA_model.pth is missing. Detection unavailable."),
                "audio_metrics": audio_metrics,
                "window_num_samples": num_samples,
                "required_samples": required_samples,
                "sample_rate": sample_rate,
                "device": str(self.device),
                "compute_latency_ms": round(compute_latency_ms, 3),
                "timestamp": time.time(),
            }

        # 2. Check if enough real audio samples have accumulated (Startup / Buffer Filling)
        total_bytes = metadata.get("total_pcm_bytes", num_samples * 2)
        if total_bytes < required_samples * 2 and num_samples < required_samples:
            compute_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "engine": self.engine_name,
                "model": "W2V2-AASIST",
                "model_checkpoint": self.model_meta.get("checkpoint_path"),
                "is_stub": False,
                "model_loaded": True,
                "status": "ANALYZING",
                "predicted_class": "ANALYZING",
                "real_probability": None,
                "synthetic_probability": None,
                "risk_score": None,
                "confidence": 0.0,
                "message": f"Accumulating continuous live audio buffer ({num_samples}/{required_samples} samples)...",
                "audio_metrics": audio_metrics,
                "window_num_samples": num_samples,
                "required_samples": required_samples,
                "sample_rate": sample_rate,
                "device": str(self.device),
                "compute_latency_ms": round(compute_latency_ms, 3),
                "timestamp": time.time(),
            }

        # 3. Input Audio Energy Gate (NO_AUDIO)
        # Prevents sending silent/ambient background noise to W2V2-AASIST out-of-domain classifier
        if rms < MIN_INPUT_RMS and peak_amp < MIN_INPUT_PEAK:
            compute_latency_ms = (time.perf_counter() - start_time) * 1000.0
            res = {
                "engine": self.engine_name,
                "model": "W2V2-AASIST",
                "model_checkpoint": self.model_meta.get("checkpoint_path"),
                "is_stub": False,
                "model_loaded": True,
                "status": "NO_AUDIO",
                "predicted_class": "NO_AUDIO",
                "real_probability": None,
                "synthetic_probability": None,
                "risk_score": None,
                "confidence": 0.0,
                "message": f"No sufficient audio energy for anti-spoof inference (RMS: {rms:.5f} < {MIN_INPUT_RMS}).",
                "audio_metrics": audio_metrics,
                "window_num_samples": num_samples,
                "required_samples": required_samples,
                "sample_rate": sample_rate,
                "device": str(self.device),
                "compute_latency_ms": round(compute_latency_ms, 3),
                "timestamp": time.time(),
            }
            with self._lock:
                self._last_result = res
            return res

        # Check non-blocking lock: if an inference pass is currently running on worker thread, return cached result
        with self._lock:
            if self._is_inferring and self._last_result is not None:
                res = dict(self._last_result)
                res["cached"] = True
                return res
            self._is_inferring = True

        try:
            import hashlib
            # 4. Deterministic 64,600-Sample Audio Preprocessing
            if audio_window.dtype != np.float32:
                audio_window = audio_window.astype(np.float32)

            # Slice latest 64,600 samples or zero-pad on the right to exactly 64,600
            if len(audio_window) > required_samples:
                processed_audio = audio_window[-required_samples:]
            elif len(audio_window) < required_samples:
                pad_width = required_samples - len(audio_window)
                processed_audio = np.pad(audio_window, (0, pad_width), mode='constant', constant_values=0.0)
            else:
                processed_audio = audio_window

            # Compute window hash, NaN/Inf check, and consecutive window differences
            window_bytes = processed_audio.tobytes()
            window_hash = hashlib.md5(window_bytes).hexdigest()[:10]
            has_nan = bool(np.isnan(processed_audio).any())
            has_inf = bool(np.isinf(processed_audio).any())

            consecutive_diff = None
            if self._prev_window is not None and len(self._prev_window) == len(processed_audio):
                rms_diff = float(np.abs(rms - (self._prev_rms if self._prev_rms is not None else 0.0)))
                mean_abs_diff = float(np.mean(np.abs(processed_audio - self._prev_window)))
                max_abs_diff = float(np.max(np.abs(processed_audio - self._prev_window)))
                is_identical = bool(self._prev_window_hash == window_hash)
                consecutive_diff = {
                    "rms_diff": round(rms_diff, 6),
                    "mean_abs_diff": round(mean_abs_diff, 6),
                    "max_abs_diff": round(max_abs_diff, 6),
                    "is_identical": is_identical,
                }

            # Update previous window tracking
            self._prev_window = processed_audio.copy()
            self._prev_window_hash = window_hash
            self._prev_rms = rms
            self.inference_count += 1

            # Save exact physical microphone speech window to tmp_live_window.npy for offline verification
            try:
                np.save("/home/jyno/Projects/Garaj/backend/tmp_live_window.npy", processed_audio)
            except Exception as e:
                logger.warning(f"Could not save tmp_live_window.npy: {e}")

            # 5. PyTorch torch.inference_mode() Forward Pass
            audio_tensor = torch.from_numpy(processed_audio).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(audio_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                # Verified W2V2-AASIST Logit Index Mapping:
                # Index 0 = Spoof / Synthetic
                # Index 1 = Bona-Fide / Real
                synthetic_prob = float(probs[0])
                real_prob = float(probs[1]) if len(probs) > 1 else float(1.0 - synthetic_prob)

            confidence = float(max(real_prob, synthetic_prob))
            predicted_class = "REAL" if real_prob >= synthetic_prob else "SYNTHETIC"
            compute_latency_ms = (time.perf_counter() - start_time) * 1000.0

            res = {
                "engine": self.engine_name,
                "model": "W2V2-AASIST",
                "model_checkpoint": self.model_meta.get("checkpoint_path"),
                "is_stub": False,
                "model_loaded": True,
                "status": "MODEL_READY",
                "predicted_class": predicted_class,
                "real_probability": round(real_prob, 4),
                "synthetic_probability": round(synthetic_prob, 4),
                "risk_score": round(synthetic_prob * 100.0, 2),
                "raw_logits": [round(float(logits[0][0]), 4), round(float(logits[0][1]), 4)],
                "confidence": round(confidence, 4),
                "audio_metrics": audio_metrics,
                "inference_count": self.inference_count,
                "window_hash": window_hash,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "consecutive_diff": consecutive_diff,
                "window_num_samples": len(processed_audio),
                "required_samples": required_samples,
                "sample_rate": sample_rate,
                "device": str(self.device),
                "compute_latency_ms": round(compute_latency_ms, 3),
                "timestamp": time.time(),
            }
            with self._lock:
                self._last_result = res
            return res

        except Exception as exc:
            compute_latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Error during W2V2-AASIST model inference: {str(exc)}")
            return {
                "engine": self.engine_name,
                "model": "W2V2-AASIST",
                "is_stub": False,
                "status": "MODEL_LOAD_ERROR",
                "error": str(exc),
                "predicted_class": "ERROR",
                "real_probability": None,
                "synthetic_probability": None,
                "confidence": 0.0,
                "required_samples": required_samples,
                "sample_rate": sample_rate,
                "compute_latency_ms": round(compute_latency_ms, 3),
                "timestamp": time.time(),
            }
        finally:
            with self._lock:
                self._is_inferring = False

    def reset_session(self, session_id: str = "") -> None:
        """Reset session state when connection resets or stops."""
        with self._lock:
            self._last_result = None
            self._is_inferring = False
            self._prev_window = None
            self._prev_window_hash = None
            self._prev_rms = None
