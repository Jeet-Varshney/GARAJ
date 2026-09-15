import time
import numpy as np
from typing import Dict, Any, Optional
from app.detection.base import BaseDetectionEngine


class IntegrationStubEngine(BaseDetectionEngine):
    """
    Streaming Integration Stub Engine.
    
    Serves as Phase 0 pipeline placeholder. Explicitly performs real-time audio feature
    extraction (RMS energy, peak amplitude, zero-crossing rate) without claiming false ML deepfake detection.
    Will be replaced in Phase 1 by PyTorch inference model (e.g. Wav2Vec2 / AASIST).
    """

    def __init__(self):
        self.engine_name = "Phase0_Integration_Stub_v1"
        self.status = "STUB_ACTIVE"

    def process_window(
        self,
        audio_window: np.ndarray,
        sample_rate: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        num_samples = len(audio_window)
        if num_samples == 0:
            return {
                "engine": self.engine_name,
                "is_stub": True,
                "status": "NO_AUDIO",
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "zero_crossing_rate": 0.0,
                "processing_time_ms": 0.0,
                "note": "Empty audio window received."
            }

        # Calculate acoustic window statistics (real mathematical metrics)
        rms = float(np.sqrt(np.mean(np.square(audio_window))))
        peak = float(np.max(np.abs(audio_window)))
        
        # Zero crossing rate (ZCR)
        zero_crossings = np.nonzero(np.diff(audio_window >= 0))[0]
        zcr = float(len(zero_crossings) / max(1, num_samples))

        compute_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "engine": self.engine_name,
            "is_stub": True,  # Explicitly indicating integration stub mode
            "model_loaded": False,
            "classification_status": "PIPELINE_STUB_ONLY",
            "window_num_samples": num_samples,
            "window_duration_sec": round(num_samples / float(sample_rate), 3),
            "sample_rate": sample_rate,
            "metrics": {
                "rms_energy": round(rms, 5),
                "peak_amplitude": round(peak, 5),
                "zero_crossing_rate": round(zcr, 5),
                "is_speech_active": rms > 0.015,
            },
            "stub_compute_latency_ms": round(compute_latency_ms, 3),
            "timestamp": time.time(),
        }

    def reset_session(self, session_id: str) -> None:
        """Reset session state if required."""
        pass
