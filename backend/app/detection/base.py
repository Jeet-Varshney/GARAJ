from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseDetectionEngine(ABC):
    """
    Abstract Base Class for Voice Deepfake / Cloning Detection Engines.
    
    This interface specifies the contract for audio inference engines.
    Future PyTorch/ONNX models (e.g. Wav2Vec2, AASIST, RawNet2) will implement
    this class directly, accepting numpy audio windows `(num_samples,)` Float32/Int16
    and returning structured inference metadata.
    """

    @abstractmethod
    def process_window(
        self,
        audio_window: np.ndarray,
        sample_rate: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes a rolling audio window.

        Args:
            audio_window: Float32 audio samples normalized between -1.0 and 1.0 of shape (samples,).
            sample_rate: Audio sampling frequency in Hz (e.g. 16000).
            metadata: Context dictionary (e.g., sequence ID, timestamps, client ID).

        Returns:
            Dict containing telemetry metrics, window statistics, and detection metrics.
        """
        pass

    @abstractmethod
    def reset_session(self, session_id: str) -> None:
        """Resets engine state for a new audio streaming session."""
        pass
