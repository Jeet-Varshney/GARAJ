import threading
import numpy as np
from typing import Tuple, Dict, Any


class RollingAudioBuffer:
    """
    Thread-safe Rolling Audio Ring Buffer for Real-Time Streaming Audio.
    
    Accepts incoming signed 16-bit PCM binary audio bytes (16 kHz, mono),
    converts them into normalized Float32 numpy arrays (-1.0 to 1.0),
    and maintains a sliding temporal window (e.g., 3.0 seconds) for ML inference.
    """

    def __init__(self, sample_rate: int = 16000, window_duration_sec: float = 4.1):
        self.sample_rate = sample_rate
        self.window_duration_sec = window_duration_sec
        self.capacity_samples = int(sample_rate * window_duration_sec)
        
        # Ring buffer storage (Float32 normalized samples)
        self._buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_samples_written = 0
        self._lock = threading.Lock()

    def reset(self):
        """Resets the buffer state."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_samples_written = 0

    def add_chunk(self, pcm_bytes: bytes) -> np.ndarray:
        """
        Ingests 16-bit signed PCM little-endian binary bytes (`pcm_s16le`).
        Returns the converted numpy array for the newly added chunk.
        """
        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        # Parse Int16 Little Endian (2 bytes per sample)
        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        # Normalize Int16 [-32768, 32767] to Float32 [-1.0, 1.0]
        float32_samples = int16_samples.astype(np.float32) / 32768.0

        num_samples = len(float32_samples)

        with self._lock:
            if num_samples >= self.capacity_samples:
                # If incoming chunk is larger than total buffer, take tail end
                self._buffer[:] = float32_samples[-self.capacity_samples:]
                self._write_pos = 0
            else:
                space_end = self.capacity_samples - self._write_pos
                if num_samples <= space_end:
                    self._buffer[self._write_pos:self._write_pos + num_samples] = float32_samples
                    self._write_pos = (self._write_pos + num_samples) % self.capacity_samples
                else:
                    # Wraparound write
                    self._buffer[self._write_pos:self.capacity_samples] = float32_samples[:space_end]
                    overflow = num_samples - space_end
                    self._buffer[:overflow] = float32_samples[space_end:]
                    self._write_pos = overflow

            self._total_samples_written += num_samples

        return float32_samples

    def get_window(self) -> np.ndarray:
        """
        Retrieves the latest sequential window of audio samples of length up to `capacity_samples`.
        Returns Float32 numpy array of shape (N,) ordered chronologically.
        """
        with self._lock:
            if self._total_samples_written == 0:
                return np.zeros(0, dtype=np.float32)

            filled_samples = min(self._total_samples_written, self.capacity_samples)

            if self._total_samples_written < self.capacity_samples:
                # Buffer has not filled yet: return active written range
                return self._buffer[:self._write_pos].copy()
            else:
                # Buffer is full: unroll circular buffer from write_pos to end + 0 to write_pos
                return np.concatenate((self._buffer[self._write_pos:], self._buffer[:self._write_pos]))

    def get_stats(self) -> Dict[str, Any]:
        """Returns buffer state metadata."""
        with self._lock:
            active_samples = min(self._total_samples_written, self.capacity_samples)
            active_duration = active_samples / float(self.sample_rate)
            is_full = self._total_samples_written >= self.capacity_samples
            
            return {
                "sample_rate": self.sample_rate,
                "capacity_samples": self.capacity_samples,
                "window_duration_sec": self.window_duration_sec,
                "active_samples": active_samples,
                "active_duration_sec": round(active_duration, 3),
                "is_full": is_full,
                "total_samples_received": self._total_samples_written,
                "total_duration_received_sec": round(self._total_samples_written / float(self.sample_rate), 3),
            }
