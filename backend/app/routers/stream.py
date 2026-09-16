import os
import time
import struct
import psutil
import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.audio.buffer import RollingAudioBuffer
from app.detection.stub_engine import IntegrationStubEngine

router = APIRouter()
logger = logging.getLogger("stream_router")

# Engine selection based on DETECTION_ENGINE env var (default: 'model')
engine_mode = os.getenv("DETECTION_ENGINE", "model").lower()

if engine_mode == "stub":
    active_engine = IntegrationStubEngine()
    print("=" * 55)
    print("Detection engine: STUB")
    print("Model status: STUB_ACTIVE")
    print("Sample rate: 16000 Hz")
    print("Window duration: 3.0 sec")
    print("=" * 55)
    logger.info("Initialized IntegrationStubEngine (Explicit Stub Mode).")
else:
    from app.detection.model_engine import ModelDetectionEngine
    active_engine = ModelDetectionEngine()
    dev_name = active_engine.model_meta.get("device_name", "CPU")
    arch_name = active_engine.model_meta.get("model_name", "W2V2-AASIST")
    backbone_name = active_engine.model_meta.get("backbone", "XLS-R 300M")
    status_str = active_engine.model_meta.get("status", active_engine.status)

    print("=" * 55)
    print("Detection engine: MODEL")
    print(f"Model architecture: {arch_name}")
    print(f"Backbone: {backbone_name}")
    print(f"Model status: {status_str}")
    print("Required samples: 64600 (4.04s @ 16kHz)")
    print(f"Device: {dev_name}")
    print("=" * 55)
    logger.info(f"Initialized ModelDetectionEngine ({arch_name}) on {dev_name}. Status: {status_str}.")


class StreamSessionTracker:
    """Tracks streaming telemetry, chunk counters, timing, and latency."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.start_time = time.time()
        self.last_chunk_time = time.time()
        self.chunks_received = 0
        self.total_pcm_bytes = 0
        self.expected_seq = 0
        self.dropped_chunks = 0
        self._chunk_timestamps = []

    def register_chunk(self, chunk_bytes_count: int, seq_id: int = None) -> Dict[str, Any]:
        now = time.time()
        self.chunks_received += 1
        self.total_pcm_bytes += chunk_bytes_count

        if seq_id is not None:
            if self.chunks_received > 1 and seq_id > self.expected_seq:
                self.dropped_chunks += (seq_id - self.expected_seq)
            self.expected_seq = seq_id + 1

        self._chunk_timestamps.append(now)
        self._chunk_timestamps = [t for t in self._chunk_timestamps if now - t <= 2.0]
        chunks_per_sec = len(self._chunk_timestamps) / 2.0 if len(self._chunk_timestamps) > 0 else 0.0

        samples_in_chunk = chunk_bytes_count // 2
        chunk_duration_ms = (samples_in_chunk / float(self.sample_rate)) * 1000.0

        self.last_chunk_time = now
        elapsed_total_sec = now - self.start_time

        return {
            "chunks_received": self.chunks_received,
            "dropped_chunks": self.dropped_chunks,
            "chunk_duration_ms": round(chunk_duration_ms, 2),
            "chunks_per_sec": round(chunks_per_sec, 2),
            "total_audio_duration_sec": round(self.total_pcm_bytes / (self.sample_rate * 2), 3),
            "session_elapsed_sec": round(elapsed_total_sec, 2),
        }


@router.websocket("/ws/stream")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Real-Time Audio WebSocket Endpoint.
    
    Accepts continuous binary PCM audio frames (16kHz mono signed 16-bit PCM).
    Binary payload format:
      - 12-byte header: uint32 sequence_id (4B), float64 client_timestamp_ms (8B)
      - Payload: signed 16-bit PCM audio bytes (`pcm_s16le`)
    """
    await websocket.accept()

    # Reset detection engine session cache for fresh connection
    reset_fn = getattr(active_engine, "reset_session", None)
    if callable(reset_fn):
        reset_fn()

    audio_buffer = RollingAudioBuffer(sample_rate=16000, window_duration_sec=4.1)
    tracker = StreamSessionTracker(sample_rate=16000)

    logger.info("WebSocket client connected to audio stream endpoint.")

    inference_task = None

    async def _async_inference_runner(current_seq_id: int, current_client_ts: float, rcv_ts: float):
        nonlocal inference_task
        try:
            window = audio_buffer.get_window()
            meta = {
                "seq_id": current_seq_id,
                "client_ts_ms": current_client_ts,
                "server_rcv_ts_ms": rcv_ts,
                "total_pcm_bytes": tracker.total_pcm_bytes,
            }
            await asyncio.to_thread(
                active_engine.process_window,
                audio_window=window,
                sample_rate=16000,
                metadata=meta
            )
        except Exception as exc:
            logger.error(f"[inference-error] Non-blocking inference pass failed: {exc}")
        finally:
            inference_task = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnected from audio stream endpoint.")
                break

            if "bytes" in message and message["bytes"]:
                receive_timestamp_ms = time.time() * 1000.0
                process_start_time = time.perf_counter()

                raw_bytes = message["bytes"]
                
                if len(raw_bytes) >= 12:
                    header = raw_bytes[:12]
                    pcm_payload = raw_bytes[12:]
                    seq_id, client_ts_ms = struct.unpack("<Id", header)
                else:
                    pcm_payload = raw_bytes
                    seq_id = tracker.chunks_received
                    client_ts_ms = receive_timestamp_ms

                chunk_array = audio_buffer.add_chunk(pcm_payload)
                session_stats = tracker.register_chunk(len(pcm_payload), seq_id)
                buffer_stats = audio_buffer.get_stats()

                # Trigger non-blocking background inference task if no inference pass is active
                if inference_task is None or inference_task.done():
                    inference_task = asyncio.create_task(
                        _async_inference_runner(seq_id, client_ts_ms, receive_timestamp_ms)
                    )

                # Fetch latest computed detection result from engine
                get_latest_fn = getattr(active_engine, "get_latest_result", None)
                engine_result = get_latest_fn() if callable(get_latest_fn) else None

                if engine_result is None:
                    engine_result = {
                        "engine": getattr(active_engine, "engine_name", "W2V2_AASIST_Engine"),
                        "model": "W2V2-AASIST",
                        "is_stub": False,
                        "status": "ANALYZING",
                        "predicted_class": "ANALYZING",
                        "real_probability": None,
                        "synthetic_probability": None,
                        "confidence": 0.0,
                        "message": "Accumulating continuous live audio samples...",
                        "window_num_samples": buffer_stats.get("active_samples", 0),
                        "required_samples": 64600,
                        "sample_rate": 16000,
                        "device": str(getattr(active_engine, "device", "cpu")),
                        "compute_latency_ms": 0.0,
                        "timestamp": time.time(),
                    }

                backend_compute_ms = (time.perf_counter() - process_start_time) * 1000.0
                network_latency_ms = max(0.0, receive_timestamp_ms - client_ts_ms) if client_ts_ms > 0 else 0.0

                cpu_usage_pct = psutil.cpu_percent(interval=None)
                ram_usage_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

                response = {
                    "status": "STREAMING_ACTIVE",
                    "seq_id": seq_id,
                    "timestamps": {
                        "client_ts_ms": round(client_ts_ms, 2),
                        "server_rcv_ts_ms": round(receive_timestamp_ms, 2),
                        "server_reply_ts_ms": round(time.time() * 1000.0, 2),
                    },
                    "latency": {
                        "network_one_way_ms": round(network_latency_ms, 2),
                        "backend_compute_ms": round(backend_compute_ms, 3),
                        "total_server_time_ms": round((time.perf_counter() - process_start_time) * 1000.0, 3),
                    },
                    "audio_format": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "format": "pcm_s16le",
                    },
                    "session_metrics": session_stats,
                    "rolling_buffer": buffer_stats,
                    "detection": engine_result,
                    "detection_stub": engine_result,
                    "system_resources": {
                        "cpu_percent": cpu_usage_pct,
                        "ram_mb": ram_usage_mb,
                    }
                }

                await websocket.send_json(response)

            elif "text" in message:
                pass

    except WebSocketDisconnect:
        logger.info("Client disconnected from audio stream endpoint.")
    except Exception as e:
        logger.error(f"Error during audio stream WebSocket connection: {str(e)}")
        try:
            await websocket.close()
        except Exception:
            pass
