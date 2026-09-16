import { useState, useEffect, useRef, useCallback } from 'react';
// @ts-ignore
import { useWebSocket } from './useWebSocket';
// @ts-ignore
import { useAudioStreamer } from './useAudioStreamer';
import type {
  RiskLevel,
  VerdictStatus,
  SecurityCheckItem,
  PipelineSpec,
  ChunkTelemetry,
  AASISTModelSpec,
  LatencyMetric,
  TerminalLogEntry,
  ActivityItem,
  LayoutViewMode,
} from '../types/garaj';

export function useGarajSecurity() {
  const [viewMode, setViewMode] = useState<LayoutViewMode>('desktop-3col');
  const [showTerminal, setShowTerminal] = useState<boolean>(true);
  const [logs, setLogs] = useState<TerminalLogEntry[]>([]);
  const [recentActivities, setRecentActivities] = useState<ActivityItem[]>([]);

  // Existing Real GARAJ WebSocket Hook
  const {
    connectionStatus,
    latestTelemetry,
    roundTripLatency,
    sentChunksCount,
    connect,
    sendAudioChunk,
  } = useWebSocket();

  // Existing Real GARAJ Audio Streamer Hook
  const {
    isStreaming,
    isSimulated,
    audioSourceType,
    clientAudioStats,
    startStreaming,
    startSimulatedStreaming,
    stopStreaming,
    analyserNode,
  } = useAudioStreamer({
    onAudioChunkReceived: (pcmArrayBuffer: ArrayBuffer) => {
      sendAudioChunk(pcmArrayBuffer);
    },
  });

  // Auto connect WebSocket on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // Handle monitoring toggle (Start/Stop Live Mic)
  const toggleMonitoring = useCallback(async (_val?: boolean) => {
    if (connectionStatus !== 'CONNECTED') {
      connect();
    }
    if (isStreaming) {
      stopStreaming();
    } else {
      await startStreaming();
    }
  }, [connectionStatus, connect, isStreaming, startStreaming, stopStreaming]);

  // Handle Test Signal toggle (Simulation mode)
  const toggleAttackSimulation = useCallback((_val?: boolean) => {
    if (connectionStatus !== 'CONNECTED') {
      connect();
    }
    if (isSimulated) {
      stopStreaming();
    } else {
      startSimulatedStreaming();
    }
  }, [connectionStatus, connect, isSimulated, startSimulatedStreaming, stopStreaming]);

  // Live Spectrum & RMS computation from analyserNode
  const [spectrumData, setSpectrumData] = useState<Uint8Array>(new Uint8Array(32).fill(0));
  const [rmsEnergy, setRmsEnergy] = useState<number>(0);
  const [peakAmplitude, setPeakAmplitude] = useState<number>(0);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isStreaming) {
      setSpectrumData(new Uint8Array(32).fill(0));
      setRmsEnergy(0);
      setPeakAmplitude(0);
      return;
    }

    const updateAudioData = () => {
      if (analyserNode) {
        const binCount = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(binCount);
        analyserNode.getByteFrequencyData(dataArray);

        const mockSpectrum = new Uint8Array(32);
        for (let i = 0; i < 32; i++) {
          const idx = Math.floor((i / 32) * binCount);
          mockSpectrum[i] = dataArray[idx] || 0;
        }
        setSpectrumData(mockSpectrum);
      } else {
        const mockSpectrum = new Uint8Array(32);
        for (let i = 0; i < 32; i++) {
          mockSpectrum[i] = Math.floor(Math.sin(Date.now() * 0.005 + i * 0.3) * 35 + 45);
        }
        setSpectrumData(mockSpectrum);
      }

      const rmsVal = parseFloat(clientAudioStats.lastPcmRms) || 0;
      const peakVal = parseFloat(clientAudioStats.lastPcmPeak) || 0;
      setRmsEnergy(rmsVal);
      setPeakAmplitude(peakVal);

      animFrameRef.current = requestAnimationFrame(updateAudioData);
    };

    updateAudioData();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isStreaming, analyserNode, clientAudioStats]);

  // TASK 5: Log WebSocket message telemetry verification fields
  useEffect(() => {
    if (!latestTelemetry) return;
    const det = latestTelemetry.detection || {};
    const metrics = det.audio_metrics || {};
    console.log(
      '[WS DETECT VERIFY]\n' +
      `status: ${det.status ?? 'N/A'}\n` +
      `predicted_class: ${det.predicted_class ?? 'N/A'}\n` +
      `RMS: ${metrics.rms ?? 'N/A'}\n` +
      `peak: ${metrics.peak_amplitude ?? 'N/A'}\n` +
      `risk_score: ${det.risk_score ?? 'N/A'}\n` +
      `real_probability: ${det.real_probability ?? 'N/A'}\n` +
      `synthetic_probability: ${det.synthetic_probability ?? 'N/A'}\n` +
      `timestamp: ${det.timestamp ?? 'N/A'}`
    );
  }, [latestTelemetry]);

  // TASK 8: Implement strict 4-level State Hierarchy:
  // 1. DISCONNECTED -> Backend Disconnected
  // 2. CONNECTED + NO_AUDIO (mic off OR status=="NO_AUDIO" OR predicted_class=="NO_AUDIO") -> NO_AUDIO
  // 3. CONNECTED + AUDIO BUT MODEL NOT READY (status=="ANALYZING") -> Waiting for detection
  // 4. CONNECTED + VALID MODEL RESULT (status=="MODEL_READY" & real_probability != null) -> Show actual REAL/SYNTHETIC result
  const detection = latestTelemetry?.detection || {};
  const isWsConnected = connectionStatus === 'CONNECTED';
  const detStatus = detection.status || 'NO_AUDIO';
  const detClass = detection.predicted_class || 'NO_AUDIO';
  const hasValidProb = detection.real_probability !== undefined && detection.real_probability !== null;

  let stateMode: 'DISCONNECTED' | 'NO_AUDIO' | 'ANALYZING' | 'MODEL_READY' = 'DISCONNECTED';
  let verdict: VerdictStatus | string = 'Backend Disconnected';
  let riskLevel: RiskLevel | string = 'NO_AUDIO';
  let realProb: number | null = null;
  let synthProb: number | null = null;
  let riskScore: number | null = null;
  let authenticityScore: number | null = null;

  if (!isWsConnected) {
    stateMode = 'DISCONNECTED';
    verdict = 'Backend Disconnected';
    riskLevel = 'NO_AUDIO';
  } else if (!isStreaming || detStatus === 'NO_AUDIO' || detClass === 'NO_AUDIO') {
    stateMode = 'NO_AUDIO';
    verdict = 'NO_AUDIO';
    riskLevel = 'NO_AUDIO';
  } else if (detStatus === 'ANALYZING' || detClass === 'ANALYZING' || !hasValidProb) {
    stateMode = 'ANALYZING';
    verdict = 'Waiting for detection';
    riskLevel = 'ACCUMULATING BUFFER';
  } else {
    stateMode = 'MODEL_READY';
    verdict = detClass as VerdictStatus;
    realProb = detection.real_probability;
    synthProb = detection.synthetic_probability;
    riskScore = detection.risk_score !== undefined && detection.risk_score !== null
      ? detection.risk_score
      : (synthProb !== null ? Math.round(synthProb * 10000) / 100 : null);
    authenticityScore = realProb !== null ? Math.round(realProb * 100) : null;
    riskLevel = verdict === 'SYNTHETIC' ? 'HIGH RISK' : 'LOW RISK';
  }

  const isSynthetic = stateMode === 'MODEL_READY' && verdict === 'SYNTHETIC';
  const isReal = stateMode === 'MODEL_READY' && verdict === 'REAL';

  const audioMetrics = detection.audio_metrics || {};
  const consecutiveDiff = detection.consecutive_diff || null;

  // Real Verified Checks
  const checks: SecurityCheckItem[] = [
    {
      id: 'audio-energy-gate',
      label: 'Input Audio Energy Gate',
      status: audioMetrics.rms !== undefined ? (audioMetrics.rms >= 0.003 ? 'Normal' : 'Caution') : 'Normal',
      description: 'Minimum RMS threshold check (MIN_RMS: 0.0030)',
      value: audioMetrics.rms !== undefined ? `RMS: ${audioMetrics.rms}` : `RMS: ${clientAudioStats.lastPcmRms}`,
    },
    {
      id: 'window-continuity',
      label: 'Consecutive Window Continuity',
      status: consecutiveDiff ? (consecutiveDiff.is_identical ? 'Caution' : 'Normal') : 'Normal',
      description: 'Delta difference between rolling 64,600 audio sample windows',
      value: consecutiveDiff ? `Delta: ${consecutiveDiff.mean_abs_diff}` : 'Unavailable',
    },
    {
      id: 'aasist-neural-pass',
      label: 'W2V2-AASIST Neural Model Pass',
      status: isReal ? 'Normal' : isSynthetic ? 'Failed' : 'Normal',
      description: 'XLS-R 300M + Graph Attention spectro-temporal evaluation',
      value: realProb !== null && synthProb !== null
        ? `Real: ${(realProb * 100).toFixed(1)}% | Spoof: ${(synthProb * 100).toFixed(1)}%`
        : !isWsConnected ? 'Backend Disconnected' : 'Waiting for 64,600 samples...',
    },
  ];

  // Pipeline spec
  const pipelineSpec: PipelineSpec = {
    wsConnection: connectionStatus === 'CONNECTED' ? 'CONNECTED' : connectionStatus === 'CONNECTING' ? 'CONNECTING' : 'DISCONNECTED',
    streamingStatus: isStreaming ? (isSimulated ? 'TEST_SIGNAL' : 'LIVE_STREAMING') : 'IDLE',
    audioEngine: isSimulated
      ? 'Simulation'
      : (audioSourceType === 'caller_audio' ? 'Caller/System Capture (AudioWorklet)' : 'Local Mic (AudioWorklet)'),
    audioFormat: '16 kHz Mono PCM_S16LE',
    sampleRate: 16000,
    channels: 1,
  };

  // Chunk telemetry
  const chunkTelemetry: ChunkTelemetry = {
    totalChunks: sentChunksCount,
    chunksPerSec: isStreaming ? 10.0 : 0,
    chunkDurationMs: 100,
    clientSent: sentChunksCount,
    dropped: 0,
    audioDurationSec: Math.round(sentChunksCount * 0.1 * 10) / 10,
  };

  // Latency metrics
  const computeLatency = detection.compute_latency_ms !== undefined && detection.compute_latency_ms !== null ? detection.compute_latency_ms : 0;
  const latencies: LatencyMetric[] = [
    { name: 'Backend Compute', valueMs: computeLatency, color: '#10B981', percentage: computeLatency > 0 ? 25 : 0 },
    { name: 'Round Trip (RTT)', valueMs: roundTripLatency, color: '#09090B', percentage: roundTripLatency > 0 ? 75 : 0 },
  ];

  // Model specification diagnostics
  const modelSpec: AASISTModelSpec = {
    architecture: detection.model || 'W2V2-AASIST (XLS-R 300M + AASIST Graph Attention)',
    requiredInputSamples: detection.required_samples || 64600,
    inputDurationSec: 4.04,
    engineStatus: detection.status || (isWsConnected ? (isStreaming ? 'ANALYZING' : 'IDLE') : 'DISCONNECTED'),
    modelPrediction: verdict,
    inferenceLatencyMs: computeLatency,
    spoofProb: synthProb !== null ? synthProb : 0,
    realProb: realProb !== null ? realProb : 0,
  };

  // Populate activities & logs from real WebSocket messages
  const prevSeqRef = useRef<number | null>(null);
  useEffect(() => {
    if (!latestTelemetry || latestTelemetry.seq_id === undefined) return;
    if (latestTelemetry.seq_id === prevSeqRef.current) return;
    prevSeqRef.current = latestTelemetry.seq_id;

    const timeStr = new Date().toLocaleTimeString();
    const seqId = latestTelemetry.seq_id;
    const detClass = latestTelemetry.detection?.predicted_class || 'UNKNOWN';

    let eventMsg = `Chunk #${seqId} processed`;
    let actStatus: 'success' | 'error' | 'info' = 'info';

    if (detClass === 'SYNTHETIC') {
      eventMsg = `ALERT: Spoof voice detected in chunk #${seqId}`;
      actStatus = 'error';
    } else if (detClass === 'REAL') {
      eventMsg = `Real voice verified in chunk #${seqId} (${(realProb! * 100).toFixed(1)}%)`;
      actStatus = 'success';
    }

    setRecentActivities((prev) => [
      ...prev.slice(-10),
      {
        id: `${Date.now()}-${seqId}`,
        time: timeStr,
        event: eventMsg,
        status: actStatus,
      },
    ]);

    setLogs((prev) => [
      ...prev.slice(-100),
      {
        id: `${Date.now()}-${seqId}-log`,
        timestamp: timeStr,
        level: detClass === 'SYNTHETIC' ? 'error' : detClass === 'REAL' ? 'success' : 'info',
        text: `[Chunk #${seqId}] Verdict: ${detClass} | Real Prob: ${realProb !== null ? (realProb * 100).toFixed(1) + '%' : 'N/A'} | Compute: ${computeLatency}ms`,
      },
    ]);
  }, [latestTelemetry, realProb, computeLatency]);

  return {
    isMonitoring: isStreaming,
    setIsMonitoring: toggleMonitoring,
    isMicActive: isStreaming && !isSimulated,
    toggleMicrophone: toggleMonitoring,
    isAttackSimulated: isSimulated,
    setIsAttackSimulated: toggleAttackSimulation,
    viewMode,
    setViewMode,
    showTerminal,
    setShowTerminal,
    authenticityScore,
    riskScore,
    realProb,
    synthProb,
    verdict,
    riskLevel,
    checks,
    pipelineSpec,
    chunkTelemetry,
    chunkFlash: false,
    latencies,
    modelSpec,
    recentActivities,
    rmsEnergy,
    peakAmplitude,
    spectrumData,
    logs,
  };
}
