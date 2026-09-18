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
  const [isPaused, setIsPaused] = useState<boolean>(false);

  // Real GARAJ WebSocket Hook
  const {
    connectionStatus,
    latestTelemetry,
    roundTripLatency,
    sentChunksCount,
    connect,
    sendAudioChunk,
    clearTelemetry,
  } = useWebSocket();

  // Real GARAJ Audio Streamer Hook
  const {
    isStreaming,
    isSimulated,
    hasMicPermission,
    permissionError,
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

  // Operational State Control Functions:
  // PAUSED: stop streaming audio, preserve latestTelemetry intact
  const pauseDetection = useCallback(() => {
    stopStreaming();
    setIsPaused(true);
  }, [stopStreaming]);

  // RESUMED / RUNNING: restart streaming audio, resume backend inference updates
  const resumeDetection = useCallback(async () => {
    setIsPaused(false);
    if (connectionStatus !== 'CONNECTED') {
      connect();
    }
    await startStreaming();
  }, [connectionStatus, connect, startStreaming]);

  // STOPPED: stop streaming, clear session & telemetry
  const stopDetection = useCallback(() => {
    stopStreaming();
    setIsPaused(false);
    clearTelemetry();
  }, [stopStreaming, clearTelemetry]);

  // Toggle function for UI Start/Pause/Resume button
  const toggleMonitoring = useCallback(async (_val?: boolean) => {
    if (isStreaming) {
      pauseDetection();
    } else {
      await resumeDetection();
    }
  }, [isStreaming, pauseDetection, resumeDetection]);

  // Handle Test Signal toggle (Simulation mode)
  const toggleAttackSimulation = useCallback((_val?: boolean) => {
    if (isSimulated) {
      stopDetection();
    } else {
      if (connectionStatus !== 'CONNECTED') {
        connect();
      }
      setIsPaused(false);
      startSimulatedStreaming();
    }
  }, [connectionStatus, connect, isSimulated, startSimulatedStreaming, stopDetection]);

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
        setSpectrumData(new Uint8Array(32).fill(0));
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

  // Log WebSocket message telemetry verification fields
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

  // Operational State Machine:
  // - RUNNING: isStreaming = true, isPaused = false
  // - PAUSED: isStreaming = false, isPaused = true (preserves latestTelemetry snapshot)
  // - STOPPED: isStreaming = false, isPaused = false (telemetry cleared)
  const isWsConnected = connectionStatus === 'CONNECTED';
  const detection = latestTelemetry?.detection || {};
  const detStatus = detection.status || 'NO_AUDIO';
  const detClass = detection.predicted_class || 'NO_AUDIO';

  const hasFiniteProb = typeof detection.real_probability === 'number' && Number.isFinite(detection.real_probability) &&
                        typeof detection.synthetic_probability === 'number' && Number.isFinite(detection.synthetic_probability);

  let stateMode: 'MIC_PERMISSION_DENIED' | 'MIC_PERMISSION_GRANTED' | 'AUDIO_STREAMING' | 'MODEL_READY' | 'PAUSED' | 'DISCONNECTED' = 'DISCONNECTED';
  let verdict: VerdictStatus | string = 'BACKEND DISCONNECTED';
  let riskLevel: RiskLevel | string = 'NO_AUDIO';
  let realProb: number | null = null;
  let synthProb: number | null = null;
  let riskScore: number | null = null;
  let authenticityScore: number | null = null;

  if (!isWsConnected && !isPaused) {
    stateMode = 'DISCONNECTED';
    verdict = 'BACKEND DISCONNECTED';
    riskLevel = 'NO_AUDIO';
  } else if (permissionError || (!hasMicPermission && !isSimulated && isStreaming)) {
    stateMode = 'MIC_PERMISSION_DENIED';
    verdict = 'MICROPHONE ACCESS REQUIRED';
    riskLevel = 'NO_AUDIO';
  } else if (isPaused) {
    // PAUSED STATE: If we have a valid previous model result from backend, PRESERVE IT AS FROZEN SNAPSHOT!
    stateMode = 'PAUSED';
    if (hasFiniteProb) {
      verdict = detClass as VerdictStatus;
      const rProb = detection.real_probability as number;
      const sProb = detection.synthetic_probability as number;
      realProb = rProb;
      synthProb = sProb;
      riskScore = detection.risk_score !== undefined && detection.risk_score !== null
        ? detection.risk_score
        : Math.round(sProb * 10000) / 100;
      authenticityScore = Math.round(rProb * 100);
      riskLevel = 'DETECTION PAUSED';
    } else {
      verdict = 'DETECTION PAUSED';
      riskLevel = 'DETECTION PAUSED';
    }
  } else if (!isStreaming) {
    stateMode = 'MIC_PERMISSION_GRANTED';
    verdict = 'NO AUDIO DETECTED';
    riskLevel = 'NO_AUDIO';
  } else if (clientAudioStats.pcmChunkCount === 0 || sentChunksCount === 0) {
    stateMode = 'MIC_PERMISSION_GRANTED';
    verdict = 'NO AUDIO DETECTED';
    riskLevel = 'NO_AUDIO';
  } else if (detStatus === 'NO_AUDIO' || detClass === 'NO_AUDIO' || !hasFiniteProb) {
    stateMode = 'AUDIO_STREAMING';
    verdict = 'WAITING FOR AUDIO';
    riskLevel = 'ACCUMULATING BUFFER';
  } else {
    // RUNNING + MODEL_READY
    stateMode = 'MODEL_READY';
    verdict = detClass as VerdictStatus;
    const rProb = detection.real_probability as number;
    const sProb = detection.synthetic_probability as number;
    realProb = rProb;
    synthProb = sProb;
    riskScore = detection.risk_score !== undefined && detection.risk_score !== null
      ? detection.risk_score
      : Math.round(sProb * 10000) / 100;
    authenticityScore = Math.round(rProb * 100);
    riskLevel = verdict === 'SYNTHETIC' ? 'HIGH RISK' : 'LOW RISK';
  }

  const isSynthetic = (stateMode === 'MODEL_READY' || stateMode === 'PAUSED') && verdict === 'SYNTHETIC';
  const isReal = (stateMode === 'MODEL_READY' || stateMode === 'PAUSED') && verdict === 'REAL';

  const audioMetrics = detection.audio_metrics || {};
  const consecutiveDiff = detection.consecutive_diff || null;

  // Real Diagnostic Checks
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
      value: (stateMode === 'MODEL_READY' || (stateMode === 'PAUSED' && hasFiniteProb)) && realProb !== null && synthProb !== null
        ? `Real: ${(realProb * 100).toFixed(1)}% | Spoof: ${(synthProb * 100).toFixed(1)}%`
        : !isWsConnected && !isPaused
        ? 'BACKEND DISCONNECTED'
        : stateMode === 'MIC_PERMISSION_DENIED'
        ? 'MICROPHONE ACCESS REQUIRED'
        : stateMode === 'PAUSED'
        ? 'DETECTION PAUSED'
        : stateMode === 'MIC_PERMISSION_GRANTED'
        ? 'NO AUDIO DETECTED'
        : 'WAITING FOR AUDIO',
    },
  ];

  // Pipeline spec
  const pipelineSpec: PipelineSpec = {
    wsConnection: connectionStatus === 'CONNECTED' ? 'CONNECTED' : connectionStatus === 'CONNECTING' ? 'CONNECTING' : 'DISCONNECTED',
    streamingStatus: isPaused ? 'PAUSED' : isStreaming ? (isSimulated ? 'TEST_SIGNAL' : 'LIVE_STREAMING') : 'IDLE',
    audioEngine: isSimulated
      ? 'Simulation'
      : 'Local Mic (AudioWorklet)',
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
    engineStatus: stateMode,
    modelPrediction: verdict,
    inferenceLatencyMs: computeLatency,
    spoofProb: synthProb !== null ? synthProb : 0,
    realProb: realProb !== null ? realProb : 0,
  };

  // Populate activities & logs from real WebSocket messages
  const prevSeqRef = useRef<number | null>(null);
  useEffect(() => {
    if (!latestTelemetry || latestTelemetry.seq_id === undefined || !isStreaming) return;
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
      eventMsg = `Real voice verified in chunk #${seqId} (${(realProb ? (realProb * 100).toFixed(1) : '0')}% )`;
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
  }, [latestTelemetry, realProb, computeLatency, isStreaming]);

  return {
    isMonitoring: isStreaming,
    isPaused,
    pauseDetection,
    resumeDetection,
    stopDetection,
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
