import { useState, useEffect, useRef, useCallback } from 'react';
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
  const [isMonitoring, setIsMonitoring] = useState<boolean>(true);
  const [isMicActive, setIsMicActive] = useState<boolean>(false);
  const [isAttackSimulated, setIsAttackSimulated] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<LayoutViewMode>('desktop-3col');
  const [showTerminal, setShowTerminal] = useState<boolean>(true);

  // Audio metrics
  const [rmsEnergy, setRmsEnergy] = useState<number>(0.0342);
  const [peakAmplitude, setPeakAmplitude] = useState<number>(0.1245);
  const [spectrumData, setSpectrumData] = useState<Uint8Array>(new Uint8Array(32).fill(40));

  // Chunk ticker
  const [chunkCount, setChunkCount] = useState<number>(517);
  const [chunkFlash, setChunkFlash] = useState<boolean>(false);

  // Terminal Logs
  const [logs, setLogs] = useState<TerminalLogEntry[]>([]);

  // Web Audio refs
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Computed Verdict & Risk based on simulation
  const authenticityScore = isAttackSimulated ? 14 : 82;
  const verdict: VerdictStatus = isMonitoring
    ? isAttackSimulated
      ? 'SYNTHETIC'
      : 'REAL'
    : 'NO_AUDIO';

  const riskLevel: RiskLevel = isAttackSimulated ? 'HIGH RISK' : 'LOW RISK';

  // Security Checks
  const checks: SecurityCheckItem[] = [
    {
      id: 'pattern-match',
      label: isAttackSimulated ? 'Voice profile mismatch detected' : 'Matches your known profile',
      status: isAttackSimulated ? 'Failed' : 'Normal',
      description: 'Spectral timbre matches user biometric baseline',
      value: isAttackSimulated ? '14.2% match' : '98.4% match',
    },
    {
      id: 'audio-flow',
      label: isAttackSimulated ? 'Artificial phase gaps & synthesis artifacts' : 'Audio flow is natural',
      status: isAttackSimulated ? 'Caution' : 'Normal',
      description: 'Continuous packet arrival without synthetic pitch jitter',
      value: isAttackSimulated ? '3.8ms jitter' : '0.4ms jitter',
    },
    {
      id: 'acoustic-energy',
      label: isAttackSimulated ? 'Neural vocoder phase anomaly' : 'Acoustic window energy clean',
      status: isAttackSimulated ? 'Failed' : 'Normal',
      description: 'Phase 1C.7 noise floor & harmonic distribution verified',
      value: isAttackSimulated ? 'Anomalous' : 'Clean',
    },
    {
      id: 'aasist-graph',
      label: isAttackSimulated ? 'W2V2-AASIST deepfake alert' : 'W2V2-AASIST deepfake score verified',
      status: isAttackSimulated ? 'Failed' : 'Normal',
      description: 'XLS-R 300M + Graph Attention embedding pass',
      value: isAttackSimulated ? 'Spoof (0.86)' : 'Real (0.82)',
    },
  ];

  // Pipeline spec
  const pipelineSpec: PipelineSpec = {
    wsConnection: 'CONNECTED',
    streamingStatus: isMonitoring ? 'LIVE_STREAMING' : 'IDLE',
    audioEngine: 'AudioWorklet',
    audioFormat: '16 kHz Mono PCM_S16LE',
    sampleRate: 16000,
    channels: 1,
  };

  // Chunk telemetry
  const chunkTelemetry: ChunkTelemetry = {
    totalChunks: chunkCount,
    chunksPerSec: isMonitoring ? 10.5 : 0,
    chunkDurationMs: 100,
    clientSent: chunkCount,
    dropped: 0,
    audioDurationSec: Math.round(chunkCount * 0.1 * 10) / 10,
  };

  // Latencies
  const latencies: LatencyMetric[] = [
    { name: 'Backend Compute', valueMs: 0.105, color: '#10B981', percentage: 15 },
    { name: 'Round Trip (RTT)', valueMs: 3.0, color: '#09090B', percentage: 55 },
    { name: 'Network One-Way', valueMs: 0.67, color: '#71717A', percentage: 20 },
    { name: 'Server Overhead', valueMs: 0.332, color: '#A1A1AA', percentage: 10 },
  ];

  // Model diagnostics
  const modelSpec: AASISTModelSpec = {
    architecture: 'W2V2-AASIST (XLS-R 300M + AASIST Graph Attention)',
    requiredInputSamples: 64600,
    inputDurationSec: 4.04,
    engineStatus: isMonitoring
      ? isAttackSimulated
        ? 'WARNING'
        : 'ACTIVE_INFERENCE'
      : 'NO_AUDIO',
    modelPrediction: isAttackSimulated
      ? 'SPOOF_DETECTED (Confidence: 86.4%)'
      : 'REAL_VOICE (Confidence: 82.1%)',
    inferenceLatencyMs: 0.222,
    spoofProb: isAttackSimulated ? 0.86 : 0.18,
    realProb: isAttackSimulated ? 0.14 : 0.82,
  };

  // Recent activity list
  const recentActivities: ActivityItem[] = [
    {
      id: 'act-1',
      time: '07:15:42 PM',
      event: isAttackSimulated
        ? 'ALERT: Deepfake neural synthesis detected in window (#517)'
        : 'Acoustic window verified (#517)',
      status: isAttackSimulated ? 'error' : 'success',
      hash: '65600_samples_ok',
    },
    {
      id: 'act-2',
      time: '07:15:32 PM',
      event: 'Biometric challenge passed (User #4092)',
      status: 'success',
    },
    {
      id: 'act-3',
      time: '07:15:22 PM',
      event: 'WebSocket streaming channel established (16kHz PCM)',
      status: 'success',
    },
    {
      id: 'act-4',
      time: '07:15:10 PM',
      event: 'W2V2-AASIST neural weights validated (LA_model.pth)',
      status: 'success',
    },
  ];

  // Live microphone capture setup
  const toggleMicrophone = useCallback(async () => {
    if (isMicActive) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
      setIsMicActive(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;

      setIsMicActive(true);

      const updateMicData = () => {
        if (!analyserRef.current) return;
        const data = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(data);
        setSpectrumData(data);

        // Compute RMS
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          sum += data[i] * data[i];
        }
        const rms = Math.sqrt(sum / data.length) / 255;
        setRmsEnergy(parseFloat(rms.toFixed(4)));
        setPeakAmplitude(parseFloat((rms * 1.8).toFixed(4)));

        animFrameRef.current = requestAnimationFrame(updateMicData);
      };

      updateMicData();
    } catch (err) {
      console.warn('Microphone access not granted or unavailable:', err);
      setIsMicActive(false);
    }
  }, [isMicActive]);

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  // Live chunk ticker & simulated spectrum
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      setChunkCount((prev) => prev + 1);
      setChunkFlash(true);
      setTimeout(() => setChunkFlash(false), 120);

      // If mic is not active, synthesize dynamic spectrum data
      if (!isMicActive) {
        const mockSpectrum = new Uint8Array(32);
        for (let i = 0; i < 32; i++) {
          const val = Math.floor(Math.sin(Date.now() * 0.005 + i * 0.3) * 35 + 45 + Math.random() * 20);
          mockSpectrum[i] = isAttackSimulated ? Math.min(255, val * 1.4) : val;
        }
        setSpectrumData(mockSpectrum);
        const energy = isAttackSimulated ? 0.1824 : parseFloat((0.025 + Math.random() * 0.02).toFixed(4));
        setRmsEnergy(energy);
        setPeakAmplitude(parseFloat((energy * 2.2).toFixed(4)));
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isMonitoring, isMicActive, isAttackSimulated]);

  // Generate continuous live diagnostic logs for the terminal
  useEffect(() => {
    const initialLogs: TerminalLogEntry[] = [
      {
        id: '1',
        timestamp: new Date().toLocaleTimeString(),
        level: 'info',
        text: 'Client Mic & AudioTx: running | Captured: 827,200 samples | Last AudioWorklet: Chunk RMS: 0.0342 | Peak: 0.1245',
      },
      {
        id: '2',
        timestamp: new Date().toLocaleTimeString(),
        level: 'info',
        text: `WebSocket & Chunk Seq ID: ${chunkCount} | Client Sent: ${chunkCount} chunks | Server Tracking: Received: ${chunkCount} chunks (${(
          chunkCount * 0.1
        ).toFixed(1)}s audio)`,
      },
      {
        id: '3',
        timestamp: new Date().toLocaleTimeString(),
        level: 'info',
        text: 'Inference Pass & Window [64600 samples] | Window Hash: 0x9f8b4a2e1c',
      },
      {
        id: '4',
        timestamp: new Date().toLocaleTimeString(),
        level: isAttackSimulated ? 'error' : 'success',
        text: isAttackSimulated
          ? 'Consecutive Window Difference: SPU SPLICING DETECTED (Phase Discontinuity: +48.2 deg)'
          : 'Consecutive Window Difference: Smooth Acoustic Continuity (Delta: 0.0014)',
      },
      {
        id: '5',
        timestamp: new Date().toLocaleTimeString(),
        level: 'info',
        text: `Window Audio Energy RMS: ${rmsEnergy} | Peak: ${peakAmplitude} | Min: 0.0001 | Max: 0.2810 (Metrics: CLEAN)`,
      },
      {
        id: '6',
        timestamp: new Date().toLocaleTimeString(),
        level: isAttackSimulated ? 'warn' : 'info',
        text: isAttackSimulated
          ? 'Raw Model Logits [Spoof(0), Real(1)]: [+3.842, -2.105]'
          : 'Raw Model Logits [Spoof(0), Real(1)]: [-1.982, +2.415]',
      },
      {
        id: '7',
        timestamp: new Date().toLocaleTimeString(),
        level: isAttackSimulated ? 'error' : 'success',
        text: isAttackSimulated
          ? 'Softmax Probabilities: Synthetic (Spoof): 86.4% | Real: 13.6%'
          : 'Softmax Probabilities: Synthetic (Spoof): 17.9% | Real: 82.1%',
      },
      {
        id: '8',
        timestamp: new Date().toLocaleTimeString(),
        level: isAttackSimulated ? 'error' : 'success',
        text: isAttackSimulated
          ? 'Predicted Class & Risk Score: SYNTHETIC (Risk Score: HIGH_RISK_ALERT)'
          : 'Predicted Class & Risk Score: REAL (Risk Score: LOW_RISK_AUTHENTICATED)',
      },
    ];
    setLogs(initialLogs);
  }, [isAttackSimulated, chunkCount]);

  return {
    isMonitoring,
    setIsMonitoring,
    isMicActive,
    toggleMicrophone,
    isAttackSimulated,
    setIsAttackSimulated,
    viewMode,
    setViewMode,
    showTerminal,
    setShowTerminal,
    authenticityScore,
    verdict,
    riskLevel,
    checks,
    pipelineSpec,
    chunkTelemetry,
    chunkFlash,
    latencies,
    modelSpec,
    recentActivities,
    rmsEnergy,
    peakAmplitude,
    spectrumData,
    logs,
  };
}
