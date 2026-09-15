export type RiskLevel = 'LOW RISK' | 'MEDIUM RISK' | 'HIGH RISK';
export type VerdictStatus = 'REAL' | 'SYNTHETIC' | 'ANALYZING' | 'NO_AUDIO';

export interface SecurityCheckItem {
  id: string;
  label: string;
  status: 'Normal' | 'Caution' | 'Failed';
  description: string;
  value?: string;
}

export interface LatencyMetric {
  name: string;
  valueMs: number;
  color: string;
  percentage: number;
}

export interface PipelineSpec {
  wsConnection: 'CONNECTED' | 'DISCONNECTED' | 'CONNECTING';
  streamingStatus: 'IDLE' | 'LIVE_STREAMING' | 'ANALYZING';
  audioEngine: string;
  audioFormat: string;
  sampleRate: number;
  channels: number;
}

export interface ChunkTelemetry {
  totalChunks: number;
  chunksPerSec: number;
  chunkDurationMs: number;
  clientSent: number;
  dropped: number;
  audioDurationSec: number;
}

export interface AASISTModelSpec {
  architecture: string;
  requiredInputSamples: number;
  inputDurationSec: number;
  engineStatus: 'ACTIVE_INFERENCE' | 'NO_AUDIO' | 'WARNING' | 'RECOMPUTING';
  modelPrediction: string;
  inferenceLatencyMs: number;
  spoofProb: number;
  realProb: number;
}

export interface TerminalLogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'success' | 'warn' | 'error';
  text: string;
}

export interface ActivityItem {
  id: string;
  time: string;
  event: string;
  status: 'success' | 'warning' | 'error';
  hash?: string;
}

export type LayoutViewMode = 'desktop-3col' | 'mobile-frame';
