export type RiskLevel = 'LOW RISK' | 'MEDIUM RISK' | 'HIGH RISK' | 'NO_AUDIO' | 'ACCUMULATING BUFFER';
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
  streamingStatus: 'IDLE' | 'LIVE_STREAMING' | 'ANALYZING' | 'TEST_SIGNAL' | 'PAUSED';
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
  engineStatus: string;
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
  status: 'success' | 'warning' | 'error' | 'info';
  hash?: string;
}

export type LayoutViewMode = 'desktop-3col' | 'mobile-frame';
