import React from 'react';
import { motion } from 'framer-motion';
import { Radio, Layers, Clock, Cpu } from 'lucide-react';
import type {
  PipelineSpec,
  ChunkTelemetry,
  LatencyMetric,
  AASISTModelSpec,
} from '../../types/garaj';

interface DiagnosticsPanelProps {
  pipelineSpec: PipelineSpec;
  chunkTelemetry: ChunkTelemetry;
  chunkFlash: boolean;
  latencies: LatencyMetric[];
  modelSpec: AASISTModelSpec;
  isAttackSimulated?: boolean;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({
  pipelineSpec,
  chunkTelemetry,
  chunkFlash,
  latencies,
  modelSpec,
  isAttackSimulated = false,
}) => {
  const totalLatencyMs = latencies.reduce((acc, l) => acc + l.valueMs, 0);

  return (
    <div className="space-y-4 sm:space-y-6 font-mono max-w-full overflow-hidden">
      {/* 2-Column Grid (Stacks on small screens & phone frames so cards never squish) */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-6">
        {/* Card A: Audio Pipeline Spec */}
        <div className="bg-white rounded-2xl p-4 sm:p-6 border border-zinc-200 shadow-sm space-y-4 min-w-0">
          <div className="flex items-center gap-2.5 border-b border-zinc-200 pb-3.5">
            <div className="p-2 rounded-xl bg-zinc-100 text-zinc-950 shrink-0">
              <Radio className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider truncate">
              AUDIO PIPELINE SPEC
            </h3>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-2 p-2.5 sm:p-3 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
              <span className="text-zinc-500 font-normal shrink-0">WebSocket:</span>
              <span className="text-zinc-950 bg-white font-extrabold px-2.5 py-0.5 rounded-full border border-zinc-300 shadow-2xs text-[11px] truncate">
                {pipelineSpec.wsConnection}
              </span>
            </div>

            <div className="flex items-center justify-between gap-2 p-2.5 sm:p-3 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
              <span className="text-zinc-500 font-normal shrink-0">Streaming:</span>
              <span className="text-zinc-950 bg-white font-extrabold px-2.5 py-0.5 rounded-full border border-zinc-300 shadow-2xs text-[11px] truncate">
                {pipelineSpec.streamingStatus}
              </span>
            </div>

            <div className="flex items-center justify-between gap-2 p-2.5 sm:p-3 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
              <span className="text-zinc-500 font-normal shrink-0">Audio Engine:</span>
              <span className="text-zinc-950 font-bold truncate text-[11px] sm:text-xs">{pipelineSpec.audioEngine}</span>
            </div>

            <div className="flex items-center justify-between gap-2 p-2.5 sm:p-3 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
              <span className="text-zinc-500 font-normal shrink-0">Format:</span>
              <span className="text-zinc-950 font-bold truncate text-[10px] sm:text-xs">{pipelineSpec.audioFormat}</span>
            </div>
          </div>
        </div>

        {/* Card B: 100ms Chunks Telemetry */}
        <div className="bg-white rounded-2xl p-4 sm:p-6 border border-zinc-200 shadow-sm space-y-4 min-w-0">
          <div className="flex items-center justify-between gap-2 border-b border-zinc-200 pb-3.5">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="p-2 rounded-xl bg-zinc-100 text-zinc-950 shrink-0">
                <Layers className="w-4 h-4" />
              </div>
              <h3 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider truncate">
                100ms CHUNK METRICS
              </h3>
            </div>
            <span className="text-[10px] text-zinc-500 font-bold shrink-0">100ms Window</span>
          </div>

          <div className="flex flex-row items-baseline justify-between gap-2 p-3 sm:p-4 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
            <div className="min-w-0">
              <span className="text-[10px] text-zinc-500 block uppercase font-bold truncate">TOTAL CHUNKS</span>
              <motion.div
                key={chunkTelemetry.totalChunks}
                initial={{ scale: 1.05 }}
                animate={{ scale: 1 }}
                className={`text-2xl sm:text-3xl font-extrabold tracking-tight transition-colors truncate ${
                  chunkFlash ? 'text-zinc-950' : 'text-zinc-900'
                }`}
              >
                {chunkTelemetry.totalChunks} <span className="text-[11px] text-zinc-500 font-normal">units</span>
              </motion.div>
            </div>

            <div className="text-right shrink-0">
              <span className="text-[10px] text-zinc-500 block uppercase font-bold">DURATION</span>
              <div className="text-zinc-950 font-extrabold text-base sm:text-lg">
                {chunkTelemetry.audioDurationSec}s
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-1.5 sm:gap-2 text-xs">
            <div className="bg-zinc-50 p-2 sm:p-2.5 rounded-xl border border-zinc-200/80 text-center min-w-0">
              <span className="text-[9px] sm:text-[10px] text-zinc-500 block font-bold truncate">RATE</span>
              <span className="text-zinc-950 font-extrabold text-[11px] sm:text-xs truncate block">{chunkTelemetry.chunksPerSec}/s</span>
            </div>
            <div className="bg-zinc-50 p-2 sm:p-2.5 rounded-xl border border-zinc-200/80 text-center min-w-0">
              <span className="text-[9px] sm:text-[10px] text-zinc-500 block font-bold truncate">SENT</span>
              <span className="text-zinc-950 font-extrabold text-[11px] sm:text-xs truncate block">{chunkTelemetry.clientSent}</span>
            </div>
            <div className="bg-zinc-50 p-2 sm:p-2.5 rounded-xl border border-zinc-200/80 text-center min-w-0">
              <span className="text-[9px] sm:text-[10px] text-zinc-500 block font-bold truncate">DROPPED</span>
              <span className="text-zinc-950 font-extrabold text-[11px] sm:text-xs truncate block">{chunkTelemetry.dropped}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Card C: Latency Breakdown */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 border border-zinc-200 shadow-sm space-y-4 min-w-0">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-200 pb-3.5 font-mono">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="p-2 rounded-xl bg-zinc-100 text-zinc-950 shrink-0">
              <Clock className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider truncate">
              LATENCY TELEMETRY
            </h3>
          </div>
          <div className="text-xs shrink-0">
            <span className="text-zinc-500 font-normal">TOTAL: </span>
            <span className="text-zinc-950 font-extrabold">~{totalLatencyMs.toFixed(3)} ms</span>
          </div>
        </div>

        <div className="space-y-3 text-xs">
          <div className="h-3 w-full bg-zinc-100 rounded-full overflow-hidden flex p-0.5 border border-zinc-200">
            {latencies.map((metric, idx) => (
              <div
                key={metric.name}
                style={{ width: `${metric.percentage}%` }}
                className={`h-full ${
                  idx === 0
                    ? 'bg-zinc-950'
                    : idx === 1
                    ? 'bg-zinc-700'
                    : idx === 2
                    ? 'bg-zinc-400'
                    : 'bg-zinc-300'
                } border-r border-white last:border-0 rounded-full transition-all`}
                title={`${metric.name}: ${metric.valueMs} ms`}
              />
            ))}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
            {latencies.map((metric) => (
              <div key={metric.name} className="p-2.5 rounded-xl bg-zinc-50 border border-zinc-200/80 min-w-0">
                <span className="text-[10px] text-zinc-500 block truncate font-normal">{metric.name}</span>
                <span className="text-zinc-950 font-extrabold text-[11px] sm:text-xs block truncate">{metric.valueMs} ms</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Card D: Detection Engine */}
      <div className={`rounded-2xl p-4 sm:p-6 border shadow-sm space-y-4 min-w-0 transition-all ${
        isAttackSimulated || modelSpec.spoofProb > 0.5
          ? 'bg-red-50/30 border-red-400 ring-1 ring-red-400/50'
          : 'bg-white border-zinc-200'
      }`}>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-200 pb-3.5">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className={`p-2 rounded-xl shrink-0 ${
              isAttackSimulated ? 'bg-red-600 text-white' : 'bg-zinc-100 text-zinc-950'
            }`}>
              <Cpu className="w-4 h-4" />
            </div>
            <h3 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider truncate">
              DETECTION ENGINE (W2V2-AASIST)
            </h3>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-[11px] font-extrabold border shrink-0 w-fit ${
              isAttackSimulated || modelSpec.spoofProb > 0.5
                ? 'bg-red-600 text-white border-red-700 animate-pulse'
                : 'bg-zinc-100 text-zinc-950 border-zinc-300'
            }`}
          >
            {isAttackSimulated ? 'SPOOF_DETECTED' : modelSpec.engineStatus}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 sm:p-3.5 rounded-xl bg-zinc-50 border border-zinc-200/80 space-y-1 min-w-0">
            <span className="text-zinc-500 text-[10px] block uppercase font-bold">MODEL ARCHITECTURE</span>
            <span className="text-zinc-950 font-extrabold text-[11px] sm:text-xs block truncate">{modelSpec.architecture}</span>
          </div>

          <div className={`p-3 sm:p-3.5 rounded-xl border space-y-1 min-w-0 ${
            isAttackSimulated || modelSpec.spoofProb > 0.5
              ? 'bg-red-100/80 border-red-300'
              : 'bg-zinc-50 border-zinc-200/80'
          }`}>
            <span className="text-zinc-500 text-[10px] block uppercase font-bold">MODEL PREDICTION</span>
            <span className={`font-extrabold text-[11px] sm:text-xs block truncate ${
              isAttackSimulated || modelSpec.spoofProb > 0.5 ? 'text-red-700' : 'text-zinc-950'
            }`}>
              {modelSpec.modelPrediction}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiagnosticsPanel;
