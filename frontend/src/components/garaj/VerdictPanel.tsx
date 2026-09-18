import React from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheck,
  ShieldAlert,
  Check,
  AlertTriangle,
  X,
  Play,
  Pause,
  Zap,
  Activity,
  Radio,
} from 'lucide-react';
import type {
  VerdictStatus,
  RiskLevel,
  SecurityCheckItem,
  ActivityItem,
} from '../../types/garaj';

interface VerdictPanelProps {
  score: number | null;
  verdict: VerdictStatus | string;
  riskLevel: RiskLevel | string;
  checks: SecurityCheckItem[];
  recentActivities: ActivityItem[];
  isMonitoring: boolean;
  isPaused?: boolean;
  onToggleMonitoring: () => void;
  isAttackSimulated: boolean;
  onToggleAttack: () => void;
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({
  score,
  verdict,
  riskLevel,
  checks,
  recentActivities,
  isMonitoring,
  isPaused = false,
  onToggleMonitoring,
  isAttackSimulated,
  onToggleAttack,
}) => {
  const isSynthetic = verdict === 'SYNTHETIC';
  const isDisconnected = verdict === 'BACKEND DISCONNECTED' || verdict === 'Backend Disconnected';
  const isNoAudio = verdict === 'NO AUDIO DETECTED' || verdict === 'NO_AUDIO' || verdict === 'No audio detected';
  const isWaiting = verdict === 'WAITING FOR AUDIO' || verdict === 'Waiting for detection' || verdict === 'Waiting for detection...';
  const isPausedState = isPaused || riskLevel === 'DETECTION PAUSED' || verdict === 'DETECTION PAUSED';
  
  const hasScore = score !== null && score !== undefined && !isDisconnected && !isNoAudio && !isWaiting;

  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = hasScore ? circumference - ((score || 0) / 100) * circumference : circumference;

  return (
    <div className={`rounded-2xl p-6 border shadow-sm flex flex-col justify-between space-y-6 transition-all ${
      isSynthetic
        ? 'bg-red-50/30 border-red-400 shadow-red-100/80 ring-1 ring-red-400/50'
        : isPausedState
        ? 'bg-amber-50/20 border-amber-300'
        : 'bg-white border-zinc-200'
    }`}>
      {/* Critical Alert Banner when Non-Real / Spoof Voice Detected */}
      {isSynthetic && !isPausedState && (
        <div className="bg-red-600 text-white font-mono text-[11px] font-extrabold py-2 px-3.5 rounded-xl flex items-center justify-between shadow-sm animate-pulse">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-white shrink-0" />
            <span>UNAUTHENTIC / SPOOF VOICE DETECTED</span>
          </div>
          <span className="bg-red-700 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider">CRITICAL ALERT</span>
        </div>
      )}

      {/* Paused Banner when Detection Stream is Paused */}
      {isPausedState && (
        <div className="bg-amber-600 text-white font-mono text-[11px] font-extrabold py-2 px-3.5 rounded-xl flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-2">
            <Pause className="w-4 h-4 text-white shrink-0 fill-current" />
            <span>{hasScore ? 'DETECTION PAUSED — FROZEN SNAPSHOT' : 'DETECTION PAUSED'}</span>
          </div>
          <span className="bg-amber-700 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider">PAUSED</span>
        </div>
      )}

      {/* Header Label */}
      <div className="flex items-center justify-between border-b border-zinc-200 pb-4 font-mono">
        <div className="flex items-center gap-2.5">
          <Activity className="w-4 h-4 text-zinc-950" />
          <div>
            <h2 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider">
              VOICE AUTHENTICITY VERDICT
            </h2>
            <p className="text-[11px] text-zinc-500 font-sans font-normal">
              Real-time W2V2-AASIST detection pipeline
            </p>
          </div>
        </div>

        <span className="text-xs font-bold text-zinc-800 bg-zinc-100 border border-zinc-200 px-3 py-1 rounded-full">
          W2V2-AASIST
        </span>
      </div>

      {/* HERO: Circular Gauge */}
      <div className="flex flex-col items-center justify-center py-2 relative">
        <div className="relative w-48 h-48 flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
            <circle
              cx="100"
              cy="100"
              r={radius}
              className="stroke-zinc-100"
              strokeWidth="12"
              fill="transparent"
            />

            <motion.circle
              cx="100"
              cy="100"
              r={radius}
              stroke={isSynthetic ? '#EF4444' : isPausedState ? '#F59E0B' : isDisconnected || isNoAudio ? '#D4D4D8' : isWaiting ? '#F59E0B' : '#09090B'}
              strokeWidth="12"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              strokeLinecap="round"
              fill="transparent"
            />
          </svg>

          {/* Center Text inside Circle */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4">
            <motion.span
              key={isDisconnected ? 'off' : isNoAudio ? 'noaudio' : isWaiting ? 'wait' : score}
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`text-3xl sm:text-4xl font-extrabold tracking-tight font-sans ${
                isSynthetic ? 'text-red-600' : isPausedState ? 'text-amber-700' : isDisconnected || isNoAudio ? 'text-zinc-400' : 'text-zinc-950'
              }`}
            >
              {hasScore ? `${score}%` : '--'}
            </motion.span>
            <span
              className={`text-[10px] sm:text-xs font-mono font-bold uppercase tracking-wider mt-1 px-3 py-0.5 rounded-full border text-center truncate max-w-full ${
                isSynthetic
                  ? 'text-red-600 bg-red-50 border-red-200'
                  : isPausedState
                  ? 'text-amber-800 bg-amber-50 border-amber-300'
                  : isDisconnected || isNoAudio
                  ? 'text-zinc-600 bg-zinc-100 border-zinc-200'
                  : isWaiting
                  ? 'text-amber-800 bg-amber-50 border-amber-200'
                  : 'text-zinc-950 bg-zinc-100 border-zinc-300'
              }`}
            >
              {verdict}
            </span>
          </div>
        </div>

        {/* Risk Level Badge */}
        <motion.div
          key={riskLevel}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4"
        >
          <div
            className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold font-mono tracking-wide border ${
              riskLevel === 'HIGH RISK'
                ? 'bg-red-50 text-red-600 border-red-200'
                : isPausedState
                ? 'bg-amber-50 text-amber-800 border-amber-300'
                : isDisconnected || isNoAudio
                ? 'bg-zinc-100 text-zinc-600 border-zinc-200'
                : isWaiting
                ? 'bg-amber-50 text-amber-800 border-amber-200'
                : 'bg-zinc-100 text-zinc-950 border-zinc-300'
            }`}
          >
            {riskLevel === 'HIGH RISK' ? (
              <ShieldAlert className="w-4 h-4 text-red-600" />
            ) : isPausedState ? (
              <Pause className="w-4 h-4 text-amber-600 fill-current" />
            ) : isDisconnected || isNoAudio ? (
              <Radio className="w-4 h-4 text-zinc-400" />
            ) : isWaiting ? (
              <Activity className="w-4 h-4 text-amber-600 animate-pulse" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-zinc-950" />
            )}
            <span>{riskLevel}</span>
          </div>
        </motion.div>
      </div>

      {/* Security Diagnostic Checks */}
      <div className="space-y-2.5 border-t border-zinc-200 pt-5 font-mono">
        <div className="flex items-center justify-between text-xs text-zinc-950 font-bold mb-2">
          <span>REAL DIAGNOSTIC CHECKS</span>
          <span className="text-zinc-500 font-normal">BACKEND BOUND</span>
        </div>

        <div className="space-y-2">
          {checks.map((check) => (
            <div
              key={check.id}
              className="flex items-center justify-between p-3 rounded-xl bg-zinc-50 border border-zinc-200/80 text-xs shadow-2xs"
            >
              <div className="flex items-center gap-3 min-w-0">
                {check.status === 'Normal' ? (
                  <Check className="w-4 h-4 text-zinc-950 shrink-0 font-bold" />
                ) : check.status === 'Caution' ? (
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                ) : (
                  <X className="w-4 h-4 text-red-600 shrink-0" />
                )}
                <div className="min-w-0">
                  <span className="text-zinc-950 font-bold block truncate">
                    {check.label}
                  </span>
                  <span className="text-[11px] text-zinc-500 block truncate font-sans font-normal">
                    {check.description}
                  </span>
                </div>
              </div>

              <span
                className={`text-[11px] font-bold px-3 py-1 rounded-lg shrink-0 border ${
                  check.status === 'Normal'
                    ? 'text-zinc-950 bg-white border-zinc-300'
                    : check.status === 'Caution'
                    ? 'text-amber-800 bg-amber-50 border-amber-200'
                    : 'text-red-700 bg-red-50 border-red-200'
                }`}
              >
                {check.value || check.status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-2 space-y-2.5 font-mono">
        <button
          onClick={onToggleMonitoring}
          className={`w-full py-3.5 px-6 rounded-full font-bold text-xs tracking-wider flex items-center justify-center gap-2.5 transition-all cursor-pointer shadow-sm ${
            isMonitoring
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-zinc-950 text-white hover:bg-zinc-800'
          }`}
        >
          {isMonitoring ? (
            <>
              <Pause className="w-4 h-4 text-white fill-current" />
              <span>PAUSE MONITORING</span>
            </>
          ) : isPausedState ? (
            <>
              <Play className="w-4 h-4 text-white fill-current" />
              <span>RESUME MONITORING</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-white fill-current" />
              <span>START MONITORING</span>
            </>
          )}
        </button>

        <button
          onClick={onToggleAttack}
          className={`w-full py-3 px-4 rounded-full font-bold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer shadow-2xs ${
            isAttackSimulated
              ? 'bg-amber-50 text-amber-800 border border-amber-300'
              : 'bg-white hover:bg-zinc-100 text-zinc-950 border border-zinc-300'
          }`}
        >
          <Zap className={`w-4 h-4 ${isAttackSimulated ? 'text-amber-600' : 'text-zinc-950'}`} />
          <span>
            {isAttackSimulated ? 'STOP TEST SIGNAL' : 'RUN TEST SIGNAL'}
          </span>
        </button>
      </div>

      {/* Recent Activity List */}
      <div className="border-t border-zinc-200 pt-4 space-y-2 font-mono">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-950 font-bold uppercase tracking-wider text-[11px]">
            RECENT ACTIVITY LOG
          </span>
          <span className="text-zinc-500 font-normal text-[11px]">REAL-TIME</span>
        </div>

        <div className="space-y-1.5 text-[11px]">
          {recentActivities.map((act) => (
            <div
              key={act.id}
              className="flex items-center gap-3 text-zinc-950 py-2 px-3 rounded-lg bg-zinc-50 border border-zinc-200/80"
            >
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  act.status === 'error' ? 'bg-red-600' : 'bg-zinc-950'
                }`}
              />
              <span className="text-zinc-500 shrink-0">{act.time}</span>
              <span className="truncate font-semibold">{act.event}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default VerdictPanel;
