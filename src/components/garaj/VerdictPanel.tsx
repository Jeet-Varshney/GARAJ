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
  ArrowUpRight,
  Activity,
} from 'lucide-react';
import type {
  VerdictStatus,
  RiskLevel,
  SecurityCheckItem,
  ActivityItem,
} from '../../types/garaj';

interface VerdictPanelProps {
  score: number;
  verdict: VerdictStatus;
  riskLevel: RiskLevel;
  checks: SecurityCheckItem[];
  recentActivities: ActivityItem[];
  isMonitoring: boolean;
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
  onToggleMonitoring,
  isAttackSimulated,
  onToggleAttack,
}) => {
  const isSynthetic = verdict === 'SYNTHETIC';
  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="bg-white rounded-2xl p-6 border border-zinc-200 shadow-sm flex flex-col justify-between space-y-6">
      {/* Header Label */}
      <div className="flex items-center justify-between border-b border-zinc-200 pb-4 font-mono">
        <div className="flex items-center gap-2.5">
          <Activity className="w-4 h-4 text-zinc-950" />
          <div>
            <h2 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider">
              VOICE AUTHENTICITY VERDICT
            </h2>
            <p className="text-[11px] text-zinc-500 font-sans font-normal">
              Neural biometric vector analysis
            </p>
          </div>
        </div>

        <span className="text-xs font-bold text-zinc-800 bg-zinc-100 border border-zinc-200 px-3 py-1 rounded-full">
          MODEL V4.2
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
              stroke={isSynthetic ? '#EF4444' : '#09090B'}
              strokeWidth="12"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 1.2, ease: 'easeOut' }}
              strokeLinecap="round"
              fill="transparent"
            />
          </svg>

          {/* Center Text inside Circle */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <motion.span
              key={score}
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`text-4xl font-extrabold tracking-tight font-sans ${
                isSynthetic ? 'text-red-600' : 'text-zinc-950'
              }`}
            >
              {score}%
            </motion.span>
            <span
              className={`text-xs font-mono font-bold uppercase tracking-widest mt-1 px-3.5 py-0.5 rounded-full border ${
                isSynthetic
                  ? 'text-red-600 bg-red-50 border-red-200'
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
                : 'bg-zinc-100 text-zinc-950 border-zinc-300'
            }`}
          >
            {riskLevel === 'HIGH RISK' ? (
              <ShieldAlert className="w-4 h-4 text-red-600" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-zinc-950" />
            )}
            <span>{riskLevel}</span>
          </div>
        </motion.div>
      </div>

      {/* Security Diagnostic Checks (Clean Light Boxes) */}
      <div className="space-y-2.5 border-t border-zinc-200 pt-5 font-mono">
        <div className="flex items-center justify-between text-xs text-zinc-950 font-bold mb-2">
          <span>VERIFIED CHECKS</span>
          <span className="text-zinc-500 font-normal">4 VERIFIED</span>
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
              ? 'bg-zinc-950 text-white hover:bg-zinc-800'
              : 'bg-zinc-900 text-white hover:bg-zinc-800'
          }`}
        >
          {isMonitoring ? (
            <>
              <Pause className="w-4 h-4 text-white fill-current" />
              <span>PAUSE MONITORING</span>
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
              ? 'bg-red-50 text-red-600 border border-red-200'
              : 'bg-white hover:bg-zinc-100 text-zinc-950 border border-zinc-300'
          }`}
        >
          <Zap className={`w-4 h-4 ${isAttackSimulated ? 'animate-bounce text-red-600' : 'text-zinc-950'}`} />
          <span>
            {isAttackSimulated ? 'ATTACK SIMULATION ACTIVE' : 'SIMULATE DEEPFAKE ATTACK'}
          </span>
        </button>
      </div>

      {/* Recent Activity List */}
      <div className="border-t border-zinc-200 pt-4 space-y-2 font-mono">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-950 font-bold uppercase tracking-wider text-[11px]">
            RECENT ACTIVITY
          </span>
          <button className="text-zinc-600 hover:text-zinc-950 font-bold flex items-center gap-0.5 text-[11px] cursor-pointer">
            <span>VIEW ALL</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
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
