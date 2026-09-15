import React from 'react';
import { INTERVIEW_PHASES } from '../context/AppContext';
import { Clock } from 'lucide-react';

interface PhaseTrackerProps {
  currentPhaseIndex: number;
  totalDurationSeconds?: number;
  elapsedSeconds?: number;
}

export const PhaseTracker: React.FC<PhaseTrackerProps> = ({
  currentPhaseIndex,
  totalDurationSeconds = 0,
  elapsedSeconds = 0,
}) => {
  const currentPhase = INTERVIEW_PHASES[currentPhaseIndex] || INTERVIEW_PHASES[0];

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="glass-panel p-4 rounded-card border border-slate-700/80 mb-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-indigo-600/20 text-indigo-400 font-bold text-sm border border-indigo-500/30 shrink-0">
            {currentPhaseIndex + 1}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-indigo-400 font-semibold">
                Phase {currentPhaseIndex + 1} of {INTERVIEW_PHASES.length}
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-xs text-slate-400 font-medium">{currentPhase.name}</span>
            </div>
            <h4 className="text-sm font-semibold text-slate-100">{currentPhase.description}</h4>
          </div>
        </div>

        {totalDurationSeconds > 0 && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 self-start sm:self-auto text-xs text-slate-300">
            <Clock className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-mono font-semibold">{formatTime(elapsedSeconds)}</span>
          </div>
        )}
      </div>

      {/* Phase step bar */}
      <div className="grid grid-cols-7 gap-1.5 mt-2">
        {INTERVIEW_PHASES.map((phase, idx) => {
          const isCompleted = idx < currentPhaseIndex;
          const isCurrent = idx === currentPhaseIndex;

          return (
            <div key={phase.id} className="group relative">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  isCompleted
                    ? 'bg-emerald-500'
                    : isCurrent
                    ? 'bg-indigo-500 animate-pulse'
                    : 'bg-slate-800 border border-slate-700'
                }`}
              />
              <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-full mb-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[10px] text-slate-200 whitespace-nowrap pointer-events-none z-10 shadow-xl">
                {phase.name}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
