import React from 'react';
import type { EvaluationScore } from '../types';
import { Award } from 'lucide-react';

interface ScoreCardProps {
  title: string;
  description: string;
  score: EvaluationScore;
  icon?: React.ReactNode;
}

export const ScoreCard: React.FC<ScoreCardProps> = ({
  title,
  description,
  score,
  icon = <Award className="w-5 h-5 text-indigo-400" />,
}) => {
  const percentage = Math.round((score.score / score.maxScore) * 100);

  const getScoreBadge = () => {
    if (percentage >= 80) {
      return {
        label: 'EXCELLENT',
        bgColor: 'bg-emerald-500/10',
        textColor: 'text-emerald-400',
        borderColor: 'border-emerald-500/30',
        barColor: 'bg-emerald-500',
      };
    }
    if (percentage >= 60) {
      return {
        label: 'PROFICIENT',
        bgColor: 'bg-indigo-500/10',
        textColor: 'text-indigo-400',
        borderColor: 'border-indigo-500/30',
        barColor: 'bg-indigo-500',
      };
    }
    if (percentage >= 40) {
      return {
        label: 'DEVELOPING',
        bgColor: 'bg-amber-500/10',
        textColor: 'text-amber-400',
        borderColor: 'border-amber-500/30',
        barColor: 'bg-amber-500',
      };
    }
    return {
      label: 'NEEDS FOCUS',
      bgColor: 'bg-red-500/10',
      textColor: 'text-red-400',
      borderColor: 'border-red-500/30',
      barColor: 'bg-red-500',
    };
  };

  const badge = getScoreBadge();

  return (
    <div className="taste-card p-5 relative overflow-hidden transition-all duration-300 hover:border-indigo-500/40">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 shrink-0">
            {icon}
          </div>
          <div>
            <h3 className="font-bold text-slate-100 text-sm tracking-tight">{title}</h3>
            <p className="text-[11px] text-slate-400 leading-snug">{description}</p>
          </div>
        </div>
        <div className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border tracking-wider ${badge.bgColor} ${badge.textColor} ${badge.borderColor}`}>
          {badge.label}
        </div>
      </div>

      <div className="mt-4 pt-2 border-t border-slate-800/80">
        <div className="flex justify-between items-baseline mb-2">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider font-mono">Performance Metric</span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-black text-slate-100 font-mono tracking-tight">{score.score}</span>
            <span className="text-xs font-semibold text-slate-500 font-mono">/ {score.maxScore}</span>
          </div>
        </div>
        
        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
          <div
            className={`h-full rounded-full transition-all duration-1000 ease-out ${badge.barColor}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
};
