import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  type?: 'spinner' | 'waveform' | 'reportSkeleton';
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Processing voice audio...',
  size = 'md',
  type = 'waveform',
}) => {
  if (type === 'reportSkeleton') {
    return (
      <div className="glass-panel p-6 rounded-card space-y-6 animate-pulse">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-slate-800" />
          <div className="space-y-2 flex-1">
            <div className="h-4 bg-slate-800 rounded w-1/3" />
            <div className="h-3 bg-slate-800 rounded w-1/4" />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-28 bg-slate-800/60 rounded-card p-4 space-y-3">
              <div className="h-4 bg-slate-700 rounded w-1/2" />
              <div className="h-3 bg-slate-700/50 rounded w-full" />
              <div className="h-2 bg-slate-700 rounded w-3/4" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (type === 'waveform') {
    return (
      <div className="flex flex-col items-center justify-center p-6 gap-3">
        <div className="flex items-center gap-1.5 h-8">
          <div className="w-1.5 h-6 bg-indigo-500 rounded-full animate-wave" style={{ animationDelay: '0ms' }} />
          <div className="w-1.5 h-10 bg-indigo-400 rounded-full animate-wave" style={{ animationDelay: '150ms' }} />
          <div className="w-1.5 h-4 bg-indigo-600 rounded-full animate-wave" style={{ animationDelay: '300ms' }} />
          <div className="w-1.5 h-8 bg-indigo-300 rounded-full animate-wave" style={{ animationDelay: '450ms' }} />
          <div className="w-1.5 h-5 bg-indigo-500 rounded-full animate-wave" style={{ animationDelay: '600ms' }} />
        </div>
        {label && <p className="text-xs font-medium text-slate-400 animate-pulse">{label}</p>}
      </div>
    );
  }

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className="flex items-center justify-center gap-2 p-4 text-indigo-400">
      <Loader2 className={`${iconSizes[size]} animate-spin`} />
      {label && <span className="text-sm font-medium text-slate-300">{label}</span>}
    </div>
  );
};
