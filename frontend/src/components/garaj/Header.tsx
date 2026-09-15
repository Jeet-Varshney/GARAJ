import React from 'react';
import { Zap, Radio, Monitor, Smartphone, AlertTriangle } from 'lucide-react';
import type { LayoutViewMode } from '../../types/garaj';

interface HeaderProps {
  viewMode: LayoutViewMode;
  onViewModeChange: (mode: LayoutViewMode) => void;
  isMonitoring: boolean;
  chunkCount: number;
  isAttackSimulated?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  viewMode,
  onViewModeChange,
  isMonitoring,
  chunkCount,
  isAttackSimulated = false,
}) => {
  return (
    <header className={`w-full text-zinc-950 border-b sticky top-0 z-50 px-4 md:px-8 py-3.5 transition-all shadow-2xs ${
      isAttackSimulated ? 'bg-red-50/40 border-red-200' : 'bg-white border-zinc-200'
    }`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand with GARAJ Lightning Logo */}
        <div className="flex items-center gap-3">
          <div className={`w-9 h-9 rounded-xl text-white flex items-center justify-center font-extrabold shadow-sm shrink-0 transition-colors ${
            isAttackSimulated ? 'bg-red-600' : 'bg-zinc-950'
          }`}>
            <Zap className="w-5 h-5 fill-current" />
          </div>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-base font-extrabold tracking-tight text-zinc-950 font-sans uppercase">
                GARAJ <span className="font-normal text-zinc-500">Voice Security</span>
              </h1>
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-[11px] font-mono font-bold border transition-colors ${
                  isAttackSimulated
                    ? 'bg-amber-100 text-amber-800 border-amber-300'
                    : isMonitoring
                    ? 'bg-zinc-100 text-zinc-900 border-zinc-200'
                    : 'bg-zinc-100 text-zinc-500 border-zinc-200'
                }`}
              >
                <span className={`w-2 h-2 rounded-full animate-pulse ${
                  isAttackSimulated ? 'bg-amber-500' : isMonitoring ? 'bg-emerald-500' : 'bg-zinc-400'
                }`} />
                {isAttackSimulated ? 'TEST SIGNAL ACTIVE' : isMonitoring ? 'Live Protection' : 'IDLE'}
              </span>
            </div>
          </div>
        </div>

        {/* Center: Device Mode Switcher */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onViewModeChange('desktop-3col')}
            className={`animated-mono-btn ${
              viewMode === 'desktop-3col' ? 'animated-mono-btn-active' : ''
            }`}
          >
            <Monitor className="w-3.5 h-3.5" />
            <span>Desktop</span>
          </button>
          <button
            onClick={() => onViewModeChange('mobile-frame')}
            className={`animated-mono-btn ${
              viewMode === 'mobile-frame' ? 'animated-mono-btn-active' : ''
            }`}
          >
            <Smartphone className="w-3.5 h-3.5" />
            <span>Mobile Presentation</span>
          </button>
        </div>

        {/* Right Telemetry Badge */}
        <div className="flex items-center gap-3 font-mono">
          <div className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs border font-mono transition-colors ${
            isAttackSimulated
              ? 'bg-amber-50 text-amber-800 border-amber-300 font-extrabold'
              : 'bg-zinc-100 border-zinc-200 text-zinc-700'
          }`}>
            {isAttackSimulated ? (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 animate-bounce" />
            ) : (
              <Radio className="w-3.5 h-3.5 text-zinc-950" />
            )}
            <span className={`font-bold uppercase ${isAttackSimulated ? 'text-amber-800' : 'text-zinc-950'}`}>
              {isAttackSimulated ? 'TEST SIGNAL' : isMonitoring ? 'CONNECTED' : 'IDLE'}
            </span>
            <span className={isAttackSimulated ? 'text-amber-300' : 'text-zinc-300'}>|</span>
            <span className={`font-semibold ${isAttackSimulated ? 'text-amber-900' : 'text-zinc-950'}`}>{chunkCount} chunks</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
