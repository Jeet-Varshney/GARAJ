import React from 'react';
import { Shield, Radio, Monitor, Smartphone } from 'lucide-react';
import type { LayoutViewMode } from '../../types/garaj';

interface HeaderProps {
  viewMode: LayoutViewMode;
  onViewModeChange: (mode: LayoutViewMode) => void;
  isMonitoring: boolean;
  chunkCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  viewMode,
  onViewModeChange,
  isMonitoring,
  chunkCount,
}) => {
  return (
    <header className="w-full bg-white text-zinc-950 border-b border-zinc-200 sticky top-0 z-50 px-4 md:px-8 py-3.5 transition-all shadow-2xs">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-zinc-950 text-white flex items-center justify-center font-extrabold shadow-sm shrink-0">
            <Shield className="w-4.5 h-4.5" />
          </div>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-base font-extrabold tracking-tight text-zinc-950 font-sans uppercase">
                GARAJ <span className="font-normal text-zinc-500">Voice Security</span>
              </h1>
              <span className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-zinc-100 border border-zinc-200 text-[11px] font-mono text-zinc-900 font-bold">
                <span className="w-2 h-2 rounded-full bg-zinc-950 animate-pulse" />
                Live Monitoring
              </span>
            </div>
          </div>
        </div>

        {/* Center: Device Mode Switcher (With Custom Animated Black Circle Fill Effect) */}
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
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-zinc-100 border border-zinc-200 text-xs text-zinc-700">
            <Radio className="w-3.5 h-3.5 text-zinc-950" />
            <span className="font-bold text-zinc-950 uppercase">
              {isMonitoring ? 'CONNECTED' : 'IDLE'}
            </span>
            <span className="text-zinc-300">|</span>
            <span className="font-semibold text-zinc-950">{chunkCount} chunks</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
