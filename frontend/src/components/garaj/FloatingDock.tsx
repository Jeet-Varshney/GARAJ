import React from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Mic, Zap, Terminal, Monitor, Smartphone } from 'lucide-react';
import type { LayoutViewMode } from '../../types/garaj';

interface FloatingDockProps {
  isMonitoring: boolean;
  onToggleMonitoring: () => void;
  isMicActive: boolean;
  onToggleMic: () => void;
  isAttackSimulated: boolean;
  onToggleAttack: () => void;
  showTerminal: boolean;
  onToggleTerminal: () => void;
  viewMode: LayoutViewMode;
  onViewModeChange: (mode: LayoutViewMode) => void;
}

export const FloatingDock: React.FC<FloatingDockProps> = ({
  isMonitoring,
  onToggleMonitoring,
  isMicActive,
  onToggleMic,
  isAttackSimulated,
  onToggleAttack,
  showTerminal,
  onToggleTerminal,
  viewMode,
  onViewModeChange,
}) => {
  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50 pointer-events-auto">
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 350, damping: 25 }}
        className="mono-floating-dock rounded-full px-5 py-2.5 flex items-center gap-2 sm:gap-3"
      >
        {/* Monitoring Toggle Button */}
        <button
          onClick={onToggleMonitoring}
          className={`relative group p-3 rounded-full transition-all duration-200 cursor-pointer ${
            isMonitoring
              ? 'bg-zinc-950 text-white shadow-xs font-bold'
              : 'bg-zinc-100 text-zinc-700 hover:text-zinc-950 hover:bg-zinc-200'
          }`}
          title={isMonitoring ? 'Pause Stream' : 'Start Stream'}
        >
          {isMonitoring ? (
            <Pause className="w-4 h-4 text-white fill-current" />
          ) : (
            <Play className="w-4 h-4 fill-current text-zinc-950" />
          )}

          <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-zinc-950 text-white text-[10px] font-mono rounded-md shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap font-bold">
            {isMonitoring ? 'Pause Stream' : 'Start Stream'}
          </span>
        </button>

        {/* Live Mic Toggle */}
        <button
          onClick={onToggleMic}
          className={`relative group p-3 rounded-full transition-all duration-200 cursor-pointer ${
            isMicActive
              ? 'bg-zinc-950 text-white shadow-xs font-bold'
              : 'bg-zinc-100 text-zinc-700 hover:text-zinc-950 hover:bg-zinc-200'
          }`}
          title="Hardware Mic"
        >
          <Mic className={`w-4 h-4 ${isMicActive ? 'animate-pulse' : ''}`} />
          <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-zinc-950 text-white text-[10px] font-mono rounded-md shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap font-bold">
            {isMicActive ? 'Mic Active' : 'Enable Mic'}
          </span>
        </button>

        <div className="w-px h-6 bg-zinc-200 mx-0.5" />

        {/* Test Signal Button */}
        <button
          onClick={onToggleAttack}
          className={`relative group p-3 rounded-full transition-all duration-200 cursor-pointer ${
            isAttackSimulated
              ? 'bg-amber-600 text-white shadow-xs'
              : 'bg-zinc-100 text-zinc-700 hover:text-zinc-950 hover:bg-zinc-200'
          }`}
          title="Test Signal"
        >
          <Zap className={`w-4 h-4 ${isAttackSimulated ? 'animate-bounce text-white' : 'text-zinc-950'}`} />
          <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-zinc-950 text-white text-[10px] font-mono rounded-md shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap font-bold">
            {isAttackSimulated ? 'Stop Test Signal' : 'Test Signal'}
          </span>
        </button>

        {/* Toggle Live Terminal */}
        <button
          onClick={onToggleTerminal}
          className={`relative group p-3 rounded-full transition-all duration-200 cursor-pointer ${
            showTerminal
              ? 'bg-zinc-950 text-white shadow-xs font-bold'
              : 'bg-zinc-100 text-zinc-700 hover:text-zinc-950 hover:bg-zinc-200'
          }`}
          title="Toggle Terminal"
        >
          <Terminal className="w-4 h-4" />
          <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-zinc-950 text-white text-[10px] font-mono rounded-md shadow-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap font-bold">
            {showTerminal ? 'Hide Terminal' : 'Show Terminal'}
          </span>
        </button>

        <div className="w-px h-6 bg-zinc-200 mx-0.5 hidden sm:block" />

        {/* Desktop / Mobile View Switcher */}
        <div className="hidden sm:flex items-center gap-1 bg-zinc-100 p-1 rounded-full border border-zinc-200">
          <button
            onClick={() => onViewModeChange('desktop-3col')}
            className={`p-2 rounded-full transition-colors cursor-pointer ${
              viewMode === 'desktop-3col'
                ? 'bg-zinc-950 text-white shadow-xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
            title="Desktop View"
          >
            <Monitor className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onViewModeChange('mobile-frame')}
            className={`p-2 rounded-full transition-colors cursor-pointer ${
              viewMode === 'mobile-frame'
                ? 'bg-zinc-950 text-white shadow-xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
            title="Mobile Presentation View"
          >
            <Smartphone className="w-3.5 h-3.5" />
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default FloatingDock;
