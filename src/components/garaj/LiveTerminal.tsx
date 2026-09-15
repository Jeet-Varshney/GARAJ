import React, { useState, useEffect, useRef } from 'react';
import { Terminal as TerminalIcon, Copy, Check, Play, Pause, RefreshCw } from 'lucide-react';
import type { TerminalLogEntry } from '../../types/garaj';

interface TerminalProps {
  commands?: string[];
  outputs?: Record<number, string[]>;
  typingSpeed?: number;
  delayBetweenCommands?: number;
  logs: TerminalLogEntry[];
  isMonitoring: boolean;
}

export const LiveTerminal: React.FC<TerminalProps> = ({
  typingSpeed = 35,
  delayBetweenCommands = 800,
  logs,
  isMonitoring,
}) => {
  const [currentCmdIndex, setCurrentCmdIndex] = useState<number>(0);
  const [displayedText, setDisplayedText] = useState<string>('');
  const [isTyping, setIsTyping] = useState<boolean>(true);
  const [completedCmds, setCompletedCmds] = useState<number[]>([]);
  const [copied, setCopied] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  const demoCommands = [
    "garaj-cli init --stream-ws wss://api.garaj.sec/v1/voice-stream",
    "audio-engine load-worklet --sample-rate 16000 --format PCM_S16LE",
    "aasist-deepfake-engine eval --window 64600_samples",
    "garaj-security verify-acoustic-continuity --phase-1c.7",
  ];

  const demoOutputs: Record<number, string[]> = {
    0: [
      "✔ WebSocket connection established (wss://api.garaj.sec/v1/voice-stream).",
      "✔ TLS 1.3 256-bit handshake verified.",
      "✔ Socket status: CONNECTED | Latency: 3.0ms RTT",
    ],
    1: [
      "✔ AudioWorklet initialized on main audio thread.",
      "✔ RingBuffer allocated (64,600 samples @ 16kHz Mono PCM).",
      "✔ Chunk size: 100ms (~1,600 samples/chunk).",
    ],
    2: [
      "✔ Loaded model weights: W2V2-AASIST (LA_model.pth).",
      "✔ Graph Attention Network pass executed in 0.222ms.",
      "✔ Logits: [Spoof: -1.982, Real: +2.415] => Softmax: Real (82.1%).",
    ],
    3: [
      "✔ Phase 1C.7 acoustic window verified.",
      "✔ Consecutive window delta: 0.0014 (No splicing artifacts).",
      "✔ Verdict: REAL (AUTHENTICATED) | Risk: LOW RISK",
    ],
  };

  useEffect(() => {
    if (isPaused) return;

    const fullCommand = demoCommands[currentCmdIndex];
    if (!fullCommand) return;

    if (isTyping) {
      if (displayedText.length < fullCommand.length) {
        const timeout = setTimeout(() => {
          setDisplayedText(fullCommand.slice(0, displayedText.length + 1));
        }, typingSpeed);
        return () => clearTimeout(timeout);
      } else {
        setIsTyping(false);
        const timeout = setTimeout(() => {
          setCompletedCmds((prev) => [...prev, currentCmdIndex]);
          if (currentCmdIndex < demoCommands.length - 1) {
            setCurrentCmdIndex((prev) => prev + 1);
            setDisplayedText('');
            setIsTyping(true);
          }
        }, delayBetweenCommands);
        return () => clearTimeout(timeout);
      }
    }
  }, [displayedText, isTyping, currentCmdIndex, isPaused, typingSpeed, delayBetweenCommands]);

  useEffect(() => {
    if (!isPaused && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayedText, completedCmds, isPaused]);

  const copyTerminalContent = () => {
    const text = demoCommands
      .map((cmd, idx) => `$ ${cmd}\n${(demoOutputs[idx] || []).join('\n')}`)
      .join('\n\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const restartTerminal = () => {
    setCurrentCmdIndex(0);
    setDisplayedText('');
    setCompletedCmds([]);
    setIsTyping(true);
  };

  return (
    <section className="w-full font-mono max-w-full overflow-hidden">
      <div className="rounded-2xl border border-zinc-200 overflow-hidden shadow-sm bg-white">
        {/* Terminal Header Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-zinc-100 border-b border-zinc-200 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-300 border border-zinc-400 shrink-0" />
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-300 border border-zinc-400 shrink-0" />
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-300 border border-zinc-400 shrink-0" />
            <span className="ml-1 text-[11px] sm:text-xs text-zinc-950 font-mono flex items-center gap-1.5 font-extrabold truncate">
              <TerminalIcon className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
              <span className="truncate">GARAJ-CLI STREAM</span>
            </span>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            <button
              onClick={() => setIsPaused(!isPaused)}
              className="p-1.5 rounded-lg bg-white text-zinc-800 hover:text-zinc-950 border border-zinc-300 transition-colors cursor-pointer"
              title={isPaused ? 'Resume Terminal' : 'Pause Terminal'}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            </button>

            <button
              onClick={restartTerminal}
              className="p-1.5 rounded-lg bg-white text-zinc-800 hover:text-zinc-950 border border-zinc-300 transition-colors cursor-pointer"
              title="Replay Terminal"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={copyTerminalContent}
              className="p-1.5 rounded-lg bg-white text-zinc-800 hover:text-zinc-950 border border-zinc-300 transition-colors cursor-pointer"
              title="Copy Commands"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-zinc-950" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Animated Terminal Content Output */}
        <div className="p-4 sm:p-5 max-h-72 overflow-y-auto bg-[#09090B] text-zinc-200 font-mono text-[11px] sm:text-xs leading-relaxed space-y-4 select-text">
          {completedCmds.map((cmdIdx) => (
            <div key={cmdIdx} className="space-y-1.5 break-words">
              <div className="flex items-start gap-1.5 text-white flex-wrap">
                <span className="text-white font-extrabold shrink-0">garaj@security-node:~$&nbsp;</span>
                <span className="text-zinc-100 font-bold break-all">{demoCommands[cmdIdx]}</span>
              </div>

              <div className="pl-3 space-y-1 text-zinc-300 border-l-2 border-zinc-800 break-words">
                {(demoOutputs[cmdIdx] || []).map((outLine, lineIdx) => (
                  <div key={lineIdx} className="flex items-start gap-2 break-all">
                    <span>{outLine}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {currentCmdIndex < demoCommands.length && (
            <div className="space-y-1.5 break-words">
              <div className="flex items-start gap-1.5 text-white flex-wrap">
                <span className="text-white font-extrabold shrink-0">garaj@security-node:~$&nbsp;</span>
                <span className="text-zinc-100 font-bold break-all">{displayedText}</span>
                <span className="w-2 h-4 bg-white animate-pulse inline-block shrink-0" />
              </div>
            </div>
          )}

          <div className="pt-3 border-t border-zinc-800 space-y-1 text-[10px] sm:text-[11px]">
            <div className="text-zinc-500 font-bold uppercase tracking-wider mb-2">
              // Live Telemetry Stream:
            </div>
            {logs.slice(0, 4).map((log) => (
              <div key={log.id} className="flex items-start gap-2 text-zinc-400 break-all">
                <span className="text-zinc-600 shrink-0">[{log.timestamp}]</span>
                <span className={log.level === 'error' ? 'text-red-400' : 'text-zinc-200'}>
                  {log.text}
                </span>
              </div>
            ))}
          </div>

          <div ref={terminalEndRef} />
        </div>

        <div className="px-4 py-2.5 bg-zinc-100 border-t border-zinc-200 flex items-center justify-between text-[10px] sm:text-[11px] text-zinc-600 font-mono gap-2">
          <div className="flex items-center gap-2 truncate">
            <span className="w-2 h-2 rounded-full bg-zinc-950 animate-pulse shrink-0" />
            <span className="font-bold text-zinc-950 truncate">STATUS: {isMonitoring ? 'STREAM_ACTIVE' : 'PAUSED'}</span>
          </div>
          <span className="font-bold text-zinc-950 shrink-0">PROCESS ID: 82940</span>
        </div>
      </div>
    </section>
  );
};

export default LiveTerminal;
