import React, { useRef, useEffect } from 'react';
import type { TranscriptMessage } from '../types';
import { Bot, User, MessageSquare } from 'lucide-react';

interface TranscriptProps {
  messages: TranscriptMessage[];
  isLive?: boolean;
}

export const Transcript: React.FC<TranscriptProps> = ({ messages, isLive = true }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isLive) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLive]);

  const formatTimestamp = (totalSeconds: number) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (messages.length === 0) {
    return (
      <div className="glass-panel p-8 rounded-card flex flex-col items-center justify-center text-center min-h-[300px]">
        <div className="w-12 h-12 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mb-3 text-slate-400">
          <MessageSquare className="w-6 h-6 text-indigo-400" />
        </div>
        <h4 className="text-base font-semibold text-slate-200 mb-1">Transcript Empty</h4>
        <p className="text-xs text-slate-400 max-w-sm">
          Conversation responses will appear here live once the voice session starts.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-4 rounded-card flex flex-col h-full max-h-[500px]">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-700/80">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">Live Transcript</h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          {messages.length} {messages.length === 1 ? 'entry' : 'entries'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 space-y-3">
        {messages.map((msg, index) => {
          const isAgent = msg.role === 'agent';

          return (
            <div
              key={index}
              className={`flex items-start gap-3 transition-opacity duration-300 ${
                isAgent ? 'flex-row' : 'flex-row-reverse'
              }`}
            >
              {/* Avatar */}
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
                  isAgent
                    ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-400'
                    : 'bg-emerald-600/20 border-emerald-500/40 text-emerald-400'
                }`}
              >
                {isAgent ? <Bot className="w-4 h-4" /> : <User className="w-4 h-4" />}
              </div>

              {/* Message Content */}
              <div className={`max-w-[80%] flex flex-col ${isAgent ? 'items-start' : 'items-end'}`}>
                <div className="flex items-center gap-2 mb-1 text-[11px]">
                  <span className={`font-semibold ${isAgent ? 'text-indigo-400' : 'text-emerald-400'}`}>
                    {isAgent ? 'AI Interviewer' : 'Candidate'}
                  </span>
                  <span className="text-slate-500">•</span>
                  <span className="font-mono text-slate-400">{formatTimestamp(msg.timestamp)}</span>
                </div>

                <div
                  className={`p-3 rounded-2xl text-sm leading-relaxed ${
                    isAgent
                      ? 'bg-slate-800 text-slate-100 border border-slate-700/80 rounded-tl-none'
                      : 'bg-indigo-600/90 text-white rounded-tr-none shadow-md shadow-indigo-600/10'
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};
