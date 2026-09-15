import React, { useState } from 'react';
import { useConversation } from '@elevenlabs/react';
import { LoadingSpinner } from '../components/LoadingSpinner';
import {
  Mic,
  PhoneOff,
  Radio,
  Bot,
  Volume2,
  Sparkles,
  Copy,
  Check,
  Code2,
  ExternalLink
} from 'lucide-react';

export const TARGET_AGENT_ID = 'agent_0001kz72n4w1eexsdt5yvrv89rm3';

interface TranscriptItem {
  id: string;
  sender: 'agent' | 'user';
  text: string;
  timestamp: string;
}

export const BasicAgentPage: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'custom' | 'widget'>('custom');
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [statusText, setStatusText] = useState<string>('Disconnected');

  const conversation = useConversation({
    onConnect: () => {
      setStatusText('Connected & Ready');
    },
    onDisconnect: () => {
      setStatusText('Disconnected');
    },
    onMessage: (message: any) => {
      if (message?.message) {
        const sender = message.source === 'user' ? 'user' : 'agent';
        setTranscript((prev) => [
          ...prev,
          {
            id: `msg-${Date.now()}-${Math.random()}`,
            sender,
            text: message.message,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          },
        ]);
      }
    },
    onError: (err: any) => {
      console.error('ElevenLabs agent connection error:', err);
      setStatusText(`Error: ${typeof err === 'string' ? err : err?.message || 'Connection failed'}`);
    },
  });

  const isConnected = conversation.status === 'connected';
  const isConnecting = conversation.status === 'connecting';
  const isSpeaking = conversation.isSpeaking;

  const handleStartAgent = async () => {
    try {
      setStatusText('Connecting to Agent...');
      await conversation.startSession({
        agentId: TARGET_AGENT_ID,
      });
    } catch (err: any) {
      console.error('Failed to start ElevenLabs agent:', err);
      setStatusText(`Failed: ${err?.message || 'Check microphone or API credentials'}`);
    }
  };

  const handleStopAgent = async () => {
    try {
      await conversation.endSession();
      setStatusText('Disconnected');
    } catch (err) {
      console.error('Failed to stop agent session:', err);
    }
  };

  const copyAgentId = () => {
    navigator.clipboard.writeText(TARGET_AGENT_ID);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-background text-textPrimary flex flex-col justify-between p-4 md:p-8 selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <header className="max-w-4xl w-full mx-auto flex items-center justify-between py-4 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-lg shadow-indigo-600/10 shrink-0">
            <Radio className="w-5 h-5 animate-pulse text-indigo-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              ElevenLabs Voice Agent
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30 font-semibold">
                LIVE
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-mono flex items-center gap-1.5 mt-0.5">
              <span>Agent ID:</span>
              <code className="text-indigo-300 font-semibold bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                {TARGET_AGENT_ID}
              </code>
              <button
                onClick={copyAgentId}
                className="hover:text-white transition-colors p-0.5 rounded"
                title="Copy Agent ID"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
              </button>
            </p>
          </div>
        </div>

        {/* Navigation links */}
        <div className="flex items-center gap-3">
          <a
            href="/simulator"
            className="text-xs text-slate-400 hover:text-indigo-300 font-mono transition-colors bg-slate-900/90 px-3 py-1.5 rounded-full border border-slate-800"
          >
            Full Mock Simulator →
          </a>
          <a
            href={`https://elevenlabs.io/app/talk-to?agent_id=${TARGET_AGENT_ID}`}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 hover:text-indigo-300 transition-colors bg-slate-900/90 px-3 py-1.5 rounded-full border border-slate-800 font-mono"
          >
            <span>ElevenLabs Dashboard</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-4xl w-full mx-auto my-8 flex-1 flex flex-col justify-center items-center space-y-8">
        
        {/* Mode Selector Tabs */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('custom')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'custom'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            1-Click Direct Launcher
          </button>
          <button
            onClick={() => setActiveTab('widget')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'widget'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Official ElevenLabs Web Widget
          </button>
        </div>

        {activeTab === 'custom' ? (
          <div className="w-full max-w-xl space-y-8 text-center">
            {/* Visualizer & Avatar Box */}
            <div className="taste-card p-8 flex flex-col items-center justify-center relative overflow-hidden border-indigo-500/30 shadow-2xl">
              {/* Background ambient glow */}
              <div
                className={`absolute inset-0 transition-opacity duration-1000 ${
                  isSpeaking
                    ? 'bg-purple-600/15 opacity-100'
                    : isConnected
                    ? 'bg-indigo-600/10 opacity-100'
                    : 'bg-transparent opacity-0'
                }`}
              />

              {/* Bot Icon with Animated Rings */}
              <div className="relative mb-6">
                {(isSpeaking || isConnected) && (
                  <div className="absolute inset-0 rounded-full voice-pulse-ring" />
                )}

                <div
                  className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 border-2 relative z-10 shadow-2xl ${
                    isSpeaking
                      ? 'bg-purple-600 border-purple-300 text-white shadow-purple-500/50 scale-105'
                      : isConnected
                      ? 'bg-indigo-600 border-indigo-400 text-white shadow-indigo-500/50'
                      : isConnecting
                      ? 'bg-amber-600 border-amber-400 text-white animate-pulse'
                      : 'bg-slate-900 border-indigo-500/30 text-indigo-400'
                  }`}
                >
                  <Bot className="w-16 h-16" />
                </div>
              </div>

              {/* Status Badge */}
              <div className="space-y-2 z-10">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-mono">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      isSpeaking
                        ? 'bg-purple-400 animate-ping'
                        : isConnected
                        ? 'bg-emerald-400 animate-pulse'
                        : isConnecting
                        ? 'bg-amber-400 animate-ping'
                        : 'bg-slate-500'
                    }`}
                  />
                  <span className="font-semibold text-slate-200">
                    {isSpeaking
                      ? 'Agent Speaking...'
                      : isConnected
                      ? 'Agent Listening — Speak Now!'
                      : isConnecting
                      ? 'Connecting WebSocket...'
                      : 'Agent Disconnected'}
                  </span>
                </div>
                <p className="text-xs text-slate-400">{statusText}</p>
              </div>

              {/* Audio Waveform while connected */}
              {isConnected && (
                <div className="mt-6 w-full max-w-xs z-10">
                  <LoadingSpinner type="waveform" label="" />
                </div>
              )}
            </div>

            {/* THE MAIN BIG START / STOP AGENT BUTTON */}
            <div className="pt-2 flex flex-col items-center gap-4">
              {!isConnected && !isConnecting ? (
                <button
                  onClick={handleStartAgent}
                  className="group relative inline-flex items-center justify-center gap-3 px-10 py-5 rounded-2xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 text-white text-xl font-extrabold shadow-2xl shadow-indigo-600/40 hover:shadow-indigo-500/60 hover:scale-105 active:scale-95 transition-all duration-200 border border-indigo-400/40 cursor-pointer"
                >
                  <span className="w-4 h-4 rounded-full bg-emerald-400 animate-ping absolute top-4 left-6" />
                  <Mic className="w-7 h-7 text-white animate-bounce" />
                  <span>START AGENT</span>
                  <Sparkles className="w-5 h-5 text-purple-200 group-hover:rotate-12 transition-transform" />
                </button>
              ) : isConnecting ? (
                <button
                  disabled
                  className="inline-flex items-center justify-center gap-3 px-10 py-5 rounded-2xl bg-slate-800 text-amber-300 text-lg font-bold border border-slate-700 cursor-not-allowed opacity-80"
                >
                  <LoadingSpinner type="spinner" size="md" label="Connecting Agent..." />
                </button>
              ) : (
                <button
                  onClick={handleStopAgent}
                  className="inline-flex items-center justify-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-red-600 to-rose-700 text-white text-lg font-bold shadow-xl shadow-red-600/30 hover:shadow-red-600/50 hover:scale-105 active:scale-95 transition-all duration-200 border border-red-400/30 cursor-pointer"
                >
                  <PhoneOff className="w-6 h-6" />
                  <span>STOP AGENT</span>
                </button>
              )}

              <p className="text-xs text-slate-400 font-mono">
                Click <strong className="text-indigo-300">START AGENT</strong> to grant mic permissions & launch real-time voice stream.
              </p>
            </div>

            {/* Live Transcript Box */}
            {transcript.length > 0 && (
              <div className="taste-card p-4 text-left space-y-3 max-h-64 overflow-y-auto">
                <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-indigo-400" /> Real-Time Live Transcript
                </h3>
                <div className="space-y-2">
                  {transcript.map((msg) => (
                    <div
                      key={msg.id}
                      className={`p-2.5 rounded-lg text-xs leading-relaxed ${
                        msg.sender === 'agent'
                          ? 'bg-indigo-950/50 border border-indigo-500/30 text-indigo-100'
                          : 'bg-slate-900 border border-slate-800 text-emerald-300'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[10px] font-mono opacity-75 mb-1">
                        <span className="font-bold uppercase">{msg.sender === 'agent' ? '🤖 ElevenLabs Agent' : '👤 You'}</span>
                        <span>{msg.timestamp}</span>
                      </div>
                      <p>{msg.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Official ElevenLabs Web Widget Container */
          <div className="w-full max-w-lg taste-card p-6 flex flex-col items-center justify-center text-center space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Code2 className="w-5 h-5 text-indigo-400" /> Embedded Official ElevenLabs Web Component
            </h3>
            <p className="text-xs text-slate-400">
              Below is the native ElevenLabs web widget element (<code className="text-indigo-300">&lt;elevenlabs-convai&gt;</code>).
            </p>

            {/* Custom Web Element for ElevenLabs ConvAI */}
            <div className="w-full flex justify-center py-4">
              {React.createElement('elevenlabs-convai', {
                'agent-id': TARGET_AGENT_ID,
              })}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="max-w-4xl w-full mx-auto text-center text-xs font-mono text-slate-500 py-4 border-t border-slate-800/80">
        ElevenLabs Agent Launcher • Agent ID: {TARGET_AGENT_ID}
      </footer>
    </div>
  );
};

export default BasicAgentPage;
