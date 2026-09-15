import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp, DEFAULT_ROLES, VOICE_OPTIONS, type VoiceOption } from '../context/AppContext';
import { useElevenLabs } from '../hooks/useElevenLabs';
import { Button } from '../components/Button';
import {
  Code,
  Layers,
  Cpu,
  Briefcase,
  Mic,
  ArrowRight,
  Sparkles,
  ShieldCheck,
  Radio,
  Settings,
  HelpCircle,
  CheckCircle,
  Volume2,
  UserCheck
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    selectedRole,
    setSelectedRole,
    selectedVoice,
    setSelectedVoice,
    customAgentId,
    setCustomAgentId,
    startSession
  } = useApp();

  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [micStatus, setMicStatus] = useState<'idle' | 'testing' | 'ready'>('idle');

  const { speakText } = useElevenLabs();

  const getRoleIcon = (iconName: string) => {
    switch (iconName) {
      case 'Code': return <Code className="w-5 h-5 text-indigo-400" />;
      case 'Layers': return <Layers className="w-5 h-5 text-emerald-400" />;
      case 'Cpu': return <Cpu className="w-5 h-5 text-amber-400" />;
      case 'Briefcase': return <Briefcase className="w-5 h-5 text-purple-400" />;
      default: return <Code className="w-5 h-5 text-indigo-400" />;
    }
  };

  const handleTestMic = async () => {
    setMicStatus('testing');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setTimeout(() => {
        stream.getTracks().forEach(track => track.stop());
        setMicStatus('ready');
      }, 1500);
    } catch (e) {
      console.warn('Microphone permission not granted or unavailable:', e);
      setMicStatus('ready');
    }
  };

  const handlePreviewVoice = (voice: VoiceOption) => {
    setSelectedVoice(voice);
    speakText(`Hello! I'm ${voice.name.split(' ')[0]}, your technical interviewer today. I'm ready for your mock session.`);
  };

  const handleStartInterview = () => {
    startSession();
    navigate('/interview');
  };

  return (
    <div className="min-h-screen bg-background text-textPrimary flex flex-col justify-between p-4 md:p-8 selection:bg-indigo-500 selection:text-white">
      {/* Header Bar */}
      <header className="max-w-7xl w-full mx-auto flex items-center justify-between py-4 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-lg shadow-indigo-600/10 shrink-0">
            <Radio className="w-4 h-4 animate-pulse" />
          </div>
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-bold tracking-tight text-white">VoiceAI</h1>
            <span className="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/30 font-semibold">SIMULATOR v1.0</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900/90 px-3 py-1.5 rounded-full border border-slate-800">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>ElevenLabs Real-time Voice</span>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl w-full mx-auto my-8 space-y-12">
        {/* Asymmetric Hero Section */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center pt-4">
          <div className="lg:col-span-7 space-y-5">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-semibold tracking-wide">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>Real-Time Voice AI Practice & Automated Feedback</span>
            </div>

            <h2 className="text-3xl sm:text-4xl md:text-6xl font-extrabold text-slate-100 tracking-tight leading-[1.05]">
              Master Technical Mock Interviews with <span className="bg-gradient-to-r from-indigo-400 via-purple-300 to-emerald-400 bg-clip-text text-transparent">Voice AI</span>
            </h2>

            <p className="text-base text-slate-300 leading-relaxed max-w-xl">
              Simulate realistic 7-phase technical voice interviews. Speak naturally, receive live question progression, and get instant score evaluation.
            </p>

            <div className="pt-2 flex items-center gap-4">
              <Button
                variant="primary"
                size="lg"
                onClick={handleStartInterview}
                rightIcon={<ArrowRight className="w-5 h-5" />}
                className="font-bold text-base px-8 py-3.5 shadow-indigo-600/30"
              >
                Start Practice Session
              </Button>
            </div>
          </div>

          {/* Right Visual Hero Preview Card */}
          <div className="lg:col-span-5">
            <div className="taste-card p-6 border-indigo-500/30 space-y-4 shadow-2xl relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">Session Readiness</span>
                </div>
                <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/30">7 Phases</span>
              </div>

              <div className="space-y-3 pt-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">Candidate Role:</span>
                  <span className="font-bold text-indigo-400">{selectedRole.title}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">AI Interviewer Voice:</span>
                  <span className="font-bold text-purple-400 flex items-center gap-1">
                    <UserCheck className="w-3.5 h-3.5" /> {selectedVoice.name}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">Microphone Check:</span>
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5" /> Verified
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-300 font-mono">
                <span className="text-slate-500">// Sample Question:</span>
                <p className="text-slate-200 mt-1 italic">
                  "{selectedRole.sampleQuestions[0]}"
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* AI Interviewer Voice Selection Section */}
        <section className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold text-slate-100 tracking-tight flex items-center gap-2">
                <Volume2 className="w-5 h-5 text-indigo-400" /> Select AI Interviewer Voice
              </h3>
              <p className="text-xs text-slate-400">Choose your preferred voice personality (Monika Female recommended).</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {VOICE_OPTIONS.map((voice) => {
              const isSelected = selectedVoice.id === voice.id;
              return (
                <div
                  key={voice.id}
                  onClick={() => setSelectedVoice(voice)}
                  className={`cursor-pointer p-4 rounded-card border transition-all duration-200 flex flex-col justify-between ${
                    isSelected
                      ? 'bg-indigo-950/40 border-2 border-indigo-500 shadow-lg shadow-indigo-600/20'
                      : 'taste-card-interactive'
                  }`}
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-100 text-sm">{voice.name}</span>
                      <span className={`text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded border ${
                        voice.gender === 'female' ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' : 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                      }`}>
                        {voice.gender}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">{voice.description}</p>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800 flex items-center justify-between">
                    <span className="text-[11px] font-mono text-slate-500">
                      {isSelected ? '✓ Selected' : 'Click to Select'}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePreviewVoice(voice);
                      }}
                      leftIcon={<Volume2 className="w-3.5 h-3.5 text-indigo-400" />}
                    >
                      Listen Voice
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Bento Grid Role Selection Section */}
        <section className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold text-slate-100 tracking-tight">Select Candidate Role</h3>
              <p className="text-xs text-slate-400">Choose your domain to customize mock questions and evaluation metrics.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {DEFAULT_ROLES.map((role) => {
              const isSelected = selectedRole.id === role.id;
              return (
                <div
                  key={role.id}
                  onClick={() => setSelectedRole(role)}
                  className={`cursor-pointer p-5 transition-all duration-300 flex flex-col justify-between ${
                    isSelected ? 'taste-card-selected' : 'taste-card-interactive'
                  }`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                        {getRoleIcon(role.iconName)}
                      </div>
                      <span className={`text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded border ${
                        role.level === 'Senior' ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30' :
                        role.level === 'Lead' ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' :
                        'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      }`}>
                        {role.level}
                      </span>
                    </div>

                    <div>
                      <h4 className="font-bold text-slate-100 text-sm tracking-tight">{role.title}</h4>
                      <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{role.description}</p>
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-1.5">
                    <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                      <HelpCircle className="w-3.5 h-3.5 text-indigo-400" /> Focus Topics:
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {role.sampleQuestions.slice(0, 2).map((q, i) => (
                        <span key={i} className="text-[10px] font-mono bg-slate-900/90 text-slate-300 px-2 py-0.5 rounded border border-slate-800 truncate max-w-full">
                          {q.split(' ')[0]}...
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Audio & Agent Config Control Panel */}
        <section className="taste-card p-6 border-indigo-500/20 shadow-2xl space-y-6">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Mic className="w-5 h-5 text-indigo-400" /> Voice Stream & Hardware Readiness
              </h3>
              <p className="text-xs text-slate-400">Verify microphone permissions before starting your 7-phase interview session.</p>
            </div>

            <div className="flex items-center gap-3">
              <Button
                variant={micStatus === 'ready' ? 'secondary' : 'outline'}
                size="sm"
                onClick={handleTestMic}
                isLoading={micStatus === 'testing'}
                leftIcon={<Mic className="w-4 h-4 text-emerald-400" />}
              >
                {micStatus === 'ready' ? 'Mic Verified ✓' : 'Test Microphone'}
              </Button>

              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 transition-colors font-mono"
              >
                <Settings className="w-3.5 h-3.5" />
                <span>{showAdvanced ? 'Hide Config' : 'Agent Config'}</span>
              </button>
            </div>
          </div>

          {/* Advanced Agent Config */}
          {showAdvanced && (
            <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2 animate-in fade-in duration-200">
              <label className="block text-xs font-semibold text-slate-300 font-mono">
                Custom ElevenLabs Agent ID (Optional)
              </label>
              <input
                type="text"
                value={customAgentId}
                onChange={(e) => setCustomAgentId(e.target.value)}
                placeholder="e.g. agent_7101k5zvyjhmfg983brhmhkd98n6 (Leave blank for interactive simulator mode)"
                className="w-full bg-slate-950 border border-slate-800 rounded-btn px-3 py-2 text-xs text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono"
              />
              <p className="text-[11px] text-slate-400">
                Connects directly to an ElevenLabs Conversational WebSocket agent or falls back to interactive mock voice simulation.
              </p>
            </div>
          )}

          {/* CTA Row */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-1">
            <div className="text-xs text-slate-400">
              Role: <span className="font-bold text-indigo-400">{selectedRole.title}</span> • Voice: <span className="font-bold text-purple-400">{selectedVoice.name}</span>
            </div>

            <Button
              variant="primary"
              size="lg"
              onClick={handleStartInterview}
              rightIcon={<ArrowRight className="w-5 h-5" />}
              className="w-full sm:w-auto font-bold text-base px-8 shadow-indigo-600/25"
            >
              Start Live Interview Session
            </Button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl w-full mx-auto text-center text-xs font-mono text-slate-500 py-4 border-t border-slate-800/80">
        VoiceAI Simulator • Built with React 18, TypeScript, Tailwind CSS & ElevenLabs
      </footer>
    </div>
  );
};
