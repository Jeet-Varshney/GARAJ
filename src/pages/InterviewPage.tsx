import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { useInterview } from '../hooks/useInterview';
import { useElevenLabs } from '../hooks/useElevenLabs';
import { PhaseTracker } from '../components/PhaseTracker';
import { Transcript } from '../components/Transcript';
import { Button } from '../components/Button';
import { ErrorBanner } from '../components/ErrorBanner';
import { LoadingSpinner } from '../components/LoadingSpinner';
import {
  Mic,
  MicOff,
  PhoneOff,
  Radio,
  Send,
  ChevronRight,
  Bot,
  Volume2,
  UserCheck,
  Zap
} from 'lucide-react';

export const InterviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { session, isMicMuted, toggleMic, selectedVoice, customAgentId, error, setError } = useApp();
  const {
    transcript: currentTranscriptMessages,
    selectedRole,
    currentPhase,
    currentPhaseIndex,
    elapsedSeconds,
    advancePhase,
    endInterviewSession,
    addMessage,
  } = useInterview();

  const [textInput, setTextInput] = useState<string>('');

  const {
    status,
    isSpeaking,
    isListening,
    isTranscribing,
    errorMessage,
    isSimulationMode,
    audioLevel,
    startSession,
    endSession,
    speakAgentResponse,
    listenAndTranscribe,
  } = useElevenLabs({
    onMessage: (role, text) => {
      addMessage(role, text);
    },
    onError: (err) => {
      setError(err);
    },
  });

  // Redirect if no active session
  useEffect(() => {
    if (!session) {
      navigate('/');
    }
  }, [session, navigate]);

  // Connect voice session on mount when session is active
  useEffect(() => {
    if (session) {
      startSession(customAgentId);
    }
    return () => {
      endSession();
    };
  }, [session]);

  const handleFinishInterview = () => {
    endSession();
    endInterviewSession();
    navigate('/report');
  };

  const handleSendTextMessage = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!textInput.trim()) return;

    const userText = textInput.trim();
    addMessage('candidate', userText);
    setTextInput('');

    // Trigger dynamic AI speech response if in simulation mode
    if (isSimulationMode) {
      setTimeout(() => {
        let aiReply = "Thank you for that explanation. Could you detail the key technical trade-offs you considered during implementation?";
        if (currentPhaseIndex === 1) {
          aiReply = `Great approach for ${selectedRole.title}. How would you optimize the data fetching and caching layer under peak traffic?`;
        } else if (currentPhaseIndex === 2) {
          aiReply = "That makes sense. How do you handle failure recovery and circuit breaking in this design?";
        } else if (currentPhaseIndex >= 3) {
          aiReply = "Understood! Let's advance to the next phase of our technical interview session.";
        }

        speakAgentResponse(aiReply);
      }, 1200);
    }
  };

  const handleManualAdvancePhase = () => {
    advancePhase();
    if (isSimulationMode && currentPhaseIndex < 6) {
      const nextPhaseName = ['Core Architecture', 'Technical Deep-Dive', 'Problem Solving & Edge Cases', 'Behavioral & Team Dynamics', 'Candidate Questions', 'Session Wrap-up'][currentPhaseIndex] || 'Next Topic';
      setTimeout(() => {
        speakAgentResponse(`Moving to Phase ${currentPhaseIndex + 2}: ${nextPhaseName}. Could you share your technical approach for this area?`);
      }, 800);
    }
  };

  if (!session) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background text-textPrimary flex flex-col justify-between p-4 md:p-6 selection:bg-indigo-500 selection:text-white">
      {/* Top Header */}
      <header className="max-w-7xl w-full mx-auto flex items-center justify-between py-3 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400">
            <Radio className="w-4 h-4 animate-pulse" />
          </div>
          <div className="flex items-baseline gap-2">
            <h2 className="text-sm font-bold text-slate-100">{selectedRole.title}</h2>
            <span className="text-[10px] font-mono text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/30 font-semibold flex items-center gap-1">
              <UserCheck className="w-3 h-3" /> Voice: {selectedVoice.name}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="danger"
            size="sm"
            onClick={handleFinishInterview}
            leftIcon={<PhoneOff className="w-3.5 h-3.5" />}
          >
            End & Evaluate Session
          </Button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl w-full mx-auto my-4 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: AI Voice Visualizer & Controls (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
          {/* Phase Tracker */}
          <PhaseTracker
            currentPhaseIndex={currentPhaseIndex}
            totalDurationSeconds={currentPhase.durationSeconds}
            elapsedSeconds={elapsedSeconds}
          />

          {/* AI Avatar Voice Hub with Kinetic Glow */}
          <div className="taste-card p-6 flex flex-col items-center justify-center text-center relative overflow-hidden flex-1 min-h-[340px] border-indigo-500/20">
            {/* Ambient Background Glow */}
            <div className={`absolute inset-0 transition-opacity duration-1000 ${
              isSpeaking ? 'bg-indigo-600/15 opacity-100' :
              (audioLevel > 0.1 || isTranscribing) ? 'bg-emerald-600/15 opacity-100' :
              'bg-transparent opacity-0'
            }`} />

            {/* Avatar Kinetic Pulse Container */}
            <div className="relative mb-6">
              {isSpeaking && (
                <div className="absolute inset-0 rounded-full voice-pulse-ring" />
              )}

              <div className={`w-28 h-28 rounded-full flex items-center justify-center transition-all duration-300 border-2 relative z-10 shadow-2xl ${
                isSpeaking
                  ? 'bg-purple-600 border-purple-300 text-white shadow-purple-500/40 scale-105'
                  : (audioLevel > 0.1 || isTranscribing)
                  ? 'bg-emerald-600 border-emerald-400 text-white shadow-emerald-500/40 scale-105'
                  : status === 'connecting'
                  ? 'bg-slate-900 border-slate-800 text-slate-500'
                  : 'bg-slate-900 border-purple-500/40 text-purple-400'
              }`}>
                <Bot className="w-14 h-14" />
              </div>
            </div>

            {/* Voice Status Indicators */}
            <div className="space-y-2 z-10">
              <div className="flex items-center justify-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${
                  isSpeaking ? 'bg-purple-400 animate-ping' :
                  (audioLevel > 0.1 || isTranscribing) ? 'bg-emerald-400 animate-ping' :
                  status === 'connected' ? 'bg-emerald-400 animate-pulse' :
                  status === 'connecting' ? 'bg-amber-400 animate-ping' :
                  'bg-red-400'
                }`} />
                <h3 className="text-base font-bold text-slate-100 tracking-tight">
                  {isSpeaking ? `${selectedVoice.name.split(' ')[0]} Speaking...` :
                   isTranscribing ? 'Transcribing Your Voice Response...' :
                   audioLevel > 0.1 ? 'Receiving Candidate Voice Input...' :
                   isListening ? 'Listening... Speak into your mic or click Speak Mic below!' :
                   status === 'connecting' ? 'Connecting Voice Stream...' :
                   'Voice Engine Ready'}
                </h3>
              </div>

              <p className="text-xs font-mono text-slate-400 flex items-center justify-center gap-1">
                {(audioLevel > 0.1 || isTranscribing) ? (
                  <span className="text-emerald-400 font-bold flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5" /> Mic Input Active (Recording Speech)
                  </span>
                ) : (
                  <>
                    <Volume2 className="w-3.5 h-3.5 text-purple-400" />
                    {isSimulationMode ? `Voice Synthesizer: ${selectedVoice.name}` : 'ElevenLabs WebSocket Active'}
                  </>
                )}
              </p>
            </div>

            {/* Audio Waveform */}
            <div className="mt-6 w-full max-w-xs">
              <LoadingSpinner type="waveform" label="" />
            </div>
          </div>

          {/* Error Banner */}
          {(errorMessage || error) && (
            <ErrorBanner
              message={errorMessage || error || ''}
              onDismiss={() => setError(null)}
              onRetry={() => startSession(customAgentId)}
            />
          )}

          {/* Voice Controls Bar */}
          <div className="taste-card p-4 flex items-center justify-between gap-3">
            <Button
              variant={isMicMuted ? 'danger' : 'secondary'}
              size="md"
              onClick={toggleMic}
              leftIcon={isMicMuted ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4 text-emerald-400" />}
            >
              {isMicMuted ? 'Mic Muted' : 'Mic Active'}
            </Button>

            <Button
              variant="outline"
              size="md"
              onClick={handleManualAdvancePhase}
              disabled={currentPhaseIndex >= 6}
              rightIcon={<ChevronRight className="w-4 h-4" />}
            >
              Next Phase
            </Button>
          </div>
        </div>

        {/* Right Column: Live Transcript & Text Input (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-4">
          <div className="flex-1">
            <Transcript messages={currentTranscriptMessages} isLive={true} />
          </div>

          {/* Push-to-Speak Voice Button & Text Input Bar */}
          <form onSubmit={handleSendTextMessage} className="taste-card p-3 flex items-center gap-2">
            <Button
              type="button"
              variant={isTranscribing ? 'danger' : 'primary'}
              size="md"
              onClick={listenAndTranscribe}
              leftIcon={<Mic className={`w-4 h-4 ${isTranscribing ? 'animate-bounce' : ''}`} />}
              className="shrink-0 font-bold"
            >
              {isTranscribing ? 'Listening...' : 'Speak Mic'}
            </Button>

            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Click 'Speak Mic' or type your answer here..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-btn px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans"
            />

            <Button
              type="submit"
              variant="secondary"
              size="md"
              rightIcon={<Send className="w-4 h-4" />}
            >
              Send
            </Button>
          </form>
        </div>
      </main>
    </div>
  );
};
