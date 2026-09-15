import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { ScoreCard } from '../components/ScoreCard';
import { Transcript } from '../components/Transcript';
import { Button } from '../components/Button';
import { LoadingSpinner } from '../components/LoadingSpinner';
import {
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Code2,
  MessageSquare,
  Sparkles,
  TrendingUp,
  Brain
} from 'lucide-react';

export const ReportPage: React.FC = () => {
  const navigate = useNavigate();
  const { session, evaluation, transcript, selectedRole, resetState } = useApp();
  const [activeTab, setActiveTab] = useState<'overview' | 'transcript'>('overview');

  const handleStartNewInterview = () => {
    resetState();
    navigate('/');
  };

  if (!evaluation) {
    return (
      <div className="min-h-screen bg-background text-textPrimary flex flex-col items-center justify-center p-6">
        <div className="max-w-md w-full text-center space-y-4">
          <LoadingSpinner type="reportSkeleton" />
          <h3 className="text-lg font-bold text-slate-200">Synthesizing Voice Evaluation...</h3>
          <p className="text-xs text-slate-400">Analyzing speech clarity, technical depth, and behavioral alignment.</p>
          <Button variant="secondary" size="sm" onClick={handleStartNewInterview}>
            Back to Role Selection
          </Button>
        </div>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 85) return 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10';
    if (score >= 70) return 'text-indigo-400 border-indigo-500/40 bg-indigo-500/10';
    return 'text-amber-400 border-amber-500/40 bg-amber-500/10';
  };

  return (
    <div className="min-h-screen bg-background text-textPrimary flex flex-col justify-between p-4 md:p-8 selection:bg-indigo-500 selection:text-white">
      {/* Design Read: AI Voice Interview Simulator Performance Evaluation Report */}

      {/* Header Bar */}
      <header className="max-w-7xl w-full mx-auto flex items-center justify-between py-4 border-b border-slate-800/80">
        <div>
          <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-indigo-400">Voice AI Performance Report</span>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2 tracking-tight">
            {selectedRole.title} <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-normal border border-slate-800">{selectedRole.level}</span>
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="primary"
            size="md"
            onClick={handleStartNewInterview}
            leftIcon={<RotateCcw className="w-4 h-4" />}
          >
            Start New Interview
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl w-full mx-auto my-8 space-y-8">
        {/* Banner Score Summary (Taste-Skill Hero Metric Card) */}
        <section className="taste-card p-6 border-indigo-500/30 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl">
          <div className="flex items-center gap-6">
            <div className={`w-24 h-24 rounded-full border-4 flex flex-col items-center justify-center font-bold shrink-0 shadow-lg ${getScoreColor(evaluation.overallScore)}`}>
              <span className="text-3xl font-black font-mono tracking-tight">{evaluation.overallScore}%</span>
              <span className="text-[10px] font-mono uppercase font-bold text-slate-400">OVERALL</span>
            </div>

            <div className="space-y-1.5">
              <div className="inline-flex items-center gap-1.5 text-xs font-mono text-indigo-400 font-bold uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5" /> Voice Evaluation Complete
              </div>
              <h2 className="text-xl font-bold text-slate-100 tracking-tight">Candidate Performance Summary</h2>
              <p className="text-xs text-slate-300 max-w-xl leading-relaxed">
                {evaluation.summary}
              </p>
            </div>
          </div>

          {/* Quick Metrics Badge */}
          <div className="flex items-center gap-4 border-t md:border-t-0 md:border-l border-slate-800 pt-4 md:pt-0 md:pl-6">
            <div className="text-center">
              <span className="text-2xl font-black font-mono text-slate-100">{transcript.length}</span>
              <p className="text-[11px] font-mono text-slate-400">MESSAGES</p>
            </div>
            <div className="w-px h-8 bg-slate-800" />
            <div className="text-center">
              <span className="text-2xl font-black font-mono text-slate-100">{session?.duration ? `${Math.round(session.duration / 60)}m` : '5m'}</span>
              <p className="text-[11px] font-mono text-slate-400">DURATION</p>
            </div>
          </div>
        </section>

        {/* Tab Navigation */}
        <div className="flex items-center gap-3 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-2 text-xs font-bold font-mono uppercase tracking-wider rounded-btn transition-all ${
              activeTab === 'overview'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            Performance Metrics
          </button>
          <button
            onClick={() => setActiveTab('transcript')}
            className={`px-4 py-2 text-xs font-bold font-mono uppercase tracking-wider rounded-btn transition-all flex items-center gap-2 ${
              activeTab === 'transcript'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" /> Full Transcript Review ({transcript.length})
          </button>
        </div>

        {activeTab === 'overview' ? (
          <>
            {/* 4 Score Cards Bento Grid */}
            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <ScoreCard
                title="Technical Skills"
                description="Core domain knowledge & architecture"
                score={evaluation.technicalSkills}
                icon={<Code2 className="w-5 h-5 text-indigo-400" />}
              />
              <ScoreCard
                title="Communication"
                description="Speech clarity & concise responses"
                score={evaluation.communication}
                icon={<MessageSquare className="w-5 h-5 text-emerald-400" />}
              />
              <ScoreCard
                title="Behavioral"
                description="Team conflict & leadership alignment"
                score={evaluation.behavioral}
                icon={<Brain className="w-5 h-5 text-purple-400" />}
              />
              <ScoreCard
                title="Problem Approach"
                description="Structured trade-offs & edge cases"
                score={evaluation.approach}
                icon={<TrendingUp className="w-5 h-5 text-amber-400" />}
              />
            </section>

            {/* Strengths & Areas for Improvement Split */}
            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Key Strengths */}
              <div className="taste-card p-6 space-y-4 border-emerald-500/20">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                  <h3 className="text-sm font-bold text-slate-100 tracking-tight uppercase font-mono">Key Strengths Identified</h3>
                </div>
                <ul className="space-y-3">
                  {evaluation.strengths.map((str, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-xs text-slate-200 leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                      <span>{str}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Areas for Improvement */}
              <div className="taste-card p-6 space-y-4 border-amber-500/20">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
                  <h3 className="text-sm font-bold text-slate-100 tracking-tight uppercase font-mono">Growth & Improvement Areas</h3>
                </div>
                <ul className="space-y-3">
                  {evaluation.improvements.map((imp, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-xs text-slate-200 leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                      <span>{imp}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          </>
        ) : (
          <section className="space-y-4">
            <Transcript messages={transcript} isLive={false} />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="max-w-7xl w-full mx-auto text-center text-xs font-mono text-slate-500 py-4 border-t border-slate-800/80">
        VoiceAI Simulator • Evaluation generated on {new Date().toLocaleDateString()}
      </footer>
    </div>
  );
};
