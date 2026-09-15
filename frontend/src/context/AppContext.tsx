import React, { createContext, useContext, useState } from 'react';
import type { InterviewSession, TranscriptMessage, Evaluation, CandidateRole } from '../types';

export interface VoiceOption {
  id: string;
  name: string;
  gender: 'female' | 'male';
  description: string;
}

export const TARGET_AGENT_ID = 'agent_0001kz72n4w1eexsdt5yvrv89rm3';

export const VOICE_OPTIONS: VoiceOption[] = [
  { id: 'elevenlabs-agent', name: 'ElevenLabs Voice Agent', gender: 'female', description: 'Real-time ElevenLabs Conversational Voice Agent' },
  { id: 'female-monika', name: 'Monika (Female)', gender: 'female', description: 'Clear, articulate female engineering interviewer voice' },
  { id: 'female-samantha', name: 'Samantha / Zira (Female)', gender: 'female', description: 'Executive female tech lead interviewer voice' },
];

export const DEFAULT_ROLES: CandidateRole[] = [
  {
    id: 'frontend-sr',
    title: 'Senior Frontend Engineer',
    description: 'React, TypeScript, System Design, State Management, Web Performance & Accessibility.',
    iconName: 'Code',
    level: 'Senior',
    sampleQuestions: [
      'How do you architect large scale React application state?',
      'Explain how you optimize Web Vitals like LCP and CLS.',
      'Walk me through designing a real-time collaborative code editor.'
    ]
  },
  {
    id: 'fullstack-mid',
    title: 'Fullstack Engineer',
    description: 'Node.js, GraphQL, PostgreSQL, Frontend frameworks, API contracts & microservices.',
    iconName: 'Layers',
    level: 'Mid-Level',
    sampleQuestions: [
      'How do you design REST vs GraphQL endpoints for high-throughput mobile apps?',
      'Describe database indexing strategies for heavy read workloads.',
      'How do you handle authentication across frontend and backend services?'
    ]
  },
  {
    id: 'sys-design-lead',
    title: 'Distributed Systems Lead',
    description: 'System Architecture, Scalability, High Availability, Message Queues & Consensus.',
    iconName: 'Cpu',
    level: 'Lead',
    sampleQuestions: [
      'Design a global rate limiter handling 1,000,000 requests per second.',
      'Compare Event Sourcing with CQRS for transactional reliability.',
      'How do you prevent cascading failures in distributed microservices?'
    ]
  },
  {
    id: 'product-mgr',
    title: 'Technical Product Manager',
    description: 'Product roadmap, AI feature specs, User Metrics, Stakeholder Management & Strategy.',
    iconName: 'Briefcase',
    level: 'Senior',
    sampleQuestions: [
      'How do you prioritize non-functional engineering debt vs revenue features?',
      'Walk through how you define metrics for a new AI voice assistant product.',
      'How do you resolve architectural disputes between frontend and backend leads?'
    ]
  }
];

export const INTERVIEW_PHASES = [
  { id: 1, name: 'Warmup & Intro', description: 'Background summary & experience overview', durationSeconds: 60 },
  { id: 2, name: 'Core Architecture', description: 'High-level system & component design', durationSeconds: 180 },
  { id: 3, name: 'Technical Deep-Dive', description: 'Code implementation & trade-off decisions', durationSeconds: 300 },
  { id: 4, name: 'Problem Solving & Edge Cases', description: 'Performance bottlenecks & security', durationSeconds: 240 },
  { id: 5, name: 'Behavioral & Team Dynamics', description: 'Handling conflict & ambiguous requirements', durationSeconds: 180 },
  { id: 6, name: 'Candidate Questions', description: 'Questions for the interviewer/team', durationSeconds: 120 },
  { id: 7, name: 'Session Wrap-up', description: 'Feedback capture & closing remarks', durationSeconds: 60 },
];

interface AppContextType {
  session: InterviewSession | null;
  transcript: TranscriptMessage[];
  evaluation: Evaluation | null;
  selectedRole: CandidateRole;
  selectedVoice: VoiceOption;
  customAgentId: string;
  isMicMuted: boolean;
  error: string | null;
  setSelectedRole: (role: CandidateRole) => void;
  setSelectedVoice: (voice: VoiceOption) => void;
  setCustomAgentId: (id: string) => void;
  startSession: () => void;
  endSession: () => void;
  addTranscriptMessage: (message: TranscriptMessage) => void;
  setEvaluation: (evalResult: Evaluation) => void;
  toggleMic: () => void;
  setError: (err: string | null) => void;
  resetState: () => void;
  updatePhaseIndex: (index: number) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedRole, setSelectedRole] = useState<CandidateRole>(DEFAULT_ROLES[0]);
  const [selectedVoice, setSelectedVoice] = useState<VoiceOption>(VOICE_OPTIONS[0]);
  const [customAgentId, setCustomAgentId] = useState<string>(TARGET_AGENT_ID);
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [isMicMuted, setIsMicMuted] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const startSession = () => {
    const newSession: InterviewSession = {
      sessionId: `session-${Date.now().toString(36)}`,
      agentId: customAgentId || TARGET_AGENT_ID,
      role: selectedRole.title,
      status: 'in-progress',
      startedAt: new Date(),
      currentPhaseIndex: 0,
    };
    setSession(newSession);
    setTranscript([]);
    setEvaluation(null);
    setError(null);
  };

  const updatePhaseIndex = (index: number) => {
    setSession(prev => prev ? { ...prev, currentPhaseIndex: index } : null);
  };

  const endSession = () => {
    if (!session) return;
    const endedAt = new Date();
    const duration = Math.round((endedAt.getTime() - session.startedAt.getTime()) / 1000);
    
    setSession(prev => prev ? {
      ...prev,
      status: 'completed',
      endedAt,
      duration,
    } : null);
  };

  const addTranscriptMessage = (msg: TranscriptMessage) => {
    setTranscript(prev => [...prev, msg]);
  };

  const toggleMic = () => {
    setIsMicMuted(prev => !prev);
  };

  const resetState = () => {
    setSession(null);
    setTranscript([]);
    setEvaluation(null);
    setError(null);
  };

  return (
    <AppContext.Provider value={{
      session,
      transcript,
      evaluation,
      selectedRole,
      selectedVoice,
      customAgentId,
      isMicMuted,
      error,
      setSelectedRole,
      setSelectedVoice,
      setCustomAgentId,
      startSession,
      endSession,
      addTranscriptMessage,
      setEvaluation,
      toggleMic,
      setError,
      resetState,
      updatePhaseIndex,
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
