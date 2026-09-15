export interface TranscriptMessage {
  role: "agent" | "candidate";
  text: string;
  timestamp: number; // Seconds since interview start
}

export interface InterviewSession {
  sessionId: string;
  agentId: string;
  role: string;
  status: "pending" | "in-progress" | "completed";
  startedAt: Date;
  endedAt?: Date;
  duration?: number;
  currentPhaseIndex: number;
}

export interface EvaluationScore {
  score: number; // 1-5
  maxScore: number;
}

export interface Evaluation {
  technicalSkills: EvaluationScore;
  communication: EvaluationScore;
  behavioral: EvaluationScore;
  approach: EvaluationScore;
  overallScore: number;
  summary: string;
  strengths: string[];
  improvements: string[];
}

export interface InterviewResult {
  sessionId: string;
  role: string;
  transcript: TranscriptMessage[];
  evaluation: Evaluation;
  completedAt: Date;
  duration: number;
}

export interface InterviewPhase {
  id: number;
  name: string;
  description: string;
  durationSeconds: number;
}

export type CandidateRole = {
  id: string;
  title: string;
  description: string;
  iconName: string;
  level: "Junior" | "Mid-Level" | "Senior" | "Lead";
  sampleQuestions: string[];
};

export type VoiceConnectionStatus = "disconnected" | "connecting" | "connected" | "speaking" | "listening" | "error";
