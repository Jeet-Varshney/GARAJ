import { useState, useEffect, useCallback, useRef } from 'react';
import { useApp, INTERVIEW_PHASES } from '../context/AppContext';
import type { Evaluation } from '../types';

export const useInterview = () => {
  const {
    session,
    transcript,
    selectedRole,
    startSession: initSession,
    endSession: finishSession,
    addTranscriptMessage,
    setEvaluation,
    updatePhaseIndex,
  } = useApp();

  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer effect when session is active
  useEffect(() => {
    if (session?.status === 'in-progress') {
      timerRef.current = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [session?.status]);

  // Phase transition logic based on elapsed time or manual trigger
  const currentPhaseIndex = session?.currentPhaseIndex ?? 0;
  const currentPhase = INTERVIEW_PHASES[currentPhaseIndex] || INTERVIEW_PHASES[0];

  const advancePhase = useCallback(() => {
    if (currentPhaseIndex < INTERVIEW_PHASES.length - 1) {
      updatePhaseIndex(currentPhaseIndex + 1);
    }
  }, [currentPhaseIndex, updatePhaseIndex]);

  // Generate mock evaluation summary based on transcript content and duration
  const generateEvaluation = useCallback((): Evaluation => {
    const messageCount = transcript.length;
    const candidateCount = transcript.filter(m => m.role === 'candidate').length;

    // Calculate score metrics based on response quality & interview participation
    const techScore = Math.min(5, Math.max(3, 3.5 + (candidateCount > 3 ? 1 : 0.5)));
    const commScore = Math.min(5, Math.max(3, 4.0 + (candidateCount > 2 ? 0.5 : 0)));
    const behavScore = Math.min(5, Math.max(3, 3.8 + (messageCount > 6 ? 0.8 : 0)));
    const apprScore = Math.min(5, Math.max(3, 4.2));

    const overall = Math.round(((techScore + commScore + behavScore + apprScore) / 20) * 100);

    return {
      technicalSkills: { score: Number(techScore.toFixed(1)), maxScore: 5 },
      communication: { score: Number(commScore.toFixed(1)), maxScore: 5 },
      behavioral: { score: Number(behavScore.toFixed(1)), maxScore: 5 },
      approach: { score: Number(apprScore.toFixed(1)), maxScore: 5 },
      overallScore: overall,
      summary: `The candidate demonstrated strong proficiency in ${selectedRole.title} concepts. Articulated trade-offs effectively, communicated structural ideas with clarity, and showed structured problem-solving under mock interview pressure.`,
      strengths: [
        'Clear articulation of architectural trade-offs and performance bottlenecks.',
        'Proactive structured approach when breaking down multi-tiered requirements.',
        'Strong technical communication with explicit rationale behind stack choices.',
        'Maintained calm and concise responses throughout all 7 interview phases.'
      ],
      improvements: [
        'Could elaborate further on edge-case error handling strategies in distributed environments.',
        'Consider incorporating concrete quantitative metrics (e.g. latency targets, SLA specs) early in system design phases.'
      ]
    };
  }, [transcript, selectedRole]);

  const endInterviewSession = useCallback(() => {
    finishSession();
    const evalResult = generateEvaluation();
    setEvaluation(evalResult);
  }, [finishSession, generateEvaluation, setEvaluation]);

  return {
    session,
    transcript,
    selectedRole,
    currentPhase,
    currentPhaseIndex,
    elapsedSeconds,
    startInterviewSession: initSession,
    endInterviewSession,
    advancePhase,
    addMessage: (role: 'agent' | 'candidate', text: string) => {
      addTranscriptMessage({
        role,
        text,
        timestamp: elapsedSeconds,
      });
    }
  };
};
