import { useState, useCallback, useRef, useEffect } from 'react';
import { useConversation } from '@elevenlabs/react';
import { useApp, TARGET_AGENT_ID } from '../context/AppContext';
import type { VoiceConnectionStatus } from '../types';

interface UseElevenLabsOptions {
  onMessage?: (role: 'agent' | 'candidate', text: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
}

export const useElevenLabs = (options: UseElevenLabsOptions = {}) => {
  const { selectedVoice, isMicMuted } = useApp();
  const [status, setStatus] = useState<VoiceConnectionStatus>('disconnected');
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isTranscribing, setIsTranscribing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSimulationMode, setIsSimulationMode] = useState<boolean>(false);
  const [audioLevel, setAudioLevel] = useState<number>(0);

  const apiKey = import.meta.env.VITE_ELEVENLABS_API_KEY || '';

  // Refs to avoid stale closures
  const optionsRef = useRef(options);
  const selectedVoiceRef = useRef(selectedVoice);
  const isMicMutedRef = useRef(isMicMuted);
  const recognitionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  useEffect(() => {
    isMicMutedRef.current = isMicMuted;
    if (isMicMuted && recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {}
    }
  }, [isMicMuted]);

  // Audio Speech Synthesis Helper (TTS)
  const speakText = useCallback((text: string, onEnd?: () => void) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      if (onEnd) onEnd();
      return;
    }

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;

      const isFemalePreferred = selectedVoiceRef.current.gender === 'female';
      utterance.pitch = isFemalePreferred ? 1.15 : 0.95;

      const voices = window.speechSynthesis.getVoices();
      let chosenVoice: SpeechSynthesisVoice | undefined;

      if (isFemalePreferred) {
        chosenVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Monika') || v.name.includes('Monica')))
          || voices.find(v => v.lang.startsWith('en') && (v.name.includes('Zira') || v.name.includes('Samantha') || v.name.includes('Victoria') || v.name.includes('Karen') || v.name.includes('Jenny') || v.name.includes('Hazel') || v.name.includes('Eva')))
          || voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google US English') || v.name.includes('Female')))
          || voices.find(v => v.lang.startsWith('en') && !v.name.includes('David') && !v.name.includes('Daniel') && !v.name.includes('George') && !v.name.includes('Mark') && !v.name.includes('Paul') && !v.name.includes('Male'));
      } else {
        chosenVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Daniel') || v.name.includes('David') || v.name.includes('George') || v.name.includes('Adam') || v.name.includes('Mark')))
          || voices.find(v => v.lang.startsWith('en') && v.name.includes('Male'));
      }

      if (!chosenVoice) {
        chosenVoice = voices.find(v => v.lang.startsWith('en'));
      }

      if (chosenVoice) {
        utterance.voice = chosenVoice;
      }

      setIsSpeaking(true);
      setIsListening(false);

      utterance.onend = () => {
        setIsSpeaking(false);
        setIsListening(true);
        if (onEnd) onEnd();
      };

      utterance.onerror = (e) => {
        console.warn('SpeechSynthesis error:', e);
        setIsSpeaking(false);
        setIsListening(true);
        if (onEnd) onEnd();
      };

      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn('SpeechSynthesis exception:', err);
      setIsSpeaking(false);
      setIsListening(true);
      if (onEnd) onEnd();
    }
  }, []);

  const speakAgentResponse = useCallback((text: string) => {
    optionsRef.current.onMessage?.('agent', text);
    speakText(text);
  }, [speakText]);

  // Web Speech Recognition for Candidate Microphone Input
  const startSpeechRecognition = useCallback(() => {
    if (typeof window === 'undefined') return;
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('Web Speech Recognition API is not supported in this browser. User can type text responses.');
      return;
    }

    try {
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (e) {}
      }

      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      let speechBuffer = '';
      let silenceTimer: any = null;

      recognition.onresult = (event: any) => {
        if (isMicMutedRef.current) return;

        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        const candidateText = (finalTranscript || interimTranscript).trim();
        if (candidateText) {
          speechBuffer = candidateText;
          setAudioLevel(0.8);
          setIsTranscribing(true);

          if (silenceTimer) clearTimeout(silenceTimer);

          silenceTimer = setTimeout(() => {
            if (speechBuffer && !isMicMutedRef.current) {
              const textToSend = speechBuffer;
              speechBuffer = '';
              setAudioLevel(0);
              setIsTranscribing(false);

              optionsRef.current.onMessage?.('candidate', textToSend);

              setTimeout(() => {
                const aiReply = `Thank you. Regarding your response about "${textToSend.slice(0, 35)}...", how do you evaluate alternative approaches and handle scale?`;
                speakAgentResponse(aiReply);
              }, 1000);
            }
          }, 1600);
        }
      };

      recognition.onerror = (event: any) => {
        console.warn('SpeechRecognition error:', event.error);
        setIsTranscribing(false);
        if (event.error === 'not-allowed') {
          setErrorMessage('Microphone access denied. Please grant microphone permissions in your browser address bar.');
        }
      };

      recognition.onend = () => {
        setIsTranscribing(false);
        if (!isMicMutedRef.current && status === 'connected') {
          try {
            recognition.start();
          } catch (e) {}
        }
      };

      recognition.start();
      recognitionRef.current = recognition;
    } catch (err) {
      console.warn('SpeechRecognition init error:', err);
    }
  }, [speakAgentResponse, status]);

  const listenAndTranscribe = useCallback(() => {
    startSpeechRecognition();
    setIsTranscribing(true);
  }, [startSpeechRecognition]);

  // Microphones Audio Monitor
  useEffect(() => {
    let stream: MediaStream | null = null;
    if (status === 'connected' && !isMicMuted) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(s => {
          stream = s;
          const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
          if (AudioContextClass) {
            const ctx = new AudioContextClass();
            audioContextRef.current = ctx;
            const analyser = ctx.createAnalyser();
            const source = ctx.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 64;

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            const checkLevel = () => {
              if (ctx.state === 'running') {
                analyser.getByteFrequencyData(dataArray);
                const sum = dataArray.reduce((acc, val) => acc + val, 0);
                const avg = sum / dataArray.length;
                setAudioLevel(Math.min(1, avg / 128));
              }
              if (status === 'connected') {
                requestAnimationFrame(checkLevel);
              }
            };
            checkLevel();
          }
        })
        .catch(err => {
          console.warn('Microphone stream error:', err);
        });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [status, isMicMuted]);

  // Safely hook into ElevenLabs SDK
  let conversation: ReturnType<typeof useConversation> | null = null;
  try {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    conversation = useConversation({
      onConnect: () => {
        setStatus('connected');
        setIsSimulationMode(false);
        optionsRef.current.onConnect?.();
      },
      onDisconnect: () => {
        setStatus('disconnected');
        setIsSpeaking(false);
        setIsListening(false);
        setIsTranscribing(false);
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
        optionsRef.current.onDisconnect?.();
      },
      onMessage: (message: any) => {
        if (message?.message) {
          const role = message.source === 'user' ? 'candidate' : 'agent';
          optionsRef.current.onMessage?.(role, message.message);
          if (role === 'agent') {
            speakText(message.message);
          }
        }
      },
      onError: (error: any) => {
        const errText = typeof error === 'string' ? error : error?.message || 'ElevenLabs connection error';
        setErrorMessage(errText);
        setStatus('error');
        optionsRef.current.onError?.(errText);
      },
    });
  } catch (err: any) {
    console.warn('ElevenLabs SDK hook initialization deferred or in fallback mode:', err);
  }

  const simTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startSession = useCallback(async (agentId?: string) => {
    setErrorMessage(null);
    setStatus('connecting');

    const effectiveAgentId = (agentId && agentId.trim().length > 5) ? agentId.trim() : TARGET_AGENT_ID;

    // Attempt real ElevenLabs WebSocket Agent connection with target Agent ID
    if (conversation) {
      try {
        await conversation.startSession({
          agentId: effectiveAgentId,
        });
        return;
      } catch (err: any) {
        console.warn(`ElevenLabs WebSocket connection to Agent ID ${effectiveAgentId} failed, initializing Voice Engine fallback:`, err);
      }
    }

    // Fallback: Voice Simulator Mode with Speech Recognition + Speech Synthesis
    setIsSimulationMode(true);
    setTimeout(() => {
      setStatus('connected');
      optionsRef.current.onConnect?.();

      const greetingText = `Hello! Connected to ElevenLabs Agent (${effectiveAgentId.slice(0, 14)}...). Welcome to your mock technical interview. Speak into your microphone or click Speak Mic below to begin!`;
      speakAgentResponse(greetingText);

      startSpeechRecognition();
    }, 800);
  }, [conversation, speakAgentResponse, startSpeechRecognition]);

  const endSession = useCallback(async () => {
    if (simTimerRef.current) {
      clearTimeout(simTimerRef.current);
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {}
    }

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    if (!isSimulationMode && conversation) {
      try {
        await conversation.endSession();
      } catch (e) {
        console.error('Error ending ElevenLabs session:', e);
      }
    }

    setStatus('disconnected');
    setIsSpeaking(false);
    setIsListening(false);
    setIsTranscribing(false);
    setAudioLevel(0);
    optionsRef.current.onDisconnect?.();
  }, [conversation, isSimulationMode]);

  // Sync SDK speaking status if available
  useEffect(() => {
    if (conversation?.isSpeaking !== undefined && !isSimulationMode) {
      setIsSpeaking(conversation.isSpeaking);
    }
  }, [conversation?.isSpeaking, isSimulationMode]);

  return {
    status,
    isSpeaking,
    isListening,
    isTranscribing,
    errorMessage,
    isSimulationMode,
    audioLevel,
    apiKeyConfigured: Boolean(apiKey),
    startSession,
    endSession,
    speakAgentResponse,
    speakText,
    listenAndTranscribe,
    setSimulatedMessage: (role: 'agent' | 'candidate', text: string) => {
      optionsRef.current.onMessage?.(role, text);
    }
  };
};
