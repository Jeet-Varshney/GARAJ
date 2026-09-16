import { useState, useRef, useCallback, useEffect } from 'react';

export function useAudioStreamer({ onAudioChunkReceived, callerStream: initialCallerStream } = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSimulated, setIsSimulated] = useState(false);
  const [hasMicPermission, setHasMicPermission] = useState(false);
  const [permissionError, setPermissionError] = useState(null);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const [clientAudioStats, setClientAudioStats] = useState({
    capturedSamples: 0,
    lastPcmRms: '0.0000',
    lastPcmPeak: '0.0000',
    audioContextState: 'uninitialized',
    sampleRate: 0,
  });

  const [audioSourceType, setAudioSourceType] = useState('remote_caller_silence');

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const callerStreamRef = useRef(initialCallerStream || null);
  const workletNodeRef = useRef(null);
  const silenceSourceRef = useRef(null);
  const analyserNodeRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const simulationIntervalRef = useRef(null);
  const simPhaseRef = useRef(0);
  const totalCapturedSamplesRef = useRef(0);

  useEffect(() => {
    if (initialCallerStream) {
      callerStreamRef.current = initialCallerStream;
    }
  }, [initialCallerStream]);

  const processChunkTelemetry = (arrayBuffer) => {
    if (!arrayBuffer) return;
    const int16View = new Int16Array(arrayBuffer);
    totalCapturedSamplesRef.current += int16View.length;
    let sumSq = 0;
    let peak = 0;
    for (let i = 0; i < int16View.length; i++) {
      const norm = int16View[i] / 32768.0;
      sumSq += norm * norm;
      const abs = Math.abs(norm);
      if (abs > peak) peak = abs;
    }
    const rms = Math.sqrt(sumSq / (int16View.length || 1));
    setClientAudioStats({
      capturedSamples: totalCapturedSamplesRef.current,
      lastPcmRms: rms.toFixed(4),
      lastPcmPeak: peak.toFixed(4),
      audioContextState: audioContextRef.current ? audioContextRef.current.state : 'uninitialized',
      sampleRate: audioContextRef.current ? audioContextRef.current.sampleRate : 0,
    });
  };

  const updateVolume = useCallback(() => {
    if (!analyserNodeRef.current) return;

    const dataArray = new Uint8Array(analyserNodeRef.current.frequencyBinCount);
    analyserNodeRef.current.getByteFrequencyData(dataArray);

    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i];
    }
    const avg = sum / dataArray.length;
    const volumePct = Math.min(100, Math.round((avg / 128.0) * 100));
    setVolumeLevel(volumePct);

    if (isStreaming && !isSimulated) {
      animFrameIdRef.current = requestAnimationFrame(updateVolume);
    }
  }, [isStreaming, isSimulated]);

  const stopStreaming = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (simulationIntervalRef.current) {
      clearInterval(simulationIntervalRef.current);
      simulationIntervalRef.current = null;
    }

    if (silenceSourceRef.current) {
      try {
        silenceSourceRef.current.stop();
        silenceSourceRef.current.disconnect();
      } catch (e) { /* ignore */ }
      silenceSourceRef.current = null;
    }

    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.port.onmessage = null;
        workletNodeRef.current.disconnect();
      } catch (e) { /* ignore */ }
      workletNodeRef.current = null;
    }

    if (mediaStreamRef.current) {
      try {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) { /* ignore */ }
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close();
      } catch (e) { /* ignore */ }
      audioContextRef.current = null;
    }

    setIsStreaming(false);
    setIsSimulated(false);
    setVolumeLevel(0);
  }, []);

  const getSystemDisplayStream = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      throw new Error('System / Tab audio capture API (getDisplayMedia) is not supported in this browser.');
    }

    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const audioTracks = displayStream.getAudioTracks();
      if (!audioTracks || audioTracks.length === 0) {
        displayStream.getTracks().forEach((track) => track.stop());
        throw new Error('No audio track selected in screen capture prompt.');
      }

      const callerAudioStream = new MediaStream([audioTracks[0]]);
      displayStream.getVideoTracks().forEach((track) => track.stop());

      audioTracks[0].onended = () => {
        stopStreaming();
      };

      return callerAudioStream;
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw new Error('Caller audio capture prompt was cancelled or denied by user.');
      }
      throw err;
    }
  };

  const startStreaming = useCallback(async (inputOption = 'caller_audio') => {
    stopStreaming();
    setPermissionError(null);

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.');
      }
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      // Initialize AudioWorklet
      await audioCtx.audioWorklet.addModule('/audio-processor.js');
      const workletNode = new AudioWorkletNode(audioCtx, 'pcm-stream-processor');
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event) => {
        const int16ArrayBuffer = event.data;
        processChunkTelemetry(int16ArrayBuffer);
        if (onAudioChunkReceived && int16ArrayBuffer) {
          onAudioChunkReceived(int16ArrayBuffer);
        }
      };

      // Lifecycle Fix: Keep AudioWorkletNode connected to silent GainNode -> destination
      const workletGain = audioCtx.createGain();
      workletGain.gain.value = 0;
      workletNode.connect(workletGain);
      workletGain.connect(audioCtx.destination);

      const analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNodeRef.current = analyserNode;

      let sourceStream = null;
      if (typeof inputOption === 'object' && inputOption !== null && inputOption instanceof MediaStream) {
        sourceStream = inputOption;
        setAudioSourceType('remote_caller_mediastream');
      } else if (callerStreamRef.current instanceof MediaStream) {
        sourceStream = callerStreamRef.current;
        setAudioSourceType('remote_caller_mediastream');
      } else if (inputOption === 'system_display') {
        sourceStream = await getSystemDisplayStream();
        setAudioSourceType('system_display_capture');
      }

      if (sourceStream) {
        mediaStreamRef.current = sourceStream;
        const sourceNode = audioCtx.createMediaStreamSource(sourceStream);
        sourceNode.connect(analyserNode);
        sourceNode.connect(workletNode);
        setHasMicPermission(true);
      } else {
        // Silent AudioSource into AudioWorklet when no active caller MediaStream exists
        // Generates 16kHz PCM zero chunks -> backend energy gate detects NO_AUDIO safely
        const silenceSource = audioCtx.createConstantSource();
        silenceSource.offset.value = 0;
        silenceSource.start();
        silenceSourceRef.current = silenceSource;
        silenceSource.connect(analyserNode);
        silenceSource.connect(workletNode);
        setAudioSourceType('remote_caller_silence');
        setHasMicPermission(true);
      }

      setIsStreaming(true);
      setIsSimulated(false);
      animFrameIdRef.current = requestAnimationFrame(updateVolume);

    } catch (err) {
      console.error('Failed to start AudioWorklet audio streaming:', err);
      let msg = err.message || 'AudioWorklet stream setup failed.';
      setPermissionError(msg);
      setIsStreaming(false);
    }
  }, [onAudioChunkReceived, stopStreaming, updateVolume]);

  const attachCallerStream = useCallback((stream) => {
    callerStreamRef.current = stream;
    if (isStreaming) {
      startStreaming(stream);
    }
  }, [isStreaming, startStreaming]);

  const startSimulatedStreaming = useCallback(() => {
    stopStreaming();
    setPermissionError(null);
    setIsStreaming(true);
    setIsSimulated(true);

    const chunkSize = 1600;
    const sampleRate = 16000;

    simulationIntervalRef.current = setInterval(() => {
      simPhaseRef.current += 1;
      const t = simPhaseRef.current * 0.1;

      const pcmBuffer = new Int16Array(chunkSize);
      for (let i = 0; i < chunkSize; i++) {
        const sampleTime = (i / sampleRate);
        const freq1 = 220 + Math.sin(t) * 40;
        const freq2 = 440 + Math.cos(t * 1.5) * 60;
        const val = 0.4 * Math.sin(2 * Math.PI * freq1 * sampleTime) +
                    0.2 * Math.sin(2 * Math.PI * freq2 * sampleTime) +
                    0.05 * (Math.random() * 2 - 1);
        pcmBuffer[i] = Math.round(Math.max(-1.0, Math.min(1.0, val)) * 32767);
      }

      const volumePct = Math.round(35 + Math.sin(t * 2) * 25);
      setVolumeLevel(volumePct);

      if (onAudioChunkReceived) {
        onAudioChunkReceived(pcmBuffer.buffer);
      }
    }, 100);
  }, [onAudioChunkReceived, stopStreaming]);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return {
    isStreaming,
    isSimulated,
    audioSourceType,
    hasMicPermission,
    permissionError,
    volumeLevel,
    clientAudioStats,
    startStreaming,
    attachCallerStream,
    startSimulatedStreaming,
    stopStreaming,
    analyserNode: analyserNodeRef.current,
  };
}
