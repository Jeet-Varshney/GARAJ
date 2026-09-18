import { useState, useRef, useCallback, useEffect } from 'react';

export function useAudioStreamer({ onAudioChunkReceived } = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSimulated, setIsSimulated] = useState(false);
  const [hasMicPermission, setHasMicPermission] = useState(false);
  const [permissionError, setPermissionError] = useState(null);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const [clientAudioStats, setClientAudioStats] = useState({
    capturedSamples: 0,
    pcmChunkCount: 0,
    lastPcmRms: '0.0000',
    lastPcmPeak: '0.0000',
    audioContextState: 'uninitialized',
    sampleRate: 0,
  });

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const analyserNodeRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const simulationIntervalRef = useRef(null);
  const totalCapturedSamplesRef = useRef(0);
  const pcmChunkCountRef = useRef(0);

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
      pcmChunkCount: pcmChunkCountRef.current,
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

    totalCapturedSamplesRef.current = 0;
    pcmChunkCountRef.current = 0;

    setIsStreaming(false);
    setIsSimulated(false);
    setVolumeLevel(0);
    setClientAudioStats({
      capturedSamples: 0,
      pcmChunkCount: 0,
      lastPcmRms: '0.0000',
      lastPcmPeak: '0.0000',
      audioContextState: 'uninitialized',
      sampleRate: 0,
    });
  }, []);

  const startStreaming = useCallback(async () => {
    stopStreaming();
    setPermissionError(null);
    totalCapturedSamplesRef.current = 0;
    pcmChunkCountRef.current = 0;

    console.log('[MIC] permission requested');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone audio capture (getUserMedia) is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      console.log('[MIC] stream acquired');
      const audioTracks = stream.getAudioTracks();
      console.log('[MIC] audio tracks =', audioTracks.length);
      if (audioTracks.length > 0) {
        console.log('[MIC] track readyState =', audioTracks[0].readyState);
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.');
      }
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      console.log('[AUDIO CONTEXT] state =', audioCtx.state);

      // Initialize AudioWorklet
      await audioCtx.audioWorklet.addModule('/audio-processor.js');
      const workletNode = new AudioWorkletNode(audioCtx, 'pcm-stream-processor');
      workletNodeRef.current = workletNode;
      console.log('[AUDIO WORKLET] active');

      workletNode.port.onmessage = (event) => {
        const int16ArrayBuffer = event.data;
        pcmChunkCountRef.current += 1;
        console.log('[PCM] chunks increasing', pcmChunkCountRef.current);
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

      mediaStreamRef.current = stream;
      const sourceNode = audioCtx.createMediaStreamSource(stream);
      sourceNode.connect(analyserNode);
      sourceNode.connect(workletNode);

      setHasMicPermission(true);
      setIsStreaming(true);
      setIsSimulated(false);
      animFrameIdRef.current = requestAnimationFrame(updateVolume);

    } catch (err) {
      console.error('[MIC] Permission or stream error:', err);
      let msg = 'MICROPHONE ACCESS REQUIRED';
      setHasMicPermission(false);
      setPermissionError(msg);
      setIsStreaming(false);
    }
  }, [onAudioChunkReceived, stopStreaming, updateVolume]);

  const startSimulatedStreaming = useCallback(() => {
    stopStreaming();
    setPermissionError(null);
    setIsStreaming(true);
    setIsSimulated(true);

    const chunkSize = 1600;

    simulationIntervalRef.current = setInterval(() => {
      const pcmBuffer = new Int16Array(chunkSize);
      pcmChunkCountRef.current += 1;
      console.log('[PCM] chunks increasing', pcmChunkCountRef.current);
      setClientAudioStats((prev) => ({
        ...prev,
        pcmChunkCount: pcmChunkCountRef.current,
        capturedSamples: pcmChunkCountRef.current * chunkSize,
      }));
      setVolumeLevel(0);

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
    hasMicPermission,
    permissionError,
    volumeLevel,
    clientAudioStats,
    startStreaming,
    startSimulatedStreaming,
    stopStreaming,
    analyserNode: analyserNodeRef.current,
  };
}
