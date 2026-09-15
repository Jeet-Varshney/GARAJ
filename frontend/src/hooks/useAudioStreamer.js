import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Resamples Float32 audio array to 16kHz Signed 16-bit PCM ArrayBuffer.
 */
function convertFloat32ToInt16PCM(float32Array, inputSampleRate = 44100, targetSampleRate = 16000) {
  if (Math.abs(inputSampleRate - targetSampleRate) < 1) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1.0, Math.min(1.0, float32Array[i]));
      int16[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return int16.buffer;
  }

  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.floor(float32Array.length / ratio);
  const int16 = new Int16Array(newLength);

  let offset = 0;
  for (let i = 0; i < newLength; i++) {
    const nextOffset = Math.floor((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < float32Array.length; j++) {
      sum += float32Array[j];
      count++;
    }
    const sample = count > 0 ? sum / count : float32Array[Math.floor(offset)] || 0;
    const s = Math.max(-1.0, Math.min(1.0, sample));
    int16[i] = s < 0 ? s * 32768 : s * 32767;
    offset = nextOffset;
  }
  return int16.buffer;
}

export function useAudioStreamer({ onAudioChunkReceived }) {
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

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const analyserNodeRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const simulationIntervalRef = useRef(null);
  const simPhaseRef = useRef(0);
  const totalCapturedSamplesRef = useRef(0);

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

    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.port.onmessage = null;
        workletNodeRef.current.disconnect();
      } catch (e) { /* ignore */ }
      workletNodeRef.current = null;
    }

    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.onaudioprocess = null;
        scriptProcessorRef.current.disconnect();
      } catch (e) { /* ignore */ }
      scriptProcessorRef.current = null;
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

  const getMicrophoneStream = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone API is not available (browser non-secure context or unsupported).');
    }

    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 1 },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err1) {
      console.warn('Strict mic constraints failed, attempting basic audio constraints:', err1);
      if (err1.name === 'AbortError') {
        await new Promise((r) => setTimeout(r, 250));
      }
      return await navigator.mediaDevices.getUserMedia({ audio: true });
    }
  };

  const startStreaming = useCallback(async () => {
    stopStreaming();
    setPermissionError(null);

    try {
      const stream = await getMicrophoneStream();
      mediaStreamRef.current = stream;
      setHasMicPermission(true);

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.');
      }
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNodeRef.current = analyserNode;
      sourceNode.connect(analyserNode);

      let workletLoaded = false;
      try {
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

        sourceNode.connect(workletNode);
        const workletGain = audioCtx.createGain();
        workletGain.gain.value = 0;
        workletNode.connect(workletGain);
        workletGain.connect(audioCtx.destination);

        workletLoaded = true;
      } catch (workletErr) {
        console.warn('AudioWorklet initialization failed, switching to ScriptProcessorNode fallback:', workletErr);
      }

      if (!workletLoaded) {
        const bufferSize = 4096;
        const scriptProcessor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
        scriptProcessorRef.current = scriptProcessor;

        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
          const inputBuffer = audioProcessingEvent.inputBuffer;
          const inputData = inputBuffer.getChannelData(0);
          const pcmBuffer = convertFloat32ToInt16PCM(inputData, inputBuffer.sampleRate, 16000);
          if (onAudioChunkReceived && pcmBuffer) {
            onAudioChunkReceived(pcmBuffer);
          }
        };

        sourceNode.connect(scriptProcessor);
        const gainNode = audioCtx.createGain();
        gainNode.gain.value = 0;
        scriptProcessor.connect(gainNode);
        gainNode.connect(audioCtx.destination);
      }

      setIsStreaming(true);
      setIsSimulated(false);
      animFrameIdRef.current = requestAnimationFrame(updateVolume);

    } catch (err) {
      console.error('Failed to start audio streaming:', err);
      let msg = err.message || 'Microphone access denied or unavailable.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Microphone permission denied. Please allow microphone access in browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No microphone hardware found on this device.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = 'Microphone hardware is busy or being used by another application.';
      } else if (err.name === 'AbortError') {
        msg = 'Microphone operation was aborted by browser or OS audio subsystem (device lock). Click Start Monitoring again or use Simulated Stream.';
      }
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
