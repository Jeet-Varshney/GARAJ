/**
 * AudioWorkletProcessor for Real-Time PCM Audio Capture & 16kHz Downsampling.
 * Runs on dedicated Web Audio API audio rendering thread.
 */

class PCMStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = 16000;
    this.chunkSize = 1600; // ~100ms chunk size at 16kHz (1600 samples)
    
    // Internal accumulation buffer for resampled Int16 PCM samples
    this.pcmBuffer = new Int16Array(this.chunkSize * 4);
    this.pcmBufferIndex = 0;
    
    // Resampling state tracking
    this.resampleRatio = sampleRate / this.targetSampleRate;
    this.resampleOffset = 0.0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const inputChannel = input[0]; // Mono input channel
    if (!inputChannel || inputChannel.length === 0) return true;

    const inputLength = inputChannel.length;

    if (Math.abs(sampleRate - this.targetSampleRate) < 1.0) {
      // Native 16kHz capture
      for (let i = 0; i < inputLength; i++) {
        const s = Math.max(-1.0, Math.min(1.0, inputChannel[i]));
        const val = s < 0 ? s * 32768 : s * 32767;
        this.pcmBuffer[this.pcmBufferIndex++] = Math.round(val);

        if (this.pcmBufferIndex >= this.chunkSize) {
          this.flushBuffer();
        }
      }
    } else {
      // Downsample to 16kHz via linear interpolation
      while (this.resampleOffset < inputLength) {
        const index = Math.floor(this.resampleOffset);
        const nextIndex = Math.min(index + 1, inputLength - 1);
        const weight = this.resampleOffset - index;

        const sample = (1 - weight) * inputChannel[index] + weight * inputChannel[nextIndex];
        
        const s = Math.max(-1.0, Math.min(1.0, sample));
        const val = s < 0 ? s * 32768 : s * 32767;
        this.pcmBuffer[this.pcmBufferIndex++] = Math.round(val);

        if (this.pcmBufferIndex >= this.chunkSize) {
          this.flushBuffer();
        }

        this.resampleOffset += this.resampleRatio;
      }
      this.resampleOffset -= inputLength;
    }

    return true;
  }

  flushBuffer() {
    if (this.pcmBufferIndex === 0) return;

    const chunkToSend = this.pcmBuffer.slice(0, this.pcmBufferIndex);
    this.port.postMessage(chunkToSend.buffer, [chunkToSend.buffer]);
    this.pcmBufferIndex = 0;
  }
}

registerProcessor('pcm-stream-processor', PCMStreamProcessor);
