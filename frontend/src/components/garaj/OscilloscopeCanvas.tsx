import React, { useEffect, useRef } from 'react';
import { Activity, Mic, Radio, Volume2 } from 'lucide-react';

interface OscilloscopeCanvasProps {
  spectrumData: Uint8Array;
  isMonitoring: boolean;
  isMicActive: boolean;
  isAttackSimulated: boolean;
  rmsEnergy: number;
  peakAmplitude: number;
  onToggleMic: () => void;
}

export const OscilloscopeCanvas: React.FC<OscilloscopeCanvasProps> = ({
  spectrumData,
  isMonitoring,
  isMicActive,
  isAttackSimulated,
  rmsEnergy,
  peakAmplitude,
  onToggleMic,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      // Grid background lines
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.04)';
      ctx.lineWidth = 1;

      for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      for (let y = 0; y < height; y += 24) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw Frequency Spectrum Bars (Subtle Light Charcoal)
      const barCount = 36;
      const barWidth = width / barCount - 3;

      for (let i = 0; i < barCount; i++) {
        const val = spectrumData[i] || 35;
        const barHeight = (val / 255) * (height * 0.45);
        const x = i * (barWidth + 3);
        const y = height - barHeight;

        ctx.fillStyle = isAttackSimulated
          ? 'rgba(239, 68, 68, 0.3)'
          : 'rgba(9, 9, 11, 0.12)';
        ctx.fillRect(x, y, barWidth, barHeight);
      }

      // Draw Waveform Line (Sharp Black Line)
      ctx.beginPath();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = isAttackSimulated ? '#EF4444' : '#09090B';

      const centerY = height / 2 - 8;
      phase += 0.07;

      for (let x = 0; x < width; x++) {
        const amp = isMonitoring ? 16 + rmsEnergy * 90 : 4;
        const freq = 0.025;
        const y =
          centerY +
          Math.sin(x * freq + phase) * amp * (isAttackSimulated ? 1.4 : 1) +
          (Math.random() - 0.5) * (isMonitoring ? 2.5 : 0.5);

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [spectrumData, isMonitoring, isAttackSimulated, rmsEnergy]);

  return (
    <div className="bg-white rounded-2xl p-6 border border-zinc-200 shadow-sm space-y-4 font-mono">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-zinc-200 pb-4">
        <div className="flex items-center gap-2.5">
          <Activity className="w-4 h-4 text-zinc-950" />
          <div>
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-zinc-950">
              SIGNAL OSCILLOSCOPE & WAVEFORM
            </h3>
            <p className="text-[11px] text-zinc-500 font-sans font-normal">
              16kHz PCM audio stream spectrum
            </p>
          </div>
        </div>

        <button
          onClick={onToggleMic}
          className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold transition-all cursor-pointer ${
            isMicActive
              ? 'bg-zinc-950 text-white shadow-xs'
              : 'bg-zinc-100 text-zinc-900 hover:bg-zinc-200 border border-zinc-200'
          }`}
        >
          <Mic className={`w-3.5 h-3.5 ${isMicActive ? 'animate-pulse' : ''}`} />
          <span>{isMicActive ? 'MIC ACTIVE' : 'ENABLE LIVE MIC'}</span>
        </button>
      </div>

      {/* Canvas Viewport (Clean Light Grey Container) */}
      <div className="relative w-full h-44 bg-zinc-50 rounded-xl overflow-hidden border border-zinc-200">
        <canvas
          ref={canvasRef}
          width={800}
          height={176}
          className="w-full h-full object-cover"
        />

        {/* Live Audio Energy Overlay */}
        <div className="absolute top-3 left-4 flex items-center gap-4 text-xs text-zinc-900 bg-white px-3.5 py-1.5 rounded-full border border-zinc-200 shadow-2xs">
          <div className="flex items-center gap-1.5">
            <Volume2 className="w-3.5 h-3.5 text-zinc-950" />
            <span className="text-zinc-500 font-normal">RMS:</span>
            <span className="text-zinc-950 font-extrabold">{rmsEnergy}</span>
          </div>
          <div>
            <span className="text-zinc-500 font-normal">PEAK:</span>
            <span className="text-zinc-950 font-extrabold">{peakAmplitude}</span>
          </div>
        </div>

        {isAttackSimulated && (
          <div className="absolute top-3 right-4 flex items-center gap-2 text-xs font-mono font-bold text-red-600 bg-red-50 px-3 py-1.5 rounded-full border border-red-200 animate-pulse">
            <Radio className="w-4 h-4 text-red-600" />
            <span>SPOOF DETECTED</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default OscilloscopeCanvas;
