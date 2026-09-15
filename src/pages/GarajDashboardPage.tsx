import React, { useState, useRef } from 'react';
import { useGarajSecurity } from '../hooks/useGarajSecurity';
import { Header } from '../components/garaj/Header';
import { VerdictPanel } from '../components/garaj/VerdictPanel';
import { OscilloscopeCanvas } from '../components/garaj/OscilloscopeCanvas';
import { DiagnosticsPanel } from '../components/garaj/DiagnosticsPanel';
import { LiveTerminal } from '../components/garaj/LiveTerminal';
import { FloatingDock } from '../components/garaj/FloatingDock';
import { Shield, Smartphone, Activity, Radio, Cpu, Terminal as TerminalIcon } from 'lucide-react';
import type { LayoutViewMode } from '../types/garaj';

export const GarajDashboardPage: React.FC = () => {
  const {
    isMonitoring,
    setIsMonitoring,
    isMicActive,
    toggleMicrophone,
    isAttackSimulated,
    setIsAttackSimulated,
    showTerminal,
    setShowTerminal,
    authenticityScore,
    verdict,
    riskLevel,
    checks,
    pipelineSpec,
    chunkTelemetry,
    chunkFlash,
    latencies,
    modelSpec,
    recentActivities,
    rmsEnergy,
    peakAmplitude,
    spectrumData,
    logs,
  } = useGarajSecurity();

  // View Mode: 'desktop-3col' (Default Full Screen) | 'mobile-frame' (4 Mobile Phone Slides Parallel)
  const [viewMode, setViewMode] = useState<LayoutViewMode>('desktop-3col');

  // Refs for smooth scrolling to specific mobile frame when tabs are clicked
  const frame1Ref = useRef<HTMLDivElement | null>(null);
  const frame2Ref = useRef<HTMLDivElement | null>(null);
  const frame3Ref = useRef<HTMLDivElement | null>(null);
  const frame4Ref = useRef<HTMLDivElement | null>(null);

  const scrollToFrame = (ref: React.RefObject<HTMLDivElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-[#09090B] mono-dot-grid flex flex-col justify-between selection:bg-zinc-950 selection:text-white pb-28">
      {/* Header */}
      <Header
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        isMonitoring={isMonitoring}
        chunkCount={chunkTelemetry.totalChunks}
        isAttackSimulated={isAttackSimulated}
      />

      {/* Main Container */}
      <main className="max-w-7xl w-full mx-auto px-4 sm:px-6 md:px-8 pt-6 flex-1 space-y-6">
        {/* DESKTOP DASHBOARD VIEW */}
        {viewMode === 'desktop-3col' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Verdict & Authenticity */}
            <div className="lg:col-span-5 space-y-6">
              <VerdictPanel
                score={authenticityScore}
                verdict={verdict}
                riskLevel={riskLevel}
                checks={checks}
                recentActivities={recentActivities}
                isMonitoring={isMonitoring}
                onToggleMonitoring={() => setIsMonitoring(!isMonitoring)}
                isAttackSimulated={isAttackSimulated}
                onToggleAttack={() => setIsAttackSimulated(!isAttackSimulated)}
              />
            </div>

            {/* Right Column: Audio Spectrum, Telemetry & Terminal */}
            <div className="lg:col-span-7 space-y-6">
              <OscilloscopeCanvas
                spectrumData={spectrumData}
                isMonitoring={isMonitoring}
                isMicActive={isMicActive}
                isAttackSimulated={isAttackSimulated}
                rmsEnergy={rmsEnergy}
                peakAmplitude={peakAmplitude}
                onToggleMic={toggleMicrophone}
              />

              <DiagnosticsPanel
                pipelineSpec={pipelineSpec}
                chunkTelemetry={chunkTelemetry}
                chunkFlash={chunkFlash}
                latencies={latencies}
                modelSpec={modelSpec}
                isAttackSimulated={isAttackSimulated}
              />

              {showTerminal && (
                <LiveTerminal
                  logs={logs}
                  isMonitoring={isMonitoring}
                  isAttackSimulated={isAttackSimulated}
                />
              )}
            </div>
          </div>
        ) : (
          /* MOBILE PRESENTATION SHOWCASE (4 PARALLEL MOBILE SCREENS SIDE-BY-SIDE) */
          <div className="space-y-6 max-w-full overflow-hidden">
            {/* Top Navigation & Jump Tabs (Clean Light Theme) */}
            <div className="bg-white p-4 rounded-2xl border border-zinc-200 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4 font-mono">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-zinc-950 text-white shrink-0">
                  <Smartphone className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-xs font-extrabold text-zinc-950 uppercase tracking-wider">
                    MOBILE PRESENTATION SHOWCASE
                  </h2>
                  <p className="text-[11px] text-zinc-500 font-sans font-normal">
                    4 Parallel Mobile Screens side-by-side for pitch slides & demo presentation
                  </p>
                </div>
              </div>

              {/* 4 Interactive Screen Tabs */}
              <div className="flex items-center gap-1.5 bg-zinc-100 p-1.5 rounded-full overflow-x-auto text-xs max-w-full border border-zinc-200">
                <button
                  onClick={() => scrollToFrame(frame1Ref)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-zinc-800 hover:text-zinc-950 hover:bg-zinc-200 transition-all font-bold whitespace-nowrap cursor-pointer"
                >
                  <Activity className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
                  <span>01. Verdict</span>
                </button>
                <button
                  onClick={() => scrollToFrame(frame2Ref)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-zinc-800 hover:text-zinc-950 hover:bg-zinc-200 transition-all font-bold whitespace-nowrap cursor-pointer"
                >
                  <Radio className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
                  <span>02. Signal</span>
                </button>
                <button
                  onClick={() => scrollToFrame(frame3Ref)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-zinc-800 hover:text-zinc-950 hover:bg-zinc-200 transition-all font-bold whitespace-nowrap cursor-pointer"
                >
                  <Cpu className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
                  <span>03. Telemetry</span>
                </button>
                <button
                  onClick={() => scrollToFrame(frame4Ref)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-zinc-800 hover:text-zinc-950 hover:bg-zinc-200 transition-all font-bold whitespace-nowrap cursor-pointer"
                >
                  <TerminalIcon className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
                  <span>04. Terminal</span>
                </button>
              </div>
            </div>

            {/* PARALLEL ROW OF 4 MOBILE PHONE FRAMES */}
            <div className="flex items-start gap-8 overflow-x-auto pb-8 pt-2 px-2 scroll-smooth">
              {/* PHONE FRAME 01: VERDICT CORE */}
              <div ref={frame1Ref} className="flex flex-col items-center gap-3 shrink-0">
                <div className="w-[360px] sm:w-[380px] bg-zinc-950 border-[6px] border-zinc-900 rounded-[40px] shadow-xl overflow-hidden relative font-sans">
                  {/* Phone Top Notch & Status Bar */}
                  <div className="bg-zinc-950 px-6 py-2 flex items-center justify-between text-[11px] font-mono font-bold text-zinc-400 border-b border-zinc-900">
                    <span>9:41 AM</span>
                    <div className="w-20 h-4 bg-zinc-900 rounded-b-xl flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-zinc-800" />
                    </div>
                    <span>5G 100%</span>
                  </div>

                  {/* Inner Phone Screen Content */}
                  <div className="p-3.5 sm:p-4 bg-[#FAFAFA] max-h-[640px] overflow-y-auto space-y-4 w-full max-w-full overflow-x-hidden">
                    <VerdictPanel
                      score={authenticityScore}
                      verdict={verdict}
                      riskLevel={riskLevel}
                      checks={checks}
                      recentActivities={recentActivities}
                      isMonitoring={isMonitoring}
                      onToggleMonitoring={() => setIsMonitoring(!isMonitoring)}
                      isAttackSimulated={isAttackSimulated}
                      onToggleAttack={() => setIsAttackSimulated(!isAttackSimulated)}
                    />
                  </div>

                  {/* Phone Bottom Home Bar */}
                  <div className="bg-zinc-950 py-2.5 flex justify-center border-t border-zinc-900">
                    <div className="w-32 h-1 bg-zinc-600 rounded-full" />
                  </div>
                </div>

                {/* Frame Badge */}
                <div className="px-4 py-1.5 rounded-full bg-white text-zinc-950 border border-zinc-300 text-xs font-mono font-extrabold flex items-center gap-2 shadow-2xs">
                  <span className="w-2 h-2 rounded-full bg-zinc-950" />
                  <span>SCREEN 01: VERDICT CORE</span>
                </div>
              </div>

              {/* PHONE FRAME 02: SIGNAL WAVEFORM */}
              <div ref={frame2Ref} className="flex flex-col items-center gap-3 shrink-0">
                <div className="w-[360px] sm:w-[380px] bg-zinc-950 border-[6px] border-zinc-900 rounded-[40px] shadow-xl overflow-hidden relative font-sans">
                  {/* Phone Top Notch & Status Bar */}
                  <div className="bg-zinc-950 px-6 py-2 flex items-center justify-between text-[11px] font-mono font-bold text-zinc-400 border-b border-zinc-900">
                    <span>9:41 AM</span>
                    <div className="w-20 h-4 bg-zinc-900 rounded-b-xl flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-zinc-800" />
                    </div>
                    <span>5G 100%</span>
                  </div>

                  {/* Inner Phone Screen Content */}
                  <div className="p-3.5 sm:p-4 bg-[#FAFAFA] max-h-[640px] overflow-y-auto space-y-4 w-full max-w-full overflow-x-hidden">
                    <OscilloscopeCanvas
                      spectrumData={spectrumData}
                      isMonitoring={isMonitoring}
                      isMicActive={isMicActive}
                      isAttackSimulated={isAttackSimulated}
                      rmsEnergy={rmsEnergy}
                      peakAmplitude={peakAmplitude}
                      onToggleMic={toggleMicrophone}
                    />
                  </div>

                  {/* Phone Bottom Home Bar */}
                  <div className="bg-zinc-950 py-2.5 flex justify-center border-t border-zinc-900">
                    <div className="w-32 h-1 bg-zinc-600 rounded-full" />
                  </div>
                </div>

                {/* Frame Badge */}
                <div className="px-4 py-1.5 rounded-full bg-white text-zinc-950 border border-zinc-300 text-xs font-mono font-extrabold flex items-center gap-2 shadow-2xs">
                  <span className="w-2 h-2 rounded-full bg-zinc-950" />
                  <span>SCREEN 02: SIGNAL WAVEFORM</span>
                </div>
              </div>

              {/* PHONE FRAME 03: TELEMETRY & MODEL */}
              <div ref={frame3Ref} className="flex flex-col items-center gap-3 shrink-0">
                <div className="w-[360px] sm:w-[380px] bg-zinc-950 border-[6px] border-zinc-900 rounded-[40px] shadow-xl overflow-hidden relative font-sans">
                  {/* Phone Top Notch & Status Bar */}
                  <div className="bg-zinc-950 px-6 py-2 flex items-center justify-between text-[11px] font-mono font-bold text-zinc-400 border-b border-zinc-900">
                    <span>9:41 AM</span>
                    <div className="w-20 h-4 bg-zinc-900 rounded-b-xl flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-zinc-800" />
                    </div>
                    <span>5G 100%</span>
                  </div>

                  {/* Inner Phone Screen Content */}
                  <div className="p-3.5 sm:p-4 bg-[#FAFAFA] max-h-[640px] overflow-y-auto space-y-4 w-full max-w-full overflow-x-hidden">
                    <DiagnosticsPanel
                      pipelineSpec={pipelineSpec}
                      chunkTelemetry={chunkTelemetry}
                      chunkFlash={chunkFlash}
                      latencies={latencies}
                      modelSpec={modelSpec}
                      isAttackSimulated={isAttackSimulated}
                    />
                  </div>

                  {/* Phone Bottom Home Bar */}
                  <div className="bg-zinc-950 py-2.5 flex justify-center border-t border-zinc-900">
                    <div className="w-32 h-1 bg-zinc-600 rounded-full" />
                  </div>
                </div>

                {/* Frame Badge */}
                <div className="px-4 py-1.5 rounded-full bg-white text-zinc-950 border border-zinc-300 text-xs font-mono font-extrabold flex items-center gap-2 shadow-2xs">
                  <span className="w-2 h-2 rounded-full bg-zinc-950" />
                  <span>SCREEN 03: TELEMETRY & MODEL</span>
                </div>
              </div>

              {/* PHONE FRAME 04: LIVE TERMINAL */}
              <div ref={frame4Ref} className="flex flex-col items-center gap-3 shrink-0">
                <div className="w-[360px] sm:w-[380px] bg-zinc-950 border-[6px] border-zinc-900 rounded-[40px] shadow-xl overflow-hidden relative font-sans">
                  {/* Phone Top Notch & Status Bar */}
                  <div className="bg-zinc-950 px-6 py-2 flex items-center justify-between text-[11px] font-mono font-bold text-zinc-400 border-b border-zinc-900">
                    <span>9:41 AM</span>
                    <div className="w-20 h-4 bg-zinc-900 rounded-b-xl flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-zinc-800" />
                    </div>
                    <span>5G 100%</span>
                  </div>

                  {/* Inner Phone Screen Content */}
                  <div className="p-3.5 sm:p-4 bg-[#FAFAFA] max-h-[640px] overflow-y-auto space-y-4 w-full max-w-full overflow-x-hidden">
                    <LiveTerminal
                      logs={logs}
                      isMonitoring={isMonitoring}
                      isAttackSimulated={isAttackSimulated}
                    />
                  </div>

                  {/* Phone Bottom Home Bar */}
                  <div className="bg-zinc-950 py-2.5 flex justify-center border-t border-zinc-900">
                    <div className="w-32 h-1 bg-zinc-600 rounded-full" />
                  </div>
                </div>

                {/* Frame Badge */}
                <div className="px-4 py-1.5 rounded-full bg-white text-zinc-950 border border-zinc-300 text-xs font-mono font-extrabold flex items-center gap-2 shadow-2xs">
                  <span className="w-2 h-2 rounded-full bg-zinc-950" />
                  <span>SCREEN 04: LIVE TERMINAL</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Floating Dock */}
      <FloatingDock
        isMonitoring={isMonitoring}
        onToggleMonitoring={() => setIsMonitoring(!isMonitoring)}
        isMicActive={isMicActive}
        onToggleMic={toggleMicrophone}
        isAttackSimulated={isAttackSimulated}
        onToggleAttack={() => setIsAttackSimulated(!isAttackSimulated)}
        showTerminal={showTerminal}
        onToggleTerminal={() => setShowTerminal(!showTerminal)}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      {/* Footer */}
      <footer className="max-w-7xl w-full mx-auto text-center text-xs font-mono text-zinc-500 py-6 border-t border-zinc-200 mt-12 font-semibold">
        <div className="flex items-center justify-center gap-2">
          <Shield className="w-4 h-4 text-zinc-950" />
          <span>GARAJ VOICE SECURITY PLATFORM • PLAIN & MINIMALIST MONOCHROME UI</span>
        </div>
      </footer>
    </div>
  );
};

export default GarajDashboardPage;
