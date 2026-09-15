import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConversationProvider } from '@elevenlabs/react';
import { AppProvider } from './context/AppContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { GarajDashboardPage } from './pages/GarajDashboardPage';
import { LandingPage } from './pages/LandingPage';
import { InterviewPage } from './pages/InterviewPage';
import { ReportPage } from './pages/ReportPage';
import { BasicAgentPage } from './pages/BasicAgentPage';

export function App() {
  return (
    <ErrorBoundary>
      <ConversationProvider>
        <AppProvider>
          <BrowserRouter>
            <Routes>
              {/* Primary GARAJ Voice Security Dashboard */}
              <Route path="/" element={<GarajDashboardPage />} />
              {/* Practice & Mock Interview Simulator Routes */}
              <Route path="/simulator" element={<LandingPage />} />
              <Route path="/agent-launcher" element={<BasicAgentPage />} />
              <Route path="/interview" element={<InterviewPage />} />
              <Route path="/report" element={<ReportPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </AppProvider>
      </ConversationProvider>
    </ErrorBoundary>
  );
}

export default App;
