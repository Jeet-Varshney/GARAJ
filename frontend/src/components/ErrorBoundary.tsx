import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { Button } from './Button';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Unhandled React Error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex items-center justify-center p-6">
          <div className="max-w-md w-full taste-card p-6 border-red-500/30 text-center space-y-4 shadow-2xl">
            <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto text-red-400">
              <AlertCircle className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100">Session Interface Exception</h2>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                {this.state.error?.message || 'An unexpected error occurred while rendering the voice session.'}
              </p>
            </div>
            <Button
              variant="primary"
              size="md"
              onClick={this.handleReset}
              leftIcon={<RotateCcw className="w-4 h-4" />}
              className="w-full"
            >
              Return to Landing Page
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
