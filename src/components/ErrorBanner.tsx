import React from 'react';
import { AlertCircle, RefreshCw, X } from 'lucide-react';
import { Button } from './Button';

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = 'Voice Connection Alert',
  message,
  onRetry,
  onDismiss,
}) => {
  return (
    <div className="bg-red-500/10 border border-red-500/30 rounded-card p-4 text-red-200 flex items-start gap-3 my-4 shadow-lg shadow-red-950/20 backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-300">
      <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      
      <div className="flex-1">
        <h4 className="text-sm font-semibold text-red-300">{title}</h4>
        <p className="text-xs text-red-200/90 mt-0.5 leading-relaxed">{message}</p>
        
        {onRetry && (
          <div className="mt-3">
            <Button
              variant="danger"
              size="sm"
              leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
              onClick={onRetry}
            >
              Retry Connection
            </Button>
          </div>
        )}
      </div>

      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-200 p-1 rounded-md transition-colors"
          aria-label="Dismiss banner"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};
