import React from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  className,
  disabled,
  ...props
}) => {
  // Taste-Skill Button Rule: Single-line CTA text, strong tactile feedback on active
  const baseStyles = 'inline-flex items-center justify-center font-semibold rounded-btn transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98] whitespace-nowrap';

  const variants = {
    primary: 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 border border-indigo-500/40 focus:ring-indigo-500',
    secondary: 'bg-slate-800/90 hover:bg-slate-700/90 text-slate-100 border border-slate-700/80 focus:ring-slate-500',
    danger: 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/20 border border-red-500/40 focus:ring-red-500',
    outline: 'border border-slate-700 bg-slate-900/40 text-slate-200 hover:bg-slate-800/80 hover:border-slate-600 focus:ring-indigo-500',
    ghost: 'text-slate-300 hover:bg-slate-800/60 hover:text-white focus:ring-slate-500',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-sm md:text-base gap-2.5 font-bold',
  };

  return (
    <button
      className={twMerge(clsx(baseStyles, variants[variant], sizes[size], className))}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin text-current shrink-0" />
      ) : leftIcon ? (
        <span className="shrink-0">{leftIcon}</span>
      ) : null}
      
      <span className="truncate">{children}</span>

      {!isLoading && rightIcon && (
        <span className="shrink-0">{rightIcon}</span>
      )}
    </button>
  );
};
