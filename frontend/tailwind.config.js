/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#080C14', // Taste off-black dark canvas
        surface: {
          DEFAULT: '#0F172A',
          hover: '#1E293B',
          glass: 'rgba(15, 23, 42, 0.65)',
          border: 'rgba(255, 255, 255, 0.08)',
        },
        garaj: {
          primary: '#0F766E', // Deep teal-green
          teal: '#14B8A6',
          emerald: '#10B981', // Success
          cyan: '#06B6D4', // Accent live data
          dark: '#080C14',
          card: '#0F172A',
          amber: '#F59E0B',
          coral: '#EF4444',
          slate: '#1E293B',
        },
        primary: {
          DEFAULT: '#14B8A6',
          hover: '#0F766E',
          glow: 'rgba(20, 184, 166, 0.3)',
        },
        success: {
          DEFAULT: '#10B981',
        },
        warning: {
          DEFAULT: '#F59E0B',
        },
        danger: {
          DEFAULT: '#EF4444',
        },
      },
      borderRadius: {
        card: '16px',
        btn: '10px',
        dock: '9999px',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Monaco', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'wave': 'wave 1.4s ease-in-out infinite',
        'radar-spin': 'radarSpin 8s linear infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        wave: {
          '0%, 100%': { transform: 'scaleY(0.3)' },
          '50%': { transform: 'scaleY(1.2)' },
        },
        radarSpin: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '0.4', filter: 'drop-shadow(0 0 8px rgba(20, 184, 166, 0.4))' },
          '50%': { opacity: '0.8', filter: 'drop-shadow(0 0 16px rgba(20, 184, 166, 0.8))' },
        }
      }
    },
  },
  plugins: [],
}
