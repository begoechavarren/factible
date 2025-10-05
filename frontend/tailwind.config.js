/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#f3f1de',
        primary: '#4b0fc0',
        accent: '#E94E3A',
        'pill-bg': 'rgba(255, 255, 255, 0.75)',
        'pill-border': 'rgba(75, 15, 192, 0.3)',
      },
      fontFamily: {
        pixel: ['VT323', 'monospace'],
        clean: ['Inter', 'sans-serif'],
      },
      letterSpacing: {
        pixel: '0.5px',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
    },
  },
  plugins: [],
}
