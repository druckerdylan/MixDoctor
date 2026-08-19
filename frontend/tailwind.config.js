/** @type {import('tailwindcss').Config} */

// Cinematic dark. Three rules the whole UI follows:
//   1. The background is near-black, not navy. Colour comes from signal only.
//   2. Every number is monospace and tabular — columns of dB values must align.
//   3. Severity has exactly four colours and they never mean anything else.
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        void: {
          DEFAULT: '#06060A', // page
          deep: '#030306', // beneath the page — vignette, modal scrim
          panel: '#0D0D13', // cards
          raised: '#14141C', // hovered / elevated
          line: '#1E1E29', // hairline borders
          lineSoft: '#16161F',
        },
        ink: {
          DEFAULT: '#F4F4F7',
          dim: '#A7A7B4',
          muted: '#70707E',
          faint: '#4A4A57',
        },
        // Health / signal. Mint reads as "instrument", not "success toast".
        signal: {
          DEFAULT: '#52F2C4',
          dim: '#2FBF97',
          glow: '#8DFFE0',
        },
        // Severity — the only four. Chart marks, badges and chrome all key off these.
        sev: {
          critical: '#FF3B58',
          major: '#FF9F1C',
          minor: '#4CC9F0',
          clean: '#52F2C4',
        },
        // Frequency heat ramp: sub -> air. Used by the spectrum and the mix map.
        spectrum: {
          sub: '#5B4BFF',
          low: '#2E7BFF',
          mid: '#4CC9F0',
          high: '#52F2C4',
          air: '#C8FF6B',
        },
        ember: '#FF6B4A',
      },
      fontFamily: {
        sans: ['"Archivo Variable"', 'Archivo', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['"Archivo Variable"', 'Archivo', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono Variable"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        // Editorial display scale — tight tracking, set at the top of the page.
        'display-xl': ['clamp(3.5rem, 11vw, 8.5rem)', { lineHeight: '0.97', letterSpacing: '-0.03em' }],
        'display-lg': ['clamp(2.5rem, 7vw, 5rem)', { lineHeight: '0.92', letterSpacing: '-0.035em' }],
        'display-md': ['clamp(1.75rem, 4vw, 2.75rem)', { lineHeight: '1.0', letterSpacing: '-0.025em' }],
        // All-caps micro labels above data.
        eyebrow: ['0.6875rem', { lineHeight: '1', letterSpacing: '0.18em' }],
        micro: ['0.625rem', { lineHeight: '1', letterSpacing: '0.14em' }],
      },
      spacing: { 18: '4.5rem', 22: '5.5rem', 30: '7.5rem' },
      borderRadius: { xl2: '1.25rem', xl3: '1.75rem' },
      boxShadow: {
        // Bloom, not drop-shadow. Lit-from-within is the whole look.
        glow: '0 0 24px -4px var(--glow-color, rgba(82,242,196,0.45))',
        'glow-lg': '0 0 60px -10px var(--glow-color, rgba(82,242,196,0.5))',
        panel: '0 1px 0 0 rgba(255,255,255,0.03) inset, 0 20px 50px -24px rgba(0,0,0,0.9)',
        lift: '0 24px 60px -28px rgba(0,0,0,0.95)',
      },
      backgroundImage: {
        'grade-signal': 'linear-gradient(135deg, #52F2C4 0%, #4CC9F0 100%)',
        'grade-heat':
          'linear-gradient(90deg, #5B4BFF 0%, #2E7BFF 22%, #4CC9F0 45%, #52F2C4 70%, #C8FF6B 100%)',
        'grade-alarm': 'linear-gradient(135deg, #FF3B58 0%, #FF9F1C 100%)',
        vignette:
          'radial-gradient(ellipse 120% 80% at 50% 0%, rgba(82,242,196,0.07) 0%, transparent 60%)',
      },
      transitionTimingFunction: {
        // Slow-out. Cinematic motion decelerates; it doesn't bounce.
        cine: 'cubic-bezier(0.16, 1, 0.3, 1)',
        drift: 'cubic-bezier(0.33, 1, 0.68, 1)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(14px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'sweep-x': { from: { transform: 'translateX(-100%)' }, to: { transform: 'translateX(200%)' } },
        breathe: { '0%,100%': { opacity: '0.45' }, '50%': { opacity: '1' } },
        'spin-slow': { to: { transform: 'rotate(360deg)' } },
      },
      animation: {
        'fade-up': 'fade-up 0.7s cubic-bezier(0.16,1,0.3,1) both',
        'sweep-x': 'sweep-x 2.2s cubic-bezier(0.4,0,0.2,1) infinite',
        breathe: 'breathe 2.6s ease-in-out infinite',
        'spin-slow': 'spin-slow 14s linear infinite',
      },
    },
  },
  plugins: [],
};
