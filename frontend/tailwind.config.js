/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'mix': {
          'primary': '#818cf8',      // Softer indigo (was #6366f1)
          'secondary': '#a78bfa',    // Softer purple (was #8b5cf6)
          'success': '#34d399',      // Softer green (was #22c55e)
          'warning': '#fbbf24',      // Softer amber (was #f59e0b)
          'danger': '#f87171',       // Softer red (was #ef4444)
          'dark': '#1a1a2e',         // Slightly adjusted dark
          'darker': '#0f0f1a',       // Deeper background
          'surface': '#232336',      // New: elevated surface
          'muted': '#2a2a40',        // New: subtle backgrounds
        }
      }
    },
  },
  plugins: [],
}
