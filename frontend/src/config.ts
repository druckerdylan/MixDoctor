// API configuration
const isDev = window.location.hostname === 'localhost';
export const API_BASE = isDev
  ? 'http://localhost:8000'
  : 'https://mixdoctor-production.up.railway.app';
// Trigger rebuild Tue Feb  3 12:23:32 EST 2026
