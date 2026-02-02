// API configuration
const isDev = window.location.hostname === 'localhost';
export const API_BASE = isDev
  ? 'http://localhost:8000'
  : 'https://mixdoctor-production.up.railway.app';
