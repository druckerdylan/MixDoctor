/// <reference types="vite/client" />

// API configuration
// In development, uses localhost. In production, uses the environment variable.
export const API_BASE = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';
