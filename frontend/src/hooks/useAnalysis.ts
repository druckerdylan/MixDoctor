import { useState, useCallback } from 'react';
import type { AnalysisResult, AnalysisStatus, StemsAnalysisResult, StemType } from '../types';

import { API_BASE } from '../config';

interface UseAnalysisReturn {
  status: AnalysisStatus;
  result: AnalysisResult | null;
  stemsResult: StemsAnalysisResult | null;
  error: string | null;
  analyze: (file: File, genre: string, reference?: string, token?: string | null, referenceFile?: File) => Promise<void>;
  analyzeStems: (stems: Map<StemType, File>, genre: string, token?: string | null) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stemsResult, setStemsResult] = useState<StemsAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File, genre: string, reference?: string, token?: string | null, referenceFile?: File) => {
    setStatus('uploading');
    setError(null);
    setResult(null);
    setStemsResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('genre', genre);
      if (reference) {
        formData.append('reference', reference);
      }
      if (referenceFile) {
        formData.append('reference_file', referenceFile);
      }

      setStatus('analyzing');

      const headers: HeadersInit = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      setStatus('complete');
    } catch (err) {
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        setError('Unable to connect to the server. Please make sure the backend is running.');
      } else {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      }
      setStatus('error');
    }
  }, []);

  const analyzeStems = useCallback(async (stems: Map<StemType, File>, genre: string, token?: string | null) => {
    setStatus('uploading');
    setError(null);
    setResult(null);
    setStemsResult(null);

    try {
      const formData = new FormData();
      formData.append('genre', genre);

      // Add stem types and files
      const stemTypes: string[] = [];
      stems.forEach((file, stemType) => {
        stemTypes.push(stemType);
        formData.append('stems', file);
      });
      formData.append('stem_types', stemTypes.join(','));

      setStatus('analyzing');

      const headers: HeadersInit = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE}/analyze-stems`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Stems analysis failed');
      }

      const data: StemsAnalysisResult = await response.json();
      setStemsResult(data);
      setStatus('complete');
    } catch (err) {
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        setError('Unable to connect to the server. Please make sure the backend is running.');
      } else {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      }
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setStemsResult(null);
    setError(null);
  }, []);

  return { status, result, stemsResult, error, analyze, analyzeStems, reset };
}
