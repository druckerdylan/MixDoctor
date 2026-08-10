import { useCallback, useEffect, useRef, useState } from 'react';

import { API_BASE } from '../config';
import type { AnalysisStatus, EngineerReport, MixAnalysis, TrackIntent } from '../types/analysis';

/** Everything the intake form collects, in one typed payload. */
export interface AnalyzeRequest {
  file: File;
  /**
   * What the file is. Genre decides the reference the mix is compared against;
   * this decides which comparisons are worth making at all — a beat's absent
   * lead is correct, a stem has no mix balance to judge, and nobody is being
   * asked to fix a released record. Omit to leave the server on `full_mix`.
   */
  intent?: TrackIntent;
  genre: string;
  referenceFile?: File | null;
  notes?: string | null;
  /**
   * Opt-in source separation. Materially slower — the server pulls the mix
   * apart before it measures anything — so it is off unless asked for, and the
   * server degrades to a normal analysis if the stage fails.
   */
  separateStems?: boolean;
}

/**
 * The write-up is a second request, so it gets its own state.
 *
 * `POST /analyze` returns every measured number in ~3s. `POST /engineer` then
 * generates the prose, and that call is bound by output length — measured at
 * 2-4 minutes. Folding them into one request would 504 behind most proxies and
 * would make the user stare at a spinner while finished results already exist.
 */
export type EngineerStatus = 'idle' | 'consulting' | 'ready' | 'failed' | 'unavailable';

export interface UseAnalysisReturn {
  status: AnalysisStatus;
  analysis: MixAnalysis | null;
  error: string | null;
  engineerStatus: EngineerStatus;
  engineerError: string | null;
  retryEngineer: () => void;
  /** Object URL for the user's own file, so the timeline can play it locally. */
  audioUrl: string | null;
  analyze: (request: AnalyzeRequest) => Promise<void>;
  /** Swap in a re-scored copy of the report already on screen. */
  applyAnalysis: (next: MixAnalysis) => void;
  reset: () => void;
}

/** Measurement is fast; this only has to cover the upload itself. */
const UPLOAD_PHASE_MS = 1_200;

/** FastAPI returns {detail: string} on HTTPException and a list on validation errors. */
function readDetail(body: unknown): string | null {
  if (typeof body !== 'object' || body === null) return null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string' && detail.trim() !== '') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) =>
        typeof entry === 'object' && entry !== null && typeof (entry as { msg?: unknown }).msg === 'string'
          ? (entry as { msg: string }).msg
          : null,
      )
      .filter((msg): msg is string => msg !== null);
    if (messages.length > 0) return messages.join('. ');
  }
  return null;
}

async function errorFromResponse(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = readDetail(body);
    if (detail) return detail;
  } catch {
    // Non-JSON body (a proxy 502 page, a gateway timeout) — fall through.
  }
  if (response.status === 413) return 'That file is too large for the server to accept.';
  if (response.status === 504) return 'The server timed out while analysing. Try a shorter excerpt.';
  return `Analysis failed (HTTP ${response.status} ${response.statusText || 'error'}).`;
}

export function useAnalysis(): UseAnalysisReturn {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [analysis, setAnalysis] = useState<MixAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [engineerStatus, setEngineerStatus] = useState<EngineerStatus>('idle');
  const [engineerError, setEngineerError] = useState<string | null>(null);

  // Refs so the unmount cleanup below can see the *current* values without
  // re-running (and therefore without revoking a URL that is still on screen).
  const audioUrlRef = useRef<string | null>(null);
  const timersRef = useRef<number[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  /** Increments on every analyze()/reset() so a stale in-flight request cannot win. */
  const runRef = useRef(0);

  const clearTimers = useCallback(() => {
    for (const id of timersRef.current) window.clearTimeout(id);
    timersRef.current = [];
  }, []);

  const abortInflight = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const releaseAudioUrl = useCallback(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      // Unmount: kill the phase timers, drop the consult, hand the blob back.
      for (const id of timersRef.current) window.clearTimeout(id);
      timersRef.current = [];
      abortRef.current?.abort();
      abortRef.current = null;
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
    },
    [],
  );

  /**
   * Ask for the write-up and fold it into the analysis already on screen.
   *
   * Stateless on the server: we post back the analysis we hold. A failure here
   * is explicitly non-fatal — every measured finding is already rendered, so
   * the page degrades to "no prose" rather than "no results".
   */
  const consult = useCallback(async (subject: MixAnalysis, run: number) => {
    abortInflight();
    const controller = new AbortController();
    abortRef.current = controller;

    setEngineerStatus('consulting');
    setEngineerError(null);

    try {
      const response = await fetch(`${API_BASE}/engineer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(subject),
        signal: controller.signal,
      });
      if (runRef.current !== run) return;

      if (!response.ok) {
        // 503 means the operator turned the layer off — that is a different
        // message from "it broke", and worth distinguishing in the UI.
        const message = await errorFromResponse(response);
        setEngineerError(message);
        setEngineerStatus(response.status === 503 ? 'unavailable' : 'failed');
        return;
      }

      const report = (await response.json()) as EngineerReport;
      if (runRef.current !== run) return;

      setAnalysis((current) => (current ? { ...current, engineer: report } : current));
      setEngineerStatus('ready');
    } catch (err) {
      if (runRef.current !== run) return;
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setEngineerError(
        err instanceof TypeError
          ? `Could not reach ${API_BASE} for the write-up.`
          : err instanceof Error
            ? err.message
            : 'The write-up could not be generated.',
      );
      setEngineerStatus('failed');
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [abortInflight]);

  const analyze = useCallback(
    async (request: AnalyzeRequest) => {
      const run = ++runRef.current;
      clearTimers();
      abortInflight();
      releaseAudioUrl();

      setAnalysis(null);
      setError(null);
      setEngineerError(null);
      setEngineerStatus('idle');
      setStatus('uploading');

      const url = URL.createObjectURL(request.file);
      audioUrlRef.current = url;
      setAudioUrl(url);

      timersRef.current.push(
        window.setTimeout(() => {
          if (runRef.current === run) setStatus('measuring');
        }, UPLOAD_PHASE_MS),
      );

      const form = new FormData();
      form.append('file', request.file);
      form.append('genre', request.genre);
      if (request.intent) form.append('intent', request.intent);
      if (request.referenceFile) form.append('reference_file', request.referenceFile);
      const notes = request.notes?.trim();
      if (notes) form.append('notes', notes);
      // Only sent when asked for: an absent field leaves the server on its own
      // default rather than us asserting one from the client.
      if (request.separateStems) form.append('separate_stems', 'true');

      try {
        const response = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form });
        if (runRef.current !== run) return;

        if (!response.ok) throw new Error(await errorFromResponse(response));

        const data = (await response.json()) as MixAnalysis;
        if (runRef.current !== run) return;

        clearTimers();
        setAnalysis(data);
        setStatus('complete');

        // Results are on screen now; the write-up catches up behind them.
        void consult(data, run);
      } catch (err) {
        if (runRef.current !== run) return;
        clearTimers();
        // A blocked/refused fetch surfaces as a bare TypeError with no useful
        // message, so name the host we actually tried to reach.
        const message =
          err instanceof TypeError
            ? `Could not reach the analysis server at ${API_BASE}. Check that it is running and reachable.`
            : err instanceof Error
              ? err.message
              : 'Something went wrong during analysis.';
        setError(message);
        setStatus('error');
      }
    },
    [abortInflight, clearTimers, consult, releaseAudioUrl],
  );

  /**
   * Swap in a re-scored report — what `POST /reassess` returns once the
   * producer has said which findings were deliberate.
   *
   * No audio is re-uploaded and nothing is re-measured: the response carries
   * the same measurements back, with the judgement on top of them updated. The
   * one thing worth guarding is the write-up, which arrives on its own slower
   * request and may have landed after the copy that was posted was taken.
   */
  const applyAnalysis = useCallback((next: MixAnalysis) => {
    setAnalysis((current) => {
      if (!current) return current;
      return next.engineer ? next : { ...next, engineer: current.engineer };
    });
  }, []);

  const retryEngineer = useCallback(() => {
    if (!analysis) return;
    void consult(analysis, runRef.current);
  }, [analysis, consult]);

  const reset = useCallback(() => {
    runRef.current += 1;
    clearTimers();
    abortInflight();
    releaseAudioUrl();
    setAudioUrl(null);
    setAnalysis(null);
    setError(null);
    setEngineerError(null);
    setEngineerStatus('idle');
    setStatus('idle');
  }, [abortInflight, clearTimers, releaseAudioUrl]);

  return {
    status,
    analysis,
    error,
    engineerStatus,
    engineerError,
    retryEngineer,
    audioUrl,
    analyze,
    applyAnalysis,
    reset,
  };
}
