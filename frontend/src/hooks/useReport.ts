/**
 * Generating the downloadable document.
 *
 * Two outputs from one endpoint. Markdown is a plain download; "Print / Save as
 * PDF" is HTML opened in a tab with `print()` called on it, because the browser
 * already renders a better PDF than any library we could ship and it costs the
 * image nothing.
 *
 * Three things in here are load-bearing:
 *
 * 1. **The tab is opened synchronously, before the fetch.** `window.open` from
 *    inside a `.then()` is a popup, and every browser blocks it. So the tab is
 *    claimed inside the click handler, shown a plain "generating" holding page,
 *    and its document replaced once the HTML arrives.
 * 2. **Nothing here can break the results page.** Every failure — offline,
 *    500, a blocked popup — ends as a string in `error` and a button that is
 *    clickable again. The report is an extra; the analysis on screen is not.
 * 3. **The object URL is revoked, but only after the download has started.**
 *    Revoking synchronously after `click()` races the download in Safari.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { API_BASE } from '../config';
import type { MixAnalysis, OwnedPlugin } from '../types/analysis';

export type ReportFormat = 'markdown' | 'html';
export type ReportStatus = 'idle' | 'generating' | 'done' | 'failed';

export interface UseReportReturn {
  status: ReportStatus;
  /** Which button is busy, so only that one shows a spinner. */
  pending: ReportFormat | null;
  error: string | null;
  downloadMarkdown: () => void;
  openPrintable: () => void;
  dismissError: () => void;
}

/** FastAPI returns `{detail: string}` on HTTPException and a list on 422. */
function readDetail(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === 'string') return message;
  }
  return null;
}

async function errorFromResponse(response: Response): Promise<string> {
  try {
    const detail = readDetail(await response.json());
    if (detail) return detail;
  } catch {
    /* Not JSON. The status line below is all we have. */
  }
  return `The server returned ${response.status} while building the document.`;
}

/** `{track-name}-mix-diagnostic-report.md`, matching what the backend suggests. */
export function reportFilename(analysis: MixAnalysis, extension = 'md'): string {
  const stem = (analysis.filename || 'mix').replace(/\.[^.]+$/, '');
  const slug =
    stem
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 80) || 'mix';
  return `${slug}-mix-diagnostic-report.${extension}`;
}

/**
 * What the new tab shows while the document is being built. Inline everything —
 * this document never reaches the network, and it is replaced wholesale by the
 * report a moment later.
 */
const HOLDING_PAGE = `<!doctype html><meta charset="utf-8"><title>Building your report…</title>
<style>
  body{margin:0;display:grid;place-items:center;min-height:100vh;background:#fff;color:#14161a;
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  div{text-align:center}
  p{color:#6a7280;margin:.4rem 0 0;font-size:13px}
  @media (prefers-color-scheme:dark){body{background:#0b0c0e;color:#f2f4f7}p{color:#8b93a1}}
</style>
<div><strong>Building your report…</strong><p>This tab will fill in automatically.</p></div>`;

export function useReport(
  analysis: MixAnalysis | null,
  plugins: readonly OwnedPlugin[] = [],
): UseReportReturn {
  const [status, setStatus] = useState<ReportStatus>('idle');
  const [pending, setPending] = useState<ReportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef(true);
  const inflight = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      abortRef.current?.abort();
    };
  }, []);

  const fetchReport = useCallback(
    async (format: ReportFormat, signal: AbortSignal): Promise<string> => {
      if (!analysis) throw new Error('There is no analysis to write up yet.');

      const query = format === 'html' ? '?format=html' : '';
      const response = await fetch(`${API_BASE}/report${query}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis, plugins }),
        signal,
      });
      if (!response.ok) throw new Error(await errorFromResponse(response));
      return response.text();
    },
    [analysis, plugins],
  );

  const run = useCallback(
    async (format: ReportFormat, onReady: (body: string) => void, onFail: () => void) => {
      if (inflight.current || !analysis) return;
      inflight.current = true;

      const controller = new AbortController();
      abortRef.current = controller;
      setPending(format);
      setStatus('generating');
      setError(null);

      try {
        const body = await fetchReport(format, controller.signal);
        onReady(body);
        if (mounted.current) setStatus('done');
      } catch (err) {
        onFail();
        if (!mounted.current || controller.signal.aborted) return;
        // A blocked or refused fetch arrives as a bare TypeError with no
        // useful message, so name the thing that is probably wrong.
        const message =
          err instanceof TypeError
            ? `Could not reach ${API_BASE} to build the document.`
            : err instanceof Error
              ? err.message
              : 'The document could not be generated.';
        setError(message);
        setStatus('failed');
      } finally {
        inflight.current = false;
        if (mounted.current) setPending(null);
      }
    },
    [analysis, fetchReport],
  );

  const downloadMarkdown = useCallback(() => {
    if (!analysis) return;
    void run(
      'markdown',
      (body) => {
        const url = URL.createObjectURL(new Blob([body], { type: 'text/markdown;charset=utf-8' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = reportFilename(analysis, 'md');
        document.body.appendChild(link);
        link.click();
        link.remove();
        // Safari cancels an in-flight download if the URL dies immediately.
        window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      },
      () => undefined,
    );
  }, [analysis, run]);

  const openPrintable = useCallback(() => {
    if (!analysis) return;

    // Claimed synchronously inside the click, or the popup blocker eats it.
    const tab = window.open('', '_blank');
    if (!tab) {
      setError(
        'Your browser blocked the new tab. Allow pop-ups for this site, or download the Markdown instead.',
      );
      setStatus('failed');
      return;
    }
    tab.document.write(HOLDING_PAGE);
    tab.document.close();

    void run(
      'html',
      (body) => {
        tab.document.open();
        tab.document.write(body);
        tab.document.close();
        tab.focus();
        // Let the replaced document lay out before the print dialogue reads it;
        // printing too early gives a blank or half-styled first page.
        tab.setTimeout(() => tab.print(), 400);
      },
      () => tab.close(),
    );
  }, [analysis, run]);

  const dismissError = useCallback(() => {
    setError(null);
    setStatus('idle');
  }, []);

  return { status, pending, error, downloadMarkdown, openPrintable, dismissError };
}

export default useReport;
