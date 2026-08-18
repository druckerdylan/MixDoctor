import { useEffect, useState } from 'react';

import { API_BASE } from '../config';

/**
 * What the server this build is pointed at can actually do.
 *
 * Deployments differ: stem separation needs torch and a couple of GB of RAM,
 * so the hosted instance may ship without it while a local one has it. Asking
 * the server beats hard-coding, and it means the UI never offers a control
 * that will quietly do nothing — the user finds out before they wait for an
 * upload rather than after.
 *
 * Defaults are deliberately pessimistic for `stems` (hide the toggle unless
 * told otherwise) and optimistic for `ai` (the write-up is requested anyway,
 * and its own failure path is already handled).
 */
export interface Capabilities {
  stems: boolean;
  ai: boolean;
  version: string;
  maxUploadBytes: number | null;
  loading: boolean;
}

const FALLBACK: Capabilities = {
  stems: false,
  ai: true,
  version: '',
  maxUploadBytes: null,
  loading: false,
};

export function useCapabilities(): Capabilities {
  const [caps, setCaps] = useState<Capabilities>({ ...FALLBACK, loading: true });

  useEffect(() => {
    const controller = new AbortController();
    let alive = true;

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/capabilities`, { signal: controller.signal });
        if (!res.ok) throw new Error(String(res.status));
        const body: unknown = await res.json();
        if (!alive || typeof body !== 'object' || body === null) return;

        const b = body as Record<string, unknown>;
        setCaps({
          stems: b.stems === true,
          ai: b.ai !== false,
          version: typeof b.version === 'string' ? b.version : '',
          maxUploadBytes: typeof b.max_upload_bytes === 'number' ? b.max_upload_bytes : null,
          loading: false,
        });
      } catch {
        // An older server has no /capabilities, and an unreachable one will
        // surface on the analyze call with a better message than anything we
        // could show here. Either way: fall back, do not block the form.
        if (alive) setCaps({ ...FALLBACK, loading: false });
      }
    })();

    return () => {
      alive = false;
      controller.abort();
    };
  }, []);

  return caps;
}
