import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

import type { AnalysisStatus } from '../types/analysis';

const EASE = [0.16, 1, 0.3, 1] as const;

/** Nominal time on each measurement pass while `status === 'measuring'`. */
const STAGE_MS = 1_050;

interface Stage {
  label: string;
  readout: string;
}

/** The order the backend actually works in. The last stage is the LLM call. */
const STAGES: Stage[] = [
  { label: 'Decoding', readout: 'PCM' },
  { label: 'Loudness & true peak', readout: 'LUFS · dBTP' },
  { label: 'Frequency balance', readout: '1/3-OCT' },
  { label: 'Stereo field & phase', readout: 'CORR' },
  { label: 'Dynamics & transients', readout: 'CREST · PSR' },
  { label: 'Low end', readout: 'KICK · SUB' },
  { label: 'Vocal balance', readout: 'CENTER dB' },
  { label: 'Masking & clarity', readout: 'MASK IDX' },
  { label: 'Consulting the engineer', readout: 'REPORT' },
];

const CONSULTING_INDEX = STAGES.length - 1;

const STATUS_LABEL: Record<AnalysisStatus, string> = {
  idle: 'Standing by',
  uploading: 'Transferring audio',
  measuring: 'Measuring',
  consulting: 'Consulting the engineer',
  complete: 'Complete',
  error: 'Failed',
};

/* ------------------------------------------------------------------ */
/* Scanning strip                                                      */
/* ------------------------------------------------------------------ */

/** Seconds for the playhead to cross the whole waveform once. */
const SCAN_PERIOD = 3.4;

/**
 * The waveform with a playhead sweeping across it, drawn straight to canvas so
 * the sweep costs nothing in React. Everything behind the playhead is lit; a
 * short tail keeps the leading edge bright.
 */
function ScanStrip({ peaks }: { peaks: number[] | null }) {
  const reduce = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0;
    let h = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = wrap.clientWidth;
      h = wrap.clientHeight;
      if (w <= 0 || h <= 0) return;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (p: number, showHead: boolean) => {
      if (w <= 0 || h <= 0) return;
      ctx.clearRect(0, 0, w, h);
      const mid = h / 2;
      const head = p * w;

      ctx.fillStyle = 'rgba(255,255,255,0.05)';
      ctx.fillRect(0, mid - 0.5, w, 1);

      const barW = 2;
      const gap = 1;
      const count = Math.max(1, Math.floor(w / (barW + gap)));

      if (peaks && peaks.length > 0) {
        const per = peaks.length / count;
        for (let i = 0; i < count; i += 1) {
          const from = Math.floor(i * per);
          const to = Math.max(from + 1, Math.floor((i + 1) * per));
          let peak = 0;
          for (let j = from; j < to && j < peaks.length; j += 1) {
            const v = peaks[j] ?? 0;
            if (v > peak) peak = v;
          }
          const x = i * (barW + gap);
          const bh = Math.max(1, peak * (mid - 2) * 2);
          if (x > head) {
            ctx.fillStyle = 'rgba(167,167,180,0.22)';
          } else {
            // Brighter within ~90px behind the playhead, cooling off after.
            const heat = Math.max(0, 1 - (head - x) / 90);
            ctx.fillStyle = `rgba(82,242,196,${(0.38 + 0.52 * heat).toFixed(3)})`;
          }
          ctx.fillRect(x, mid - bh / 2, barW, bh);
        }
      } else {
        // No decodable preview: a scope with no signal, ticked rather than faked.
        for (let i = 0; i < count; i += 4) {
          const x = i * (barW + gap);
          ctx.fillStyle = x <= head ? 'rgba(82,242,196,0.4)' : 'rgba(167,167,180,0.14)';
          ctx.fillRect(x, mid - 3, 1, 6);
        }
      }

      if (!showHead) return;
      const glow = ctx.createLinearGradient(head - 70, 0, head, 0);
      glow.addColorStop(0, 'rgba(82,242,196,0)');
      glow.addColorStop(1, 'rgba(82,242,196,0.2)');
      ctx.fillStyle = glow;
      ctx.fillRect(head - 70, 0, 70, h);
      ctx.fillStyle = '#8DFFE0';
      ctx.fillRect(head, 0, 1.5, h);
    };

    resize();

    if (reduce) {
      draw(1, false);
      const ro = new ResizeObserver(() => {
        resize();
        draw(1, false);
      });
      ro.observe(wrap);
      return () => ro.disconnect();
    }

    let raf = 0;
    let last = performance.now();
    let clock = 0;

    const frame = (now: number) => {
      clock += Math.min(0.05, (now - last) / 1000);
      last = now;
      draw((clock % SCAN_PERIOD) / SCAN_PERIOD, true);
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      if (raf) return;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    };
    const onVisibility = () => (document.visibilityState === 'visible' ? start() : stop());

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    document.addEventListener('visibilitychange', onVisibility);
    if (document.visibilityState === 'visible') start();

    return () => {
      stop();
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [peaks, reduce]);

  return (
    <div ref={wrapRef} className="relative h-24 w-full sm:h-32">
      <canvas ref={canvasRef} className="block h-full w-full" aria-hidden="true" />
      {!peaks && (
        <span className="pointer-events-none absolute inset-0 grid place-items-center">
          <span className="eyebrow text-ink-faint">Waveform preview unavailable</span>
        </span>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Stage glyph                                                         */
/* ------------------------------------------------------------------ */

type StageState = 'pending' | 'active' | 'done';

function StageGlyph({ state }: { state: StageState }) {
  const reduce = useReducedMotion();

  if (state === 'done') {
    return (
      <span
        aria-hidden="true"
        className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-signal/40 bg-signal/10 text-signal"
      >
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
          <path
            d="M2.5 6.3 4.8 8.6 9.5 3.9"
            stroke="currentColor"
            strokeWidth={1.7}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    );
  }

  if (state === 'active') {
    return (
      <span aria-hidden="true" className="relative grid h-5 w-5 shrink-0 place-items-center">
        <svg viewBox="0 0 20 20" className="absolute inset-0 h-5 w-5 text-signal">
          <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeOpacity={0.22} strokeWidth={1.5} />
          <motion.circle
            cx="10"
            cy="10"
            r="8"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeDasharray="14 36"
            style={{ transformOrigin: '10px 10px' }}
            animate={reduce ? undefined : { rotate: 360 }}
            transition={reduce ? undefined : { duration: 1.4, repeat: Infinity, ease: 'linear' }}
          />
        </svg>
        <span className="h-1.5 w-1.5 rounded-full bg-signal shadow-glow" />
      </span>
    );
  }

  return (
    <span
      aria-hidden="true"
      className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-void-line"
    >
      <span className="h-1 w-1 rounded-full bg-ink-faint/60" />
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* AnalyzingSequence                                                   */
/* ------------------------------------------------------------------ */

export interface AnalyzingSequenceProps {
  status: AnalysisStatus;
  filename: string;
  genre: string;
  peaks: number[] | null;
  onCancel?: (() => void) | null;
}

export default function AnalyzingSequence({
  status,
  filename,
  genre,
  peaks,
  onCancel = null,
}: AnalyzingSequenceProps) {
  const reduce = useReducedMotion();
  const startedRef = useRef<number>(Date.now());
  const measuringStartRef = useRef<number | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    if (status === 'measuring' && measuringStartRef.current === null) {
      measuringStartRef.current = Date.now();
    }
  }, [status]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, []);

  // Stage index is clamped by the *real* status: the measurement passes never
  // run ahead of the server, and the last one holds until 'consulting' lands.
  let active: number;
  if (status === 'complete') {
    active = STAGES.length;
  } else if (status === 'consulting') {
    active = CONSULTING_INDEX;
  } else if (status === 'measuring') {
    const elapsed = now - (measuringStartRef.current ?? now);
    active = Math.min(CONSULTING_INDEX - 1, Math.floor(elapsed / STAGE_MS));
  } else {
    active = 0;
  }

  const seconds = (now - startedRef.current) / 1000;

  return (
    <section className="mx-auto flex min-h-[calc(100svh-3.5rem)] max-w-[1800px] items-center px-4 py-16 sm:min-h-[calc(100svh-4rem)] sm:px-6 lg:px-10">
      <motion.div
        initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18 }}
        animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: EASE }}
        className="mx-auto w-full max-w-3xl"
      >
        {/* Header ------------------------------------------------------ */}
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <span
                className={`h-1.5 w-1.5 rounded-full bg-signal ${reduce ? '' : 'animate-breathe'}`}
                aria-hidden="true"
              />
              <p className="eyebrow text-signal-dim">{STATUS_LABEL[status]}</p>
            </div>
            <h1 className="display mt-4 truncate text-display-md text-ink" title={filename}>
              {filename}
            </h1>
            <p className="mt-2 font-mono text-micro uppercase text-ink-faint">
              {genre} targets · {peaks ? `${peaks.length} peak buckets` : 'no local preview'}
            </p>
          </div>

          <div className="shrink-0 text-right">
            <p className="eyebrow mb-1.5 text-ink-faint">Elapsed</p>
            <p className="stat text-2xl text-ink sm:text-3xl">
              {seconds.toFixed(1)}
              <span className="ml-1 text-sm text-ink-muted">s</span>
            </p>
          </div>
        </div>

        {/* Scanner ----------------------------------------------------- */}
        <div className="panel mt-8 overflow-hidden p-4 sm:p-5">
          <ScanStrip peaks={peaks} />
        </div>

        {/* One quiet announcement per stage, rather than re-reading the list. */}
        <p role="status" aria-live="polite" className="sr-only">
          {STATUS_LABEL[status]}
          {active < STAGES.length ? ` — step ${active + 1} of ${STAGES.length}: ${STAGES[active]?.label ?? ''}` : ''}
        </p>

        {/* Stages ------------------------------------------------------ */}
        <ol className="mt-8" aria-label="Analysis progress">
          {STAGES.map((stage, i) => {
            const state: StageState = i < active ? 'done' : i === active ? 'active' : 'pending';
            return (
              <motion.li
                key={stage.label}
                initial={reduce ? { opacity: 0 } : { opacity: 0, y: 8 }}
                animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: EASE, delay: 0.1 + i * 0.05 }}
                className="border-t border-void-line/60"
              >
                <div className="flex items-center gap-3 py-3">
                  <StageGlyph state={state} />
                  <span
                    className={[
                      'font-mono text-eyebrow uppercase tracking-[0.18em] transition-colors duration-500',
                      state === 'active'
                        ? 'text-ink'
                        : state === 'done'
                          ? 'text-ink-muted'
                          : 'text-ink-faint/70',
                    ].join(' ')}
                  >
                    {stage.label}
                  </span>
                  <span
                    className={[
                      'stat ml-auto text-micro uppercase transition-colors duration-500',
                      state === 'done'
                        ? 'text-signal-dim'
                        : state === 'active'
                          ? 'text-ink-muted'
                          : 'text-ink-faint/50',
                    ].join(' ')}
                  >
                    {state === 'pending' ? '—' : stage.readout}
                  </span>
                </div>
                {state === 'active' && (
                  <div className="shimmer h-px w-full bg-void-line" aria-hidden="true" />
                )}
              </motion.li>
            );
          })}
          <li className="border-t border-void-line/60" aria-hidden="true" />
        </ol>

        {/* Footer ------------------------------------------------------ */}
        <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs leading-relaxed text-ink-faint">
            Most tracks finish in 10–40 seconds. Long or high-sample-rate files take longer — the
            measurement passes run on the full file, not an excerpt.
          </p>
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="shrink-0 font-mono text-micro uppercase text-ink-faint underline-offset-4 transition-colors duration-200 hover:text-ink-dim hover:underline"
            >
              Cancel
            </button>
          )}
        </div>
      </motion.div>
    </section>
  );
}
