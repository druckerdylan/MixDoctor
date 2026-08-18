import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import {
  ACCEPT_ATTR,
  ACCEPTED_EXTENSIONS,
  MAX_UPLOAD_BYTES,
  PREVIEW_MAX_BYTES,
  formatBytes,
} from '../config';

/* ------------------------------------------------------------------ */
/* Peak extraction                                                     */
/* ------------------------------------------------------------------ */

export const PEAK_BUCKETS = 600;

/**
 * Decode the file just far enough to draw it, then throw the PCM away.
 *
 * We keep exactly `buckets` normalized peaks (a few kB) rather than the decoded
 * AudioBuffer, which for a 6-minute 48k stereo master is ~140 MB of float32 and
 * has no business living in React state.
 *
 * Returns null — never throws — if the format is one the browser cannot decode.
 * The preview is a nicety; upload must never depend on it.
 */
export async function decodePeaks(file: File, buckets = PEAK_BUCKETS): Promise<number[] | null> {
  if (file.size > PREVIEW_MAX_BYTES) return null;
  if (typeof OfflineAudioContext === 'undefined') return null;

  try {
    const bytes = await file.arrayBuffer();
    // Length must be >= 1; the value is irrelevant, we only want the decoder.
    const ctx = new OfflineAudioContext(1, 128, 44100);
    const buffer = await ctx.decodeAudioData(bytes);
    if (!buffer || buffer.length === 0) return null;

    const left = buffer.getChannelData(0);
    const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
    const step = buffer.length / buckets;

    const peaks = new Array<number>(buckets);
    let max = 0;

    for (let b = 0; b < buckets; b += 1) {
      const start = Math.floor(b * step);
      const end = Math.min(buffer.length, Math.floor((b + 1) * step));
      let peak = 0;
      for (let i = start; i < end; i += 1) {
        const l = left[i] ?? 0;
        const a = l < 0 ? -l : l;
        if (a > peak) peak = a;
        if (right) {
          const r = right[i] ?? 0;
          const ar = r < 0 ? -r : r;
          if (ar > peak) peak = ar;
        }
      }
      peaks[b] = peak;
      if (peak > max) max = peak;
    }

    if (max <= 0) return null;
    // Normalize so a quiet rough mix still reads as a waveform, not a flat line.
    for (let b = 0; b < buckets; b += 1) peaks[b] = (peaks[b] ?? 0) / max;
    return peaks;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Waveform canvas                                                     */
/* ------------------------------------------------------------------ */

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, size] as const;
}

export interface WaveformPreviewProps {
  peaks: number[] | null;
  className?: string;
  /** 0..1. Everything left of it is drawn lit; null draws the whole thing idle. */
  progress?: number | null;
  litColor?: string;
  idleColor?: string;
  /** Draw a bright vertical playhead at `progress`. */
  playhead?: boolean;
}

/**
 * A static peak-envelope waveform. Hand-drawn to canvas: mirrored bars around a
 * center line, one bar every 3 device-independent pixels, so the shape reads at
 * 320px and at 1400px without re-decoding.
 */
export function WaveformPreview({
  peaks,
  className,
  progress = null,
  litColor = '#52F2C4',
  idleColor = 'rgba(167,167,180,0.32)',
  playhead = false,
}: WaveformPreviewProps) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w <= 0 || size.h <= 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(size.w * dpr);
    canvas.height = Math.floor(size.h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const mid = size.h / 2;

    // Center line — reads as the zero-crossing axis even with no audio loaded.
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fillRect(0, mid - 0.5, size.w, 1);

    if (!peaks || peaks.length === 0) return;

    const barW = 2;
    const gap = 1;
    const count = Math.max(1, Math.floor(size.w / (barW + gap)));
    const per = peaks.length / count;
    const cut = progress === null ? -1 : progress * size.w;

    for (let i = 0; i < count; i += 1) {
      const from = Math.floor(i * per);
      const to = Math.max(from + 1, Math.floor((i + 1) * per));
      let peak = 0;
      for (let j = from; j < to && j < peaks.length; j += 1) {
        const v = peaks[j] ?? 0;
        if (v > peak) peak = v;
      }
      const x = i * (barW + gap);
      // 1px floor so silence is still a visible line rather than a gap.
      const h = Math.max(1, peak * (mid - 1) * 2);
      ctx.fillStyle = x <= cut ? litColor : idleColor;
      ctx.fillRect(x, mid - h / 2, barW, h);
    }

    if (playhead && progress !== null) {
      const x = Math.max(0, Math.min(size.w - 1, progress * size.w));
      const glow = ctx.createLinearGradient(x - 40, 0, x, 0);
      glow.addColorStop(0, 'rgba(82,242,196,0)');
      glow.addColorStop(1, 'rgba(82,242,196,0.22)');
      ctx.fillStyle = glow;
      ctx.fillRect(x - 40, 0, 40, size.h);
      ctx.fillStyle = '#8DFFE0';
      ctx.fillRect(x, 0, 1.5, size.h);
    }
  }, [peaks, size.w, size.h, progress, litColor, idleColor, playhead]);

  return (
    <div ref={wrapRef} className={className}>
      <canvas ref={canvasRef} className="block h-full w-full" aria-hidden="true" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot === -1 ? '' : name.slice(dot).toLowerCase();
}

function validate(file: File): string | null {
  const ext = extensionOf(file.name);
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return `${ext || 'That file type'} isn't supported. Use ${ACCEPTED_EXTENSIONS.join(', ')}.`;
  }
  if (file.size === 0) return 'That file is empty.';
  if (file.size > MAX_UPLOAD_BYTES) {
    return `${formatBytes(file.size)} is over the ${formatBytes(MAX_UPLOAD_BYTES)} limit. Bounce a shorter section or a 24-bit WAV at 48k.`;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* DropZone                                                            */
/* ------------------------------------------------------------------ */

export interface DropZoneProps {
  /** Unique — used to tie the label to the hidden input. */
  id: string;
  file: File | null;
  onFile: (file: File | null) => void;
  /** Fires with the decoded preview peaks, or null when decoding was skipped. */
  onPeaks?: (peaks: number[] | null) => void;
  /** Peaks already decoded for `file` — lets a remount skip re-decoding. */
  initialPeaks?: number[] | null;
  title: string;
  hint: string;
  /** Slimmer variant for the optional reference slot. */
  compact?: boolean;
  disabled?: boolean;
}

export default function DropZone({
  id,
  file,
  onFile,
  onPeaks,
  initialPeaks = null,
  title,
  hint,
  compact = false,
  disabled = false,
}: DropZoneProps) {
  const reduce = useReducedMotion();
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [peaks, setPeaks] = useState<number[] | null>(initialPeaks);
  const [decoding, setDecoding] = useState(false);

  // dragenter/dragleave fire for every nested element. Counting them is the
  // only way to know when the pointer has truly left the panel.
  const depth = useRef(0);
  // Guards against a slow decode of file A landing after the user picks file B.
  const decodeToken = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    (next: File | null) => {
      decodeToken.current += 1;
      const token = decodeToken.current;

      if (!next) {
        setError(null);
        setPeaks(null);
        setDecoding(false);
        onFile(null);
        onPeaks?.(null);
        if (inputRef.current) inputRef.current.value = '';
        return;
      }

      const problem = validate(next);
      if (problem) {
        setError(problem);
        setPeaks(null);
        setDecoding(false);
        onFile(null);
        onPeaks?.(null);
        if (inputRef.current) inputRef.current.value = '';
        return;
      }

      setError(null);
      setPeaks(null);
      onFile(next);

      setDecoding(true);
      void decodePeaks(next).then((result) => {
        if (decodeToken.current !== token) return;
        setDecoding(false);
        setPeaks(result);
        onPeaks?.(result);
      });
    },
    [onFile, onPeaks],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      depth.current = 0;
      setDragging(false);
      if (disabled) return;
      const dropped = event.dataTransfer.files?.[0] ?? null;
      if (dropped) accept(dropped);
    },
    [accept, disabled],
  );

  const waveHeight = compact ? 'h-10' : 'h-20 sm:h-24';

  return (
    <div>
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          depth.current += 1;
          if (!disabled) setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          depth.current = Math.max(0, depth.current - 1);
          if (depth.current === 0) setDragging(false);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        data-dragging={dragging || undefined}
        className={[
          'group relative isolate overflow-hidden rounded-xl2 border bg-void-panel',
          'transition-all duration-300 ease-cine',
          // The file input is transparent and full-bleed, so its own focus ring
          // would be invisible. Put the ring on the panel instead.
          'focus-within:border-signal focus-within:outline focus-within:outline-2',
          'focus-within:outline-offset-2 focus-within:outline-signal',
          compact ? 'p-4' : 'p-5 sm:p-7',
          dragging
            ? 'border-signal bg-void-raised shadow-glow-lg'
            : 'border-void-line hover:border-ink-faint/60 hover:bg-void-raised',
          disabled ? 'pointer-events-none opacity-50' : '',
        ].join(' ')}
      >
        {/* Heat wash that only appears while a file is over the panel. */}
        <div
          aria-hidden="true"
          className={[
            'pointer-events-none absolute inset-0 -z-10 bg-grade-heat transition-opacity duration-500 ease-cine',
            dragging ? 'opacity-[0.09]' : 'opacity-0',
          ].join(' ')}
        />

        <input
          ref={inputRef}
          id={id}
          type="file"
          accept={ACCEPT_ATTR}
          disabled={disabled}
          // A canceled picker must not wipe an already-loaded file.
          onChange={(e) => {
            const picked = e.target.files?.[0];
            if (picked) accept(picked);
          }}
          className={
            file
              ? 'sr-only'
              : 'absolute inset-0 z-20 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed'
          }
        />

        {file ? (
          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="eyebrow mb-1.5 text-signal-dim">{title}</p>
                <p className="truncate font-display text-sm font-semibold tracking-tight text-ink sm:text-base">
                  {file.name}
                </p>
                <p className="stat mt-1 text-xs text-ink-muted">
                  {formatBytes(file.size)}
                  <span className="mx-2 text-ink-faint">/</span>
                  {(extensionOf(file.name) || '—').replace('.', '').toUpperCase()}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <label
                  htmlFor={id}
                  className="cursor-pointer rounded-full border border-void-line px-3 py-1.5 font-mono text-micro uppercase text-ink-muted transition-colors duration-200 hover:border-ink-faint hover:text-ink"
                >
                  Replace
                </label>
                <button
                  type="button"
                  onClick={() => accept(null)}
                  aria-label={`Remove ${file.name}`}
                  className="rounded-full border border-transparent px-3 py-1.5 font-mono text-micro uppercase text-ink-faint transition-colors duration-200 hover:border-sev-critical/40 hover:text-sev-critical"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className={`relative ${waveHeight} overflow-hidden rounded-lg bg-void-deep/70`}>
              {peaks ? (
                <WaveformPreview peaks={peaks} className="h-full w-full px-1" />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <span
                    className={`eyebrow ${decoding && !reduce ? 'animate-breathe' : ''} text-ink-faint`}
                  >
                    {decoding ? 'Reading waveform' : 'Preview unavailable — upload is unaffected'}
                  </span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div
            className={`pointer-events-none flex items-center gap-4 ${compact ? '' : 'sm:gap-6'}`}
          >
            <div
              className={[
                'grid shrink-0 place-items-center rounded-xl border transition-colors duration-300 ease-cine',
                compact ? 'h-10 w-10' : 'h-14 w-14',
                dragging
                  ? 'border-signal/50 bg-signal/10 text-signal'
                  : 'border-void-line bg-void-deep text-ink-faint group-hover:text-ink-muted',
              ].join(' ')}
              aria-hidden="true"
            >
              <svg
                width={compact ? 16 : 22}
                height={compact ? 16 : 22}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.6}
                strokeLinecap="round"
              >
                <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
                <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" />
              </svg>
            </div>

            <div className="min-w-0">
              <p className="eyebrow mb-1.5 text-ink-faint">{title}</p>
              <p
                className={`font-display font-semibold tracking-tight text-ink ${compact ? 'text-sm' : 'text-base sm:text-lg'}`}
              >
                {dragging ? 'Release to load' : 'Drop a file, or click to browse'}
              </p>
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">{hint}</p>
            </div>
          </div>
        )}
      </div>

      <AnimatePresence initial={false}>
        {error && (
          <motion.p
            role="alert"
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: -4 }}
            animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="mt-2.5 flex items-start gap-2 text-xs leading-relaxed text-sev-critical"
          >
            <span aria-hidden="true" className="mt-[2px] font-mono">
              ✕
            </span>
            <span>{error}</span>
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}
