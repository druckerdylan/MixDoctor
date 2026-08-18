/**
 * Timeline — the "I can hear it" view.
 *
 * Waveform on canvas (peaks outside, RMS body inside, mirrored), one issue lane
 * per dimension that actually has moments, a scrubbable playhead across all of
 * it. Everything wide scrolls inside this component; the page never widens.
 */

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  DIMENSION_LABELS,
  DIMENSION_SHORT,
  DIMENSIONS,
  SEVERITY_RANK,
  SEVERITY_VAR,
  formatTime,
  type Dimension,
  type Finding,
  type MixAnalysis,
  type Moment,
  type Severity,
} from '../../types/analysis';
import useAudioPlayback from '../../hooks/useAudioPlayback';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const RULER_H = 26;
const LANE_H = 26;
const LANE_GAP = 5;
const ZOOMS = [1, 2, 4, 8] as const;

type Zoom = (typeof ZOOMS)[number];

interface LaneItem {
  key: string;
  finding: Finding;
  moment: Moment;
}

interface Lane {
  dimension: Dimension;
  /** Short form — the gutter is narrow and truncation reads as a bug. */
  label: string;
  fullLabel: string;
  worst: Severity;
  items: LaneItem[];
}

interface HoverInfo {
  key: string;
  x: number;
  top: number;
  finding: Finding;
  moment: Moment;
}

export interface TimelineProps {
  analysis: MixAnalysis;
  audioUrl: string | null;
  selected: Dimension | null;
  onSeek: (t: number) => void;
}

/* ------------------------------------------------------------------ utils */

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Pick a ruler interval that keeps ticks at least ~64px apart. */
function tickStep(duration: number, px: number): number {
  const candidates = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600];
  for (const s of candidates) {
    if (duration > 0 && (s / duration) * px >= 64) return s;
  }
  return 600;
}

function drawWaveform(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  peaks: number[],
  rms: number[],
  lit: boolean,
): void {
  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const mid = h / 2;
  ctx.fillStyle = lit ? 'rgba(82,242,196,0.40)' : 'rgba(30,30,41,1)';
  ctx.fillRect(0, mid - 0.5, w, 1);

  const n = peaks.length;
  if (n === 0) return;

  let max = 0;
  for (let i = 0; i < n; i += 1) {
    const v = Math.abs(peaks[i] ?? 0);
    if (v > max) max = v;
  }
  if (!(max > 0)) return;

  const norm = 1 / max;
  const step = 3;
  const barW = 2;
  const cols = Math.max(1, Math.floor(w / step));
  const rn = rms.length;
  const amp = Math.max(1, mid - 1);

  const peakFill = lit ? 'rgba(82,242,196,0.38)' : 'rgba(112,112,126,0.34)';
  const rmsFill = lit ? 'rgba(141,255,224,0.95)' : 'rgba(180,180,196,0.78)';

  for (let c = 0; c < cols; c += 1) {
    const a = Math.floor((c / cols) * n);
    const b = Math.max(a + 1, Math.floor(((c + 1) / cols) * n));
    let p = 0;
    for (let i = a; i < b && i < n; i += 1) {
      const v = Math.abs(peaks[i] ?? 0);
      if (v > p) p = v;
    }
    let r = 0;
    if (rn > 0) {
      const ra = Math.floor((c / cols) * rn);
      const rb = Math.max(ra + 1, Math.floor(((c + 1) / cols) * rn));
      for (let i = ra; i < rb && i < rn; i += 1) {
        const v = Math.abs(rms[i] ?? 0);
        if (v > r) r = v;
      }
    }
    const x = c * step;
    const ph = Math.max(0.7, p * norm * amp);
    ctx.fillStyle = peakFill;
    ctx.fillRect(x, mid - ph, barW, ph * 2);
    if (r > 0) {
      const rh = Math.max(0.7, Math.min(ph, r * norm * amp));
      ctx.fillStyle = rmsFill;
      ctx.fillRect(x, mid - rh, barW, rh * 2);
    }
  }
}

/* ------------------------------------------------------------- sub-pieces */

const WaveCanvas = memo(function WaveCanvas({
  width,
  height,
  peaks,
  rms,
  lit,
}: {
  width: number;
  height: number;
  peaks: number[];
  rms: number[];
  lit: boolean;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || width <= 0 || height <= 0) return;
    drawWaveform(el, width, height, peaks, rms, lit);
  }, [width, height, peaks, rms, lit]);
  return <canvas ref={ref} aria-hidden="true" className="absolute left-0 top-0" />;
});

const Ruler = memo(function Ruler({
  width,
  duration,
}: {
  width: number;
  duration: number;
}) {
  const step = tickStep(duration, width);
  const ticks: number[] = [];
  for (let t = 0; t <= duration + 0.001; t += step) ticks.push(t);
  return (
    <div className="relative" style={{ height: RULER_H }}>
      <div className="absolute inset-x-0 bottom-0 h-px bg-void-line" />
      {ticks.map((t) => {
        const left = duration > 0 ? (t / duration) * width : 0;
        if (left > width - 2) return null;
        return (
          <div key={t} className="absolute bottom-0 top-0" style={{ left }}>
            <div className="absolute bottom-0 h-2 w-px bg-ink-faint/50" />
            {/* Drop the label rather than let it run past the right edge. */}
            {left < width - 38 && (
              <span className="absolute bottom-2.5 left-1 whitespace-nowrap font-mono text-micro text-ink-faint">
                {formatTime(t)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
});

/**
 * Only *critical* moments mark the waveform. Washing the long major regions
 * across it too turned the waveform muddy and cost more legibility than the
 * redundancy with the lanes was worth.
 */
const WaveTint = memo(function WaveTint({
  lanes,
  width,
  duration,
}: {
  lanes: Lane[];
  width: number;
  duration: number;
}) {
  if (duration <= 0) return null;
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0">
      {lanes.flatMap((lane) =>
        lane.items
          .filter((i) => i.finding.severity === 'critical')
          .map((i) => {
            const left = clamp((i.moment.t_start / duration) * width, 0, width);
            const right = clamp((i.moment.t_end / duration) * width, 0, width);
            return (
              <div
                key={`tint-${i.key}`}
                className="absolute inset-y-0"
                style={{
                  left,
                  width: Math.max(2, right - left),
                  background: SEVERITY_VAR.critical,
                  opacity: 0.16,
                }}
              />
            );
          }),
      )}
    </div>
  );
});

const Lanes = memo(function Lanes({
  lanes,
  width,
  duration,
  baseTop,
  selected,
  activeKey,
  onHover,
  onPick,
}: {
  lanes: Lane[];
  width: number;
  duration: number;
  baseTop: number;
  selected: Dimension | null;
  activeKey: string | null;
  onHover: (info: HoverInfo | null) => void;
  onPick: (t: number) => void;
}) {
  return (
    <div>
      {lanes.map((lane, li) => {
        const dimmed = selected !== null && selected !== lane.dimension;
        const laneTop = baseTop + li * (LANE_H + LANE_GAP);
        return (
          <div
            key={lane.dimension}
            className="relative transition-opacity duration-500 ease-cine"
            style={{ height: LANE_H, marginTop: li === 0 ? 0 : LANE_GAP, opacity: dimmed ? 0.18 : 1 }}
          >
            <div className="absolute inset-0 rounded-[3px] bg-void-deep/70 ring-1 ring-inset ring-void-lineSoft" />
            {lane.items.map((item) => {
              const { moment, finding } = item;
              const left = clamp(duration > 0 ? (moment.t_start / duration) * width : 0, 0, width);
              const rawW = duration > 0 ? ((moment.t_end - moment.t_start) / duration) * width : 0;
              const barW = Math.max(4, Math.min(rawW, Math.max(4, width - left)));
              const intensity = clamp(Number.isFinite(moment.intensity) ? moment.intensity : 0.6, 0, 1);
              const barH = Math.round((0.42 + 0.58 * intensity) * (LANE_H - 6));
              const isActive = activeKey === item.key;
              return (
                <button
                  key={item.key}
                  type="button"
                  className="group absolute bottom-[3px] rounded-[2px] outline-offset-2 transition-[filter,opacity] duration-200 ease-cine hover:!opacity-100"
                  style={{
                    left,
                    width: barW,
                    height: barH,
                    background: SEVERITY_VAR[finding.severity],
                    opacity: 0.34 + 0.62 * intensity,
                    filter: isActive ? 'brightness(1.35)' : undefined,
                    boxShadow: isActive
                      ? `0 0 12px -2px ${SEVERITY_VAR[finding.severity]}`
                      : undefined,
                  }}
                  aria-label={`${finding.title} — ${moment.label} at ${formatTime(moment.t_start)}, severity ${finding.severity}`}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onPick(moment.t_start);
                  }}
                  onMouseEnter={() =>
                    onHover({ key: item.key, x: left + barW / 2, top: laneTop, finding, moment })
                  }
                  onMouseLeave={() => onHover(null)}
                  onFocus={() =>
                    onHover({ key: item.key, x: left + barW / 2, top: laneTop, finding, moment })
                  }
                  onBlur={() => onHover(null)}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
});

/* ------------------------------------------------------------------ icons */

function PlayIcon({ playing }: { playing: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" fill="currentColor">
      {playing ? (
        <>
          <rect x="6" y="4.5" width="4" height="15" rx="1" />
          <rect x="14" y="4.5" width="4" height="15" rx="1" />
        </>
      ) : (
        <path d="M8 4.8v14.4a.6.6 0 0 0 .92.5l11.3-7.2a.6.6 0 0 0 0-1L8.92 4.3A.6.6 0 0 0 8 4.8Z" />
      )}
    </svg>
  );
}

function NextIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" fill="currentColor">
      <path d="M6 5.4v13.2a.6.6 0 0 0 .93.5l9.4-6.6a.6.6 0 0 0 0-1L6.93 4.9A.6.6 0 0 0 6 5.4Z" />
      <rect x="17.6" y="4.6" width="2.6" height="14.8" rx="1" />
    </svg>
  );
}

/* -------------------------------------------------------------- component */

export function Timeline({ analysis, audioUrl, selected, onSeek }: TimelineProps) {
  const reduced = useReducedMotion();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);

  const [viewportW, setViewportW] = useState(0);
  const [zoom, setZoom] = useState<Zoom>(1);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [guideT, setGuideT] = useState<number | null>(null);
  const drag = useRef<{ id: number; type: string; startX: number; moved: boolean } | null>(null);

  const { isPlaying, currentTime, duration, toggle, seek, ready, error } =
    useAudioPlayback(audioUrl);

  const measuredDuration = analysis.measurements?.duration_seconds ?? 0;
  const dur = duration > 0.1 ? duration : measuredDuration > 0.1 ? measuredDuration : 0;

  const peaks = analysis.waveform_peaks ?? [];
  const rms = analysis.waveform_rms ?? [];

  /* ---- lanes ------------------------------------------------------ */

  const lanes = useMemo<Lane[]>(() => {
    const byDim = new Map<Dimension, LaneItem[]>();
    for (const finding of analysis.findings ?? []) {
      if (!finding?.moments?.length) continue;
      const list = byDim.get(finding.dimension) ?? [];
      finding.moments.forEach((moment, i) => {
        if (!Number.isFinite(moment?.t_start)) return;
        list.push({ key: `${finding.id}-${i}`, finding, moment });
      });
      if (list.length) byDim.set(finding.dimension, list);
    }
    const out: Lane[] = [];
    byDim.forEach((items, dimension) => {
      let worst: Severity = 'clean';
      for (const it of items) {
        if (SEVERITY_RANK[it.finding.severity] < SEVERITY_RANK[worst]) worst = it.finding.severity;
      }
      items.sort((a, b) => a.moment.t_start - b.moment.t_start);
      out.push({
        dimension,
        label: DIMENSION_SHORT[dimension] ?? dimension,
        fullLabel: DIMENSION_LABELS[dimension] ?? dimension,
        worst,
        items,
      });
    });
    out.sort((a, b) => {
      const s = SEVERITY_RANK[a.worst] - SEVERITY_RANK[b.worst];
      if (s !== 0) return s;
      return DIMENSIONS.indexOf(a.dimension) - DIMENSIONS.indexOf(b.dimension);
    });
    return out;
  }, [analysis.findings]);

  const momentTimes = useMemo(() => {
    const pool = selected ? lanes.filter((l) => l.dimension === selected) : lanes;
    const ts = pool.flatMap((l) => l.items.map((i) => i.moment.t_start));
    return Array.from(new Set(ts.map((t) => Math.round(t * 100) / 100))).sort((a, b) => a - b);
  }, [lanes, selected]);

  const hasMoments = lanes.length > 0;

  /* ---- geometry --------------------------------------------------- */

  // Layout effect: measuring after paint would flash a 240px-wide waveform.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = (): void => setViewportW(el.clientWidth);
    update();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', update);
      return () => window.removeEventListener('resize', update);
    }
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const contentW = Math.max(240, viewportW * zoom);
  const waveH = viewportW > 0 && viewportW < 480 ? 96 : viewportW < 900 ? 124 : 148;
  const lanesH = hasMoments ? lanes.length * LANE_H + Math.max(0, lanes.length - 1) * LANE_GAP : 0;
  const plotH = RULER_H + waveH + (hasMoments ? lanesH + 10 : 0);
  const playX = dur > 0 ? clamp((currentTime / dur) * contentW, 0, contentW) : 0;

  /* ---- seeking ----------------------------------------------------- */

  const commitSeek = useCallback(
    (t: number, notify: boolean) => {
      const clamped = dur > 0 ? clamp(t, 0, dur) : Math.max(0, t);
      seek(clamped);
      if (notify) onSeek(clamped);
    },
    [dur, onSeek, seek],
  );

  const timeFromClientX = useCallback(
    (clientX: number): number => {
      const el = plotRef.current;
      if (!el || dur <= 0) return 0;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0) return 0;
      return clamp(((clientX - rect.left) / rect.width) * dur, 0, dur);
    },
    [dur],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0 || dur <= 0) return;
      drag.current = { id: e.pointerId, type: e.pointerType, startX: e.clientX, moved: false };
      // Touch is left uncaptured so a zoomed timeline can still be panned; a
      // touch that does not travel is treated as a tap-to-seek on release.
      if (e.pointerType === 'touch') return;
      e.currentTarget.setPointerCapture(e.pointerId);
      const t = timeFromClientX(e.clientX);
      setGuideT(t);
      commitSeek(t, false);
    },
    [commitSeek, dur, timeFromClientX],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (dur <= 0) return;
      const d = drag.current;
      if (d && Math.abs(e.clientX - d.startX) > 4) d.moved = true;
      if (d && d.type !== 'touch') {
        const t = timeFromClientX(e.clientX);
        setGuideT(t);
        commitSeek(t, false);
      } else if (!d && e.pointerType !== 'touch') {
        setGuideT(timeFromClientX(e.clientX));
      }
    },
    [commitSeek, dur, timeFromClientX],
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const d = drag.current;
      drag.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      if (!d || dur <= 0) return;
      if (d.type === 'touch' && d.moved) return; // that was a pan, not a tap
      commitSeek(timeFromClientX(e.clientX), true);
    },
    [commitSeek, dur, timeFromClientX],
  );

  const onPointerCancel = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    drag.current = null;
    setGuideT(null);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  const jumpNext = useCallback(() => {
    if (!momentTimes.length) return;
    const next = momentTimes.find((t) => t > currentTime + 0.2);
    const target = next ?? momentTimes[0] ?? 0;
    commitSeek(target, true);
  }, [commitSeek, currentTime, momentTimes]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === ' ' || e.key === 'Spacebar') {
        if (!ready) return;
        e.preventDefault();
        toggle();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        commitSeek(currentTime + (e.shiftKey ? 15 : 5), false);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        commitSeek(currentTime - (e.shiftKey ? 15 : 5), false);
      } else if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        jumpNext();
      } else if (e.key === 'Home') {
        e.preventDefault();
        commitSeek(0, false);
      }
    },
    [commitSeek, currentTime, jumpNext, ready, toggle],
  );

  // Keep the playhead on screen while playing at zoom.
  useEffect(() => {
    if (!isPlaying || zoom === 1) return;
    const el = scrollRef.current;
    if (!el) return;
    const left = el.scrollLeft;
    const right = left + el.clientWidth;
    if (playX < left + 32 || playX > right - 96) {
      el.scrollTo({ left: Math.max(0, playX - el.clientWidth * 0.3), behavior: 'auto' });
    }
  }, [isPlaying, playX, zoom]);

  const handlePick = useCallback(
    (t: number) => {
      commitSeek(t, true);
    },
    [commitSeek],
  );

  /* ---- motion ------------------------------------------------------ */

  const container = {
    hidden: {},
    show: { transition: { staggerChildren: reduced ? 0 : 0.07, delayChildren: reduced ? 0 : 0.05 } },
  };
  const item = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: reduced ? 0.001 : 0.6, ease: EASE },
    },
  };

  // Wide enough for the longest DIMENSION_SHORT label ("Dynamics") uncut.
  const gutterW = 'w-[84px] sm:w-[104px]';
  const readOnly = audioUrl === null || error !== null;
  const showTransport = !readOnly;
  const momentCount = lanes.reduce((n, l) => n + l.items.length, 0);

  return (
    <motion.section
      variants={container}
      initial="hidden"
      // Mount-triggered, NOT whileInView: a missed IntersectionObserver
      // callback would leave the whole panel stuck at opacity 0, and this is
      // the primary content of the view.
      animate="show"
      className="panel overflow-hidden"
      aria-labelledby="timeline-heading"
    >
      {/* -------- header + transport -------- */}
      <motion.div
        variants={item}
        className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4 px-4 pb-3 pt-4 sm:px-5"
      >
        <div>
          <p className="eyebrow">Timeline</p>
          <h2 id="timeline-heading" className="display mt-1.5 text-xl text-ink sm:text-2xl">
            When it happens
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {showTransport && (
            <>
              <button
                type="button"
                onClick={toggle}
                disabled={!ready}
                aria-label={isPlaying ? 'Pause' : 'Play'}
                aria-pressed={isPlaying}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-signal text-void-deep transition-all duration-300 ease-cine hover:bg-signal-glow disabled:cursor-not-allowed disabled:opacity-30"
                style={{ boxShadow: isPlaying ? '0 0 22px -4px rgba(82,242,196,0.7)' : undefined }}
              >
                <PlayIcon playing={isPlaying} />
              </button>
              <div className="stat text-sm text-ink">
                {formatTime(currentTime)}
                <span className="text-ink-faint"> / {formatTime(dur)}</span>
              </div>
            </>
          )}

          {hasMoments && (
            <button
              type="button"
              onClick={jumpNext}
              className="inline-flex items-center gap-2 rounded-full border border-void-line px-3 py-1.5 font-mono text-micro uppercase text-ink-dim transition-colors duration-200 hover:border-ink-faint hover:text-ink"
              aria-label="Jump to next issue"
            >
              <NextIcon />
              Next issue
            </button>
          )}

          <div
            className="flex items-center gap-0.5 rounded-full border border-void-line p-0.5"
            role="group"
            aria-label="Timeline zoom"
          >
            {ZOOMS.map((z) => (
              <button
                key={z}
                type="button"
                onClick={() => setZoom(z)}
                aria-pressed={zoom === z}
                aria-label={`Zoom ${z} times`}
                className={`rounded-full px-2.5 py-1 font-mono text-micro transition-colors duration-200 ${
                  zoom === z ? 'bg-void-raised text-signal' : 'text-ink-faint hover:text-ink-dim'
                }`}
              >
                {z}×
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {audioUrl === null && (
        <p className="px-4 pb-3 font-mono text-micro uppercase text-ink-faint sm:px-5">
          Audio not attached to this analysis — timeline is read-only
        </p>
      )}
      {error !== null && (
        <p className="px-4 pb-3 font-mono text-micro uppercase sev-major sm:px-5">{error}</p>
      )}

      <div className="hairline" />

      {/* -------- gutter + scrolling plot -------- */}
      <motion.div variants={item} className="flex">
        {/* Sticky gutter: lives outside the scroller, so labels never travel. */}
        <div className={`${gutterW} shrink-0 border-r border-void-lineSoft bg-void-panel pl-4 pr-2 sm:pl-5`}>
          <div style={{ height: RULER_H }} className="flex items-end pb-1">
            <span className="eyebrow text-[9px]">Time</span>
          </div>
          <div style={{ height: waveH }} className="flex flex-col justify-center gap-1.5">
            <span className="flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="h-[3px] w-2.5 rounded-sm"
                style={{ background: readOnly ? 'rgba(82,242,196,0.45)' : 'rgba(112,112,126,0.7)' }}
              />
              <span className="eyebrow text-[9px] leading-none">Peak</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="h-[3px] w-2.5 rounded-sm"
                style={{ background: readOnly ? '#8DFFE0' : 'rgba(180,180,196,0.9)' }}
              />
              <span className="font-mono text-[9px] uppercase leading-none tracking-[0.14em] text-ink-muted">
                RMS
              </span>
            </span>
          </div>
          {hasMoments && (
            <div style={{ paddingTop: 10 }}>
              {lanes.map((lane, li) => {
                const dimmed = selected !== null && selected !== lane.dimension;
                return (
                  <div
                    key={lane.dimension}
                    className="flex items-center gap-1.5 transition-opacity duration-500 ease-cine"
                    style={{
                      height: LANE_H,
                      marginTop: li === 0 ? 0 : LANE_GAP,
                      opacity: dimmed ? 0.28 : 1,
                    }}
                  >
                    <span
                      aria-hidden="true"
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: SEVERITY_VAR[lane.worst] }}
                    />
                    <span
                      className="truncate font-mono text-[9px] uppercase leading-none tracking-[0.12em] text-ink-muted"
                      title={lane.fullLabel}
                    >
                      {lane.label}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div
          ref={scrollRef}
          tabIndex={0}
          role="group"
          aria-label="Mix timeline. Space plays or pauses, arrow keys scrub, N jumps to the next issue."
          onKeyDown={onKeyDown}
          className="no-scrollbar relative min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
        >
          <div
            ref={plotRef}
            className="relative select-none"
            style={{ width: contentW, height: plotH, touchAction: 'pan-x pan-y' }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
            onPointerLeave={() => {
              if (!drag.current) setGuideT(null);
            }}
          >
            <Ruler width={contentW} duration={dur} />

            <div className="relative" style={{ height: waveH }}>
              {peaks.length > 0 ? (
                <>
                  {/* Read-only (no audio): there is no "played" region to
                      distinguish, so light the whole waveform rather than
                      leaving a dead gray slab. */}
                  <WaveCanvas
                    width={contentW}
                    height={waveH}
                    peaks={peaks}
                    rms={rms}
                    lit={readOnly}
                  />
                  {!readOnly && (
                    <div
                      className="absolute inset-y-0 left-0 overflow-hidden"
                      style={{ width: playX }}
                      aria-hidden="true"
                    >
                      <WaveCanvas width={contentW} height={waveH} peaks={peaks} rms={rms} lit />
                    </div>
                  )}
                  <WaveTint lanes={lanes} width={contentW} duration={dur} />
                </>
              ) : (
                <div className="flex h-full items-center">
                  <div className="h-px w-full bg-void-line" />
                  <span className="absolute left-2 font-mono text-micro uppercase text-ink-faint">
                    Waveform unavailable
                  </span>
                </div>
              )}
            </div>

            {hasMoments && (
              <div style={{ paddingTop: 10 }}>
                <Lanes
                  lanes={lanes}
                  width={contentW}
                  duration={dur}
                  baseTop={RULER_H + waveH + 10}
                  selected={selected}
                  activeKey={hover?.key ?? null}
                  onHover={setHover}
                  onPick={handlePick}
                />
              </div>
            )}

            {/* hover guide */}
            {guideT !== null && dur > 0 && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute bottom-0"
                style={{ left: clamp((guideT / dur) * contentW, 0, contentW), top: RULER_H }}
              >
                <div className="h-full w-px bg-ink-faint/40" />
              </div>
            )}

            {/* playhead */}
            {dur > 0 && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute bottom-0"
                style={{ left: playX, top: RULER_H - 6 }}
              >
                <div
                  className="h-full w-px"
                  style={{ background: '#8DFFE0', boxShadow: '0 0 10px 0 rgba(82,242,196,0.75)' }}
                />
                <div
                  className="absolute -left-[3px] top-0 h-1.5 w-1.5 rotate-45"
                  style={{ background: '#8DFFE0' }}
                />
              </div>
            )}

            {/* tooltip */}
            {hover && (
              <div
                className="pointer-events-none absolute z-20 w-max max-w-[240px] rounded-lg border border-void-line bg-void-raised/95 px-3 py-2 shadow-lift backdrop-blur-sm"
                style={{
                  left: clamp(hover.x, 96, Math.max(96, contentW - 96)),
                  top: Math.max(4, hover.top - 12),
                  transform: 'translate(-50%, -100%)',
                }}
                aria-hidden="true"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`sev-chip sev-${hover.finding.severity}`}
                    style={{ paddingTop: 2, paddingBottom: 2 }}
                  >
                    {hover.finding.severity}
                  </span>
                  <span className="stat text-xs text-ink">
                    {formatTime(hover.moment.t_start)}–{formatTime(hover.moment.t_end)}
                  </span>
                </div>
                <p className="mt-1.5 text-[13px] font-medium leading-snug text-ink">
                  {hover.finding.title}
                </p>
                {hover.moment.label && (
                  <p className="mt-0.5 text-xs leading-snug text-ink-dim">{hover.moment.label}</p>
                )}
                <p className="eyebrow mt-1.5 text-[9px]">
                  Intensity {Math.round(clamp(hover.moment.intensity ?? 0, 0, 1) * 100)}%
                </p>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* -------- footer -------- */}
      {hasMoments ? (
        <motion.div
          variants={item}
          className="flex flex-wrap items-center justify-between gap-3 border-t border-void-lineSoft px-4 py-2.5 sm:px-5"
        >
          <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            {momentCount} flagged moment{momentCount === 1 ? '' : 's'}
            {selected ? ` · filtered to ${DIMENSION_LABELS[selected]}` : ''} · click a bar to jump
          </p>
          <p className="hidden font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint sm:block">
            Space play · ←/→ scrub · N next
          </p>
        </motion.div>
      ) : (
        <motion.div
          variants={item}
          className="flex flex-wrap items-center gap-3 border-t border-void-lineSoft px-4 py-4 sm:px-5"
        >
          <span className="sev-chip sev-clean">Clean</span>
          <p className="text-sm leading-snug text-ink-dim">
            Nothing flagged at a specific moment. Whatever this mix needs, it needs it{' '}
            <span className="text-ink">everywhere</span> — not in one bad bar.
          </p>
        </motion.div>
      )}
    </motion.section>
  );
}

export default Timeline;
