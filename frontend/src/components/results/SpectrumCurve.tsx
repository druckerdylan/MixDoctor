/**
 * SpectrumCurve — 1/3-octave analyzer with the genre target laid over it.
 *
 * The single most useful pixel here is the shaded delta between the measured
 * curve and the target: warm where the mix is hot, cool where it is shy. Every
 * axis is labelled and hovering gives exact numbers, because a producer is
 * going to act on this.
 */

import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  SEVERITY_VAR,
  formatDb,
  formatHz,
  type Band,
  type ReferenceDelta,
  type Resonance,
  type SpectralMeasurement,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const F_MIN = 20;
const F_MAX = 20000;
const L_MIN = Math.log10(F_MIN);
const L_SPAN = Math.log10(F_MAX) - L_MIN;

const FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
/** Below ~520px the full decade set collides; keep a readable subset. */
const FREQ_TICKS_NARROW = [20, 100, 500, 2000, 10000, 20000];

/**
 * Named regions are *context*, so they are drawn neutral. Tinting them by
 * character (blue mud, red harsh) collided head-on with the hot/cool shading,
 * where blue and warm already mean "shy" and "hot vs target".
 */
const REGIONS: { name: string; lo: number; hi: number }[] = [
  { name: 'Mud', lo: 150, hi: 400 },
  { name: 'Harsh', lo: 2000, hi: 5000 },
  { name: 'Air', lo: 10000, hi: 20000 },
];

const C_MEASURED = '#52F2C4';
const C_TARGET = '#9C9CAB';
const C_REFERENCE = '#C8FF6B';
const C_HOT = '#FF6B4A';
const C_COOL = '#2E7BFF';

export interface SpectrumCurveProps {
  spectral: SpectralMeasurement;
  reference: ReferenceDelta | null;
}

/* ------------------------------------------------------------------ math */

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Fritsch–Carlson tangents: smooth, and provably no overshoot between points. */
function monotoneTangents(xs: number[], ys: number[]): number[] {
  const n = xs.length;
  const m = new Array<number>(n).fill(0);
  if (n < 2) return m;
  const secants = new Array<number>(n - 1).fill(0);
  for (let i = 0; i < n - 1; i += 1) {
    const dx = xs[i + 1] - xs[i];
    secants[i] = dx === 0 ? 0 : (ys[i + 1] - ys[i]) / dx;
  }
  m[0] = secants[0];
  m[n - 1] = secants[n - 2];
  for (let i = 1; i < n - 1; i += 1) {
    const a = secants[i - 1];
    const b = secants[i];
    m[i] = a * b <= 0 ? 0 : (a + b) / 2;
  }
  for (let i = 0; i < n - 1; i += 1) {
    const s = secants[i];
    if (s === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / s;
    const b = m[i + 1] / s;
    const sum = a * a + b * b;
    if (sum > 9) {
      const t = 3 / Math.sqrt(sum);
      m[i] = t * a * s;
      m[i + 1] = t * b * s;
    }
  }
  return m;
}

/** Cubic-Hermite sampler over sorted xs. Flat outside the known range. */
function makeInterpolator(xs: number[], ys: number[]): (x: number) => number {
  const n = xs.length;
  if (n === 0) return () => 0;
  if (n === 1) return () => ys[0];
  const m = monotoneTangents(xs, ys);
  return (x: number): number => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i += 1;
    const h = xs[i + 1] - xs[i];
    if (h === 0) return ys[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i] +
      (t3 - 2 * t2 + t) * h * m[i] +
      (-2 * t3 + 3 * t2) * ys[i + 1] +
      (t3 - t2) * h * m[i + 1]
    );
  };
}

/** Monotone cubic SVG path. Reversing the point order retraces the same curve. */
function smoothPath(pts: [number, number][]): string {
  if (pts.length === 0) return '';
  if (pts.length === 1) return `M ${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)}`;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const m = monotoneTangents(xs, ys);
  let d = `M ${xs[0].toFixed(2)} ${ys[0].toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const h = xs[i + 1] - xs[i];
    const c1x = xs[i] + h / 3;
    const c1y = ys[i] + (m[i] * h) / 3;
    const c2x = xs[i + 1] - h / 3;
    const c2y = ys[i + 1] - (m[i + 1] * h) / 3;
    d += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${xs[
      i + 1
    ].toFixed(2)} ${ys[i + 1].toFixed(2)}`;
  }
  return d;
}

function niceStep(raw: number): number {
  const steps = [1, 2, 3, 5, 6, 10, 12, 15, 20, 25, 30];
  for (const s of steps) if (raw <= s) return s;
  return 40;
}

/* ------------------------------------------------------------- component */

export function SpectrumCurve({ spectral, reference }: SpectrumCurveProps) {
  const reduced = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(0);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [activeRes, setActiveRes] = useState<number | null>(null);

  const rawId = useId();
  const uid = rawId.replace(/[^a-zA-Z0-9_-]/g, '');

  // Layout effect: the chart is width-driven, so measuring after paint would
  // flash the "no spectrum" fallback before the first real frame.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = (): void => setW(el.clientWidth);
    update();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', update);
      return () => window.removeEventListener('resize', update);
    }
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const h = w < 420 ? 252 : w < 720 ? 300 : 356;
  // Leaves room for the dB ticks *and* the rotated axis unit beside them.
  const padL = w < 480 ? 42 : 52;
  const padR = 14;
  const padT = 26;
  const padB = 46;
  const plotW = Math.max(1, w - padL - padR);
  const plotH = Math.max(1, h - padT - padB);

  /* ---- series ------------------------------------------------------- */

  const series = useMemo(() => {
    const centers = spectral?.third_octave_centers ?? [];
    const levels = spectral?.third_octave_db ?? [];
    const n = Math.min(centers.length, levels.length);
    const kept: { hz: number; db: number; src: number }[] = [];
    for (let i = 0; i < n; i += 1) {
      const f = centers[i];
      const v = levels[i];
      if (!Number.isFinite(f) || !Number.isFinite(v)) continue;
      if (f < F_MIN * 0.9 || f > F_MAX * 1.05) continue;
      kept.push({ hz: clamp(f, F_MIN, F_MAX), db: v, src: i });
    }
    // Interpolation and path building both assume ascending x.
    kept.sort((a, b) => a.hz - b.hz);
    return {
      hz: kept.map((k) => k.hz),
      db: kept.map((k) => k.db),
      src: kept.map((k) => k.src),
    };
  }, [spectral]);

  const logHz = useMemo(() => series.hz.map((f) => Math.log10(f)), [series.hz]);

  /**
   * Target line. Macro bands report an aggregate level, so their target_db is
   * not on the same scale as a single 1/3-octave bin. What *is* scale-free is
   * deviation_db, so each band anchors at (mean measured level in the band −
   * its deviation) and we interpolate those anchors across log frequency.
   */
  const target = useMemo<number[] | null>(() => {
    const bands: Band[] = spectral?.bands ?? [];
    if (!bands.length || series.hz.length < 2) return null;
    const ax: number[] = [];
    const ay: number[] = [];
    const sorted = [...bands]
      .filter((b) => Number.isFinite(b?.center_hz) && Number.isFinite(b?.deviation_db))
      .sort((a, b) => a.center_hz - b.center_hz);
    for (const band of sorted) {
      let sum = 0;
      let count = 0;
      for (let i = 0; i < series.hz.length; i += 1) {
        const f = series.hz[i];
        if (f >= band.low_hz && f <= band.high_hz) {
          sum += series.db[i];
          count += 1;
        }
      }
      if (count === 0) continue;
      const lx = Math.log10(clamp(band.center_hz, F_MIN, F_MAX));
      if (ax.length && Math.abs(lx - ax[ax.length - 1]) < 1e-6) continue;
      ax.push(lx);
      ay.push(sum / count - band.deviation_db);
    }
    if (ax.length < 2) return null;
    const interp = makeInterpolator(ax, ay);
    return logHz.map((lx) => interp(lx));
  }, [spectral, series, logHz]);

  /**
   * The reference is given as a per-bin delta against the raw centre array, so
   * each kept bin looks its delta up by its original index. Anything shorter
   * than the centre array cannot be aligned safely and is dropped.
   */
  const refCurve = useMemo<number[] | null>(() => {
    const delta = reference?.third_octave_delta_db;
    if (!delta?.length) return null;
    const centers = spectral?.third_octave_centers ?? [];
    if (delta.length < centers.length || series.hz.length < 2) return null;
    let usableBins = 0;
    const out = series.src.map((srcIndex, k) => {
      const d = delta[srcIndex];
      if (!Number.isFinite(d)) return series.db[k];
      usableBins += 1;
      return series.db[k] - d;
    });
    return usableBins >= 2 ? out : null;
  }, [reference, spectral, series]);

  const domain = useMemo(() => {
    const all: number[] = [...series.db];
    if (target) all.push(...target);
    if (refCurve) all.push(...refCurve);
    const finite = all.filter((v) => Number.isFinite(v));
    if (!finite.length) return { lo: -60, hi: 0, step: 10 };
    let lo = Math.min(...finite) - 3;
    let hi = Math.max(...finite) + 3;
    if (hi - lo < 12) {
      const mid = (hi + lo) / 2;
      lo = mid - 6;
      hi = mid + 6;
    }
    const step = niceStep((hi - lo) / 6);
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step) * step;
    return { lo, hi, step };
  }, [series.db, target, refCurve]);

  const xOf = useCallback(
    (hz: number): number => padL + ((Math.log10(clamp(hz, F_MIN, F_MAX)) - L_MIN) / L_SPAN) * plotW,
    [padL, plotW],
  );
  const xOfLog = useCallback(
    (lx: number): number => padL + ((lx - L_MIN) / L_SPAN) * plotW,
    [padL, plotW],
  );
  const yOf = useCallback(
    (db: number): number =>
      padT + ((domain.hi - clamp(db, domain.lo, domain.hi)) / (domain.hi - domain.lo)) * plotH,
    [domain, padT, plotH],
  );

  const usable = series.hz.length >= 4 && w > 0;

  const paths = useMemo(() => {
    if (!usable) return null;
    const measuredPts: [number, number][] = logHz.map((lx, i) => [xOfLog(lx), yOf(series.db[i])]);
    const measuredD = smoothPath(measuredPts);
    const refD = refCurve
      ? smoothPath(logHz.map((lx, i) => [xOfLog(lx), yOf(refCurve[i])] as [number, number]))
      : null;
    if (!target) {
      return { measuredD, targetD: null, betweenD: null, refD };
    }
    const targetPts: [number, number][] = logHz.map((lx, i) => [xOfLog(lx), yOf(target[i])]);
    const targetD = smoothPath(targetPts);
    // Monotone-cubic tangents are order-independent, so feeding the reversed
    // points back in retraces the identical curve — that is what closes the
    // measured-vs-target region exactly rather than approximately.
    const reversedTargetD = smoothPath([...targetPts].reverse()).replace(/^M/, 'L');
    const betweenD = `${measuredD} ${reversedTargetD} Z`;
    return { measuredD, targetD, betweenD, refD };
  }, [usable, logHz, series.db, target, refCurve, xOfLog, yOf]);

  const resonances = useMemo<Resonance[]>(() => {
    const list = (spectral?.resonances ?? []).filter(
      (r) => Number.isFinite(r?.freq_hz) && r.freq_hz >= F_MIN && r.freq_hz <= F_MAX,
    );
    return [...list].sort((a, b) => (b.prominence_db ?? 0) - (a.prominence_db ?? 0)).slice(0, 6);
  }, [spectral]);

  const measuredAt = useMemo(
    () => makeInterpolator(logHz, series.db),
    [logHz, series.db],
  );

  /* ---- hover -------------------------------------------------------- */

  const onMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (!usable) return;
      const rect = e.currentTarget.getBoundingClientRect();
      if (rect.width <= 0) return;
      const px = ((e.clientX - rect.left) / rect.width) * w;
      if (px < padL - 8 || px > padL + plotW + 8) {
        setHoverIdx(null);
        return;
      }
      const lx = L_MIN + (clamp(px, padL, padL + plotW) - padL) / plotW * L_SPAN;
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < logHz.length; i += 1) {
        const d = Math.abs(logHz[i] - lx);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      setHoverIdx(best);
    },
    [usable, w, padL, plotW, logHz],
  );

  const hoverHz = hoverIdx !== null ? series.hz[hoverIdx] : null;
  const hoverDb = hoverIdx !== null ? series.db[hoverIdx] : null;
  const hoverTarget = hoverIdx !== null && target ? target[hoverIdx] : null;
  const hoverDev = hoverDb !== null && hoverTarget !== null ? hoverDb - hoverTarget : null;
  const hoverRegion =
    hoverHz !== null ? REGIONS.find((r) => hoverHz >= r.lo && hoverHz <= r.hi) ?? null : null;

  const dbTicks = useMemo(() => {
    const out: number[] = [];
    for (let v = domain.lo; v <= domain.hi + 1e-6; v += domain.step) out.push(Math.round(v));
    return out;
  }, [domain]);

  /* ---- pin stacking so labels do not collide ------------------------ */

  const pins = useMemo(() => {
    if (!usable) return [];
    const sortedByX = [...resonances].sort((a, b) => a.freq_hz - b.freq_hz);
    let lastX = -Infinity;
    let level = 0;
    return sortedByX.map((r) => {
      const x = xOf(r.freq_hz);
      if (x - lastX < 62) level = (level + 1) % 3;
      else level = 0;
      lastX = x;
      const yCurve = yOf(measuredAt(Math.log10(clamp(r.freq_hz, F_MIN, F_MAX))));
      const yPin = Math.max(padT + 8, yCurve - 22 - level * 15);
      return { res: r, x, yCurve, yPin };
    });
  }, [usable, resonances, xOf, yOf, measuredAt, padT]);

  const summary = useMemo(() => {
    const bands = spectral?.bands ?? [];
    if (!bands.length) return 'Third-octave spectrum of the mix.';
    const worst = [...bands]
      .filter((b) => Number.isFinite(b?.deviation_db))
      .sort((a, b) => Math.abs(b.deviation_db) - Math.abs(a.deviation_db))[0];
    if (!worst) return 'Third-octave spectrum of the mix.';
    return `Third-octave spectrum against the genre target. Largest deviation: ${worst.name} at ${formatDb(worst.deviation_db)} dB.`;
  }, [spectral]);

  const fade = {
    hidden: reduced ? { opacity: 0 } : { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: reduced ? 0.001 : 0.6, ease: EASE } },
  };

  /* ---- render ------------------------------------------------------- */

  return (
    <motion.section
      variants={{ hidden: {}, show: { transition: { staggerChildren: reduced ? 0 : 0.07 } } }}
      initial="hidden"
      // Mount-triggered for the same reason as Timeline: a scroll-triggered
      // reveal that never fires hides the chart permanently.
      animate="show"
      className="panel overflow-hidden"
      aria-labelledby="spectrum-heading"
    >
      {/* -------- header + live readout -------- */}
      <motion.div
        variants={fade}
        className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4 px-4 pb-3 pt-4 sm:px-5"
      >
        <div>
          <p className="eyebrow">Spectrum</p>
          <h2 id="spectrum-heading" className="display mt-1.5 text-xl text-ink sm:text-2xl">
            Tonal balance
          </h2>
        </div>

        <div className="flex items-end gap-4 sm:gap-6">
          <Readout label="Freq" value={hoverHz !== null ? `${formatHz(hoverHz)} Hz` : '—'} />
          <Readout label="Level" value={hoverDb !== null ? `${formatDb(hoverDb)} dB` : '—'} />
          <Readout
            label="Target"
            value={hoverTarget !== null ? `${formatDb(hoverTarget)} dB` : '—'}
          />
          <Readout
            label="Dev"
            value={hoverDev !== null ? `${hoverDev >= 0 ? '▲' : '▼'} ${formatDb(hoverDev)} dB` : '—'}
            tone={
              hoverDev === null
                ? undefined
                : hoverDev > 1.5
                  ? C_HOT
                  : hoverDev < -1.5
                    ? '#4CC9F0'
                    : undefined
            }
          />
        </div>
      </motion.div>

      <div className="hairline" />

      {/* -------- plot -------- */}
      <motion.div variants={fade} ref={wrapRef} className="relative w-full px-1 pt-1">
        {w === 0 ? (
          // Pre-measurement frame. Reserve height so nothing jumps.
          <div style={{ height: h }} aria-hidden="true" />
        ) : !usable ? (
          <div className="flex items-center gap-3 px-3 py-10">
            <span className="eyebrow">No spectrum</span>
            <p className="text-sm text-ink-dim">
              This analysis did not return third-octave data.
            </p>
          </div>
        ) : (
          <svg
            width="100%"
            height={h}
            viewBox={`0 0 ${w} ${h}`}
            role="img"
            aria-label={summary}
            className="block touch-pan-y"
            onPointerMove={onMove}
            onPointerLeave={() => setHoverIdx(null)}
          >
            <defs>
              <linearGradient
                id={`${uid}-hot`}
                gradientUnits="userSpaceOnUse"
                x1="0"
                y1={padT}
                x2="0"
                y2={padT + plotH}
              >
                <stop offset="0%" stopColor={C_HOT} stopOpacity="0.46" />
                <stop offset="100%" stopColor={C_HOT} stopOpacity="0.07" />
              </linearGradient>
              <linearGradient
                id={`${uid}-cool`}
                gradientUnits="userSpaceOnUse"
                x1="0"
                y1={padT}
                x2="0"
                y2={padT + plotH}
              >
                <stop offset="0%" stopColor={C_COOL} stopOpacity="0.07" />
                <stop offset="100%" stopColor={C_COOL} stopOpacity="0.46" />
              </linearGradient>

              {paths?.targetD && (
                <>
                  <clipPath id={`${uid}-above`}>
                    <path
                      d={`${paths.targetD} L ${padL + plotW} ${padT} L ${padL} ${padT} Z`}
                    />
                  </clipPath>
                  <clipPath id={`${uid}-below`}>
                    <path
                      d={`${paths.targetD} L ${padL + plotW} ${padT + plotH} L ${padL} ${
                        padT + plotH
                      } Z`}
                    />
                  </clipPath>
                </>
              )}
              <clipPath id={`${uid}-plot`}>
                <rect x={padL} y={padT} width={plotW} height={plotH} />
              </clipPath>
            </defs>

            {/* named regions */}
            {REGIONS.map((r) => {
              const x0 = xOf(r.lo);
              const x1 = xOf(r.hi);
              return (
                <g key={r.name}>
                  <rect
                    x={x0}
                    y={padT}
                    width={Math.max(0, x1 - x0)}
                    height={plotH}
                    fill="#F4F4F7"
                    opacity={0.028}
                  />
                  <text
                    x={(x0 + x1) / 2}
                    y={padT + plotH - 7}
                    textAnchor="middle"
                    className="font-mono"
                    fontSize={9}
                    letterSpacing="0.14em"
                    fill="#4A4A57"
                  >
                    {r.name.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {/* dB grid */}
            {dbTicks.map((v) => (
              <g key={`db-${v}`}>
                <line
                  x1={padL}
                  y1={yOf(v)}
                  x2={padL + plotW}
                  y2={yOf(v)}
                  stroke="#1E1E29"
                  strokeWidth={1}
                />
                <text
                  x={padL - 7}
                  y={yOf(v) + 3}
                  textAnchor="end"
                  className="font-mono"
                  fontSize={9}
                  fill="#4A4A57"
                >
                  {v}
                </text>
              </g>
            ))}

            {/* frequency grid */}
            {(w < 520 ? FREQ_TICKS_NARROW : FREQ_TICKS).map((f) => (
              <g key={`f-${f}`}>
                <line
                  x1={xOf(f)}
                  y1={padT}
                  x2={xOf(f)}
                  y2={padT + plotH}
                  stroke="#16161F"
                  strokeWidth={1}
                />
                <text
                  x={xOf(f)}
                  y={padT + plotH + 15}
                  textAnchor={f === F_MIN ? 'start' : f === F_MAX ? 'end' : 'middle'}
                  className="font-mono"
                  fontSize={9}
                  fill="#70707E"
                >
                  {formatHz(f)}
                </text>
              </g>
            ))}

            {/* delta shading */}
            {paths?.betweenD && paths.targetD && (
              <g clipPath={`url(#${uid}-plot)`}>
                <path
                  d={paths.betweenD}
                  fill={`url(#${uid}-hot)`}
                  clipPath={`url(#${uid}-above)`}
                />
                <path
                  d={paths.betweenD}
                  fill={`url(#${uid}-cool)`}
                  clipPath={`url(#${uid}-below)`}
                />
              </g>
            )}

            {/* reference */}
            {paths?.refD && (
              <path
                d={paths.refD}
                fill="none"
                stroke={C_REFERENCE}
                strokeWidth={1.4}
                strokeDasharray="5 4"
                strokeLinecap="round"
                opacity={0.75}
                clipPath={`url(#${uid}-plot)`}
              />
            )}

            {/* target */}
            {paths?.targetD && (
              <path
                d={paths.targetD}
                fill="none"
                stroke={C_TARGET}
                strokeWidth={1.4}
                strokeLinecap="round"
                opacity={0.85}
                clipPath={`url(#${uid}-plot)`}
              />
            )}

            {/* measured */}
            {paths && (
              <path
                d={paths.measuredD}
                fill="none"
                stroke={C_MEASURED}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                clipPath={`url(#${uid}-plot)`}
                style={{ filter: 'drop-shadow(0 0 5px rgba(82,242,196,0.45))' }}
              />
            )}

            {/* spectral centroid */}
            {Number.isFinite(spectral?.spectral_centroid_hz) &&
              spectral.spectral_centroid_hz >= F_MIN && (
                <g>
                  <line
                    x1={xOf(spectral.spectral_centroid_hz)}
                    y1={padT}
                    x2={xOf(spectral.spectral_centroid_hz)}
                    y2={padT + plotH}
                    stroke="#F4F4F7"
                    strokeWidth={1}
                    strokeDasharray="2 5"
                    opacity={0.22}
                  />
                  <text
                    x={clamp(xOf(spectral.spectral_centroid_hz) + 5, padL, padL + plotW - 4)}
                    y={padT - 10}
                    textAnchor={
                      xOf(spectral.spectral_centroid_hz) > padL + plotW - 90 ? 'end' : 'start'
                    }
                    className="font-mono"
                    fontSize={9}
                    letterSpacing="0.12em"
                    fill="#4A4A57"
                  >
                    CENTROID {formatHz(spectral.spectral_centroid_hz)}
                  </text>
                </g>
              )}

            {/* resonance pins */}
            {pins.map(({ res, x, yCurve, yPin }, i) => {
              const on = activeRes === i;
              const color = SEVERITY_VAR[res.severity] ?? C_MEASURED;
              return (
                <g
                  key={`pin-${res.freq_hz}-${i}`}
                  tabIndex={0}
                  role="button"
                  aria-label={`Resonance at ${formatHz(res.freq_hz)} hertz, Q ${res.q.toFixed(
                    1,
                  )}, ${formatDb(res.prominence_db)} decibels prominent, ${res.severity}`}
                  onMouseEnter={() => setActiveRes(i)}
                  onMouseLeave={() => setActiveRes(null)}
                  onFocus={() => setActiveRes(i)}
                  onBlur={() => setActiveRes(null)}
                  style={{ cursor: 'help' }}
                >
                  <line
                    x1={x}
                    y1={yPin + 4}
                    x2={x}
                    y2={yCurve}
                    stroke={color}
                    strokeWidth={1}
                    opacity={on ? 0.95 : 0.5}
                    strokeDasharray="2 3"
                  />
                  <circle cx={x} cy={yCurve} r={on ? 3.4 : 2.4} fill={color} />
                  <text
                    x={x}
                    y={yPin}
                    textAnchor="middle"
                    className="font-mono"
                    fontSize={9}
                    letterSpacing="0.08em"
                    fill={on ? '#F4F4F7' : color}
                    opacity={on ? 1 : 0.85}
                  >
                    {formatHz(res.freq_hz)}
                  </text>
                  {on && (
                    <text
                      x={clamp(x, padL + 40, padL + plotW - 40)}
                      y={yPin - 12}
                      textAnchor="middle"
                      className="font-mono"
                      fontSize={9}
                      fill="#A7A7B4"
                    >
                      Q {res.q.toFixed(1)} · {formatDb(res.prominence_db)} dB
                    </text>
                  )}
                </g>
              );
            })}

            {/* hover guide */}
            {hoverIdx !== null && hoverHz !== null && hoverDb !== null && (
              <g pointerEvents="none">
                <line
                  x1={xOf(hoverHz)}
                  y1={padT}
                  x2={xOf(hoverHz)}
                  y2={padT + plotH}
                  stroke="#F4F4F7"
                  strokeWidth={1}
                  opacity={0.28}
                />
                {hoverTarget !== null && (
                  <circle cx={xOf(hoverHz)} cy={yOf(hoverTarget)} r={2.6} fill={C_TARGET} />
                )}
                <circle
                  cx={xOf(hoverHz)}
                  cy={yOf(hoverDb)}
                  r={3.6}
                  fill="#06060A"
                  stroke={C_MEASURED}
                  strokeWidth={1.6}
                />
              </g>
            )}

            {/* axis frame */}
            <line
              x1={padL}
              y1={padT + plotH}
              x2={padL + plotW}
              y2={padT + plotH}
              stroke="#1E1E29"
              strokeWidth={1}
            />
            <text
              x={padL}
              y={h - 8}
              className="font-mono"
              fontSize={9}
              letterSpacing="0.14em"
              fill="#4A4A57"
            >
              FREQUENCY (Hz)
            </text>
            {/* The dB unit belongs to the y axis, so it is set along it. */}
            <text
              x={11}
              y={padT + plotH / 2}
              transform={`rotate(-90 11 ${padT + plotH / 2})`}
              textAnchor="middle"
              className="font-mono"
              fontSize={9}
              letterSpacing="0.14em"
              fill="#4A4A57"
            >
              LEVEL (dB)
            </text>
            {hoverRegion && (
              <text
                x={padL + plotW}
                y={h - 8}
                textAnchor="end"
                className="font-mono"
                fontSize={9}
                letterSpacing="0.14em"
                fill="#70707E"
              >
                {hoverRegion.name.toUpperCase()} REGION
              </text>
            )}
          </svg>
        )}
      </motion.div>

      {/* -------- legend + resonance chips -------- */}
      <motion.div
        variants={fade}
        className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-void-lineSoft px-4 py-3 sm:px-5"
      >
        <LegendKey color={C_MEASURED} label="Measured" />
        {target && <LegendKey color={C_TARGET} label="Genre target" />}
        {paths?.refD && <LegendKey color={C_REFERENCE} label="Reference" dashed />}
        {target && (
          <>
            <LegendKey color={C_HOT} label="Hot vs target" swatch="fill" />
            <LegendKey color={C_COOL} label="Shy vs target" swatch="fill" />
          </>
        )}
      </motion.div>

      {resonances.length > 0 && (
        <motion.div variants={fade} className="border-t border-void-lineSoft px-4 py-3 sm:px-5">
          <p className="eyebrow mb-2">Detected resonances</p>
          <div className="no-scrollbar mask-fade-r -mx-1 flex gap-2 overflow-x-auto px-1 pb-0.5">
            {resonances.map((r, i) => {
              const idx = pins.findIndex((p) => p.res === r);
              return (
                <button
                  key={`chip-${r.freq_hz}-${i}`}
                  type="button"
                  onMouseEnter={() => setActiveRes(idx)}
                  onMouseLeave={() => setActiveRes(null)}
                  onFocus={() => setActiveRes(idx)}
                  onBlur={() => setActiveRes(null)}
                  className={`sev-chip shrink-0 sev-${r.severity} transition-transform duration-200 ease-cine hover:-translate-y-0.5`}
                  aria-label={`Resonance ${formatHz(r.freq_hz)} hertz, Q ${r.q.toFixed(1)}, ${formatDb(
                    r.prominence_db,
                  )} decibels`}
                >
                  <span className="text-ink">{formatHz(r.freq_hz)} Hz</span>
                  <span className="opacity-70">Q{r.q.toFixed(1)}</span>
                  <span className="opacity-70">{formatDb(r.prominence_db)} dB</span>
                </button>
              );
            })}
          </div>
        </motion.div>
      )}
    </motion.section>
  );
}

/* ------------------------------------------------------------- fragments */

function Readout({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="min-w-[52px]">
      <p className="eyebrow text-[9px]">{label}</p>
      <p className="stat mt-1 whitespace-nowrap text-xs text-ink sm:text-sm" style={{ color: tone }}>
        {value}
      </p>
    </div>
  );
}

function LegendKey({
  color,
  label,
  dashed,
  swatch,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  swatch?: 'fill';
}) {
  return (
    <span className="flex items-center gap-2">
      {swatch === 'fill' ? (
        <span
          aria-hidden="true"
          className="h-2.5 w-4 rounded-sm"
          style={{ background: color, opacity: 0.55 }}
        />
      ) : (
        <svg width="18" height="8" aria-hidden="true">
          <line
            x1="0"
            y1="4"
            x2="18"
            y2="4"
            stroke={color}
            strokeWidth="2"
            strokeDasharray={dashed ? '4 3' : undefined}
            strokeLinecap="round"
          />
        </svg>
      )}
      <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-muted">
        {label}
      </span>
    </span>
  );
}

export default SpectrumCurve;
