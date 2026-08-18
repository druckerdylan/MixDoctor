/**
 * SectionMap — the arrangement, measured section by section.
 *
 * A single integrated LUFS for a whole record hides the thing producers
 * actually get wrong: the chorus that does not lift. This draws the track as
 * blocks sized by duration and shaded by each section's own loudness, then
 * says the consequence out loud underneath.
 *
 * Color here is a *sequential ramp*, not the severity palette. Severity means
 * "how bad", and a loud section is not bad — it is loud. Mixing the two
 * vocabularies is how a chart starts lying. The ramp is anchored to a fixed
 * 12 LU window below the loudest section, so a flat track renders as a flat
 * band of near-identical blocks rather than being auto-stretched into a
 * dynamic-looking one.
 */

import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  SEVERITY_VAR,
  formatTime,
  type Section,
  type SectionAnalysis,
  type Severity,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const SEV_CLASS: Record<Severity, string> = {
  critical: 'sev-critical',
  major: 'sev-major',
  minor: 'sev-minor',
  clean: 'sev-clean',
};

const SEV_GLYPH: Record<Severity, string> = {
  critical: '▲',
  major: '●',
  minor: '◆',
  clean: '✓',
};

/** How far below the loudest section the ramp bottoms out. */
const RAMP_SPAN_LU = 12;

/**
 * Sequential ramp, dark to bright, one hue family. Reads as magnitude at a
 * glance and stays legible on a near-black page at every stop.
 */
const RAMP: string[] = ['#101E28', '#164452', '#1C7A72', '#3FC79C', '#A8FFE6'];

/** Below these widths the block cannot hold the label without lying about it. */
const PX_FOR_LABEL = 46;
const PX_FOR_LUFS = 74;
const PX_FOR_TIME = 104;

/* ------------------------------------------------------------------ utils */

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

/** Position 0–1 along RAMP, linearly interpolated in sRGB. */
function rampColor(t: number): string {
  const x = clamp(t, 0, 1) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(RAMP[i] ?? '#101E28');
  const b = hexToRgb(RAMP[i + 1] ?? '#A8FFE6');
  const mix = (j: 0 | 1 | 2): number => Math.round((a[j] ?? 0) + ((b[j] ?? 0) - (a[j] ?? 0)) * f);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

/** Perceived brightness, so the label can pick a side rather than guess. */
function isBright(t: number): boolean {
  const x = clamp(t, 0, 1) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(RAMP[i] ?? '#101E28');
  const b = hexToRgb(RAMP[i + 1] ?? '#A8FFE6');
  const lum =
    0.2126 * ((a[0] ?? 0) + ((b[0] ?? 0) - (a[0] ?? 0)) * f) +
    0.7152 * ((a[1] ?? 0) + ((b[1] ?? 0) - (a[1] ?? 0)) * f) +
    0.0722 * ((a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * f);
  return lum > 150;
}

/** "chorus" -> "Chorus", "section 3" -> "Section 3". */
function titleCase(label: string): string {
  const s = label.trim();
  if (!s) return 'Section';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/* ------------------------------------------------------------------ types */

interface Block {
  key: string;
  label: string;
  start: number;
  end: number;
  lufs: number;
  hasLufs: boolean;
  crest: number;
  width: number;
  leftPct: number;
  widthPct: number;
  t: number;
  color: string;
  bright: boolean;
  isLoudest: boolean;
  isQuietest: boolean;
}

/* ------------------------------------------------------------------ props */

export interface SectionMapProps {
  sections: SectionAnalysis;
  duration: number;
  onSeek?: (t: number) => void;
}

export default function SectionMap({ sections, duration, onSeek }: SectionMapProps) {
  const reduce = useReducedMotion() ?? false;

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [viewportW, setViewportW] = useState(0);

  // Layout effect: measuring after paint would flash a strip with every label
  // hidden and then pop them all in.
  useLayoutEffect(() => {
    const el = wrapRef.current;
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

  const list = useMemo<Section[]>(
    () => (Array.isArray(sections?.sections) ? sections.sections.filter(Boolean) : []),
    [sections?.sections],
  );

  const total = useMemo(() => {
    let maxEnd = 0;
    for (const s of list) maxEnd = Math.max(maxEnd, finite(s.t_end), finite(s.t_start));
    return Math.max(finite(duration), maxEnd);
  }, [list, duration]);

  const stripW = Math.max(viewportW, 480, list.length * 78);

  const blocks = useMemo<Block[]>(() => {
    if (total <= 0) return [];

    const lufsValues = list
      .map((s) => s.integrated_lufs)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const maxLufs = lufsValues.length ? Math.max(...lufsValues) : 0;

    return list
      .map((s, i) => {
        const start = clamp(finite(s.t_start), 0, total);
        const end = clamp(finite(s.t_end, start), start, total);
        const hasLufs = typeof s.integrated_lufs === 'number' && Number.isFinite(s.integrated_lufs);
        const lufs = hasLufs ? s.integrated_lufs : maxLufs - RAMP_SPAN_LU;
        const t = clamp((lufs - (maxLufs - RAMP_SPAN_LU)) / RAMP_SPAN_LU, 0, 1);
        return {
          key: `${i}-${s.label ?? 'section'}-${start.toFixed(2)}`,
          label: titleCase(typeof s.label === 'string' ? s.label : `section ${i + 1}`),
          start,
          end,
          lufs,
          hasLufs,
          crest: finite(s.crest_factor_db),
          width: end - start,
          leftPct: (start / total) * 100,
          widthPct: Math.max(0.4, ((end - start) / total) * 100),
          t,
          color: rampColor(t),
          bright: isBright(t),
          isLoudest: Boolean(s.is_loudest),
          isQuietest: Boolean(s.is_quietest),
        };
      })
      .filter((b) => b.width > 0.01);
  }, [list, total]);

  const loudest = blocks.find((b) => b.isLoudest) ?? null;
  const quietest = blocks.find((b) => b.isQuietest) ?? null;

  const lift = finite(sections?.peak_lift_db);
  const spread = finite(sections?.loudness_spread_lu);
  const swing = finite(sections?.low_end_swing_db);
  const notes = Array.isArray(sections?.notes) ? sections.notes.filter((n) => Boolean(n)) : [];

  const liftSeverity: Severity = lift < 0.8 ? 'major' : lift < 1.8 ? 'minor' : 'clean';

  const liftWords = (() => {
    const named = loudest ? `the ${loudest.label.toLowerCase()}` : null;
    const value = `${Math.abs(lift).toFixed(1)} LU`;
    if (lift < 0.8) {
      return `Your loudest section is only ${value} above the median — ${
        named ? `${named} is not lifting` : 'nothing is lifting'
      }. Everything is arriving at the same level, so nothing reads as an arrival.`;
    }
    if (lift < 1.8) {
      return `Your loudest section sits ${value} above the median. That is a nudge, not a lift${
        named ? `; if ${named} is meant to be the payoff, it is not landing yet` : ''
      }.`;
    }
    if (lift <= 5.0) {
      return `Your loudest section sits ${value} above the median. The track has a top and you can hear where it is.`;
    }
    return `Your loudest section sits ${value} above the median. That is a big jump — worth checking that the quiet sections are shaped, not simply under-mixed.`;
  })();

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 14 },
    whileInView: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.15 },
    transition: { duration: reduce ? 0.25 : 0.6, ease: EASE, delay: reduce ? 0 : delay },
  });

  /* --------------------------------------------------------------- gates */

  if (!sections || !sections.available) return null;
  if (list.length < 2) return null;
  if (blocks.length < 2) return null;

  const interactive = typeof onSeek === 'function';
  const rampLow = blocks.reduce((m, b) => Math.min(m, b.lufs), Number.POSITIVE_INFINITY);
  const rampHigh = blocks.reduce((m, b) => Math.max(m, b.lufs), Number.NEGATIVE_INFINITY);

  return (
    <section className="panel overflow-hidden p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <p className="eyebrow">Arrangement</p>
          <h3 className="display mt-2 text-[clamp(1.15rem,2.2vw,1.6rem)] leading-none tracking-[-0.03em] text-ink">
            Section by section
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`sev-chip ${SEV_CLASS[liftSeverity]}`}>
            <span aria-hidden="true">{SEV_GLYPH[liftSeverity]}</span>
            {liftSeverity === 'clean' ? 'Lift is there' : `Lift ${lift.toFixed(1)} LU`}
          </span>
          <span className="sev-chip text-ink-muted">{blocks.length} sections</span>
        </div>
      </div>

      {/* ---------------------------------------------------------- strip */}
      <motion.div {...rise(0.06)} className="mt-6">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <p className="eyebrow">Shaded by integrated loudness</p>
          <p className="eyebrow">
            {Number.isFinite(rampLow) ? rampLow.toFixed(1) : '—'} →{' '}
            {Number.isFinite(rampHigh) ? rampHigh.toFixed(1) : '—'} LUFS
          </p>
        </div>

        {/* Wide arrangements scroll inside this box; the page never widens. */}
        <div ref={wrapRef} className="-mx-1 overflow-x-auto px-1 pb-1">
          <div
            className="relative h-[86px] select-none sm:h-[96px]"
            style={{ width: stripW }}
            role="list"
            aria-label={`Track arrangement, ${blocks.length} sections across ${formatTime(total)}.`}
          >
            {blocks.map((b, i) => {
              const px = (b.widthPct / 100) * stripW;
              const showLabel = px >= PX_FOR_LABEL;
              const showLufs = px >= PX_FOR_LUFS && b.hasLufs;
              const showTime = px >= PX_FOR_TIME;
              const text = b.bright ? '#04040A' : '#F4F4F7';
              const sub = b.bright ? 'rgba(4,4,10,0.62)' : 'rgba(167,167,180,0.9)';
              const marked = b.isLoudest || b.isQuietest;

              const label = `${b.label}, ${formatTime(b.start)} to ${formatTime(b.end)}${
                b.hasLufs ? `, ${b.lufs.toFixed(1)} LUFS` : ''
              }${b.isLoudest ? ', loudest section' : ''}${b.isQuietest ? ', quietest section' : ''}`;

              const inner = (
                <>
                  <span
                    aria-hidden="true"
                    className="absolute inset-0"
                    style={{ background: b.color }}
                  />
                  {/* A top sheen keeps flat blocks from reading as dead slabs. */}
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 top-0 h-1/2"
                    style={{
                      background: 'linear-gradient(180deg, rgba(255,255,255,0.07), transparent)',
                    }}
                  />
                  {marked ? (
                    <span
                      aria-hidden="true"
                      className="absolute inset-x-0 bottom-0 h-[3px]"
                      style={{ background: b.isLoudest ? '#8DFFE0' : 'rgba(167,167,180,0.75)' }}
                    />
                  ) : null}

                  <span className="relative flex h-full flex-col justify-between p-2 text-left sm:p-2.5">
                    <span className="min-w-0">
                      {showLabel ? (
                        <span
                          className="display block truncate text-[12.5px] leading-none tracking-[-0.02em]"
                          style={{ color: text }}
                        >
                          {b.label}
                        </span>
                      ) : null}
                      {showLufs ? (
                        <span
                          className="stat mt-1.5 block truncate text-[11.5px] leading-none"
                          style={{ color: text }}
                        >
                          {b.lufs.toFixed(1)}
                          <span className="ml-1 text-[8.5px]" style={{ color: sub }}>
                            LUFS
                          </span>
                        </span>
                      ) : null}
                    </span>

                    <span className="flex min-w-0 items-end justify-between gap-1">
                      {showTime ? (
                        <span
                          className="stat truncate text-[9px] leading-none"
                          style={{ color: sub }}
                        >
                          {formatTime(b.start)}
                        </span>
                      ) : (
                        <span />
                      )}
                      {marked && showLabel ? (
                        <span
                          aria-hidden="true"
                          className="shrink-0 font-mono text-[8.5px] uppercase leading-none tracking-[0.12em]"
                          style={{ color: text }}
                        >
                          {b.isLoudest ? '▲ loudest' : '▼ quietest'}
                        </span>
                      ) : null}
                    </span>
                  </span>
                </>
              );

              const style: React.CSSProperties = {
                left: `${b.leftPct}%`,
                width: `calc(${b.widthPct}% - 2px)`,
              };

              return (
                <motion.div
                  key={b.key}
                  role="listitem"
                  className="absolute inset-y-0"
                  style={style}
                  initial={reduce ? { opacity: 0 } : { opacity: 0, scaleY: 0.72 }}
                  whileInView={{ opacity: 1, scaleY: 1 }}
                  viewport={{ once: true, amount: 0.3 }}
                  transition={{
                    duration: reduce ? 0.2 : 0.55,
                    ease: EASE,
                    delay: reduce ? 0 : 0.035 * i,
                  }}
                >
                  {interactive ? (
                    <button
                      type="button"
                      onClick={() => onSeek?.(b.start)}
                      aria-label={`Jump to ${label}`}
                      title={`${b.label} · ${formatTime(b.start)}–${formatTime(b.end)}${
                        b.hasLufs ? ` · ${b.lufs.toFixed(1)} LUFS` : ''
                      } · crest ${b.crest.toFixed(1)} dB`}
                      className="relative block h-full w-full overflow-hidden rounded-[4px] outline-offset-2 ring-1 ring-inset ring-white/5 transition-[filter,transform] duration-300 ease-cine hover:z-10 hover:brightness-125"
                    >
                      {inner}
                    </button>
                  ) : (
                    <div
                      title={`${b.label} · ${formatTime(b.start)}–${formatTime(b.end)}`}
                      className="relative h-full w-full overflow-hidden rounded-[4px] ring-1 ring-inset ring-white/5"
                    >
                      {inner}
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="mt-2 flex items-center justify-between gap-4">
          <span className="stat text-[9px] text-ink-faint">0:00</span>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-ink-faint">
              quiet
            </span>
            <span
              aria-hidden="true"
              className="h-[6px] w-24 rounded-full"
              style={{ background: `linear-gradient(90deg, ${RAMP.join(', ')})` }}
            />
            <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-ink-faint">
              loud
            </span>
          </div>
          <span className="stat text-[9px] text-ink-faint">{formatTime(total)}</span>
        </div>

        {interactive ? (
          <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            Click a section to jump to it
          </p>
        ) : null}
      </motion.div>

      {/* --------------------------------------------------------- the point */}
      <motion.p
        {...rise(0.12)}
        className="display mt-7 max-w-3xl text-pretty text-[clamp(1rem,1.9vw,1.35rem)] leading-[1.25] tracking-[-0.02em] text-ink"
      >
        {liftWords}
      </motion.p>

      {loudest && quietest && loudest.key !== quietest.key ? (
        <motion.p {...rise(0.15)} className="mt-3 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">
          Loudest is <span className="text-ink-dim">{loudest.label.toLowerCase()}</span> at{' '}
          <span className="stat text-ink-dim">{formatTime(loudest.start)}</span>
          {loudest.hasLufs ? (
            <>
              {' '}
              (<span className="stat text-ink-dim">{loudest.lufs.toFixed(1)} LUFS</span>)
            </>
          ) : null}
          ; quietest is <span className="text-ink-dim">{quietest.label.toLowerCase()}</span> at{' '}
          <span className="stat text-ink-dim">{formatTime(quietest.start)}</span>
          {quietest.hasLufs ? (
            <>
              {' '}
              (<span className="stat text-ink-dim">{quietest.lufs.toFixed(1)} LUFS</span>)
            </>
          ) : null}
          .
        </motion.p>
      ) : null}

      {/* ------------------------------------------------------- the numbers */}
      <div className="hairline my-7" />

      <motion.div {...rise(0.18)} className="grid grid-cols-2 gap-x-5 gap-y-6 sm:grid-cols-4">
        <div className="min-w-0">
          <p className="eyebrow whitespace-nowrap">Peak lift</p>
          <p
            className="stat mt-2 whitespace-nowrap text-[clamp(1rem,1.9vw,1.25rem)] leading-none"
            style={liftSeverity === 'clean' ? undefined : { color: SEVERITY_VAR[liftSeverity] }}
          >
            {lift.toFixed(1)}
            <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">LU</span>
          </p>
          <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">Loudest vs the median</p>
        </div>

        <div className="min-w-0">
          <p className="eyebrow whitespace-nowrap">Loudness spread</p>
          <p
            className="stat mt-2 whitespace-nowrap text-[clamp(1rem,1.9vw,1.25rem)] leading-none"
            style={spread < 1.5 ? { color: SEVERITY_VAR.major } : undefined}
          >
            {spread.toFixed(1)}
            <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">LU</span>
          </p>
          <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">
            {spread < 1.5 ? 'Flat across the whole track' : 'Quietest to loudest section'}
          </p>
        </div>

        <div className="min-w-0">
          <p className="eyebrow whitespace-nowrap">Low-end swing</p>
          <p className="stat mt-2 whitespace-nowrap text-[clamp(1rem,1.9vw,1.25rem)] leading-none">
            {swing.toFixed(1)}
            <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">dB</span>
          </p>
          <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">
            Sub and low-bass variation between sections
          </p>
        </div>

        <div className="min-w-0">
          <p className="eyebrow whitespace-nowrap">Sections</p>
          <p className="stat mt-2 whitespace-nowrap text-[clamp(1rem,1.9vw,1.25rem)] leading-none">
            {blocks.length}
            <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">
              / {formatTime(total)}
            </span>
          </p>
          <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">
            Detected from the audio, not from markers
          </p>
        </div>
      </motion.div>

      {notes.length ? (
        <div className="mt-6 rounded-xl border border-void-line/70 px-4 py-3">
          <p className="eyebrow">Arrangement notes</p>
          <ul className="mt-2.5 space-y-1.5">
            {notes.map((n, i) => (
              <li key={`${n}-${i}`} className="flex gap-2.5">
                <span className="mt-[3px] shrink-0 text-[9px] text-ink-faint" aria-hidden="true">
                  ●
                </span>
                <span className="text-[12px] leading-snug text-ink-muted">{n}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
