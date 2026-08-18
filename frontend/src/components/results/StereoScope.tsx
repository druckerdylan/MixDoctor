/**
 * StereoScope — the stereo field, told honestly.
 *
 * The goniometer here is SYNTHESISED. We do not ship the samples to the
 * browser, so the figure is a parametric Lissajous built from the measured
 * correlation and width: two cosines phase-offset by acos(correlation),
 * plotted as mid (vertical) against side (horizontal) exactly like a real
 * vectorscope. It is a faithful *picture of the numbers* and it is labeled as
 * such — it is never presented as live audio.
 */

import { useId, useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  SEVERITY_VAR,
  formatDb,
  formatHz,
  type PhaseMeasurement,
  type Severity,
  type StereoMeasurement,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const SEV_CLASS: Record<Severity, string> = {
  critical: 'sev-critical',
  major: 'sev-major',
  minor: 'sev-minor',
  clean: 'sev-clean',
};

/** Shape as well as hue — severity is never carried by color alone. */
const SEV_GLYPH: Record<Severity, string> = {
  critical: '▲',
  major: '●',
  minor: '◆',
  clean: '✓',
};

interface MacroBand {
  key: string;
  label: string;
  lo: number;
  hi: number;
  /** Above this width the band is smearing energy that should be centered. */
  widthLimit: number | null;
  /** Below this correlation the band will lose level in mono. */
  corrFloor: number;
}

/** Core macro-band order, low to high. Mirrors MACRO_BANDS in the analyzer. */
const MACRO_BANDS: MacroBand[] = [
  { key: 'sub', label: 'Sub', lo: 20, hi: 60, widthLimit: 0.18, corrFloor: 0.85 },
  { key: 'low_bass', label: 'Low bass', lo: 60, hi: 120, widthLimit: 0.3, corrFloor: 0.7 },
  { key: 'upper_bass', label: 'Upper bass', lo: 120, hi: 250, widthLimit: 0.45, corrFloor: 0.5 },
  { key: 'low_mid', label: 'Low mid', lo: 250, hi: 500, widthLimit: null, corrFloor: 0.15 },
  { key: 'mid', label: 'Mid', lo: 500, hi: 1000, widthLimit: null, corrFloor: 0.05 },
  { key: 'upper_mid', label: 'Upper mid', lo: 1000, hi: 2000, widthLimit: null, corrFloor: 0.0 },
  { key: 'presence', label: 'Presence', lo: 2000, hi: 5000, widthLimit: null, corrFloor: -0.05 },
  { key: 'brilliance', label: 'Brilliance', lo: 5000, hi: 10000, widthLimit: null, corrFloor: -0.1 },
  { key: 'air', label: 'Air', lo: 10000, hi: 20000, widthLimit: null, corrFloor: -0.15 },
];

/* ------------------------------------------------------------------ maths */

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** Scope geometry, in the 240-unit viewBox. */
const VB = 240;
const CX = VB / 2;
const CY = VB / 2;
const R_BEZEL = 100;
/** Half-height of the mid axis at full modulation. */
const R_MID = 62;

/**
 * One closed Lissajous figure.
 *
 * L(t) = cos t, R(t) = cos(t + φ) with φ = acos(correlation). Plotting
 * mid = (L+R)/2 up the screen and side = (R−L)/2 across it gives the standard
 * vectorscope orientation: correlation +1 collapses to a vertical line,
 * 0 opens into a circle, −1 lies flat on the horizontal.
 */
function lissajous(correlation: number, sideGain: number, amp: number, samples = 420): string {
  const phi = Math.acos(clamp(correlation, -1, 1));
  const parts: string[] = [];
  for (let i = 0; i <= samples; i += 1) {
    const t = (i / samples) * Math.PI * 2;
    const l = Math.cos(t);
    const r = Math.cos(t + phi);
    const mid = (l + r) / 2;
    const side = (r - l) / 2;
    const x = side * R_MID * sideGain * amp;
    const y = -mid * R_MID * amp;
    parts.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`);
  }
  parts.push('Z');
  return parts.join('');
}

function correlationVerdict(corr: number, monoLossDb: number): { severity: Severity; words: string } {
  const loss = Math.abs(monoLossDb);
  if (corr < 0) {
    return {
      severity: 'critical',
      words: 'Out of phase. Material is canceling — a mono sum will lose real level.',
    };
  }
  if (corr < 0.25) {
    return {
      severity: 'major',
      words: 'Very wide and only loosely correlated. Check this on a single speaker before you print.',
    };
  }
  if (corr < 0.55) {
    return {
      severity: 'minor',
      words:
        loss > 1.2
          ? 'Wide. It survives a mono sum, but it gives up some level doing it.'
          : 'Wide and still mono-compatible. This is normal for a modern master.',
    };
  }
  if (corr > 0.94) {
    return {
      severity: 'minor',
      words: 'Almost fully correlated — close to mono. There is width available if the arrangement wants it.',
    };
  }
  return { severity: 'clean', words: 'Well correlated. Sums to mono cleanly.' };
}

/* -------------------------------------------------------------- fragments */

function StatCell({
  label,
  value,
  unit,
  tone,
  note,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: string;
  note?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="eyebrow">{label}</p>
      <p className="stat mt-2 text-[19px] leading-none" style={tone ? { color: tone } : undefined}>
        {value}
        {unit ? <span className="ml-1 text-[11px] text-ink-faint">{unit}</span> : null}
      </p>
      {note ? <p className="mt-1.5 text-[11.5px] leading-snug text-ink-muted">{note}</p> : null}
    </div>
  );
}

/* ------------------------------------------------------------ correlation */

function CorrelationMeter({
  value,
  severity,
  verdictWords,
  monoWords,
}: {
  value: number;
  severity: Severity;
  verdictWords: string;
  monoWords: string;
}) {
  const reduce = useReducedMotion() ?? false;
  const v = clamp(value, -1, 1);
  const pct = ((v + 1) / 2) * 100;
  const color = SEVERITY_VAR[severity];

  const ticks = [-1, -0.5, 0, 0.5, 1];

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="eyebrow">Correlation −1 … +1</p>
        <p className="stat text-[15px]" style={{ color }}>
          {v >= 0 ? '+' : ''}
          {v.toFixed(2)}
        </p>
      </div>

      <div
        className="relative mt-3 h-9"
        role="img"
        aria-label={`Phase correlation ${v.toFixed(2)} on a scale from minus one to plus one. Below zero is out of phase.`}
      >
        {/* Rail. The negative half is a marked hazard zone, not just a color. */}
        <div className="absolute inset-x-0 top-1.5 h-3 overflow-hidden rounded-full border border-void-line bg-void-deep">
          <div
            className="absolute inset-y-0 left-0 w-1/2"
            style={{
              background:
                'repeating-linear-gradient(135deg, rgba(255,59,88,0.30) 0 5px, rgba(255,59,88,0.10) 5px 10px)',
            }}
          />
          <div
            className="absolute inset-y-0 right-0 w-1/2"
            style={{
              background: 'linear-gradient(90deg, rgba(255,159,28,0.20), rgba(82,242,196,0.22))',
            }}
          />
          <div className="absolute inset-y-0 left-1/2 w-px bg-ink-faint/70" />
        </div>

        {/* Measured value */}
        <motion.div
          className="absolute top-0"
          style={{ left: `${pct}%` }}
          initial={reduce ? { opacity: 0 } : { opacity: 0, scale: 0.7 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: reduce ? 0.2 : 0.6, ease: EASE, delay: reduce ? 0 : 0.15 }}
        >
          <div className="-translate-x-1/2">
            <div
              className="h-6 w-[3px] rounded-full"
              style={{ background: color, boxShadow: `0 0 12px -1px ${color}` }}
            />
            <div
              className="mx-auto -mt-[3px] h-1.5 w-1.5 rotate-45"
              style={{ background: color }}
              aria-hidden="true"
            />
          </div>
        </motion.div>

        {/* Scale */}
        <div className="absolute inset-x-0 top-6 flex justify-between">
          {ticks.map((t) => (
            <span
              key={t}
              className="stat text-[10px] text-ink-faint"
              style={{
                position: 'absolute',
                left: `${((t + 1) / 2) * 100}%`,
                transform: 'translateX(-50%)',
              }}
            >
              {t > 0 ? `+${t}` : t}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-7 flex items-start gap-2.5">
        <span className={`mt-[2px] shrink-0 text-[11px] ${SEV_CLASS[severity]}`} aria-hidden="true">
          {SEV_GLYPH[severity]}
        </span>
        <div className="min-w-0">
          <p className="text-[13.5px] leading-snug text-ink">{verdictWords}</p>
          <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">{monoWords}</p>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- bands */

interface BandRow {
  key: string;
  label: string;
  lo: number;
  hi: number;
  width: number | null;
  corr: number | null;
  monoLoss: number | null;
  flagged: boolean;
  reason: string | null;
  /** Position of the "as wide as this range should ever get" tick, 0–100. */
  widthLimitPct: number | null;
}

function BandTable({ rows }: { rows: BandRow[] }) {
  const reduce = useReducedMotion() ?? false;
  const flaggedCount = rows.filter((r) => r.flagged).length;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="eyebrow">Width &amp; correlation by band</p>
        {flaggedCount > 0 ? (
          <span className="sev-chip sev-major">
            <span aria-hidden="true">●</span>
            {flaggedCount} low band{flaggedCount === 1 ? '' : 's'} too wide
          </span>
        ) : (
          <span className="sev-chip sev-clean">
            <span aria-hidden="true">✓</span>
            Low end held together
          </span>
        )}
      </div>

      <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
        <table className="w-full min-w-[540px] border-collapse text-left">
          <thead>
            <tr className="border-b border-void-line">
              <th scope="col" className="eyebrow pb-2 pr-3 font-normal">
                Band
              </th>
              <th scope="col" className="eyebrow pb-2 pr-3 font-normal">
                Range
              </th>
              <th scope="col" className="eyebrow pb-2 pr-3 font-normal">
                Width 0–1
              </th>
              <th scope="col" className="eyebrow pb-2 pr-3 text-right font-normal">
                Corr
              </th>
              <th scope="col" className="eyebrow pb-2 text-right font-normal">
                Mono loss
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const w = r.width === null ? null : clamp(r.width, 0, 1);
              const tone = r.flagged ? SEVERITY_VAR.major : '#52F2C4';
              return (
                <tr key={r.key} className="border-b border-void-lineSoft last:border-0">
                  <th scope="row" className="py-2.5 pr-3 text-left align-middle">
                    <span
                      className={`text-[12.5px] font-medium ${r.flagged ? 'text-ink' : 'text-ink-dim'}`}
                    >
                      {r.label}
                    </span>
                    {r.flagged ? (
                      <span className="ml-2 align-middle text-[10px] text-sev-major" aria-hidden="true">
                        ●
                      </span>
                    ) : null}
                  </th>
                  <td className="py-2.5 pr-3 align-middle">
                    <span className="stat text-[11px] text-ink-faint">
                      {formatHz(r.lo)}–{formatHz(r.hi)}
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 align-middle">
                    <div className="flex items-center gap-2.5">
                      <div className="relative h-[5px] w-full min-w-[80px] overflow-hidden rounded-full bg-void-line/70">
                        {w === null ? null : (
                          <motion.div
                            className="absolute inset-y-0 left-0 rounded-full"
                            style={{
                              width: '100%',
                              transformOrigin: 'left center',
                              background: r.flagged
                                ? `linear-gradient(90deg, color-mix(in srgb, ${tone} 45%, transparent), ${tone})`
                                : 'linear-gradient(90deg, rgba(167,167,180,0.35), rgba(167,167,180,0.8))',
                              boxShadow: r.flagged ? `0 0 10px -2px ${tone}` : undefined,
                            }}
                            initial={{ scaleX: reduce ? w : 0 }}
                            whileInView={{ scaleX: w }}
                            viewport={{ once: true, amount: 0.3 }}
                            transition={{
                              duration: reduce ? 0 : 0.7,
                              ease: EASE,
                              delay: reduce ? 0 : 0.03 * i,
                            }}
                          />
                        )}
                        {r.widthLimitPct !== null ? (
                          <span
                            aria-hidden="true"
                            className="absolute inset-y-[-3px] w-px bg-ink-faint"
                            style={{ left: `${r.widthLimitPct}%` }}
                          />
                        ) : null}
                      </div>
                      <span className="stat w-[38px] shrink-0 text-right text-[11.5px] text-ink-dim">
                        {w === null ? '—' : w.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="py-2.5 pr-3 text-right align-middle">
                    <span
                      className="stat text-[11.5px]"
                      style={{
                        color:
                          r.corr === null ? '#4A4A57' : r.corr < 0 ? SEVERITY_VAR.critical : '#A7A7B4',
                      }}
                    >
                      {r.corr === null ? '—' : `${r.corr >= 0 ? '+' : ''}${r.corr.toFixed(2)}`}
                    </span>
                  </td>
                  <td className="py-2.5 text-right align-middle">
                    <span
                      className="stat text-[11.5px]"
                      style={{
                        color:
                          r.monoLoss === null
                            ? '#4A4A57'
                            : Math.abs(r.monoLoss) >= 3
                              ? SEVERITY_VAR.critical
                              : Math.abs(r.monoLoss) >= 1.5
                                ? SEVERITY_VAR.major
                                : '#A7A7B4',
                      }}
                    >
                      {r.monoLoss === null ? '—' : `${formatDb(r.monoLoss)} dB`}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {rows.some((r) => r.flagged && r.reason) ? (
        <ul className="mt-4 space-y-2">
          {rows
            .filter((r) => r.flagged && r.reason)
            .map((r) => (
              <li key={`why-${r.key}`} className="flex gap-2.5">
                <span className="mt-[2px] shrink-0 text-[10px] text-sev-major" aria-hidden="true">
                  ●
                </span>
                <p className="text-[12.5px] leading-snug text-ink-dim">
                  <span className="text-ink">{r.label}</span> — {r.reason}
                </p>
              </li>
            ))}
        </ul>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ props */

export interface StereoScopeProps {
  stereo: StereoMeasurement;
  phase: PhaseMeasurement;
}

export default function StereoScope({ stereo, phase }: StereoScopeProps) {
  const reduce = useReducedMotion() ?? false;
  const uid = useId().replace(/:/g, '');
  const glowId = `scope-glow-${uid}`;
  const fadeId = `scope-fade-${uid}`;

  const isMono = Boolean(stereo?.is_mono_source);
  const correlation = finite(stereo?.correlation, 1);
  const width = finite(stereo?.width, 0);
  const monoLoss = finite(stereo?.mono_sum_loss_db ?? phase?.mono_sum_loss_db, 0);
  const balance = finite(stereo?.balance_db, 0);
  const lowSide = finite(stereo?.low_end_side_energy_db, 0);
  const polarityInverted = Boolean(phase?.polarity_inverted);

  const verdict = useMemo(
    () => correlationVerdict(correlation, monoLoss),
    [correlation, monoLoss],
  );

  const sideGain = useMemo(() => Math.min(1.5, 0.7 + 1.4 * clamp(width, 0, 1)), [width]);

  const traces = useMemo(() => {
    // A bright primary trace plus three tighter ghosts — the persistence of a
    // real scope, without pretending to be one.
    const ghosts = [0.94, 0.86, 0.74];
    return {
      main: lissajous(correlation, sideGain, 1),
      ghosts: ghosts.map((a) => lissajous(correlation, sideGain, a)),
    };
  }, [correlation, sideGain]);

  const bandRows = useMemo<BandRow[]>(() => {
    const bw = stereo?.band_width ?? {};
    const bc = stereo?.band_correlation ?? {};
    const bl = stereo?.band_mono_loss_db ?? phase?.band_mono_loss_db ?? {};

    return MACRO_BANDS.map((b) => {
      const rawW = bw[b.key];
      const rawC = bc[b.key];
      const rawL = bl[b.key];
      const w = typeof rawW === 'number' && Number.isFinite(rawW) ? rawW : null;
      const c = typeof rawC === 'number' && Number.isFinite(rawC) ? rawC : null;
      const l = typeof rawL === 'number' && Number.isFinite(rawL) ? rawL : null;

      const tooWide = b.widthLimit !== null && w !== null && w > b.widthLimit;
      const tooLoose = c !== null && c < b.corrFloor && b.widthLimit !== null;
      const flagged = Boolean(tooWide || tooLoose);

      let reason: string | null = null;
      if (tooWide && tooLoose) {
        reason = `width ${w?.toFixed(2)} and correlation ${c?.toFixed(2)} — energy below ${formatHz(b.hi)} Hz should sit in the middle. Mono this band.`;
      } else if (tooWide) {
        reason = `width ${w?.toFixed(2)} against a ${b.widthLimit?.toFixed(2)} ceiling for this range. Narrow it or mono it.`;
      } else if (tooLoose) {
        reason = `correlation ${c?.toFixed(2)} — this will lose weight the moment it is summed.`;
      }

      return {
        key: b.key,
        label: b.label,
        lo: b.lo,
        hi: b.hi,
        width: w,
        corr: c,
        monoLoss: l,
        flagged,
        reason,
        widthLimitPct: b.widthLimit === null ? null : b.widthLimit * 100,
      };
    });
  }, [stereo?.band_width, stereo?.band_correlation, stereo?.band_mono_loss_db, phase?.band_mono_loss_db]);

  const hasBandData = bandRows.some((r) => r.width !== null || r.corr !== null);

  const monoWords = phase?.mono_compatible
    ? `Summing to mono costs ${Math.abs(monoLoss).toFixed(1)} dB. That is inside the margin an engineer would accept.`
    : `Summing to mono costs ${Math.abs(monoLoss).toFixed(1)} dB${
        phase?.worst_band ? `, worst in ${phase.worst_band.replace(/_/g, ' ')}` : ''
      }. Playback on a phone speaker, a club sub or a shop PA will not sound like your room.`;

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 14 },
    whileInView: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.15 },
    transition: { duration: reduce ? 0.25 : 0.6, ease: EASE, delay: reduce ? 0 : delay },
  });

  return (
    <section className="panel overflow-hidden p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <p className="eyebrow">Stereo field</p>
          <h3 className="display mt-2 text-[clamp(1.15rem,2.2vw,1.6rem)] leading-none tracking-[-0.03em] text-ink">
            {isMono ? 'Mono source' : 'Width, phase & mono fold-down'}
          </h3>
        </div>
        <span className={`sev-chip ${SEV_CLASS[isMono ? 'minor' : verdict.severity]}`}>
          <span aria-hidden="true">{SEV_GLYPH[isMono ? 'minor' : verdict.severity]}</span>
          {isMono
            ? 'Single channel'
            : phase?.mono_compatible
              ? 'Mono safe'
              : 'Mono risk'}
        </span>
      </div>

      {/* Five-alarm: polarity flip. Nothing else on this page matters until it is fixed. */}
      {polarityInverted ? (
        <motion.div
          {...rise(0.02)}
          className="relative mt-5 overflow-hidden rounded-xl2 border p-4 sm:p-5"
          style={{
            borderColor: 'rgba(255,59,88,0.45)',
            background:
              'linear-gradient(100deg, rgba(255,59,88,0.14), rgba(255,59,88,0.04) 55%, transparent)',
          }}
        >
          <span aria-hidden="true" className="absolute inset-y-0 left-0 w-[3px] bg-sev-critical" />
          <div className="relative flex items-start gap-3">
            <span className="mt-[2px] shrink-0 text-[13px] text-sev-critical" aria-hidden="true">
              ▲
            </span>
            <div className="min-w-0">
              <p className="eyebrow text-sev-critical">Polarity inverted</p>
              <p className="display mt-2 text-[15px] leading-snug tracking-[-0.02em] text-ink">
                {isMono
                  ? 'This file is polarity-inverted.'
                  : 'One channel is flipped against the other.'}
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">
                {isMono
                  ? 'Flip the polarity before you layer this with anything else. On its own it sounds identical; against another source carrying the same material it will cancel.'
                  : 'This is not a taste call. Flip the polarity on the offending channel or stem before you touch anything else — every measurement below is being read through a mix that partially cancels itself.'}
              </p>
            </div>
          </div>
        </motion.div>
      ) : null}

      {isMono ? (
        /* ---------------------------------------------------- mono source */
        <motion.div {...rise(0.06)} className="mt-6">
          <div className="panel-raised flex flex-col items-center justify-center gap-4 px-5 py-12 text-center">
            <svg viewBox="0 0 64 64" width="52" height="52" aria-hidden="true" className="text-ink-faint">
              <circle cx="32" cy="32" r="24" fill="none" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="32" y1="8" x2="32" y2="56" stroke="#52F2C4" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
            <div>
              <p className="display text-[17px] leading-snug tracking-[-0.02em] text-ink">
                This file has one channel.
              </p>
              <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-ink-dim">
                There is no stereo field to measure, so nothing is drawn here. Correlation, width and
                per-band imaging only exist once there are two channels to compare — we will not
                invent them.
              </p>
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
              <span className="sev-chip sev-minor">
                <span aria-hidden="true">◆</span>
                Mono · guaranteed fold-down safe
              </span>
              <span className="sev-chip sev-clean">
                <span aria-hidden="true">✓</span>
                0.0 dB mono sum loss
              </span>
            </div>
          </div>
        </motion.div>
      ) : (
        <>
          <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)] lg:gap-9">
            {/* ------------------------------------------------ goniometer */}
            <motion.div {...rise(0.06)} className="min-w-0">
              <div className="relative mx-auto w-full max-w-[320px]">
                <svg
                  viewBox={`0 0 ${VB} ${VB}`}
                  width="100%"
                  height="100%"
                  role="img"
                  aria-label={`Vectorscope representation of the measured stereo field. Correlation ${correlation.toFixed(
                    2,
                  )}, width ${width.toFixed(2)}. Drawn from the measured values, not from live audio.`}
                  className="block overflow-visible"
                  style={{ aspectRatio: '1 / 1' }}
                >
                  <defs>
                    <filter id={glowId} x="-40%" y="-40%" width="180%" height="180%">
                      <feGaussianBlur stdDeviation="2.4" result="b" />
                      <feMerge>
                        <feMergeNode in="b" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                    <radialGradient id={fadeId} cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#52F2C4" stopOpacity="0.16" />
                      <stop offset="100%" stopColor="#52F2C4" stopOpacity="0" />
                    </radialGradient>
                  </defs>

                  <circle cx={CX} cy={CY} r={R_BEZEL} fill="#030306" stroke="#1E1E29" strokeWidth="1" />
                  <circle cx={CX} cy={CY} r={R_BEZEL - 1} fill={`url(#${fadeId})`} />

                  <g transform={`translate(${CX} ${CY})`}>
                    {/* Reticle */}
                    {[0.34, 0.67, 1].map((f) => (
                      <circle
                        key={f}
                        cx={0}
                        cy={0}
                        r={R_BEZEL * f * 0.92}
                        fill="none"
                        stroke="#1E1E29"
                        strokeWidth="0.7"
                      />
                    ))}
                    <line x1={0} y1={-R_BEZEL * 0.94} x2={0} y2={R_BEZEL * 0.94} stroke="#2A2A38" strokeWidth="0.8" />
                    <line x1={-R_BEZEL * 0.94} y1={0} x2={R_BEZEL * 0.94} y2={0} stroke="#1E1E29" strokeWidth="0.7" />
                    {/* L / R diagonals */}
                    <line
                      x1={-R_BEZEL * 0.66}
                      y1={-R_BEZEL * 0.66}
                      x2={R_BEZEL * 0.66}
                      y2={R_BEZEL * 0.66}
                      stroke="#1E1E29"
                      strokeWidth="0.7"
                      strokeDasharray="3 4"
                    />
                    <line
                      x1={R_BEZEL * 0.66}
                      y1={-R_BEZEL * 0.66}
                      x2={-R_BEZEL * 0.66}
                      y2={R_BEZEL * 0.66}
                      stroke="#1E1E29"
                      strokeWidth="0.7"
                      strokeDasharray="3 4"
                    />

                    {/* Ghost traces */}
                    {traces.ghosts.map((d, i) => (
                      <motion.path
                        key={i}
                        d={d}
                        fill="none"
                        stroke="#52F2C4"
                        strokeWidth={0.8}
                        opacity={0.14 - i * 0.03}
                        initial={reduce ? { opacity: 0.14 - i * 0.03 } : { pathLength: 0 }}
                        whileInView={reduce ? undefined : { pathLength: 1 }}
                        viewport={{ once: true, amount: 0.3 }}
                        transition={{ duration: reduce ? 0 : 1.1, ease: EASE, delay: reduce ? 0 : 0.2 + i * 0.06 }}
                      />
                    ))}

                    {/* Soft body */}
                    <path d={traces.main} fill="#52F2C4" opacity={0.07} />

                    {/* Primary trace */}
                    <motion.path
                      d={traces.main}
                      fill="none"
                      stroke="#52F2C4"
                      strokeWidth={1.6}
                      strokeLinecap="round"
                      filter={`url(#${glowId})`}
                      initial={reduce ? { opacity: 1 } : { pathLength: 0, opacity: 0.6 }}
                      whileInView={reduce ? { opacity: 1 } : { pathLength: 1, opacity: 1 }}
                      viewport={{ once: true, amount: 0.3 }}
                      transition={{ duration: reduce ? 0 : 1.25, ease: EASE, delay: reduce ? 0 : 0.15 }}
                    />

                    {/* Axis labels */}
                    <text x={0} y={-R_BEZEL - 7} textAnchor="middle" className="fill-ink-muted" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      M
                    </text>
                    <text x={0} y={R_BEZEL + 15} textAnchor="middle" className="fill-ink-faint" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      −M
                    </text>
                    <text x={-R_BEZEL - 8} y={3.5} textAnchor="end" className="fill-ink-faint" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      −S
                    </text>
                    <text x={R_BEZEL + 8} y={3.5} textAnchor="start" className="fill-ink-faint" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      +S
                    </text>
                    <text x={-R_BEZEL * 0.63} y={-R_BEZEL * 0.63} textAnchor="middle" className="fill-ink-muted" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      L
                    </text>
                    <text x={R_BEZEL * 0.63} y={-R_BEZEL * 0.63} textAnchor="middle" className="fill-ink-muted" style={{ font: '600 9px ui-monospace, monospace', letterSpacing: '0.14em' }}>
                      R
                    </text>
                  </g>
                </svg>
              </div>

              <p className="mt-4 text-center text-[11.5px] leading-snug text-ink-muted">
                <span className="text-ink-dim">Representation, not live audio.</span> The figure is
                generated from the measured correlation ({correlation >= 0 ? '+' : ''}
                {correlation.toFixed(2)}) and width ({width.toFixed(2)}): vertical is mono, horizontal
                is side.
              </p>
            </motion.div>

            {/* ----------------------------------------------- meters */}
            <motion.div {...rise(0.12)} className="min-w-0">
              <CorrelationMeter
                value={correlation}
                severity={verdict.severity}
                verdictWords={verdict.words}
                monoWords={monoWords}
              />

              <div className="hairline my-6" />

              <div className="grid grid-cols-2 gap-x-5 gap-y-6 sm:grid-cols-4">
                <StatCell
                  label="Width"
                  value={width.toFixed(2)}
                  note={width > 0.7 ? 'Very wide' : width < 0.2 ? 'Near mono' : 'Typical range'}
                />
                <StatCell
                  label="Mono sum"
                  value={formatDb(-Math.abs(monoLoss))}
                  unit="dB"
                  tone={
                    Math.abs(monoLoss) >= 3
                      ? SEVERITY_VAR.critical
                      : Math.abs(monoLoss) >= 1.5
                        ? SEVERITY_VAR.major
                        : undefined
                  }
                  note="Level lost folding to mono"
                />
                <StatCell
                  label="Low-end side"
                  value={formatDb(lowSide)}
                  unit="dB"
                  tone={lowSide > -12 ? SEVERITY_VAR.major : undefined}
                  note="Side energy below 120 Hz"
                />
                <StatCell
                  label="L / R balance"
                  value={formatDb(balance)}
                  unit="dB"
                  tone={Math.abs(balance) >= 1 ? SEVERITY_VAR.major : undefined}
                  note={
                    Math.abs(balance) < 0.3
                      ? 'Centered'
                      : `${Math.abs(balance).toFixed(1)} dB toward ${balance > 0 ? 'right' : 'left'}`
                  }
                />
              </div>

              {phase?.worst_band ? (
                <p className="mt-6 text-[12.5px] leading-snug text-ink-muted">
                  Worst band for phase:{' '}
                  <span className="text-ink-dim">{phase.worst_band.replace(/_/g, ' ')}</span> at
                  correlation{' '}
                  <span className="stat text-ink-dim">
                    {finite(phase.worst_band_correlation).toFixed(2)}
                  </span>
                  .
                </p>
              ) : null}
            </motion.div>
          </div>

          {hasBandData ? (
            <>
              <div className="hairline my-7" />
              <motion.div {...rise(0.18)}>
                <BandTable rows={bandRows} />
              </motion.div>
            </>
          ) : null}
        </>
      )}
    </section>
  );
}
