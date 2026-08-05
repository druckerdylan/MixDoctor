/**
 * LoudnessTargets — what actually happens to this master when it leaves the room.
 *
 * The single fact most producers get wrong is that streaming turns loud
 * masters down, so the loudness war was lost years ago and mastering hot only
 * costs dynamics. So the delta is the hero here: one diverging bar per
 * platform, centred on that platform's target, reading "turned down 4.2 LU"
 * in plain words.
 */

import { useEffect, useMemo, useRef } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  SEVERITY_VAR,
  type LoudnessMeasurement,
  type PlatformTarget,
  type Severity,
  type Verdict,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

/** The scale everyone reads streaming loudness on. */
const S_MIN = -24;
const S_MAX = -4;
/** Half-domain of the per-platform delta bars, in LU. */
const DELTA_SPAN = 8;

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

const VERDICT_SEV: Record<Verdict, Severity> = {
  good: 'clean',
  watch: 'major',
  problem: 'critical',
};

/* ------------------------------------------------------------------ utils */

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** Position on the −24…−4 LUFS rail, as a percentage. */
function railPct(lufs: number): number {
  return clamp(((lufs - S_MIN) / (S_MAX - S_MIN)) * 100, 0, 100);
}

function lufs(v: number, digits = 1): string {
  return v.toFixed(digits);
}

/* -------------------------------------------------------------- fragments */

function Stat({
  label,
  value,
  unit,
  note,
  tone,
}: {
  label: string;
  value: string;
  unit: string;
  note: string;
  tone?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="eyebrow whitespace-nowrap">{label}</p>
      <p className="stat mt-2 whitespace-nowrap text-[clamp(1rem,2vw,1.3rem)] leading-none" style={tone ? { color: tone } : undefined}>
        {value}
        <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">{unit}</span>
      </p>
      <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">{note}</p>
    </div>
  );
}

/* --------------------------------------------------------------- the rail */

interface RailGroup {
  target: number;
  names: string[];
}

function LufsRail({ integrated, groups }: { integrated: number; groups: RailGroup[] }) {
  const reduce = useReducedMotion() ?? false;

  const quietEnd = railPct(-16);
  const hotStart = railPct(-11);
  const mixPct = railPct(integrated);
  const offScaleLoud = integrated > S_MAX;
  const offScaleQuiet = integrated < S_MIN;

  const ticks: number[] = [];
  for (let v = S_MIN; v <= S_MAX; v += 2) ticks.push(v);

  // When the scale has to scroll (narrow screens), open on the mix marker
  // rather than at -24 LUFS, where there is nothing to see.
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const overflow = el.scrollWidth - el.clientWidth;
    if (overflow <= 0) return;
    const target = (mixPct / 100) * el.scrollWidth - el.clientWidth / 2;
    el.scrollLeft = Math.max(0, Math.min(overflow, target));
  }, [mixPct]);

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="eyebrow">Integrated loudness · LUFS</p>
        <p className="eyebrow">
          scale {S_MIN} → {S_MAX}
        </p>
      </div>

      {/* Vertical budget, top to bottom: rail 0–22, axis ticks 22–26, axis
          numbers 26–36, platform stack 38–64. Nothing overlaps anything.
          Below ~600px the tick labels would collide, so the scale scrolls
          inside itself rather than dropping data or stacking on top of it. */}
      {/* Negative margin + matching padding: the end labels are centred on the
          scale ends, so they need room to hang past them without being clipped. */}
      <div ref={scrollRef} className="-mx-3 overflow-x-auto px-3 pb-1">
      <div
        className="relative mt-12 h-[66px] min-w-[560px] select-none"
        role="img"
        aria-label={`This mix measures ${lufs(integrated)} LUFS integrated. Platform targets sit at ${groups
          .map((g) => `${lufs(g.target)} for ${g.names.join(', ')}`)
          .join('; ')}.`}
      >
        {/* Rail */}
        <div className="absolute inset-x-0 top-0 flex h-[22px] items-center overflow-hidden rounded-lg border border-void-line bg-void-deep">
          <div
            className="absolute inset-0"
            style={{
              background: `linear-gradient(90deg,
                rgba(76,201,240,0.07) 0%,
                rgba(76,201,240,0.10) ${quietEnd}%,
                rgba(82,242,196,0.22) ${quietEnd}%,
                rgba(82,242,196,0.22) ${hotStart}%,
                rgba(255,159,28,0.20) ${hotStart}%,
                rgba(255,59,88,0.30) 100%)`,
            }}
          />
          {/* Target zone edges, so the band reads as a defined region not a wash. */}
          <div className="absolute inset-y-0 w-px bg-signal/45" style={{ left: `${quietEnd}%` }} />
          <div className="absolute inset-y-0 w-px bg-sev-major/45" style={{ left: `${hotStart}%` }} />

          {/* Zone caption, printed on the track like a real meter scale. */}
          <span
            className="absolute whitespace-nowrap font-mono text-[8px] uppercase tracking-[0.16em] text-signal-dim/70"
            style={{ left: `${(quietEnd + hotStart) / 2}%`, transform: 'translateX(-50%)' }}
          >
            streaming range
          </span>
        </div>

        {/* Axis ticks */}
        <div className="pointer-events-none absolute inset-x-0 top-[22px]" aria-hidden="true">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute top-0 w-px bg-void-line"
              style={{ left: `${railPct(t)}%`, height: t % 4 === 0 ? 4 : 2.5 }}
            />
          ))}
        </div>

        {/* Platform target ticks */}
        {groups.map((g) => {
          const p = railPct(g.target);
          const extra = g.names.length - 1;
          // First word only — "Apple Music" and "Club / DJ" are too wide to
          // sit under a tick without colliding with the next one.
          const first = (g.names[0] ?? '').split(/[\s/]+/)[0] ?? '';
          // The axis already prints a number every 4 LU; don't say it twice.
          const onAxisLabel = Math.abs(g.target % 4) < 0.05;
          return (
            <div key={g.target} className="absolute top-0" style={{ left: `${p}%` }}>
              <div className="-translate-x-1/2">
                <div className="mx-auto h-[22px] w-[2px] bg-ink-dim/70" />
                <div className="mt-4 flex flex-col items-center gap-1">
                  {/* Placeholder keeps every platform name on one baseline. */}
                  <span className="stat whitespace-nowrap text-[10px] leading-none text-ink-dim">
                    {onAxisLabel ? ' ' : lufs(g.target, 0)}
                  </span>
                  <span className="max-w-[92px] truncate whitespace-nowrap text-center font-mono text-[9px] uppercase leading-none tracking-[0.1em] text-ink-faint">
                    {first}
                    {extra > 0 ? ` +${extra}` : ''}
                  </span>
                </div>
              </div>
            </div>
          );
        })}

        {/* The mix */}
        <motion.div
          className="absolute top-0"
          style={{ left: `${mixPct}%` }}
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: -8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: reduce ? 0.2 : 0.7, ease: EASE, delay: reduce ? 0 : 0.2 }}
        >
          <div className="-translate-x-1/2">
            <div className="absolute bottom-full left-1/2 mb-2 -translate-x-1/2">
              <div
                className="flex items-center gap-2 whitespace-nowrap rounded-full border px-2.5 py-1"
                style={{
                  borderColor: 'rgba(82,242,196,0.4)',
                  background: 'rgba(82,242,196,0.10)',
                  boxShadow: '0 0 22px -6px rgba(82,242,196,0.6)',
                }}
              >
                <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-signal-dim">
                  {offScaleLoud ? 'off scale' : offScaleQuiet ? 'below scale' : 'this mix'}
                </span>
                <span className="stat text-[12px] text-signal">{lufs(integrated)}</span>
              </div>
            </div>
            <div
              className="h-[22px] w-[3px] rounded-full bg-signal"
              style={{ boxShadow: '0 0 14px -1px rgba(82,242,196,0.9)' }}
            />
          </div>
        </motion.div>

        {/* Scale labels — every 4 LU, above the platform stack. */}
        <div className="absolute inset-x-0 top-[27px]" aria-hidden="true">
          {ticks
            .filter((t) => t % 4 === 0)
            .map((t) => (
              <span
                key={t}
                className="stat absolute top-0 text-[10px] leading-none text-ink-faint"
                style={{ left: `${railPct(t)}%`, transform: 'translateX(-50%)' }}
              >
                {t}
              </span>
            ))}
        </div>
      </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------- platform rows */

interface Row {
  key: string;
  platform: string;
  note: string;
  targetLufs: number;
  ceiling: number;
  /** Measured minus target. Positive = louder than the platform wants. */
  over: number;
  turnedDown: boolean;
  peakOk: boolean;
  severity: Severity;
}

function DeltaBar({ over, turnedDown, index }: { over: number; turnedDown: boolean; index: number }) {
  const reduce = useReducedMotion() ?? false;
  const clamped = clamp(over, -DELTA_SPAN, DELTA_SPAN);
  const frac = Math.abs(clamped) / DELTA_SPAN / 2; // fraction of full bar width
  const overflowed = Math.abs(over) > DELTA_SPAN;
  const positive = clamped >= 0;

  const color = turnedDown
    ? SEVERITY_VAR.major
    : positive
      ? SEVERITY_VAR.clean
      : 'rgba(167,167,180,0.75)';

  return (
    <div className="relative h-[30px] w-full min-w-[170px]">
      {/* Track */}
      <div className="absolute inset-x-0 top-[9px] h-[10px] rounded-full border border-void-line bg-void-deep" />

      {/* ±4 LU guides */}
      {[-4, 4].map((g) => (
        <span
          key={g}
          aria-hidden="true"
          className="absolute top-[7px] h-[14px] w-px bg-void-line"
          style={{ left: `${50 + (g / DELTA_SPAN) * 50}%` }}
        />
      ))}

      {/* Target (zero) line */}
      <span
        aria-hidden="true"
        className="absolute top-[4px] h-[20px] w-px bg-ink-faint"
        style={{ left: '50%' }}
      />

      {/* The delta */}
      <motion.div
        className="absolute top-[10px] h-[8px]"
        style={{
          left: positive ? '50%' : `${50 - frac * 100}%`,
          transformOrigin: positive ? 'left center' : 'right center',
          width: `${frac * 100}%`,
          borderRadius: positive ? '0 4px 4px 0' : '4px 0 0 4px',
          background: positive
            ? `linear-gradient(90deg, color-mix(in srgb, ${color} 40%, transparent), ${color})`
            : `linear-gradient(270deg, color-mix(in srgb, ${color} 40%, transparent), ${color})`,
          boxShadow: turnedDown ? `0 0 14px -3px ${color}` : undefined,
        }}
        initial={{ scaleX: reduce ? 1 : 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: reduce ? 0 : 0.75, ease: EASE, delay: reduce ? 0 : 0.06 * index }}
      />

      {overflowed ? (
        <span
          aria-hidden="true"
          className="absolute top-[8px] stat text-[10px]"
          style={{ [positive ? 'right' : 'left']: 0, color }}
        >
          {positive ? '»' : '«'}
        </span>
      ) : null}

      <div className="absolute inset-x-0 top-[21px] flex justify-between">
        <span className="stat text-[9px] text-ink-faint">−{DELTA_SPAN} LU</span>
        <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-ink-faint">target</span>
        <span className="stat text-[9px] text-ink-faint">+{DELTA_SPAN} LU</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ props */

export interface LoudnessTargetsProps {
  targets: PlatformTarget[];
  loudness: LoudnessMeasurement;
}

export default function LoudnessTargets({ targets, loudness }: LoudnessTargetsProps) {
  const reduce = useReducedMotion() ?? false;

  const integrated = finite(loudness?.integrated_lufs, S_MIN);
  const truePeak = finite(loudness?.true_peak_dbtp, -99);
  const hasLoudness = typeof loudness?.integrated_lufs === 'number' && Number.isFinite(loudness.integrated_lufs);

  const rows = useMemo<Row[]>(() => {
    return (targets ?? []).map((t, i) => {
      const target = finite(t.target_lufs, -14);
      // Recompute from the measurement so the geometry can never disagree with
      // the number printed next to it; the flags stay the backend's call.
      const over = hasLoudness ? integrated - target : finite(t.delta_lufs, 0);
      return {
        key: `${t.platform}-${i}`,
        platform: t.platform || 'Platform',
        note: t.note ?? '',
        targetLufs: target,
        ceiling: finite(t.max_true_peak_dbtp, -1),
        over,
        turnedDown: Boolean(t.will_be_turned_down),
        peakOk: t.peak_ok !== false,
        severity: VERDICT_SEV[t.verdict] ?? 'minor',
      };
    });
  }, [targets, integrated, hasLoudness]);

  const groups = useMemo<RailGroup[]>(() => {
    const map = new Map<number, string[]>();
    for (const r of rows) {
      const key = Math.round(r.targetLufs * 10) / 10;
      const list = map.get(key);
      if (list) list.push(r.platform);
      else map.set(key, [r.platform]);
    }
    return [...map.entries()]
      .map(([target, names]) => ({ target, names }))
      .sort((a, b) => a.target - b.target);
  }, [rows]);

  const turnedDownRows = rows.filter((r) => r.turnedDown);
  const peakFails = rows.filter((r) => !r.peakOk);
  const strictestCeiling = rows.length
    ? Math.min(...rows.map((r) => r.ceiling))
    : -1;

  const worstTurnDown = turnedDownRows.reduce<Row | null>(
    (acc, r) => (acc === null || Math.abs(r.over) > Math.abs(acc.over) ? r : acc),
    null,
  );

  const headlineSev: Severity = peakFails.length
    ? 'critical'
    : turnedDownRows.length
      ? 'major'
      : 'clean';

  const headline = peakFails.length
    ? `True peak is over the ceiling on ${peakFails.length} platform${peakFails.length === 1 ? '' : 's'}.`
    : worstTurnDown
      ? `You will be turned down ${Math.abs(worstTurnDown.over).toFixed(1)} LU on ${worstTurnDown.platform}.`
      : rows.length
        ? 'Nothing here gets turned down. The master lands inside every target.'
        : 'No platform targets in this analysis.';

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 14 },
    whileInView: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.15 },
    transition: { duration: reduce ? 0.25 : 0.6, ease: EASE, delay: reduce ? 0 : delay },
  });

  const tpTone =
    truePeak > strictestCeiling ? SEVERITY_VAR.critical : truePeak > strictestCeiling - 0.3 ? SEVERITY_VAR.major : undefined;

  return (
    <section className="panel overflow-hidden p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <p className="eyebrow">Delivery</p>
          <h3 className="display mt-2 text-[clamp(1.15rem,2.2vw,1.6rem)] leading-none tracking-[-0.03em] text-ink">
            Loudness &amp; platform targets
          </h3>
        </div>
        <span className={`sev-chip ${SEV_CLASS[headlineSev]}`}>
          <span aria-hidden="true">{SEV_GLYPH[headlineSev]}</span>
          {peakFails.length
            ? `${peakFails.length} peak fail${peakFails.length === 1 ? '' : 's'}`
            : turnedDownRows.length
              ? `${turnedDownRows.length} turned down`
              : 'Delivery clean'}
        </span>
      </div>

      <motion.p
        {...rise(0.04)}
        className="display mt-4 max-w-3xl text-pretty text-[clamp(1rem,1.9vw,1.35rem)] leading-[1.25] tracking-[-0.02em] text-ink"
      >
        {headline}
      </motion.p>

      {rows.length ? (
        <motion.div {...rise(0.1)} className="mt-9">
          <LufsRail integrated={integrated} groups={groups} />
        </motion.div>
      ) : null}

      {/* -------------------------------------------------- platform rows */}
      {rows.length ? (
        <motion.div {...rise(0.16)} className="mt-10">
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="eyebrow">Per platform</p>
            <p className="eyebrow">
              true peak {truePeak > -99 ? `${truePeak.toFixed(2)} dBTP` : '—'}
            </p>
          </div>

          <ul className="divide-y divide-void-lineSoft">
            {rows.map((r, i) => {
              const under = r.over < -0.05;
              const onTarget = Math.abs(r.over) <= 0.5;
              const outcomeSev: Severity = r.turnedDown ? 'major' : onTarget ? 'clean' : under ? 'minor' : 'clean';
              const outcome = r.turnedDown
                ? `Turned down ${Math.abs(r.over).toFixed(1)} LU`
                : onTarget
                  ? 'On target'
                  : under
                    ? `${Math.abs(r.over).toFixed(1)} LU under · left alone`
                    : `${Math.abs(r.over).toFixed(1)} LU over · not normalised`;

              return (
                <li
                  key={r.key}
                  /* Fixed outer columns so every row's bar and chips line up
                     down the page — a jagged column of deltas is unreadable. */
                  className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-5 gap-y-3 py-4
                             lg:grid-cols-[minmax(140px,1.1fr)_78px_minmax(180px,1.7fr)_336px] lg:items-center"
                >
                  {/* Name */}
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        aria-hidden="true"
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: SEVERITY_VAR[r.severity] }}
                      />
                      <span className="display truncate text-[14px] leading-none tracking-[-0.02em] text-ink">
                        {r.platform}
                      </span>
                    </div>
                    {r.note ? (
                      <p className="mt-1.5 line-clamp-2 pl-[14px] text-[11.5px] leading-snug text-ink-muted">
                        {r.note}
                      </p>
                    ) : null}
                  </div>

                  {/* Target */}
                  <div className="text-right lg:text-right">
                    <p className="eyebrow lg:hidden">target</p>
                    <p className="stat mt-1 whitespace-nowrap text-[13px] text-ink-dim lg:mt-0">
                      {lufs(r.targetLufs)}
                      <span className="ml-1 text-[9px] text-ink-faint">LUFS</span>
                    </p>
                  </div>

                  {/* Delta */}
                  <div className="col-span-2 lg:col-span-1">
                    <DeltaBar over={r.over} turnedDown={r.turnedDown} index={i} />
                  </div>

                  {/* Outcome */}
                  <div className="col-span-2 flex flex-wrap items-center gap-2 lg:col-span-1 lg:justify-end">
                    <span className={`sev-chip ${SEV_CLASS[outcomeSev]} normal-case`}>
                      <span aria-hidden="true">{SEV_GLYPH[outcomeSev]}</span>
                      <span className="font-mono tracking-normal">{outcome}</span>
                    </span>
                    <span className={`sev-chip ${r.peakOk ? 'sev-clean' : 'sev-critical'}`}>
                      <span aria-hidden="true">{r.peakOk ? '✓' : '▲'}</span>
                      {r.peakOk ? 'TP ok' : 'TP over'} {r.ceiling.toFixed(1)}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </motion.div>
      ) : (
        <div className="mt-8 rounded-xl border border-void-line px-4 py-8 text-center">
          <p className="eyebrow">No platform targets were returned for this analysis</p>
        </div>
      )}

      {/* --------------------------------------------------- true peak line */}
      {rows.length ? (
        <motion.div
          {...rise(0.2)}
          className="mt-6 flex items-start gap-3 rounded-xl border border-void-line/70 px-4 py-3"
        >
          <span
            className={`mt-[2px] shrink-0 text-[11px] ${truePeak > strictestCeiling ? 'sev-critical' : 'sev-clean'}`}
            aria-hidden="true"
          >
            {truePeak > strictestCeiling ? '▲' : '✓'}
          </span>
          <p className="text-[12.5px] leading-snug text-ink-dim">
            True peak <span className="stat text-ink">{truePeak.toFixed(2)} dBTP</span> against the
            strictest ceiling here, <span className="stat text-ink">{strictestCeiling.toFixed(1)} dBTP</span>
            {truePeak > strictestCeiling ? (
              <>
                {' '}
                — over by{' '}
                <span className="stat text-sev-critical">
                  {(truePeak - strictestCeiling).toFixed(2)} dB
                </span>
                . Lossy transcodes will add more on top of that; pull the limiter ceiling down.
              </>
            ) : (
              <>
                {' '}
                — {Math.abs(truePeak - strictestCeiling).toFixed(2)} dB of headroom. Safe through a
                lossy transcode.
              </>
            )}
          </p>
        </motion.div>
      ) : null}

      {/* ------------------------------------------------------- the numbers */}
      <div className="hairline my-7" />

      <motion.div {...rise(0.24)}>
        <p className="eyebrow mb-4">The measurement</p>
        <div className="grid grid-cols-2 gap-x-5 gap-y-6 sm:grid-cols-3 lg:grid-cols-6">
          <Stat
            label="Integrated"
            value={lufs(integrated)}
            unit="LUFS"
            note="Whole-programme loudness"
          />
          <Stat
            label="Short-term max"
            value={lufs(finite(loudness?.short_term_max_lufs, integrated))}
            unit="LUFS"
            note="Loudest 3 s window"
          />
          <Stat
            label="LRA"
            value={finite(loudness?.loudness_range_lu).toFixed(1)}
            unit="LU"
            note={
              finite(loudness?.loudness_range_lu) < 3
                ? 'Very flat — over-compressed'
                : 'Macro dynamic range'
            }
            tone={finite(loudness?.loudness_range_lu) < 3 ? SEVERITY_VAR.major : undefined}
          />
          <Stat
            label="PLR"
            value={finite(loudness?.plr_db).toFixed(1)}
            unit="dB"
            note="Peak to loudness"
            tone={finite(loudness?.plr_db) < 6 ? SEVERITY_VAR.major : undefined}
          />
          <Stat
            label="PSR"
            value={finite(loudness?.psr_median_db).toFixed(1)}
            unit="dB"
            note={`Median · p10 ${finite(loudness?.psr_p10_db).toFixed(1)}`}
            tone={finite(loudness?.psr_median_db) < 5 ? SEVERITY_VAR.major : undefined}
          />
          <Stat
            label="True peak"
            value={truePeak.toFixed(2)}
            unit="dBTP"
            note={`Sample peak ${finite(loudness?.sample_peak_dbfs).toFixed(2)} dBFS`}
            tone={tpTone}
          />
        </div>
      </motion.div>
    </section>
  );
}
