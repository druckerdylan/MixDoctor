/**
 * StemBalance — the separated sources, measured one at a time.
 *
 * Every other panel in this report reads a two-track. That means "vocal
 * balance" is a center-channel proxy, "compression" is whatever the master bus
 * shows, and "the music is burying the vocal" is an inference from a congested
 * band. With stems those become readings: each source has its own loudness
 * against the full mix, its own crest, its own gain reduction, and the masking
 * between two sources is a measured difference rather than a guess.
 *
 * So the hierarchy here is deliberate. The level rail answers the single
 * question producers actually ask — *are my vocals too quiet* — and the
 * masking list underneath states the mix problem as a sentence, because
 * "Music is burying Vocals by 4.2 dB in 1–2 kHz, worst at 1:04" is the
 * clearest thing this product can say.
 *
 * One thing this file owns that the backend does not: the per-stem level
 * window. `targets.py` has genre windows for loudness, width, crest and
 * vocal-to-instrument, but nothing for stem-versus-mix level, so the zones
 * below are a frontend table and are labeled as "typical for <genre>" rather
 * than as a measured target.
 */

import { useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  SEVERITY_RANK,
  SEVERITY_VAR,
  STEM_KINDS,
  STEM_LABELS,
  formatDb,
  formatTime,
  type MaskingPair,
  type Severity,
  type StemAnalysis,
  type StemKind,
  type StemMeasurement,
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

/** The level rail, in dB relative to the full mix. */
const RAIL_MIN = -30;
const RAIL_MAX = 0;
const RAIL_TICKS = [-30, -24, -18, -12, -6, 0];

/** Full scale for the masking overlap bar, matching the analyzer's own cap. */
const OVERLAP_FULL_DB = 24;

/* ------------------------------------------------------------- level zones */

type Zone = [number, number];

/**
 * Where each source usually sits against the full mix, per genre family.
 * These are conventions, not physics: a rap record puts the vocal further
 * forward than a house record, and a folk record puts the drums further back
 * than either. Treated as guidance in the copy for exactly that reason.
 */
const ZONE_FAMILIES: Record<string, Record<StemKind, Zone>> = {
  vocal_forward: {
    vocals: [-10.0, -3.5],
    drums: [-9.0, -3.0],
    bass: [-12.0, -4.5],
    other: [-8.0, -1.5],
  },
  rhythm_forward: {
    vocals: [-12.0, -5.0],
    drums: [-7.5, -2.0],
    bass: [-9.5, -2.5],
    other: [-8.0, -1.5],
  },
  electronic: {
    vocals: [-16.0, -6.0],
    drums: [-8.0, -2.0],
    bass: [-8.5, -1.5],
    other: [-6.0, -0.5],
  },
  neutral: {
    vocals: [-13.0, -4.0],
    drums: [-9.0, -2.5],
    bass: [-11.0, -3.0],
    other: [-8.0, -1.5],
  },
};

const NEUTRAL_ZONES = ZONE_FAMILIES.neutral as Record<StemKind, Zone>;

/**
 * The analysis carries the *normalized* genre key ("hip_hop", "rnb", "lofi"),
 * not the label the user picked. Printing the key raw looks like a leaked
 * internal, so it gets a display form; the matching below runs on the key.
 */
const GENRE_LABELS: Record<string, string> = {
  hip_hop: 'Hip-Hop',
  rnb: 'R&B',
  edm: 'EDM',
  dnb: 'Drum & Bass',
  lofi: 'Lo-Fi',
  indie: 'Indie',
};

function prettyGenre(genre: string | undefined): string | null {
  const raw = (genre ?? '').trim();
  if (!raw) return null;
  const key = raw.toLowerCase();
  const mapped = GENRE_LABELS[key];
  if (mapped) return mapped;
  return raw
    .replace(/_/g, ' ')
    .replace(/\b[a-z]/g, (c) => c.toUpperCase());
}

function zonesFor(genre: string | undefined): Record<StemKind, Zone> {
  const g = (genre ?? '').toLowerCase();
  if (!g) return NEUTRAL_ZONES;
  if (/hip|trap|rap|rock|metal|punk/.test(g)) {
    return ZONE_FAMILIES.rhythm_forward ?? NEUTRAL_ZONES;
  }
  if (/edm|electro|house|techno|drum\s*&|dnb|d&b|ambient|lo-?fi|cinematic/.test(g)) {
    return ZONE_FAMILIES.electronic ?? NEUTRAL_ZONES;
  }
  if (/pop|r&b|rnb|soul|country|acoustic|folk|indie|alternat|jazz/.test(g)) {
    return ZONE_FAMILIES.vocal_forward ?? NEUTRAL_ZONES;
  }
  return NEUTRAL_ZONES;
}

/* ------------------------------------------------------------------ utils */

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function isNum(v: number | undefined | null): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function railPct(db: number): number {
  return clamp(((db - RAIL_MIN) / (RAIL_MAX - RAIL_MIN)) * 100, 0, 100);
}

/** "1–2 kHz", "250–500 Hz", "500 Hz–1 kHz" — never "1000.0–2000.0". */
function bandRange(lo: number, hi: number): string {
  const trim = (hz: number): string => {
    const k = hz / 1000;
    if (k >= 10) return k.toFixed(0);
    return Number.isInteger(k) ? k.toFixed(0) : k.toFixed(1);
  };
  if (lo >= 1000) return `${trim(lo)}–${trim(hi)} kHz`;
  if (hi >= 1000) return `${Math.round(lo)} Hz–${trim(hi)} kHz`;
  return `${Math.round(lo)}–${Math.round(hi)} Hz`;
}

/** Distance outside the window; negative below, positive above, 0 inside. */
function zoneMiss(value: number, [lo, hi]: Zone): number {
  if (value < lo) return value - lo;
  if (value > hi) return value - hi;
  return 0;
}

function severityFromMiss(miss: number): Severity {
  const m = Math.abs(miss);
  if (m <= 0.001) return 'clean';
  if (m <= 2.0) return 'minor';
  if (m <= 4.5) return 'major';
  return 'critical';
}

/**
 * Gain reduction on a single source. `dynamics.py` treats 3 dB on the master
 * bus as the point worth flagging; the same threshold applies here, with the
 * upper steps reserved for "this element has been squashed on its own".
 */
function severityFromGr(gr: number): Severity {
  if (gr >= 8.0) return 'critical';
  if (gr >= 5.0) return 'major';
  if (gr >= 3.0) return 'minor';
  return 'clean';
}

function worse(a: Severity, b: Severity): Severity {
  return SEVERITY_RANK[a] <= SEVERITY_RANK[b] ? a : b;
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
  unit?: string;
  note?: string;
  tone?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="eyebrow whitespace-nowrap">{label}</p>
      <p
        className="stat mt-2 whitespace-nowrap text-[clamp(1rem,1.9vw,1.25rem)] leading-none"
        style={tone ? { color: tone } : undefined}
      >
        {value}
        {unit ? <span className="ml-1 text-[10px] tracking-[0.1em] text-ink-faint">{unit}</span> : null}
      </p>
      {note ? <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">{note}</p> : null}
    </div>
  );
}

/** One compact mono readout inside a stem row. */
function Readout({
  label,
  value,
  unit,
  tone,
  flagged = false,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: string;
  flagged?: boolean;
}) {
  return (
    <span className="inline-flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">{label}</span>
      <span
        className={`stat text-[11.5px] ${flagged ? '' : 'text-ink-dim'}`}
        style={flagged && tone ? { color: tone } : undefined}
      >
        {value}
        {unit ? <span className="ml-0.5 text-[9px] text-ink-faint">{unit}</span> : null}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------- level rail */

function LevelRail({
  ratio,
  zone,
  severity,
  index,
  label,
}: {
  ratio: number;
  zone: Zone;
  severity: Severity;
  index: number;
  label: string;
}) {
  const reduce = useReducedMotion() ?? false;

  const pct = railPct(ratio);
  const zoneLo = railPct(zone[0]);
  const zoneHi = railPct(zone[1]);
  const flagged = severity !== 'clean';
  const tone = flagged ? SEVERITY_VAR[severity] : 'rgba(167,167,180,0.85)';
  const belowScale = ratio < RAIL_MIN;
  const aboveScale = ratio > RAIL_MAX;

  return (
    <div
      className="relative h-[30px] w-full min-w-[180px]"
      role="img"
      aria-label={`${label} measures ${formatDb(ratio)} dB against the full mix. Typical range for this genre is ${formatDb(
        zone[0],
      )} to ${formatDb(zone[1])} dB.`}
    >
      {/* Track */}
      <div className="absolute inset-x-0 top-[8px] h-[12px] overflow-hidden rounded-full border border-void-line bg-void-deep">
        {/* Genre zone */}
        <div
          className="absolute inset-y-0"
          style={{
            left: `${zoneLo}%`,
            width: `${Math.max(0.8, zoneHi - zoneLo)}%`,
            background: 'rgba(82,242,196,0.16)',
          }}
        />
        <div className="absolute inset-y-0 w-px bg-signal/40" style={{ left: `${zoneLo}%` }} />
        <div className="absolute inset-y-0 w-px bg-signal/40" style={{ left: `${zoneHi}%` }} />

        {/* Measured level, growing from the quiet end toward the mix. */}
        <motion.div
          className="absolute inset-y-[3px] left-0 rounded-full"
          style={{
            width: `${pct}%`,
            transformOrigin: 'left center',
            background: `linear-gradient(90deg, color-mix(in srgb, ${tone} 22%, transparent), ${tone})`,
            boxShadow: flagged ? `0 0 12px -3px ${tone}` : undefined,
          }}
          initial={{ scaleX: reduce ? 1 : 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: reduce ? 0 : 0.7, ease: EASE, delay: reduce ? 0 : 0.05 * index }}
        />
      </div>

      {/* The reading itself */}
      <div className="absolute top-[3px]" style={{ left: `${pct}%` }}>
        <div className="-translate-x-1/2">
          <div
            className="h-[22px] w-[3px] rounded-full"
            style={{ background: tone, boxShadow: `0 0 12px -1px ${tone}` }}
          />
        </div>
      </div>

      {belowScale || aboveScale ? (
        <span
          aria-hidden="true"
          className="stat absolute top-[7px] text-[10px]"
          style={{ [aboveScale ? 'right' : 'left']: 0, color: tone }}
        >
          {aboveScale ? '»' : '«'}
        </span>
      ) : null}

      {/* Scale */}
      <div className="pointer-events-none absolute inset-x-0 top-[22px]" aria-hidden="true">
        {RAIL_TICKS.map((t) => (
          <span
            key={t}
            className="stat absolute top-0 text-[9px] leading-none text-ink-faint"
            style={{ left: `${railPct(t)}%`, transform: 'translateX(-50%)' }}
          >
            {t === 0 ? 'mix' : t}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ types */

interface StemRow {
  kind: StemKind;
  label: string;
  ratio: number;
  zone: Zone;
  miss: number;
  severity: Severity;
  crest: number;
  gr: number;
  grSeverity: Severity;
  punch: number;
  active: number;
  micro: number;
  onsets: number;
}

interface MaskRow {
  key: string;
  masker: string;
  maskee: string;
  overlap: number;
  severity: Severity;
  range: string;
  detail: string;
  worst: string[];
}

/* ------------------------------------------------------------------ props */

export interface StemBalanceProps {
  stems: StemAnalysis;
  /**
   * Optional, and only used to pick the level window. Nothing here breaks
   * without it — the neutral zone table is the fallback.
   */
  genre?: string;
}

export default function StemBalance({ stems, genre }: StemBalanceProps) {
  const reduce = useReducedMotion() ?? false;

  const zones = useMemo(() => zonesFor(genre), [genre]);
  const genreLabel = useMemo(() => prettyGenre(genre), [genre]);

  const list = useMemo<StemMeasurement[]>(
    () => (Array.isArray(stems?.stems) ? stems.stems : []),
    [stems?.stems],
  );

  const rows = useMemo<StemRow[]>(() => {
    const byKind = new Map<StemKind, StemMeasurement>();
    for (const m of list) {
      if (m && m.present && STEM_KINDS.includes(m.kind)) byKind.set(m.kind, m);
    }
    const out: StemRow[] = [];
    for (const kind of STEM_KINDS) {
      const m = byKind.get(kind);
      if (!m) continue;
      const zone = zones[kind] ?? NEUTRAL_ZONES[kind];
      const ratio = finite(m.level_ratio_db, RAIL_MIN);
      const miss = zoneMiss(ratio, zone);
      const gr = clamp(finite(m.gain_reduction_estimate_db, 0), 0, 24);
      out.push({
        kind,
        label: STEM_LABELS[kind] ?? kind,
        ratio,
        zone,
        miss,
        severity: severityFromMiss(miss),
        crest: finite(m.crest_factor_db, 0),
        gr,
        grSeverity: severityFromGr(gr),
        punch: clamp(finite(m.transient_punch, 0), 0, 1),
        active: clamp(finite(m.active_ratio, 0), 0, 1),
        micro: finite(m.micro_dynamics_db, 0),
        onsets: Math.max(0, Math.round(finite(m.onset_count, 0))),
      });
    }
    return out;
  }, [list, zones]);

  const missing = useMemo<StemKind[]>(() => {
    const seen = new Set(rows.map((r) => r.kind));
    const known = new Set(list.map((m) => m?.kind));
    return STEM_KINDS.filter((k) => !seen.has(k) && known.has(k));
  }, [rows, list]);

  const masking = useMemo<MaskRow[]>(() => {
    const pairs: MaskingPair[] = Array.isArray(stems?.masking_pairs) ? stems.masking_pairs : [];
    return pairs
      .filter((p) => p && p.masker !== p.maskee)
      .map((p, i) => {
        const moments = Array.isArray(p.moments) ? p.moments : [];
        const worst = [...moments]
          .filter((m) => isNum(m?.t_start))
          .sort((a, b) => finite(b.value ?? b.intensity) - finite(a.value ?? a.intensity))
          .slice(0, 3)
          .sort((a, b) => finite(a.t_start) - finite(b.t_start))
          .map((m) => {
            const s = finite(m.t_start);
            const e = Math.max(s, finite(m.t_end, s));
            return e - s < 1 ? formatTime(s) : `${formatTime(s)}–${formatTime(e)}`;
          });
        return {
          key: `${p.masker}-${p.maskee}-${p.band}-${i}`,
          masker: STEM_LABELS[p.masker] ?? p.masker,
          maskee: STEM_LABELS[p.maskee] ?? p.maskee,
          overlap: finite(p.overlap_db, 0),
          severity: p.severity ?? 'minor',
          range: bandRange(finite(p.low_hz, 0), finite(p.high_hz, 0)),
          detail: typeof p.detail === 'string' ? p.detail : '',
          worst,
        };
      })
      .sort((a, b) => {
        const s = SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity];
        return s !== 0 ? s : b.overlap - a.overlap;
      });
  }, [stems?.masking_pairs]);

  /* ------------------------------------------------------------ headline */

  const squashed = useMemo(
    () => rows.filter((r) => r.gr >= 5.0).sort((a, b) => b.gr - a.gr),
    [rows],
  );
  const offZone = useMemo(
    () => rows.filter((r) => r.severity !== 'clean').sort((a, b) => Math.abs(b.miss) - Math.abs(a.miss)),
    [rows],
  );

  const worstMask = masking[0] ?? null;
  const topSquash = squashed[0] ?? null;
  const topOff = offZone[0] ?? null;

  const headlineSeverity: Severity = useMemo(() => {
    let s: Severity = 'clean';
    for (const r of rows) s = worse(s, worse(r.severity, r.grSeverity));
    for (const m of masking) s = worse(s, m.severity);
    return s;
  }, [rows, masking]);

  const headline = (() => {
    if (worstMask && (worstMask.severity === 'critical' || worstMask.severity === 'major')) {
      return `${worstMask.masker} is burying ${worstMask.maskee} by ${worstMask.overlap.toFixed(
        1,
      )} dB in ${worstMask.range}.`;
    }
    if (topOff && topOff.severity !== 'minor') {
      const where = genreLabel ? `where ${genreLabel} usually puts them` : 'the usual range';
      return topOff.miss < 0
        ? `${topOff.label} sit ${Math.abs(topOff.miss).toFixed(1)} dB under ${where}.`
        : `${topOff.label} sit ${topOff.miss.toFixed(1)} dB above ${where}.`;
    }
    if (topSquash) {
      return `${topSquash.label} are carrying about ${topSquash.gr.toFixed(
        1,
      )} dB of gain reduction on their own — that is a channel compressor, not the master bus.`;
    }
    if (worstMask) {
      return `${worstMask.masker} crowds ${worstMask.maskee} by ${worstMask.overlap.toFixed(1)} dB in ${
        worstMask.range
      }, but not enough to call it a problem.`;
    }
    if (rows.length) {
      return 'Every separated source lands inside the level range this genre usually sits in, and nothing is burying anything else.';
    }
    return 'Separation ran, but no source came back loud enough to measure.';
  })();

  const chipWord =
    headlineSeverity === 'clean'
      ? 'Balance clean'
      : masking.some((m) => m.severity === 'critical' || m.severity === 'major')
        ? `${masking.filter((m) => m.severity === 'critical' || m.severity === 'major').length} masking conflict${
            masking.filter((m) => m.severity === 'critical' || m.severity === 'major').length === 1 ? '' : 's'
          }`
        : squashed.length
          ? `${squashed.length} over-compressed`
          : `${offZone.length} off balance`;

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 14 },
    whileInView: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.15 },
    transition: { duration: reduce ? 0.25 : 0.6, ease: EASE, delay: reduce ? 0 : delay },
  });

  /* --------------------------------------------------------------- gates */

  if (!stems || !stems.available) return null;

  const model = typeof stems.model_name === 'string' && stems.model_name ? stems.model_name : 'separation';
  const sepSeconds = Math.max(0, finite(stems.separation_ms, 0)) / 1000;
  const warnings = Array.isArray(stems.warnings) ? stems.warnings.filter((w) => Boolean(w)) : [];

  const vocalToInstrument = isNum(stems.vocal_to_instrument_db) ? stems.vocal_to_instrument_db : null;
  const kickToBass = isNum(stems.kick_to_bass_db) ? stems.kick_to_bass_db : null;
  const kickHz = isNum(stems.kick_fundamental_hz) ? stems.kick_fundamental_hz : null;
  const bassHz = isNum(stems.bass_fundamental_hz) ? stems.bass_fundamental_hz : null;
  const hasDirect = vocalToInstrument !== null || kickToBass !== null || kickHz !== null || bassHz !== null;

  return (
    <section className="panel overflow-hidden p-5 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <p className="eyebrow">Separated sources</p>
          <h3 className="display mt-2 text-[clamp(1.15rem,2.2vw,1.6rem)] leading-none tracking-[-0.03em] text-ink">
            What each element is actually doing
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`sev-chip ${SEV_CLASS[headlineSeverity]}`}>
            <span aria-hidden="true">{SEV_GLYPH[headlineSeverity]}</span>
            {chipWord}
          </span>
          <span className="sev-chip text-ink-muted">
            {model}
            {sepSeconds >= 0.1 ? ` · ${sepSeconds.toFixed(1)}s` : ''}
          </span>
        </div>
      </div>

      <motion.p
        {...rise(0.04)}
        className="display mt-4 max-w-3xl text-pretty text-[clamp(1rem,1.9vw,1.35rem)] leading-[1.25] tracking-[-0.02em] text-ink"
      >
        {headline}
      </motion.p>

      <motion.p {...rise(0.06)} className="mt-3 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">
        These are direct readings, not center-channel estimates. Each source was pulled out of the
        bounce and measured on its own, which is the only way a 2-track can tell you the vocal is
        quiet rather than that the 1–2 kHz region is busy.
      </motion.p>

      {/* ------------------------------------------------------- the rails */}
      {rows.length ? (
        <motion.div {...rise(0.1)} className="mt-9">
          <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="eyebrow">Level against the full mix</p>
            <p className="eyebrow">shaded band = typical for {genreLabel ?? 'this material'}</p>
          </div>

          <ul className="divide-y divide-void-lineSoft">
            {rows.map((r, i) => {
              const grFlagged = r.gr >= 5.0;
              return (
                <li
                  key={r.kind}
                  className="relative grid gap-x-6 gap-y-3 py-4 lg:grid-cols-[128px_minmax(0,1fr)_88px] lg:items-center"
                  style={
                    grFlagged
                      ? {
                          background:
                            'linear-gradient(90deg, rgba(255,159,28,0.07), transparent 42%)',
                        }
                      : undefined
                  }
                >
                  {grFlagged ? (
                    <span
                      aria-hidden="true"
                      className="absolute inset-y-0 -left-3 w-[2px] rounded-full"
                      style={{ background: SEVERITY_VAR[r.grSeverity] }}
                    />
                  ) : null}

                  {/* Name */}
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        aria-hidden="true"
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: SEVERITY_VAR[worse(r.severity, r.grSeverity)] }}
                      />
                      <span className="display truncate text-[14px] leading-none tracking-[-0.02em] text-ink">
                        {r.label}
                      </span>
                    </div>
                    <p className="mt-1.5 pl-[14px] font-mono text-[9px] uppercase tracking-[0.12em] text-ink-faint">
                      {r.severity === 'clean'
                        ? 'in the usual range'
                        : r.miss < 0
                          ? `${Math.abs(r.miss).toFixed(1)} dB under range`
                          : `${r.miss.toFixed(1)} dB over range`}
                    </p>
                  </div>

                  {/* Rail + readouts */}
                  <div className="min-w-0">
                    <LevelRail
                      ratio={r.ratio}
                      zone={r.zone}
                      severity={r.severity}
                      index={i}
                      label={r.label}
                    />
                    <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1.5">
                      <Readout label="crest" value={r.crest.toFixed(1)} unit="dB" />
                      <Readout
                        label="gain red."
                        value={r.gr.toFixed(1)}
                        unit="dB"
                        flagged={r.grSeverity !== 'clean'}
                        tone={SEVERITY_VAR[r.grSeverity]}
                      />
                      <Readout label="punch" value={r.punch.toFixed(2)} />
                      <Readout label="active" value={`${Math.round(r.active * 100)}%`} />
                      {r.onsets > 0 ? <Readout label="hits" value={String(r.onsets)} /> : null}
                    </div>
                  </div>

                  {/* The number */}
                  <div className="lg:text-right">
                    <p className="eyebrow lg:hidden">vs mix</p>
                    <p
                      className="stat mt-1 whitespace-nowrap text-[17px] leading-none lg:mt-0"
                      style={{
                        color: r.severity === 'clean' ? undefined : SEVERITY_VAR[r.severity],
                      }}
                    >
                      {formatDb(r.ratio)}
                      <span className="ml-1 text-[10px] text-ink-faint">dB</span>
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>

          {/* Over-compression is the claim a 2-track literally cannot make. */}
          {squashed.length ? (
            <div className="mt-5 space-y-2">
              {squashed.map((r) => (
                <div
                  key={`gr-${r.kind}`}
                  className="flex items-start gap-3 rounded-xl border px-4 py-3"
                  style={{
                    borderColor: 'color-mix(in srgb, ' + SEVERITY_VAR[r.grSeverity] + ' 32%, transparent)',
                    background: 'color-mix(in srgb, ' + SEVERITY_VAR[r.grSeverity] + ' 7%, transparent)',
                  }}
                >
                  <span
                    className={`mt-[2px] shrink-0 text-[11px] ${SEV_CLASS[r.grSeverity]}`}
                    aria-hidden="true"
                  >
                    {SEV_GLYPH[r.grSeverity]}
                  </span>
                  <p className="text-[12.5px] leading-snug text-ink-dim">
                    <span className="text-ink">{r.label}</span> read{' '}
                    <span className="stat" style={{ color: SEVERITY_VAR[r.grSeverity] }}>
                      {r.gr.toFixed(1)} dB
                    </span>{' '}
                    of gain reduction on their own, with{' '}
                    <span className="stat text-ink-dim">{r.crest.toFixed(1)} dB</span> of crest left.
                    That is compression on this element, not on the master — back the channel
                    compressor off before you touch the bus.
                  </p>
                </div>
              ))}
            </div>
          ) : null}

          {missing.length ? (
            <p className="mt-5 text-[12px] leading-snug text-ink-muted">
              No{' '}
              {missing.map((k, i) => (
                <span key={k}>
                  {i > 0 ? (i === missing.length - 1 ? ' or ' : ', ') : ''}
                  <span className="text-ink-dim">{(STEM_LABELS[k] ?? k).toLowerCase()}</span>
                </span>
              ))}{' '}
              source came back above the noise floor, so {missing.length === 1 ? 'it is' : 'they are'}{' '}
              not reported. An instrumental has no vocal, and we will not print a level for an
              artefact.
            </p>
          ) : null}
        </motion.div>
      ) : (
        <div className="mt-8 rounded-xl border border-void-line px-4 py-8 text-center">
          <p className="eyebrow">Separation ran but returned no measurable source</p>
        </div>
      )}

      {/* -------------------------------------------------- direct readings */}
      {hasDirect ? (
        <>
          <div className="hairline my-7" />
          <motion.div {...rise(0.16)}>
            <p className="eyebrow mb-4">Measured across sources</p>
            <div className="grid grid-cols-2 gap-x-5 gap-y-6 sm:grid-cols-4">
              {vocalToInstrument !== null ? (
                <Stat
                  label="Vocal vs music"
                  value={formatDb(vocalToInstrument)}
                  unit="dB"
                  note={
                    vocalToInstrument < -4
                      ? 'Vocal is behind the music'
                      : vocalToInstrument > 4
                        ? 'Vocal is out in front'
                        : 'Sitting in the pocket'
                  }
                  tone={
                    Math.abs(vocalToInstrument) > 4 ? SEVERITY_VAR.major : undefined
                  }
                />
              ) : null}
              {kickToBass !== null ? (
                <Stat
                  label="Kick vs bass"
                  value={formatDb(kickToBass)}
                  unit="dB"
                  note="Two objects compared, not one split"
                  tone={Math.abs(kickToBass) < 1.5 ? SEVERITY_VAR.major : undefined}
                />
              ) : null}
              {kickHz !== null ? (
                <Stat
                  label="Kick fundamental"
                  value={kickHz.toFixed(0)}
                  unit="Hz"
                  note={
                    bassHz !== null && Math.abs(kickHz - bassHz) < 8
                      ? 'Same note as the bass — they will fight'
                      : 'Measured on the drum stem'
                  }
                  tone={
                    bassHz !== null && Math.abs(kickHz - bassHz) < 8 ? SEVERITY_VAR.major : undefined
                  }
                />
              ) : null}
              {bassHz !== null ? (
                <Stat
                  label="Bass fundamental"
                  value={bassHz.toFixed(0)}
                  unit="Hz"
                  note="Measured on the bass stem"
                />
              ) : null}
            </div>
          </motion.div>
        </>
      ) : null}

      {/* ------------------------------------------------------ the masking */}
      <div className="hairline my-7" />

      <motion.div {...rise(0.2)}>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <p className="eyebrow">What is burying what</p>
          <p className="eyebrow">
            {masking.length} measured pair{masking.length === 1 ? '' : 's'}
          </p>
        </div>

        {masking.length ? (
          <ul className="space-y-3">
            {masking.map((m, i) => {
              const tone = SEVERITY_VAR[m.severity];
              const frac = clamp(Math.abs(m.overlap) / OVERLAP_FULL_DB, 0, 1);
              return (
                <li
                  key={m.key}
                  className="relative overflow-hidden rounded-xl border px-4 py-3.5 sm:px-5"
                  style={{
                    borderColor: `color-mix(in srgb, ${tone} 30%, transparent)`,
                    background: `color-mix(in srgb, ${tone} 6%, transparent)`,
                  }}
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-y-0 left-0 w-[3px]"
                    style={{ background: tone }}
                  />

                  <div className="flex flex-wrap items-start justify-between gap-x-5 gap-y-2">
                    <p className="min-w-0 max-w-2xl text-pretty text-[14px] leading-snug text-ink">
                      <span className="display tracking-[-0.02em]">{m.masker}</span> is burying{' '}
                      <span className="display tracking-[-0.02em]">{m.maskee}</span> by{' '}
                      <span className="stat" style={{ color: tone }}>
                        {Math.abs(m.overlap).toFixed(1)} dB
                      </span>{' '}
                      in <span className="stat text-ink-dim">{m.range}</span>
                    </p>
                    <span className={`sev-chip shrink-0 ${SEV_CLASS[m.severity]}`}>
                      <span aria-hidden="true">{SEV_GLYPH[m.severity]}</span>
                      {m.severity}
                    </span>
                  </div>

                  {/* Magnitude, drawn. Full scale is the analyzer's own 24 dB cap. */}
                  <div className="relative mt-3 h-[6px] overflow-hidden rounded-full bg-void-deep">
                    <motion.div
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${frac * 100}%`,
                        transformOrigin: 'left center',
                        background: `linear-gradient(90deg, color-mix(in srgb, ${tone} 35%, transparent), ${tone})`,
                      }}
                      initial={{ scaleX: reduce ? 1 : 0 }}
                      whileInView={{ scaleX: 1 }}
                      viewport={{ once: true, amount: 0.4 }}
                      transition={{
                        duration: reduce ? 0 : 0.65,
                        ease: EASE,
                        delay: reduce ? 0 : 0.05 * i,
                      }}
                    />
                  </div>

                  {m.worst.length ? (
                    <p className="mt-3 font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
                      worst at{' '}
                      <span className="stat text-[10px] normal-case tracking-normal text-ink-dim">
                        {m.worst.join(' · ')}
                      </span>
                    </p>
                  ) : null}

                  {m.detail ? (
                    <p className="mt-2 text-[12px] leading-relaxed text-ink-muted">{m.detail}</p>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-void-line px-4 py-4">
            <span className="sev-chip sev-clean">
              <span aria-hidden="true">✓</span>
              Nothing masked
            </span>
            <p className="text-[13px] leading-snug text-ink-dim">
              No source sits far enough over another, in a band they both occupy, for long enough to
              count. The arrangement is making room for itself.
            </p>
          </div>
        )}
      </motion.div>

      {/* --------------------------------------------------------- caveats */}
      {warnings.length ? (
        <div className="mt-6 rounded-xl border border-void-line/70 px-4 py-3">
          <p className="eyebrow">Separation notes</p>
          <ul className="mt-2.5 space-y-1.5">
            {warnings.map((w, i) => (
              <li key={`${w}-${i}`} className="flex gap-2.5">
                <span className="mt-[3px] shrink-0 text-[9px] text-ink-faint" aria-hidden="true">
                  ●
                </span>
                <span className="text-[12px] leading-snug text-ink-muted">{w}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
