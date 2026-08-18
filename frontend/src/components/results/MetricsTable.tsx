/**
 * MetricsTable — show your work.
 *
 * Everything the analyzer actually measured, grouped the way an engineer would
 * look for it. No editorialising, no derived opinions: label, value, unit. Time
 * series are summarized (range and sample count) rather than dumped, because a
 * thousand raw floats is noise, not evidence.
 */

import { useCallback, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  formatHz,
  formatTime,
  type Measurements,
  type Moment,
  type VocalProminence,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

/**
 * Prominence in words. Deliberately non-judgmental: "tucked" describes where
 * the lead sits, not whether putting it there was a mistake.
 */
const PROMINENCE_LABEL: Record<VocalProminence, string> = {
  absent: 'No lead detected',
  tucked: 'Tucked under the bed',
  balanced: 'Sitting in the bed',
  forward: 'Out in front',
};

/* ------------------------------------------------------------------ rows */

type Row =
  | {
      kind: 'num';
      label: string;
      value: number;
      unit?: string;
      digits?: number;
      signed?: boolean;
      note?: string;
    }
  | { kind: 'text'; label: string; value: string; note?: string }
  | {
      kind: 'bool';
      label: string;
      value: boolean;
      yes: string;
      no: string;
      /** Which state, if either, deserves the alarm color. */
      alarmOn?: boolean;
    }
  | {
      kind: 'map';
      label: string;
      value: Record<string, number> | undefined | null;
      unit?: string;
      digits?: number;
      note?: string;
    }
  | { kind: 'series'; label: string; value: number[] | undefined | null; unit?: string }
  | { kind: 'moments'; label: string; value: Moment[] | undefined | null }
  | { kind: 'list'; label: string; value: string[] | undefined | null };

interface Section {
  id: string;
  title: string;
  blurb: string;
  rows: Row[];
}

/* ------------------------------------------------------------------ utils */

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function num(v: number, digits: number, signed: boolean): string {
  if (!Number.isFinite(v)) return '—';
  const body = Math.abs(v).toFixed(digits);
  if (v < 0) return `−${body}`;
  return signed ? `+${body}` : body;
}

function humanKey(k: string): string {
  return k.replace(/_/g, ' ');
}

function seriesSummary(s: number[] | undefined | null, unit: string | undefined): string {
  if (!s || !s.length) return 'not captured';
  let lo = Infinity;
  let hi = -Infinity;
  let sum = 0;
  let n = 0;
  for (const v of s) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    n += 1;
  }
  if (!n) return `${s.length} pts · no finite values`;
  const u = unit ? ` ${unit}` : '';
  return `${s.length} pts · ${lo.toFixed(1)} to ${hi.toFixed(1)}${u} · mean ${(sum / n).toFixed(1)}`;
}

function thirdOctaveSummary(
  db: number[] | undefined | null,
  centers: number[] | undefined | null,
): string {
  const n = db?.length ?? 0;
  if (!n) return '—';
  const c = centers ?? [];
  if (!c.length) return `${n} bands`;
  return `${n} bands · ${formatHz(finite(c[0], 20))}–${formatHz(finite(c[c.length - 1], 20000))} Hz`;
}

function momentsSummary(m: Moment[] | undefined | null): string {
  if (!m || !m.length) return 'none detected';
  const total = m.reduce((acc, x) => acc + Math.max(0, finite(x.t_end) - finite(x.t_start)), 0);
  const first = m[0];
  return `${m.length} · ${total.toFixed(1)} s total · first at ${formatTime(finite(first?.t_start))}`;
}

/* ------------------------------------------------------------- row render */

function ValueCell({ row }: { row: Row }) {
  switch (row.kind) {
    case 'num':
      return (
        <span className="stat text-[13px] text-ink">
          {num(row.value, row.digits ?? 1, row.signed ?? false)}
          {row.unit ? <span className="ml-1.5 text-[10px] text-ink-faint">{row.unit}</span> : null}
        </span>
      );
    case 'text':
      return <span className="stat text-[13px] text-ink">{row.value || '—'}</span>;
    case 'bool': {
      // Only bools with a stated alarm state get a severity color. The rest
      // are facts about the file, not verdicts, so they stay neutral.
      const alarmed = row.alarmOn !== undefined && row.value === row.alarmOn;
      const cls =
        row.alarmOn === undefined ? 'text-ink-muted' : alarmed ? 'sev-critical' : 'sev-clean';
      return (
        <span className={`sev-chip ${cls}`}>
          <span aria-hidden="true">
            {row.alarmOn === undefined ? '·' : alarmed ? '▲' : '✓'}
          </span>
          {row.value ? row.yes : row.no}
        </span>
      );
    }
    case 'map': {
      const entries = Object.entries(row.value ?? {}).filter(([, v]) => Number.isFinite(v));
      if (!entries.length) return <span className="stat text-[13px] text-ink-faint">—</span>;
      return (
        <span className="flex flex-wrap gap-x-3 gap-y-1.5">
          {entries.map(([k, v]) => (
            <span key={k} className="whitespace-nowrap">
              <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">
                {humanKey(k)}
              </span>{' '}
              <span className="stat text-[12px] text-ink-dim">
                {num(v, row.digits ?? 2, false)}
                {row.unit ? <span className="text-ink-faint">{row.unit}</span> : null}
              </span>
            </span>
          ))}
        </span>
      );
    }
    case 'series':
      return <span className="stat text-[12px] text-ink-dim">{seriesSummary(row.value, row.unit)}</span>;
    case 'moments':
      return <span className="stat text-[12px] text-ink-dim">{momentsSummary(row.value)}</span>;
    case 'list': {
      const list = (row.value ?? []).filter(Boolean);
      if (!list.length) return <span className="stat text-[13px] text-ink-faint">none</span>;
      return (
        <span className="flex flex-wrap gap-x-2 gap-y-1">
          {list.map((s, i) => (
            <span
              key={`${s}-${i}`}
              className="rounded border border-void-line px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim"
            >
              {humanKey(s)}
            </span>
          ))}
        </span>
      );
    }
    default:
      return null;
  }
}

function rowNote(row: Row): string | null {
  if (row.kind === 'num' || row.kind === 'text' || row.kind === 'map') return row.note ?? null;
  return null;
}

/* ----------------------------------------------------------- the sections */

function buildSections(m: Measurements): Section[] {
  const loudness = m.loudness;
  const dyn = m.dynamics;
  const spec = m.spectral;
  const stereo = m.stereo;
  const phase = m.phase;
  const low = m.low_end;
  const vocal = m.vocal;
  const trans = m.transients;
  const clarity = m.clarity;
  const clip = m.clipping;

  const sections: Section[] = [];

  if (loudness) {
    sections.push({
      id: 'loudness',
      title: 'Loudness',
      blurb: 'ITU-R BS.1770-4 gating, K-weighted.',
      rows: [
        { kind: 'num', label: 'Integrated', value: loudness.integrated_lufs, unit: 'LUFS' },
        { kind: 'num', label: 'Momentary max', value: loudness.momentary_max_lufs, unit: 'LUFS', note: '400 ms window' },
        { kind: 'num', label: 'Short-term max', value: loudness.short_term_max_lufs, unit: 'LUFS', note: '3 s window' },
        { kind: 'num', label: 'Loudness range', value: loudness.loudness_range_lu, unit: 'LU' },
        { kind: 'num', label: 'True peak', value: loudness.true_peak_dbtp, unit: 'dBTP', digits: 2, note: '4× oversampled' },
        { kind: 'num', label: 'Sample peak', value: loudness.sample_peak_dbfs, unit: 'dBFS', digits: 2 },
        { kind: 'num', label: 'PLR', value: loudness.plr_db, unit: 'dB', note: 'Peak to loudness ratio' },
        { kind: 'num', label: 'PSR — median', value: loudness.psr_median_db, unit: 'dB' },
        { kind: 'num', label: 'PSR — 10th pct', value: loudness.psr_p10_db, unit: 'dB', note: 'The most squashed passages' },
        { kind: 'series', label: 'Short-term series', value: loudness.short_term_series, unit: 'LUFS' },
        { kind: 'series', label: 'Momentary series', value: loudness.momentary_series, unit: 'LUFS' },
      ],
    });
  }

  if (dyn) {
    sections.push({
      id: 'dynamics',
      title: 'Dynamics',
      blurb: 'How much movement survived the compressor and the limiter.',
      rows: [
        { kind: 'num', label: 'Crest factor', value: dyn.crest_factor_db, unit: 'dB' },
        { kind: 'num', label: 'Peak to loudness', value: dyn.peak_to_loudness_db, unit: 'dB' },
        { kind: 'num', label: 'Macro dynamics', value: dyn.macro_dynamics_lu, unit: 'LU', note: 'Section to section' },
        { kind: 'num', label: 'Micro dynamics', value: dyn.micro_dynamics_db, unit: 'dB', note: 'Beat to beat' },
        { kind: 'num', label: 'DR value', value: dyn.dr_value, unit: '', note: 'Offline DR meter equivalent' },
        { kind: 'num', label: 'RMS', value: dyn.rms_db, unit: 'dBFS' },
        { kind: 'num', label: 'Pumping index', value: dyn.pumping_index, unit: '', digits: 2, note: '0 = steady, 1 = breathing hard' },
        { kind: 'num', label: 'Pumping rate', value: dyn.pumping_rate_hz, unit: 'Hz', digits: 2 },
        { kind: 'num', label: 'Gain reduction est.', value: dyn.gain_reduction_estimate_db, unit: 'dB', note: 'Inferred, not metered' },
        { kind: 'series', label: 'Crest series', value: dyn.crest_series, unit: 'dB' },
      ],
    });
  }

  if (spec) {
    sections.push({
      id: 'spectrum',
      title: 'Spectrum',
      blurb: '1/3-octave analysis against the genre target curve.',
      rows: [
        { kind: 'num', label: 'Spectral tilt', value: spec.spectral_tilt_db_per_decade, unit: 'dB/dec', signed: true },
        { kind: 'num', label: 'Spectral centroid', value: spec.spectral_centroid_hz, unit: 'Hz', digits: 0 },
        { kind: 'num', label: 'Mud ratio', value: spec.mud_ratio_db, unit: 'dB', signed: true, note: '150–400 Hz against low bass' },
        { kind: 'num', label: 'Mud to mid', value: spec.mud_to_mid_db, unit: 'dB', signed: true },
        { kind: 'num', label: 'Boxiness', value: spec.boxiness_db, unit: 'dB', signed: true, note: '300–600 Hz' },
        { kind: 'num', label: 'Harshness index', value: spec.harshness_index, unit: '', digits: 2, note: '2–5 kHz' },
        { kind: 'num', label: 'Sibilance index', value: spec.sibilance_index, unit: '', digits: 2, note: '5–9 kHz' },
        { kind: 'num', label: 'Sharpness', value: spec.sharpness_acum, unit: 'acum', digits: 2 },
        {
          kind: 'text',
          label: 'Resonances found',
          value: spec.resonances?.length
            ? `${spec.resonances.length} · worst ${formatHz(finite(spec.resonances[0]?.freq_hz))} Hz at ${finite(
                spec.resonances[0]?.prominence_db,
              ).toFixed(1)} dB`
            : 'none above threshold',
        },
        {
          kind: 'text',
          label: '1/3-octave bins',
          value: thirdOctaveSummary(spec.third_octave_db, spec.third_octave_centers),
          note: 'Plotted in full above',
        },
      ],
    });
  }

  if (stereo) {
    sections.push({
      id: 'stereo',
      title: 'Stereo',
      blurb: 'Imaging and how much of it survives a mono fold-down.',
      rows: [
        { kind: 'bool', label: 'Mono source', value: stereo.is_mono_source, yes: 'Single channel', no: 'Two channels' },
        { kind: 'num', label: 'Correlation', value: stereo.correlation, unit: '', digits: 3, signed: true },
        { kind: 'num', label: 'Width', value: stereo.width, unit: '', digits: 3 },
        { kind: 'num', label: 'Mono sum loss', value: stereo.mono_sum_loss_db, unit: 'dB' },
        { kind: 'num', label: 'Low-end side energy', value: stereo.low_end_side_energy_db, unit: 'dB', signed: true },
        { kind: 'num', label: 'L / R balance', value: stereo.balance_db, unit: 'dB', signed: true, note: 'Positive = right' },
        { kind: 'map', label: 'Width by band', value: stereo.band_width, digits: 2 },
        { kind: 'map', label: 'Correlation by band', value: stereo.band_correlation, digits: 2 },
        { kind: 'map', label: 'Mono loss by band', value: stereo.band_mono_loss_db, digits: 1, unit: ' dB' },
        { kind: 'series', label: 'Correlation series', value: stereo.correlation_series },
        { kind: 'series', label: 'Width series', value: stereo.width_series },
      ],
    });
  }

  if (phase) {
    sections.push({
      id: 'phase',
      title: 'Phase',
      blurb: 'Polarity and inter-channel coherence.',
      rows: [
        { kind: 'bool', label: 'Polarity inverted', value: phase.polarity_inverted, yes: 'Inverted', no: 'Correct', alarmOn: true },
        { kind: 'bool', label: 'Mono compatible', value: phase.mono_compatible, yes: 'Yes', no: 'No', alarmOn: false },
        { kind: 'num', label: 'Correlation', value: phase.correlation, unit: '', digits: 3, signed: true },
        { kind: 'num', label: 'Mono sum loss', value: phase.mono_sum_loss_db, unit: 'dB' },
        { kind: 'text', label: 'Worst band', value: phase.worst_band ? humanKey(phase.worst_band) : 'none' },
        { kind: 'num', label: 'Worst band correlation', value: phase.worst_band_correlation, unit: '', digits: 3, signed: true },
        { kind: 'map', label: 'Mono loss by band', value: phase.band_mono_loss_db, digits: 1, unit: ' dB' },
        { kind: 'moments', label: 'Problem moments', value: phase.problem_moments },
      ],
    });
  }

  if (low) {
    sections.push({
      id: 'low_end',
      title: 'Low End',
      blurb: 'Kick and bass: the two things that must not fight.',
      rows: [
        { kind: 'bool', label: 'Kick detected', value: low.kick_detected, yes: 'Yes', no: 'No' },
        { kind: 'num', label: 'Kick fundamental', value: low.kick_fundamental_hz, unit: 'Hz', digits: 0 },
        { kind: 'num', label: 'Kick count', value: low.kick_count, unit: 'hits', digits: 0 },
        { kind: 'num', label: 'Bass fundamental', value: low.bass_fundamental_hz, unit: 'Hz', digits: 0 },
        { kind: 'num', label: 'Sub energy', value: low.sub_energy_db, unit: 'dB', signed: true, note: '20–60 Hz' },
        { kind: 'num', label: 'Kick / bass collision', value: low.kick_bass_collision_db, unit: 'dB', note: 'Shared energy at the fundamental' },
        { kind: 'bool', label: 'Sidechain detected', value: low.has_sidechain, yes: 'Yes', no: 'No' },
        { kind: 'num', label: 'Ducking depth', value: low.ducking_depth_db, unit: 'dB' },
        { kind: 'num', label: 'Kick definition', value: low.kick_definition_db, unit: 'dB' },
        { kind: 'num', label: 'Low-end mono ratio', value: low.low_end_mono_ratio, unit: '', digits: 2, note: '1.0 = fully centered' },
        { kind: 'num', label: 'Sub rumble', value: low.sub_rumble_db, unit: 'dB', note: 'Below 25 Hz' },
        { kind: 'moments', label: 'Collision moments', value: low.collision_moments },
      ],
    });
  }

  if (vocal) {
    sections.push({
      id: 'vocal',
      title: 'Vocal',
      blurb:
        'Only meaningful when a lead vocal was found. A tucked lead is a production ' +
        'decision — the point of a beat someone is going to rap over — not a fault.',
      rows: [
        { kind: 'bool', label: 'Vocal present', value: vocal.vocal_present, yes: 'Detected', no: 'Not detected' },
        {
          kind: 'num',
          label: 'Voice-test confidence',
          value: vocal.vocal_confidence,
          unit: '',
          digits: 2,
          note: 'Center energy, syllabic modulation and consonant articulation combined',
        },
        {
          kind: 'text',
          label: 'Lead prominence',
          value: PROMINENCE_LABEL[vocal.vocal_prominence] ?? 'Not assessed',
          note: 'Where the lead sits against the instrument bed',
        },
        { kind: 'num', label: 'Center energy ratio', value: vocal.center_energy_ratio, unit: '', digits: 2 },
        { kind: 'num', label: 'Vocal to instruments', value: vocal.vocal_to_instrument_db, unit: 'dB', signed: true },
        { kind: 'num', label: 'Intelligibility', value: vocal.intelligibility_index, unit: '', digits: 2 },
        { kind: 'num', label: 'Presence balance', value: vocal.presence_balance_db, unit: 'dB', signed: true, note: '2–6 kHz' },
        { kind: 'num', label: 'Sibilance', value: vocal.sibilance_db, unit: 'dB', signed: true },
        { kind: 'num', label: 'Level consistency', value: vocal.consistency_db, unit: 'dB', note: 'Spread of vocal level' },
        { kind: 'list', label: 'Masked bands', value: vocal.masked_bands },
        { kind: 'moments', label: 'Buried moments', value: vocal.buried_moments },
        { kind: 'moments', label: 'Too loud moments', value: vocal.loud_moments },
      ],
    });
  }

  if (trans) {
    sections.push({
      id: 'transients',
      title: 'Transients',
      blurb: 'Punch, attack and how much the limiter rounded off.',
      rows: [
        { kind: 'num', label: 'Onset density', value: trans.onset_density, unit: '/s', digits: 2 },
        { kind: 'num', label: 'Estimated tempo', value: trans.estimated_tempo, unit: 'BPM', digits: 0 },
        { kind: 'num', label: 'Attack time', value: trans.attack_time_ms, unit: 'ms' },
        { kind: 'num', label: 'Punch index', value: trans.punch_index, unit: '', digits: 2 },
        { kind: 'num', label: 'Transient to sustain', value: trans.transient_to_sustain_db, unit: 'dB', signed: true },
        { kind: 'num', label: 'Smearing index', value: trans.smearing_index, unit: '', digits: 2, note: 'High = soft, blurred hits' },
        { kind: 'map', label: 'Punch by band', value: trans.band_punch, digits: 2 },
        { kind: 'moments', label: 'Weak moments', value: trans.weak_moments },
      ],
    });
  }

  if (clarity) {
    sections.push({
      id: 'clarity',
      title: 'Clarity',
      blurb: 'Masking and congestion — why a busy mix stops reading.',
      rows: [
        { kind: 'num', label: 'Clarity index', value: clarity.clarity_index, unit: '', digits: 2 },
        { kind: 'num', label: 'Spectral flatness', value: clarity.spectral_flatness, unit: '', digits: 3 },
        { kind: 'num', label: 'Spectral contrast', value: clarity.spectral_contrast, unit: 'dB' },
        { kind: 'num', label: 'Masking index', value: clarity.masking_index, unit: '', digits: 2 },
        { kind: 'num', label: 'Definition', value: clarity.definition_db, unit: 'dB', signed: true },
        {
          kind: 'text',
          label: 'Worst congested band',
          value: clarity.worst_congested_band ? humanKey(clarity.worst_congested_band) : 'none',
        },
        { kind: 'map', label: 'Congestion by band', value: clarity.band_congestion, digits: 2 },
        { kind: 'moments', label: 'Congested moments', value: clarity.congested_moments },
      ],
    });
  }

  if (clip) {
    sections.push({
      id: 'clipping',
      title: 'Clipping',
      blurb: 'Hard overs, flat-topped samples and inter-sample peaks.',
      rows: [
        { kind: 'num', label: 'Sample peak', value: clip.sample_peak_dbfs, unit: 'dBFS', digits: 2 },
        { kind: 'num', label: 'True peak', value: clip.true_peak_dbtp, unit: 'dBTP', digits: 2 },
        { kind: 'num', label: 'Clipped samples', value: clip.clipped_samples, unit: '', digits: 0 },
        { kind: 'num', label: 'Clipped', value: clip.clip_percentage, unit: '%', digits: 4 },
        { kind: 'num', label: 'Longest flat run', value: clip.longest_flat_run, unit: 'samples', digits: 0 },
        { kind: 'num', label: 'Flat runs', value: clip.flat_run_count, unit: '', digits: 0 },
        { kind: 'num', label: 'Inter-sample overs', value: clip.inter_sample_overs, unit: '', digits: 0 },
        { kind: 'bool', label: 'Float over unity', value: clip.is_float_over_unity, yes: 'Yes', no: 'No', alarmOn: true },
        { kind: 'num', label: 'Distortion index', value: clip.distortion_index, unit: '', digits: 3 },
        { kind: 'moments', label: 'Clip events', value: clip.events },
      ],
    });
  }

  return sections;
}

/* ------------------------------------------------------------------ props */

export interface MetricsTableProps {
  measurements: Measurements;
}

export default function MetricsTable({ measurements }: MetricsTableProps) {
  const reduce = useReducedMotion() ?? false;
  const sections = useMemo(() => (measurements ? buildSections(measurements) : []), [measurements]);

  const [open, setOpen] = useState<Record<string, boolean>>({ loudness: true });

  const toggle = useCallback((id: string) => {
    setOpen((prev) => ({ ...prev, [id]: !prev[id] }));
  }, []);

  const allOpen = sections.length > 0 && sections.every((s) => open[s.id]);

  const setAll = useCallback(
    (value: boolean) => {
      const next: Record<string, boolean> = {};
      for (const s of sections) next[s.id] = value;
      setOpen(next);
    },
    [sections],
  );

  const fileFacts = useMemo(() => {
    if (!measurements) return [];
    const facts: { label: string; value: string }[] = [
      { label: 'Duration', value: formatTime(finite(measurements.duration_seconds)) },
      { label: 'Sample rate', value: `${(finite(measurements.sample_rate) / 1000).toFixed(1)} kHz` },
      { label: 'Channels', value: measurements.is_mono ? 'Mono' : 'Stereo' },
    ];
    const orig = finite(measurements.original_sample_rate);
    if (orig && orig !== finite(measurements.sample_rate)) {
      facts.push({ label: 'Source rate', value: `${(orig / 1000).toFixed(1)} kHz` });
    }
    if (typeof measurements.bit_depth === 'number' && Number.isFinite(measurements.bit_depth)) {
      facts.push({ label: 'Bit depth', value: `${measurements.bit_depth}-bit` });
    }
    return facts;
  }, [measurements]);

  if (!sections.length) {
    return (
      <div className="panel flex min-h-[120px] items-center justify-center p-8">
        <p className="eyebrow">No measurements in this analysis</p>
      </div>
    );
  }

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-4 p-5 sm:p-7">
        <div className="min-w-0">
          <p className="eyebrow">Show your work</p>
          <h3 className="display mt-2 text-[clamp(1.15rem,2.2vw,1.6rem)] leading-none tracking-[-0.03em] text-ink">
            Every measurement
          </h3>
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
            {fileFacts.map((f) => (
              <span key={f.label} className="whitespace-nowrap">
                <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                  {f.label}
                </span>{' '}
                <span className="stat text-[12px] text-ink-dim">{f.value}</span>
              </span>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={() => setAll(!allOpen)}
          className="btn-ghost shrink-0 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.14em]"
        >
          {allOpen ? 'Collapse all' : 'Expand all'}
        </button>
      </div>

      <div className="hairline" />

      <ul>
        {sections.map((s) => {
          const isOpen = Boolean(open[s.id]);
          return (
            <li key={s.id} className="border-b border-void-lineSoft last:border-0">
              <h4>
                <button
                  type="button"
                  onClick={() => toggle(s.id)}
                  aria-expanded={isOpen}
                  aria-controls={`metrics-${s.id}`}
                  className="group flex w-full items-center gap-4 px-5 py-4 text-left transition-colors duration-300 ease-cine hover:bg-void-raised sm:px-7"
                >
                  <span
                    aria-hidden="true"
                    className="grid h-5 w-5 shrink-0 place-items-center rounded border border-void-line text-[10px] text-ink-muted transition-colors duration-300 group-hover:border-ink-faint group-hover:text-ink-dim"
                  >
                    {isOpen ? '−' : '+'}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="display block text-[14px] leading-none tracking-[-0.02em] text-ink">
                      {s.title}
                    </span>
                    <span className="mt-1.5 block truncate text-[11.5px] leading-snug text-ink-muted">
                      {s.blurb}
                    </span>
                  </span>
                  <span className="stat shrink-0 text-[11px] text-ink-faint">
                    {s.rows.length.toString().padStart(2, '0')}
                  </span>
                </button>
              </h4>

              {isOpen ? (
                <motion.div
                  id={`metrics-${s.id}`}
                  initial={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: reduce ? 0.15 : 0.4, ease: EASE }}
                  className="px-5 pb-6 sm:px-7"
                >
                  <dl className="grid grid-cols-1 gap-x-10 gap-y-0 md:grid-cols-2">
                    {s.rows.map((row) => {
                      const note = rowNote(row);
                      // Long-form values get the full width rather than being
                      // squeezed into half a column.
                      const wide =
                        row.kind === 'map' || row.kind === 'list' || row.kind === 'series';
                      return (
                        <div
                          key={`${s.id}-${row.label}`}
                          className={`flex flex-col gap-1 border-t border-void-lineSoft py-2.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-6 ${
                            wide ? 'md:col-span-2' : ''
                          }`}
                        >
                          <dt className="min-w-0 shrink-0">
                            <span className="text-[12.5px] leading-snug text-ink-dim">{row.label}</span>
                            {note ? (
                              <span className="ml-2 text-[11px] leading-snug text-ink-faint">{note}</span>
                            ) : null}
                          </dt>
                          <dd className="min-w-0 sm:text-right">
                            <ValueCell row={row} />
                          </dd>
                        </div>
                      );
                    })}
                  </dl>
                </motion.div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
