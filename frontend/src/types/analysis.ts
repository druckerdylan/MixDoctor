/**
 * Mirrors backend/analysis/types.py exactly. If a field changes there, change
 * it here in the same commit — this file is the wire contract, not a
 * convenience copy.
 */

export type Severity = 'critical' | 'major' | 'minor' | 'clean';
export type Verdict = 'good' | 'watch' | 'problem';

export type Dimension =
  | 'frequency_balance'
  | 'mud'
  | 'harshness'
  | 'stereo_width'
  | 'dynamic_range'
  | 'clipping'
  | 'phase'
  | 'low_end'
  | 'vocal_balance'
  | 'compression'
  | 'loudness'
  | 'limiter'
  | 'transients'
  | 'clarity';

/** Diagnostic order: what makes a mix sound broken before what makes it sound unfinished. */
export const DIMENSIONS: Dimension[] = [
  'clipping',
  'phase',
  'loudness',
  'limiter',
  'dynamic_range',
  'compression',
  'frequency_balance',
  'mud',
  'harshness',
  'low_end',
  'vocal_balance',
  'stereo_width',
  'transients',
  'clarity',
];

export const DIMENSION_LABELS: Record<Dimension, string> = {
  frequency_balance: 'Frequency Balance',
  mud: 'Mud & Low-Mid',
  harshness: 'Harshness',
  stereo_width: 'Stereo Width',
  dynamic_range: 'Dynamic Range',
  clipping: 'Clipping',
  phase: 'Phase',
  low_end: 'Kick / 808',
  vocal_balance: 'Vocal Balance',
  compression: 'Compression',
  loudness: 'Loudness',
  limiter: 'Limiter',
  transients: 'Transients',
  clarity: 'Clarity',
};

/** Short labels for the radar/mix-map, where space is tight. */
export const DIMENSION_SHORT: Record<Dimension, string> = {
  frequency_balance: 'Balance',
  mud: 'Mud',
  harshness: 'Harsh',
  stereo_width: 'Width',
  dynamic_range: 'Dynamics',
  clipping: 'Clip',
  phase: 'Phase',
  low_end: 'Low End',
  vocal_balance: 'Vocal',
  compression: 'Comp',
  loudness: 'Loudness',
  limiter: 'Limiter',
  transients: 'Punch',
  clarity: 'Clarity',
};

export const SEVERITY_RANK: Record<Severity, number> = {
  critical: 0,
  major: 1,
  minor: 2,
  clean: 3,
};

export interface Band {
  name: string;
  low_hz: number;
  high_hz: number;
  center_hz: number;
  level_db: number;
  target_db: number;
  deviation_db: number;
  verdict: Verdict;
}

export interface Moment {
  t_start: number;
  t_end: number;
  intensity: number;
  value?: number | null;
  label: string;
}

export interface Evidence {
  label: string;
  value: number;
  unit: string;
  target?: number | null;
  target_range?: [number, number] | null;
  verdict: Verdict;
  detail: string;
}

/**
 * Is this objectively wrong, or is it just different from the genre reference?
 *
 * A `defect` is wrong regardless of genre, intent or taste: clipping, an
 * inverted channel, a mix that vanishes in mono. A `deviation` is a measured
 * difference from what records in this genre usually do — it may well be
 * deliberate, it caps at `major`, and it must never be presented as damage.
 */
export type FindingKind = 'defect' | 'deviation';

/** What the uploaded file is, which decides which findings are asked for. */
export type TrackIntent =
  | 'full_mix'
  | 'beat'
  | 'instrumental'
  | 'stem'
  | 'reference'
  | 'demo';

export const TRACK_INTENT_LABELS: Record<TrackIntent, string> = {
  full_mix: 'Full mix',
  beat: 'Beat (instrumental for a topline)',
  instrumental: 'Instrumental',
  stem: 'Single stem',
  reference: 'Reference track',
  demo: 'Demo / rough',
};

/** Short forms for the results meta line, which is already dense. */
export const TRACK_INTENT_SHORT: Record<TrackIntent, string> = {
  full_mix: 'Full mix',
  beat: 'Beat',
  instrumental: 'Instrumental',
  stem: 'Stem',
  reference: 'Reference',
  demo: 'Demo',
};

export interface Finding {
  id: string;
  dimension: Dimension;
  title: string;
  kind: FindingKind;
  severity: Severity;
  confidence: number;
  detail: string;
  evidence: Evidence[];
  band_hz?: [number, number] | null;
  moments: Moment[];
  impact: number;
  /**
   * How far outside its window the measurement sits, in tolerance units.
   * 0 = inside, 1.5 = a defect's `major`, 3.0 = a defect's `critical`.
   *
   * This is the magnitude `severity` buckets away. A deviation's label caps at
   * `major` however far out it is, so two findings can share a label and not a
   * distance — sort or emphasise by this, not by `severity` alone.
   */
  miss_ratio: number;
}

export interface DimensionScore {
  dimension: Dimension;
  label: string;
  score: number;
  severity: Severity;
  headline: string;
  finding_ids: string[];
}

export interface LoudnessMeasurement {
  integrated_lufs: number;
  momentary_max_lufs: number;
  short_term_max_lufs: number;
  loudness_range_lu: number;
  true_peak_dbtp: number;
  sample_peak_dbfs: number;
  plr_db: number;
  psr_p10_db: number;
  psr_median_db: number;
  short_term_series: number[];
  momentary_series: number[];
  times: number[];
}

export interface ClippingMeasurement {
  sample_peak_dbfs: number;
  true_peak_dbtp: number;
  clipped_samples: number;
  clip_percentage: number;
  longest_flat_run: number;
  flat_run_count: number;
  inter_sample_overs: number;
  is_float_over_unity: boolean;
  distortion_index: number;
  events: Moment[];
}

export interface Resonance {
  freq_hz: number;
  q: number;
  prominence_db: number;
  severity: Severity;
  moments: Moment[];
}

export interface SpectralMeasurement {
  third_octave_centers: number[];
  third_octave_db: number[];
  bands: Band[];
  spectral_tilt_db_per_decade: number;
  spectral_centroid_hz: number;
  mud_ratio_db: number;
  mud_to_mid_db: number;
  boxiness_db: number;
  harshness_index: number;
  sibilance_index: number;
  sharpness_acum: number;
  resonances: Resonance[];
  band_series: Record<string, number[]>;
  times: number[];
}

export interface StereoMeasurement {
  is_mono_source: boolean;
  correlation: number;
  width: number;
  band_correlation: Record<string, number>;
  band_width: Record<string, number>;
  mono_sum_loss_db: number;
  band_mono_loss_db: Record<string, number>;
  low_end_side_energy_db: number;
  balance_db: number;
  correlation_series: number[];
  width_series: number[];
  times: number[];
}

export interface PhaseMeasurement {
  correlation: number;
  polarity_inverted: boolean;
  mono_compatible: boolean;
  mono_sum_loss_db: number;
  worst_band?: string | null;
  worst_band_correlation: number;
  band_mono_loss_db: Record<string, number>;
  problem_moments: Moment[];
}

export interface DynamicsMeasurement {
  crest_factor_db: number;
  peak_to_loudness_db: number;
  macro_dynamics_lu: number;
  micro_dynamics_db: number;
  dr_value: number;
  rms_db: number;
  pumping_index: number;
  pumping_rate_hz: number;
  gain_reduction_estimate_db: number;
  crest_series: number[];
  times: number[];
}

export interface TransientMeasurement {
  onset_density: number;
  estimated_tempo: number;
  attack_time_ms: number;
  punch_index: number;
  transient_to_sustain_db: number;
  band_punch: Record<string, number>;
  smearing_index: number;
  weak_moments: Moment[];
}

export interface LowEndMeasurement {
  kick_detected: boolean;
  kick_fundamental_hz: number;
  kick_count: number;
  bass_fundamental_hz: number;
  sub_energy_db: number;
  kick_bass_collision_db: number;
  ducking_depth_db: number;
  has_sidechain: boolean;
  kick_definition_db: number;
  low_end_mono_ratio: number;
  sub_rumble_db: number;
  collision_moments: Moment[];
}

/**
 * Where the lead sits against the instrument bed. A tucked lead is a production
 * decision — the whole point of a beat someone is going to rap over — not a
 * balance fault, so the UI must never present it as one.
 */
export type VocalProminence = 'absent' | 'tucked' | 'balanced' | 'forward';

export interface VocalMeasurement {
  vocal_present: boolean;
  /** Graded 0-1 confidence that a real lead voice is there. `vocal_present` is
   *  this crossing a threshold, so prefer the float wherever nuance matters. */
  vocal_confidence: number;
  vocal_prominence: VocalProminence;
  center_energy_ratio: number;
  vocal_to_instrument_db: number;
  intelligibility_index: number;
  presence_balance_db: number;
  sibilance_db: number;
  consistency_db: number;
  masked_bands: string[];
  buried_moments: Moment[];
  loud_moments: Moment[];
}

export interface ClarityMeasurement {
  clarity_index: number;
  spectral_flatness: number;
  spectral_contrast: number;
  masking_index: number;
  band_congestion: Record<string, number>;
  worst_congested_band?: string | null;
  definition_db: number;
  congested_moments: Moment[];
}

export type StemKind = 'vocals' | 'drums' | 'bass' | 'other';

export const STEM_KINDS: StemKind[] = ['vocals', 'drums', 'bass', 'other'];

export const STEM_LABELS: Record<StemKind, string> = {
  vocals: 'Vocals',
  drums: 'Drums',
  bass: 'Bass',
  other: 'Music',
};

export interface StemMeasurement {
  kind: StemKind;
  present: boolean;
  level_lufs: number;
  level_ratio_db: number;
  peak_dbfs: number;
  crest_factor_db: number;
  micro_dynamics_db: number;
  gain_reduction_estimate_db: number;
  spectral_centroid_hz: number;
  band_levels_db: Record<string, number>;
  band_occupancy: Record<string, number>;
  stereo_width: number;
  correlation: number;
  transient_punch: number;
  onset_count: number;
  active_ratio: number;
  level_series: number[];
}

export interface MaskingPair {
  masker: StemKind;
  maskee: StemKind;
  band: string;
  low_hz: number;
  high_hz: number;
  overlap_db: number;
  severity: Severity;
  detail: string;
  moments: Moment[];
}

export interface StemAnalysis {
  available: boolean;
  model_name: string;
  separation_ms: number;
  stems: StemMeasurement[];
  masking_pairs: MaskingPair[];
  vocal_to_instrument_db?: number | null;
  kick_to_bass_db?: number | null;
  kick_fundamental_hz?: number | null;
  bass_fundamental_hz?: number | null;
  warnings: string[];
}

export interface Section {
  index: number;
  label: string;
  t_start: number;
  t_end: number;
  integrated_lufs: number;
  crest_factor_db: number;
  stereo_width: number;
  band_levels_db: Record<string, number>;
  is_loudest: boolean;
  is_quietest: boolean;
}

export interface SectionAnalysis {
  available: boolean;
  sections: Section[];
  loudness_spread_lu: number;
  peak_lift_db: number;
  low_end_swing_db: number;
  notes: string[];
}

/** A plugin the producer owns. `capabilities` is what actually shapes advice. */
export interface OwnedPlugin {
  name: string;
  manufacturer?: string | null;
  category: string;
  capabilities: string[];
}

export interface Measurements {
  duration_seconds: number;
  sample_rate: number;
  original_sample_rate: number;
  is_mono: boolean;
  bit_depth?: number | null;
  loudness: LoudnessMeasurement;
  clipping: ClippingMeasurement;
  spectral: SpectralMeasurement;
  stereo: StereoMeasurement;
  phase: PhaseMeasurement;
  dynamics: DynamicsMeasurement;
  transients: TransientMeasurement;
  low_end: LowEndMeasurement;
  vocal: VocalMeasurement;
  clarity: ClarityMeasurement;
  /** Both always present; check `.available` rather than null-guarding. */
  stems: StemAnalysis;
  sections: SectionAnalysis;
}

export interface PlatformTarget {
  platform: string;
  target_lufs: number;
  max_true_peak_dbtp: number;
  delta_lufs: number;
  will_be_turned_down: boolean;
  peak_ok: boolean;
  verdict: Verdict;
  note: string;
}

export interface ReferenceDelta {
  integrated_lufs: number;
  dynamic_range_db: number;
  stereo_width: number;
  true_peak_dbtp: number;
  band_deltas: Record<string, number>;
  third_octave_delta_db: number[];
  similarity: number;
  biggest_gaps: string[];
}

export interface Move {
  order: number;
  target: string;
  action: string;
  tool: string;
  settings: Record<string, string>;
  why: string;
}

export interface Prescription {
  finding_id: string;
  dimension: Dimension;
  headline: string;
  diagnosis: string;
  root_cause: string;
  moves: Move[];
  alternative: string;
  do_not: string;
  done_when: string;
  expected_gain: number;
  minutes: number;
  difficulty: 'easy' | 'moderate' | 'advanced';
}

export interface EngineerReport {
  verdict: string;
  the_one_thing: string;
  strengths: string[];
  prescriptions: Prescription[];
  session_plan: string[];
  mastering_verdict: string;
  reference_notes: string[];
}

export interface MixAnalysis {
  schema_version: number;
  filename: string;
  genre: string;
  intent: TrackIntent;
  health_score: number;
  grade: string;
  ceiling_score: number;
  mastering_ready: boolean;
  mastering_blockers: string[];
  dimensions: DimensionScore[];
  findings: Finding[];
  measurements: Measurements;
  platform_targets: PlatformTarget[];
  reference?: ReferenceDelta | null;
  engineer?: EngineerReport | null;
  waveform_peaks: number[];
  waveform_rms: number[];
  analysis_ms: number;
  warnings: string[];
}

export type AnalysisStatus =
  | 'idle'
  | 'uploading'
  | 'measuring'
  | 'consulting'
  | 'complete'
  | 'error';

/** Severity -> CSS custom property, so charts and chrome stay in sync. */
export const SEVERITY_VAR: Record<Severity, string> = {
  critical: 'var(--sev-critical)',
  major: 'var(--sev-major)',
  minor: 'var(--sev-minor)',
  clean: 'var(--sev-clean)',
};

export function severityFromScore(score: number): Severity {
  if (score < 45) return 'critical';
  if (score < 68) return 'major';
  if (score < 85) return 'minor';
  return 'clean';
}

export function formatTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rem = Math.floor(s % 60);
  return `${m}:${rem.toString().padStart(2, '0')}`;
}

/** Signed dB, always with an explicit sign — engineers read the sign first. */
export function formatDb(value: number, digits = 1): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export function formatHz(hz: number): string {
  if (hz >= 1000) {
    const k = hz / 1000;
    return `${k >= 10 ? k.toFixed(0) : k.toFixed(1)}k`;
  }
  return `${Math.round(hz)}`;
}
