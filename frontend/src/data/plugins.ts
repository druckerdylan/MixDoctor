/**
 * The plugin catalog, tagged by capability.
 *
 * Mirrors backend/analysis/capabilities.py. The point is not the brand name —
 * it is what the box can *do*, because that is what changes the shape of a
 * prescription. "Cut the 318 Hz resonance" is three different instructions
 * depending on whether the producer owns a resonance suppressor, a dynamic EQ,
 * or only a static one.
 *
 * If a slug changes in capabilities.py it must change here in the same commit:
 * the backend drops unknown slugs silently (`normalise()`), so a stale tag does
 * not error — it just quietly stops shaping advice.
 */

import type { Dimension } from '../types/analysis';

/* ------------------------------------------------------------------ */
/* Capability vocabulary — the 34 slugs, verbatim from the backend.     */
/* ------------------------------------------------------------------ */

export const CAPABILITY_LABELS: Record<string, string> = {
  // EQ
  eq_static: 'Static EQ',
  eq_dynamic: 'Dynamic EQ',
  eq_linear_phase: 'Linear-phase EQ',
  eq_mid_side: 'Mid/Side EQ',
  eq_match: 'Match EQ',
  // Dynamics
  comp_single: 'Compressor',
  comp_multiband: 'Multiband compression',
  comp_sidechain_ext: 'External sidechain',
  comp_parallel: 'Parallel / blend',
  comp_bus: 'Bus compressor',
  comp_opto: 'Opto compressor',
  comp_fet: 'FET compressor',
  comp_vca: 'VCA compressor',
  expander: 'Expander / gate',
  // Loudness
  limiter: 'Limiter',
  limiter_truepeak: 'True-peak limiter',
  clipper: 'Clipper',
  // Spectral repair
  deesser: 'De-esser',
  resonance_suppressor: 'Resonance suppressor',
  spectral_repair: 'Spectral repair',
  // Color
  saturation: 'Saturation',
  tape: 'Tape',
  exciter: 'Exciter',
  // Space
  imager: 'Stereo imager',
  mono_maker: 'Mono maker',
  reverb: 'Reverb',
  delay: 'Delay',
  transient_shaper: 'Transient shaper',
  // Measurement
  meter_loudness: 'Loudness meter',
  meter_spectrum: 'Spectrum analyzer',
  meter_correlation: 'Correlation meter',
  reference_matching: 'Reference tool',
  // Restoration / pitch
  pitch_correction: 'Pitch correction',
  noise_reduction: 'Noise reduction',
};

/** What each capability unlocks, written for a human rather than the model. */
export const CAPABILITY_BLURBS: Record<string, string> = {
  eq_static: 'Fixed bells, shelves and filters.',
  eq_dynamic: 'A band that only acts when the region crosses a threshold.',
  eq_linear_phase: 'No phase smear across the crossover. Pre-rings on transients.',
  eq_mid_side: 'Treat the center and the sides independently.',
  eq_match: 'Fit the spectrum to a captured reference curve.',
  comp_single: 'Standard downward compression.',
  comp_multiband: 'Compress one region without the rest ducking with it.',
  comp_sidechain_ext: "Duck one source from another's signal.",
  comp_parallel: 'Built-in dry/wet — density without losing the transient.',
  comp_bus: 'Glue-style bus compression, slow and program-dependent.',
  comp_opto: 'Slow, level-dependent release. Forgiving on vocals.',
  comp_fet: 'Very fast attack. Aggressive on drums and rock vocals.',
  comp_vca: 'Precise and punchy, tight control of attack and release.',
  expander: 'Push down what sits below a threshold.',
  limiter: 'A hard ceiling.',
  limiter_truepeak: 'Oversampled ceiling detection that survives a lossy encoder.',
  clipper: 'Shave peaks before the limiter so the limiter works less.',
  deesser: 'Frequency-selective ducking aimed at sibilance.',
  resonance_suppressor: 'Tracks and ducks narrow resonances dynamically.',
  spectral_repair: 'Remove a sound from the spectrogram directly.',
  saturation: 'Harmonic generation for density and perceived loudness.',
  tape: 'Compression, HF softening and wow/flutter as one character.',
  exciter: 'Synthesise harmonics above what is already there.',
  imager: 'Widen or narrow, usually per band.',
  mono_maker: 'Force everything below a frequency to mono.',
  reverb: 'Space and depth.',
  delay: 'Echo, and width via short offsets.',
  transient_shaper: 'Attack and sustain independent of level.',
  meter_loudness: 'LUFS, LRA and true-peak readout.',
  meter_spectrum: 'See the balance while working.',
  meter_correlation: 'Watch phase and mono compatibility.',
  reference_matching: 'A/B against a commercial record, level-matched.',
  pitch_correction: 'Tuning.',
  noise_reduction: 'Broadband noise, hum and clicks.',
};

/** Grouping for the summary panel — same order as the backend file's sections. */
export const CAPABILITY_GROUPS: { label: string; capabilities: string[] }[] = [
  { label: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_linear_phase', 'eq_mid_side', 'eq_match'] },
  {
    label: 'Dynamics',
    capabilities: [
      'comp_single',
      'comp_multiband',
      'comp_sidechain_ext',
      'comp_parallel',
      'comp_bus',
      'comp_opto',
      'comp_fet',
      'comp_vca',
      'expander',
    ],
  },
  { label: 'Loudness', capabilities: ['limiter', 'limiter_truepeak', 'clipper'] },
  { label: 'Repair', capabilities: ['deesser', 'resonance_suppressor', 'spectral_repair', 'noise_reduction'] },
  { label: 'Color', capabilities: ['saturation', 'tape', 'exciter'] },
  { label: 'Space', capabilities: ['imager', 'mono_maker', 'reverb', 'delay', 'transient_shaper'] },
  {
    label: 'Measurement',
    capabilities: ['meter_loudness', 'meter_spectrum', 'meter_correlation', 'reference_matching'],
  },
  { label: 'Pitch', capabilities: ['pitch_correction'] },
];

/** Flat vocabulary, in display order. */
export const ALL_CAPABILITIES: string[] = CAPABILITY_GROUPS.flatMap((g) => g.capabilities);

/**
 * Everyone has these, whatever DAW they use. Advice may always assume them, so
 * a producer who has configured nothing still gets an executable plan.
 * Mirrors STOCK_CAPABILITIES in capabilities.py.
 */
export const STOCK_CAPABILITIES: string[] = [
  'eq_static',
  'comp_single',
  'limiter',
  'expander',
  'reverb',
  'delay',
  'saturation',
  'meter_spectrum',
];

/**
 * Which capability actually solves which finding. Mirrors
 * CAPABILITY_FOR_DIMENSION in capabilities.py — ordered best-first, so the
 * head of each list is the tool you would reach for and the tail is the
 * fallback you would settle for.
 */
export const CAPABILITY_FOR_DIMENSION: Record<Dimension, string[]> = {
  clipping: ['limiter_truepeak', 'clipper', 'limiter'],
  limiter: ['limiter_truepeak', 'clipper', 'comp_multiband'],
  loudness: ['limiter_truepeak', 'clipper', 'comp_bus'],
  dynamic_range: ['comp_bus', 'comp_parallel', 'transient_shaper', 'clipper'],
  compression: ['comp_parallel', 'transient_shaper', 'comp_multiband', 'comp_opto'],
  frequency_balance: ['eq_static', 'eq_match', 'eq_linear_phase'],
  mud: ['eq_dynamic', 'resonance_suppressor', 'comp_multiband', 'eq_static'],
  harshness: ['resonance_suppressor', 'eq_dynamic', 'deesser', 'eq_static'],
  phase: ['mono_maker', 'eq_mid_side', 'meter_correlation'],
  stereo_width: ['imager', 'eq_mid_side', 'mono_maker', 'delay'],
  low_end: ['comp_sidechain_ext', 'eq_dynamic', 'mono_maker', 'comp_multiband'],
  vocal_balance: ['comp_opto', 'eq_dynamic', 'deesser', 'comp_parallel'],
  transients: ['transient_shaper', 'clipper', 'comp_parallel'],
  clarity: ['resonance_suppressor', 'eq_dynamic', 'comp_multiband', 'eq_mid_side'],
};

/** Drop unknown slugs and de-duplicate, preserving order. Mirrors `normalise()`. */
export function normaliseCapabilities(caps: readonly string[] | null | undefined): string[] {
  if (!Array.isArray(caps)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of caps) {
    if (typeof raw !== 'string') continue;
    const slug = raw.trim().toLowerCase();
    if (slug in CAPABILITY_LABELS && !seen.has(slug)) {
      seen.add(slug);
      out.push(slug);
    }
  }
  return out;
}

/** Sort a capability set into the canonical display order. */
export function orderCapabilities(caps: readonly string[]): string[] {
  const owned = new Set(normaliseCapabilities(caps));
  return ALL_CAPABILITIES.filter((c) => owned.has(c));
}

/* ------------------------------------------------------------------ */
/* Catalog                                                           */
/* ------------------------------------------------------------------ */

export interface CatalogPlugin {
  name: string;
  manufacturer: string;
  category: string;
  capabilities: string[];
}

type Entry = { name: string; category: string; capabilities: string[] };

/**
 * Kept keyed by manufacturer because that is how a producer thinks about what
 * they own ("I have the FabFilter bundle"), and because the browse column is
 * grouped the same way. `PLUGIN_CATALOG` is the flattened view.
 */
const CATALOG_BY_MANUFACTURER: Record<string, Entry[]> = {
  FabFilter: [
    { name: 'Pro-Q 4', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_linear_phase', 'eq_mid_side'] },
    { name: 'Pro-Q 3', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_linear_phase', 'eq_mid_side'] },
    { name: 'Pro-C 2', category: 'Compressor', capabilities: ['comp_single', 'comp_sidechain_ext', 'comp_parallel'] },
    { name: 'Pro-L 2', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'Pro-MB', category: 'Multiband', capabilities: ['comp_multiband', 'comp_sidechain_ext', 'comp_parallel', 'expander'] },
    { name: 'Pro-DS', category: 'De-esser', capabilities: ['deesser'] },
    { name: 'Pro-G', category: 'Gate', capabilities: ['expander', 'comp_sidechain_ext'] },
    { name: 'Pro-R 2', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Pro-R', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Saturn 2', category: 'Saturator', capabilities: ['saturation', 'exciter'] },
    { name: 'Timeless 3', category: 'Delay', capabilities: ['delay'] },
    { name: 'Volcano 3', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Simplon', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Twin 3', category: 'Instrument', capabilities: [] },
  ],

  Waves: [
    { name: 'SSL E-Channel', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'comp_vca', 'expander'] },
    { name: 'SSL G-Master Buss Compressor', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'CLA-2A', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'CLA-3A', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'CLA-76', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'API 2500', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'API 550A', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'API 550B', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'API 560', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'H-Delay', category: 'Delay', capabilities: ['delay'] },
    { name: 'H-Reverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'R-Verb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Renaissance Compressor', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'Renaissance EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Renaissance Bass', category: 'Bass Enhancer', capabilities: ['exciter'] },
    { name: 'L1 Ultramaximizer', category: 'Limiter', capabilities: ['limiter'] },
    { name: 'L2 Ultramaximizer', category: 'Limiter', capabilities: ['limiter'] },
    { name: 'L3 Multimaximizer', category: 'Limiter', capabilities: ['limiter', 'comp_multiband'] },
    { name: 'Linear Phase EQ', category: 'EQ', capabilities: ['eq_static', 'eq_linear_phase'] },
    { name: 'Linear Phase Multiband Compressor', category: 'Multiband', capabilities: ['comp_multiband', 'eq_linear_phase'] },
    { name: 'C4 Multiband Compressor', category: 'Multiband', capabilities: ['comp_multiband', 'expander'] },
    { name: 'C6 Multiband Compressor', category: 'Multiband', capabilities: ['comp_multiband', 'eq_dynamic'] },
    { name: 'F6 Floating-Band Dynamic EQ', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_mid_side', 'comp_sidechain_ext'] },
    { name: 'Smack Attack', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'J37 Tape', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'Kramer Tape', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'Abbey Road Vinyl', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Vocal Rider', category: 'Utility', capabilities: ['comp_single'] },
    { name: 'Bass Rider', category: 'Utility', capabilities: ['comp_single'] },
    { name: 'Sibilance', category: 'De-esser', capabilities: ['deesser'] },
    { name: 'DeEsser', category: 'De-esser', capabilities: ['deesser'] },
    { name: 'S1 Stereo Imager', category: 'Stereo Imager', capabilities: ['imager', 'eq_mid_side'] },
    { name: 'Center', category: 'Stereo Imager', capabilities: ['imager', 'eq_mid_side'] },
    { name: 'Brauer Motion', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'Doubler', category: 'Chorus', capabilities: ['imager', 'delay'] },
    { name: 'MetaFlanger', category: 'Modulation', capabilities: [] },
    { name: 'MondoMod', category: 'Modulation', capabilities: [] },
    { name: 'Enigma', category: 'Modulation', capabilities: [] },
    { name: 'PuigTec EQP-1A', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'PuigTec MEQ-5', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'PuigChild 660', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'PuigChild 670', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'Scheps Omni Channel', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'deesser', 'saturation'] },
    { name: 'Scheps 73', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'Scheps Parallel Particles', category: 'Saturator', capabilities: ['saturation', 'exciter', 'comp_parallel'] },
    { name: 'Greg Wells MixCentric', category: 'Utility', capabilities: ['comp_bus', 'eq_static', 'saturation'] },
    { name: 'Greg Wells VoiceCentric', category: 'Utility', capabilities: ['comp_single', 'eq_static'] },
    { name: 'Greg Wells PianoCentric', category: 'Utility', capabilities: ['comp_single', 'eq_static'] },
    { name: 'Greg Wells ToneCentric', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'CLA MixHub', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'saturation'] },
    { name: 'NLS Non-Linear Summer', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'VU Meter', category: 'Analyzer', capabilities: [] },
    { name: 'PAZ Analyzer', category: 'Analyzer', capabilities: ['meter_spectrum', 'meter_correlation'] },
    { name: 'WLM Plus Loudness Meter', category: 'Analyzer', capabilities: ['meter_loudness'] },
  ],

  iZotope: [
    {
      name: 'Ozone 11',
      category: 'Mastering Suite',
      capabilities: [
        'eq_static',
        'eq_dynamic',
        'eq_linear_phase',
        'eq_mid_side',
        'eq_match',
        'comp_single',
        'comp_multiband',
        'limiter',
        'limiter_truepeak',
        'imager',
        'mono_maker',
        'exciter',
        'saturation',
        'meter_loudness',
        'meter_spectrum',
        'reference_matching',
      ],
    },
    { name: 'Ozone 11 Equalizer', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_linear_phase', 'eq_mid_side', 'eq_match'] },
    { name: 'Ozone 11 Dynamics', category: 'Multiband', capabilities: ['comp_multiband', 'comp_single', 'expander'] },
    { name: 'Ozone 11 Imager', category: 'Stereo Imager', capabilities: ['imager', 'mono_maker', 'meter_correlation'] },
    { name: 'Ozone 11 Maximizer', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'Ozone Imager 2', category: 'Stereo Imager', capabilities: ['imager', 'meter_correlation'] },
    {
      name: 'Neutron 4',
      category: 'Channel Strip',
      capabilities: [
        'eq_static',
        'eq_dynamic',
        'eq_mid_side',
        'comp_single',
        'comp_multiband',
        'comp_sidechain_ext',
        'expander',
        'transient_shaper',
        'saturation',
        'imager',
        'meter_spectrum',
      ],
    },
    { name: 'Neutron 4 Compressor', category: 'Compressor', capabilities: ['comp_single', 'comp_multiband', 'comp_sidechain_ext'] },
    { name: 'Neutron 4 EQ', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_mid_side'] },
    { name: 'Neutron 4 Gate', category: 'Gate', capabilities: ['expander', 'comp_sidechain_ext'] },
    { name: 'Neutron 4 Transient Shaper', category: 'Transient', capabilities: ['transient_shaper'] },
    {
      name: 'Nectar 4',
      category: 'Vocal Suite',
      capabilities: ['eq_static', 'eq_dynamic', 'comp_single', 'deesser', 'expander', 'pitch_correction', 'saturation', 'reverb', 'delay', 'imager'],
    },
    { name: 'RX 11', category: 'Restoration', capabilities: ['spectral_repair', 'noise_reduction', 'deesser'] },
    { name: 'Insight 2', category: 'Analyzer', capabilities: ['meter_loudness', 'meter_spectrum', 'meter_correlation'] },
    { name: 'Tonal Balance Control', category: 'Analyzer', capabilities: ['meter_spectrum', 'reference_matching'] },
    { name: 'Vinyl', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Trash 2', category: 'Distortion', capabilities: ['saturation'] },
    { name: 'Stutter Edit 2', category: 'Creative', capabilities: [] },
    { name: 'VocalSynth 2', category: 'Vocal Effects', capabilities: ['saturation'] },
  ],

  oeksound: [
    { name: 'soothe2', category: 'Resonance Suppressor', capabilities: ['resonance_suppressor'] },
    { name: 'spiff', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'bloom', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic'] },
  ],

  Soundtheory: [{ name: 'Gullfoss', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic'] }],

  Wavesfactory: [
    { name: 'Trackspacer 2.5', category: 'Utility', capabilities: ['comp_sidechain_ext'] },
    { name: 'Spectre', category: 'Saturator', capabilities: ['saturation', 'exciter'] },
  ],

  Sonible: [
    { name: 'smart:EQ 4', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_match'] },
    { name: 'smart:comp 2', category: 'Compressor', capabilities: ['comp_single', 'comp_multiband', 'comp_sidechain_ext'] },
    { name: 'smart:limit', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak', 'meter_loudness'] },
    { name: 'smart:reverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'smart:deess', category: 'De-esser', capabilities: ['deesser'] },
    { name: 'true:balance', category: 'Analyzer', capabilities: ['meter_loudness', 'meter_spectrum'] },
    { name: 'pure:comp', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'pure:EQ', category: 'EQ', capabilities: ['eq_static'] },
  ],

  Youlean: [{ name: 'Youlean Loudness Meter 2', category: 'Analyzer', capabilities: ['meter_loudness'] }],

  'Valhalla DSP': [
    { name: 'ValhallaDSP Room', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ValhallaDSP Vintage Verb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ValhallaDSP Plate', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ValhallaDSP Shimmer', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ValhallaDSP Supermassive', category: 'Reverb', capabilities: ['reverb', 'delay'] },
    { name: 'ValhallaDSP Delay', category: 'Delay', capabilities: ['delay'] },
    { name: 'ValhallaDSP Freq Echo', category: 'Delay', capabilities: ['delay'] },
    { name: 'ValhallaDSP Space Modulator', category: 'Modulation', capabilities: [] },
  ],

  Soundtoys: [
    { name: 'Decapitator', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Radiator', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'EchoBoy', category: 'Delay', capabilities: ['delay'] },
    { name: 'EchoBoy Jr.', category: 'Delay', capabilities: ['delay'] },
    { name: 'PrimalTap', category: 'Delay', capabilities: ['delay'] },
    { name: 'Crystallizer', category: 'Delay', capabilities: ['delay'] },
    { name: 'PanMan', category: 'Panner', capabilities: ['imager'] },
    { name: 'MicroShift', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'Little Plate', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Devil-Loc', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'Devil-Loc Deluxe', category: 'Compressor', capabilities: ['comp_single', 'comp_parallel', 'saturation'] },
    { name: 'Sie-Q', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'FilterFreak', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Little AlterBoy', category: 'Pitch', capabilities: ['pitch_correction'] },
    { name: 'Tremolator', category: 'Modulation', capabilities: [] },
    { name: 'PhaseMistress', category: 'Modulation', capabilities: [] },
    { name: 'Effect Rack', category: 'Multi-FX', capabilities: [] },
  ],

  'Plugin Alliance': [
    { name: 'bx_console SSL 4000 E', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'comp_vca', 'expander'] },
    { name: 'bx_console SSL 4000 G', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'comp_vca', 'expander'] },
    { name: 'bx_console Focusrite SC', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander'] },
    { name: 'bx_console AMEK 9099', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander'] },
    { name: 'bx_digital V3', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side', 'mono_maker', 'imager', 'deesser', 'meter_correlation'] },
    { name: 'bx_masterdesk', category: 'Mastering Suite', capabilities: ['eq_static', 'comp_bus', 'limiter', 'limiter_truepeak', 'imager', 'meter_loudness'] },
    { name: 'bx_limiter True Peak', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'bx_subsynth', category: 'Bass Enhancer', capabilities: ['exciter'] },
    { name: 'bx_saturator V2', category: 'Saturator', capabilities: ['saturation', 'eq_mid_side'] },
    { name: 'bx_stereomaker', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'Shadow Hills Mastering Compressor', category: 'Compressor', capabilities: ['comp_bus', 'comp_opto', 'comp_vca'] },
    { name: 'Lindell 80 Series', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'saturation'] },
    { name: 'Lindell 50 Series', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'Lindell 254E', category: 'Compressor', capabilities: ['comp_vca'] },
    { name: 'Lindell 354E', category: 'Compressor', capabilities: ['comp_vca'] },
    { name: 'Lindell 7X-500', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'Lindell PEX-500', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'SPL Iron', category: 'Compressor', capabilities: ['comp_bus'] },
    { name: 'SPL Transient Designer Plus', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'SPL De-Verb Plus', category: 'Utility', capabilities: [] },
    { name: 'Vertigo VSC-2', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'Vertigo VSM-3', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Vertigo VSE-2', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Dangerous BAX EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Elysia Alpha Compressor', category: 'Compressor', capabilities: ['comp_bus'] },
    { name: 'Elysia Karacter', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Millennia NSEQ-2', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Millennia TCL-2', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'Unfiltered Audio BYOME', category: 'Multi-FX', capabilities: ['eq_static', 'comp_single', 'saturation', 'delay', 'reverb'] },
    { name: 'Unfiltered Audio Sandman Pro', category: 'Delay', capabilities: ['delay'] },
    { name: 'ADPTR Metric AB', category: 'Analyzer', capabilities: ['meter_loudness', 'meter_spectrum', 'meter_correlation', 'reference_matching'] },
    { name: 'ADPTR Streamliner', category: 'Analyzer', capabilities: ['meter_loudness'] },
    { name: 'Megaverb', category: 'Reverb', capabilities: ['reverb'] },
  ],

  'Universal Audio': [
    { name: 'UA 1176 Classic Limiter', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'UA 1176LN', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'UA 1176AE', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'UA LA-2A Classic Leveler', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'UA LA-3A Classic Leveler', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'UA Teletronix LA-2A', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'UA Neve 1073', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'UA Neve 1073 Legacy', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'UA Neve 31102', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'UA Neve 88RS', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'saturation'] },
    { name: 'UA API 2500', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'UA API 550A', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA API Vision Channel Strip', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'saturation'] },
    { name: 'UA SSL E Channel Strip', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'comp_vca', 'expander'] },
    { name: 'UA SSL G Bus Compressor', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'UA Fairchild 660', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'UA Fairchild 670', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'UA Pultec EQP-1A', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA Pultec MEQ-5', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA Pultec HLF-3C', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA Manley Massive Passive', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA Manley Variable Mu', category: 'Compressor', capabilities: ['comp_bus'] },
    { name: 'UA Studer A800', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'UA Ampex ATR-102', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'UA Oxide Tape', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'UA Capitol Chambers', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'UA Pure Plate', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'UA Lexicon 224', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'UA EMT 140', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'UA EMT 250', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'UA Galaxy Tape Echo', category: 'Delay', capabilities: ['delay', 'tape'] },
    { name: 'UA EP-34 Tape Echo', category: 'Delay', capabilities: ['delay', 'tape'] },
    { name: 'UA Cooper Time Cube', category: 'Delay', capabilities: ['delay'] },
    { name: 'UA Precision Limiter', category: 'Limiter', capabilities: ['limiter'] },
    { name: 'UA Precision Multiband', category: 'Multiband', capabilities: ['comp_multiband', 'expander'] },
    { name: 'UA Precision EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'UA Precision De-Esser', category: 'De-esser', capabilities: ['deesser'] },
  ],

  'Slate Digital': [
    { name: 'FG-X Mastering Processor', category: 'Limiter', capabilities: ['limiter', 'comp_bus', 'meter_loudness'] },
    { name: 'FG-X 2', category: 'Limiter', capabilities: ['limiter', 'comp_bus', 'meter_loudness'] },
    { name: 'Virtual Mix Rack', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'saturation'] },
    { name: 'Virtual Tape Machines', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'Virtual Console Collection', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Virtual Buss Compressors', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca', 'comp_parallel'] },
    { name: 'Virtual Preamp Collection', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'Fresh Air', category: 'Exciter', capabilities: ['exciter'] },
    { name: 'Infinity EQ', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_mid_side'] },
    { name: 'Custom Series EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'MetaTune', category: 'Pitch', capabilities: ['pitch_correction'] },
    { name: 'Verbsuite Classics', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ANA 2', category: 'Instrument', capabilities: [] },
  ],

  Softube: [
    { name: 'Tube-Tech CL 1B', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'Tube-Tech PE 1C', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Tube-Tech ME 1B', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Summit Audio TLA-100A', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'Summit Audio EQF-100', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Weiss DS1-MK3', category: 'De-esser', capabilities: ['deesser', 'comp_multiband', 'limiter'] },
    { name: 'Weiss Compressor/Limiter', category: 'Compressor', capabilities: ['comp_single', 'comp_multiband', 'limiter'] },
    { name: 'Weiss EQ1', category: 'EQ', capabilities: ['eq_static', 'eq_linear_phase'] },
    { name: 'Weiss MM-1', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak', 'comp_multiband'] },
    { name: 'Drawmer 1973', category: 'Multiband', capabilities: ['comp_multiband', 'comp_fet'] },
    { name: 'Drawmer S73', category: 'Multiband', capabilities: ['comp_multiband'] },
    { name: 'Harmonics', category: 'Saturator', capabilities: ['saturation', 'exciter'] },
    { name: 'Saturation Knob', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'TSAR-1R', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'TSAR-1', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Console 1', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'saturation'] },
    { name: 'Tape', category: 'Tape', capabilities: ['tape', 'saturation'] },
    { name: 'Fix Flanger', category: 'Modulation', capabilities: [] },
    { name: 'Fix Phaser', category: 'Modulation', capabilities: [] },
    { name: 'Fix Doubler', category: 'Chorus', capabilities: ['imager', 'delay'] },
  ],

  Eventide: [
    { name: 'H3000 Factory', category: 'Multi-FX', capabilities: ['delay', 'imager'] },
    { name: 'H910', category: 'Pitch', capabilities: ['imager', 'delay'] },
    { name: 'H949', category: 'Pitch', capabilities: ['imager', 'delay'] },
    { name: 'MicroPitch', category: 'Pitch', capabilities: ['imager', 'delay'] },
    { name: 'Quadravox', category: 'Pitch', capabilities: ['imager', 'delay'] },
    { name: 'Octavox', category: 'Pitch', capabilities: ['imager', 'delay'] },
    { name: 'Instant Phaser', category: 'Modulation', capabilities: [] },
    { name: 'Instant Flanger', category: 'Modulation', capabilities: [] },
    { name: 'Rotary Mod', category: 'Modulation', capabilities: [] },
    { name: 'Undulator', category: 'Modulation', capabilities: [] },
    { name: 'TriceraChorus', category: 'Chorus', capabilities: ['imager'] },
    { name: 'Omnipressor', category: 'Compressor', capabilities: ['comp_single', 'expander'] },
    { name: 'UltraChannel', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'delay', 'imager'] },
    { name: 'EChannel', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander'] },
    { name: 'UltraTap', category: 'Delay', capabilities: ['delay', 'reverb'] },
    { name: 'Blackhole', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'MangledVerb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'ShimmerVerb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Spring', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Tverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Fission', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'Physion', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'Elevate', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak', 'comp_multiband', 'transient_shaper'] },
    { name: 'Saturnu', category: 'Multiband', capabilities: ['saturation', 'comp_multiband'] },
  ],

  'Native Instruments': [
    { name: 'Guitar Rig 7', category: 'Amp Sim', capabilities: ['saturation'] },
    { name: 'Replika XT', category: 'Delay', capabilities: ['delay'] },
    { name: 'Replika', category: 'Delay', capabilities: ['delay'] },
    { name: 'Raum', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Supercharger', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'Supercharger GT', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'Solid Bus Comp', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca', 'comp_parallel'] },
    { name: 'Solid Dynamics', category: 'Compressor', capabilities: ['comp_single', 'expander'] },
    { name: 'Solid EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Passive EQ', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side'] },
    { name: 'Enhanced EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Vari Comp', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'VC 2A', category: 'Compressor', capabilities: ['comp_opto'] },
    { name: 'VC 76', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'VC 160', category: 'Compressor', capabilities: ['comp_vca'] },
    { name: 'Transient Master', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'Driver', category: 'Distortion', capabilities: ['saturation'] },
    { name: 'Crush Pack', category: 'Distortion', capabilities: ['saturation'] },
    { name: 'Choral', category: 'Chorus', capabilities: ['imager'] },
    { name: 'Phasis', category: 'Modulation', capabilities: [] },
    { name: 'Flair', category: 'Modulation', capabilities: [] },
    { name: 'Mod Pack', category: 'Modulation', capabilities: [] },
  ],

  Sonnox: [
    { name: 'Oxford EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Oxford Dynamics', category: 'Compressor', capabilities: ['comp_single', 'expander', 'limiter', 'saturation'] },
    { name: 'Oxford Limiter v3', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'Oxford Inflator', category: 'Exciter', capabilities: ['saturation', 'exciter'] },
    { name: 'Oxford TransMod', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'Oxford Envolution', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'Oxford Reverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Oxford SuprEsser', category: 'De-esser', capabilities: ['deesser', 'eq_dynamic'] },
    { name: 'Claro', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side', 'meter_spectrum'] },
    { name: 'Codec Toolbox', category: 'Analyzer', capabilities: ['meter_loudness'] },
    { name: 'Restore', category: 'Restoration', capabilities: ['noise_reduction'] },
  ],

  'PSP Audioware': [
    { name: 'PSP VintageWarmer2', category: 'Compressor', capabilities: ['comp_single', 'comp_multiband', 'saturation', 'limiter'] },
    { name: 'PSP MasterComp', category: 'Compressor', capabilities: ['comp_bus'] },
    { name: 'PSP oldTimer', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'PSP BussPressor', category: 'Compressor', capabilities: ['comp_bus', 'comp_vca'] },
    { name: 'PSP FETpressor', category: 'Compressor', capabilities: ['comp_fet', 'comp_parallel'] },
    { name: '2445', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Nexcellence', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Echo', category: 'Delay', capabilities: ['delay'] },
    { name: 'Lexicon PSP 42', category: 'Delay', capabilities: ['delay'] },
    { name: 'Xenon', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'InfiniStrip', category: 'Channel Strip', capabilities: ['eq_static', 'comp_single', 'expander', 'deesser', 'saturation'] },
  ],

  'Tokyo Dawn Labs': [
    { name: 'TDR Nova', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'comp_multiband'] },
    { name: 'TDR Nova GE', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_mid_side', 'comp_multiband', 'comp_sidechain_ext'] },
    { name: 'TDR Kotelnikov', category: 'Compressor', capabilities: ['comp_bus'] },
    { name: 'TDR Kotelnikov GE', category: 'Compressor', capabilities: ['comp_bus', 'comp_parallel'] },
    { name: 'TDR Molotok', category: 'Compressor', capabilities: ['comp_bus', 'comp_parallel'] },
    { name: 'TDR SlickEQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'TDR SlickEQ GE', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'TDR SlickEQ M', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side', 'saturation'] },
    { name: 'TDR VOS SlickEQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'TDR Limiter 6 GE', category: 'Limiter', capabilities: ['comp_single', 'clipper', 'limiter', 'limiter_truepeak'] },
  ],

  Voxengo: [
    { name: 'SPAN', category: 'Analyzer', capabilities: ['meter_spectrum', 'meter_correlation'] },
    { name: 'SPAN Plus', category: 'Analyzer', capabilities: ['meter_spectrum', 'meter_correlation', 'meter_loudness'] },
    { name: 'Elephant', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak', 'meter_loudness'] },
    { name: 'CurveEQ', category: 'EQ', capabilities: ['eq_static', 'eq_linear_phase', 'eq_match'] },
    { name: 'MSED', category: 'Utility', capabilities: ['eq_mid_side'] },
    { name: 'Marvel GEQ', category: 'EQ', capabilities: ['eq_static', 'eq_linear_phase'] },
  ],

  'Nugen Audio': [
    { name: 'ISL 2', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak'] },
    { name: 'VisLM 2', category: 'Analyzer', capabilities: ['meter_loudness'] },
    { name: 'Monofilter', category: 'Utility', capabilities: ['mono_maker'] },
    { name: 'Stereoizer', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'Stereoplacer', category: 'Stereo Imager', capabilities: ['imager', 'eq_mid_side'] },
    { name: 'MasterCheck', category: 'Analyzer', capabilities: ['meter_loudness', 'reference_matching'] },
  ],

  MeldaProduction: [
    { name: 'MAutoDynamicEq', category: 'EQ', capabilities: ['eq_static', 'eq_dynamic', 'eq_mid_side', 'eq_linear_phase', 'comp_sidechain_ext'] },
    { name: 'MEqualizer', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'MCompressor', category: 'Compressor', capabilities: ['comp_single', 'comp_parallel', 'comp_sidechain_ext'] },
    { name: 'MMultiBandDynamics', category: 'Multiband', capabilities: ['comp_multiband', 'expander', 'comp_sidechain_ext'] },
    { name: 'MLimiterX', category: 'Limiter', capabilities: ['limiter', 'limiter_truepeak', 'clipper'] },
    { name: 'MSaturator', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'MStereoExpander', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'MAnalyzer', category: 'Analyzer', capabilities: ['meter_spectrum', 'meter_correlation', 'meter_loudness'] },
    { name: 'MTransient', category: 'Transient', capabilities: ['transient_shaper'] },
  ],

  Klanghelm: [
    { name: 'DC8C 3', category: 'Compressor', capabilities: ['comp_single', 'comp_parallel', 'comp_sidechain_ext'] },
    { name: 'MJUC', category: 'Compressor', capabilities: ['comp_bus', 'comp_parallel'] },
    { name: 'SDRR2', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'IVGI2', category: 'Saturator', capabilities: ['saturation'] },
    { name: 'VUMT Deluxe', category: 'Analyzer', capabilities: [] },
  ],

  Antares: [
    { name: 'Auto-Tune Pro X', category: 'Pitch', capabilities: ['pitch_correction'] },
    { name: 'Auto-Tune EFX+', category: 'Pitch', capabilities: ['pitch_correction'] },
  ],

  Celemony: [{ name: 'Melodyne 5', category: 'Pitch', capabilities: ['pitch_correction'] }],

  Kilohearts: [
    { name: 'kHs Compressor', category: 'Compressor', capabilities: ['comp_single', 'comp_sidechain_ext'] },
    { name: 'kHs Limiter', category: 'Limiter', capabilities: ['limiter'] },
    { name: 'kHs EQ', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'kHs Reverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'kHs Delay', category: 'Delay', capabilities: ['delay'] },
    { name: 'kHs Chorus', category: 'Chorus', capabilities: ['imager'] },
    { name: 'kHs Distortion', category: 'Distortion', capabilities: ['saturation'] },
    { name: 'kHs Transient Shaper', category: 'Transient', capabilities: ['transient_shaper'] },
    { name: 'kHs Gate', category: 'Gate', capabilities: ['expander'] },
    { name: 'kHs Stereo', category: 'Stereo Imager', capabilities: ['imager'] },
    { name: 'kHs Phaser', category: 'Modulation', capabilities: [] },
    { name: 'kHs Flanger', category: 'Modulation', capabilities: [] },
    { name: 'Multipass', category: 'Multiband', capabilities: ['comp_multiband'] },
    { name: 'Snap Heap', category: 'Multi-FX', capabilities: [] },
    { name: 'Disperser', category: 'Creative', capabilities: [] },
    { name: 'Phase Plant', category: 'Instrument', capabilities: [] },
  ],

  'Xfer Records': [
    { name: 'OTT', category: 'Multiband', capabilities: ['comp_multiband', 'comp_parallel'] },
    { name: 'LFOTool', category: 'Utility', capabilities: [] },
    { name: 'Cthulhu', category: 'Creative', capabilities: [] },
    { name: 'Serum FX', category: 'Multi-FX', capabilities: ['eq_static', 'comp_single', 'saturation', 'delay', 'reverb'] },
  ],

  Goodhertz: [
    { name: 'Vulf Compressor', category: 'Compressor', capabilities: ['comp_single', 'saturation'] },
    { name: 'Tone Control', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side'] },
    { name: 'Tiltshift', category: 'EQ', capabilities: ['eq_static'] },
    { name: 'Midside', category: 'Stereo Imager', capabilities: ['eq_mid_side', 'imager'] },
    { name: 'Panpot', category: 'Panner', capabilities: ['imager'] },
    { name: 'CanOpener Studio', category: 'Utility', capabilities: [] },
    { name: 'Lossy', category: 'Creative', capabilities: [] },
    { name: 'Wow Control', category: 'Modulation', capabilities: [] },
    { name: 'Trem Control', category: 'Modulation', capabilities: [] },
  ],

  'Pulsar Audio': [
    { name: 'Pulsar Mu', category: 'Compressor', capabilities: ['comp_bus', 'comp_parallel'] },
    { name: 'Pulsar 1178', category: 'Compressor', capabilities: ['comp_fet', 'comp_parallel'] },
    { name: 'Pulsar Smasher', category: 'Compressor', capabilities: ['comp_single', 'comp_parallel', 'saturation'] },
    { name: 'Pulsar Massive', category: 'EQ', capabilities: ['eq_static', 'eq_mid_side'] },
    { name: 'Pulsar 8200', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Pulsar Echorec', category: 'Delay', capabilities: ['delay'] },
  ],

  'Cherry Audio': [
    { name: 'Stardust 201', category: 'Delay', capabilities: ['delay', 'tape'] },
    { name: 'Galactic Reverb', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'PSY-301', category: 'Chorus', capabilities: ['imager'] },
  ],

  Arturia: [
    { name: 'Comp TUBE-STA', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'Comp FET-76', category: 'Compressor', capabilities: ['comp_fet'] },
    { name: 'Comp VCA-65', category: 'Compressor', capabilities: ['comp_vca'] },
    { name: 'Comp DIODE-609', category: 'Compressor', capabilities: ['comp_single'] },
    { name: 'Pre 1973', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'Pre TridA', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'Pre V76', category: 'EQ', capabilities: ['eq_static', 'saturation'] },
    { name: 'Filter MINI', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Filter MS-20', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Filter SEM', category: 'Filter', capabilities: ['eq_static'] },
    { name: 'Delay TAPE-201', category: 'Delay', capabilities: ['delay', 'tape'] },
    { name: 'Delay MEMORY-BRIGADE', category: 'Delay', capabilities: ['delay'] },
    { name: 'Delay ETERNITY', category: 'Delay', capabilities: ['delay'] },
    { name: 'Rev PLATE-140', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Rev INTENSITY', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Rev SPRING-636', category: 'Reverb', capabilities: ['reverb'] },
    { name: 'Chorus JUN-6', category: 'Chorus', capabilities: ['imager'] },
    { name: 'Chorus DIMENSION-D', category: 'Chorus', capabilities: ['imager'] },
    { name: 'Flanger BL-20', category: 'Modulation', capabilities: [] },
    { name: 'Phaser BI-TRON', category: 'Modulation', capabilities: [] },
    { name: 'Rotary CLS-222', category: 'Modulation', capabilities: [] },
    { name: 'Bus FORCE', category: 'Channel Strip', capabilities: ['eq_static', 'comp_bus', 'expander', 'saturation', 'limiter'] },
  ],
};

/** Flattened catalog — the list the picker actually searches. */
export const PLUGIN_CATALOG: CatalogPlugin[] = Object.entries(CATALOG_BY_MANUFACTURER).flatMap(
  ([manufacturer, entries]) =>
    entries.map((entry) => ({
      name: entry.name,
      manufacturer,
      category: entry.category,
      capabilities: normaliseCapabilities(entry.capabilities),
    })),
);

export const MANUFACTURERS: string[] = Object.keys(CATALOG_BY_MANUFACTURER).sort((a, b) =>
  a.localeCompare(b),
);

/** Every category present in the catalog, alphabetical. */
export const CATEGORIES: string[] = Array.from(
  new Set(PLUGIN_CATALOG.map((p) => p.category)),
).sort((a, b) => a.localeCompare(b));

/**
 * What a category implies when we know nothing else — used to pre-tick
 * capabilities on the custom-plugin form, and to reconstruct capabilities for
 * rows that come back from the server (which stores name/category only).
 */
export const CAPABILITIES_FOR_CATEGORY: Record<string, string[]> = {
  EQ: ['eq_static'],
  Filter: ['eq_static'],
  Compressor: ['comp_single'],
  Multiband: ['comp_multiband'],
  Limiter: ['limiter'],
  Clipper: ['clipper'],
  Gate: ['expander'],
  'De-esser': ['deesser'],
  'Resonance Suppressor': ['resonance_suppressor'],
  Reverb: ['reverb'],
  Delay: ['delay'],
  Saturator: ['saturation'],
  Distortion: ['saturation'],
  Tape: ['tape', 'saturation'],
  Exciter: ['exciter'],
  'Bass Enhancer': ['exciter'],
  'Stereo Imager': ['imager'],
  Panner: ['imager'],
  Chorus: ['imager'],
  Transient: ['transient_shaper'],
  'Channel Strip': ['eq_static', 'comp_single', 'expander'],
  'Mastering Suite': ['eq_static', 'comp_multiband', 'limiter', 'imager', 'meter_loudness'],
  'Vocal Suite': ['eq_static', 'comp_single', 'deesser'],
  Analyzer: ['meter_spectrum'],
  Restoration: ['noise_reduction', 'spectral_repair'],
  Pitch: ['pitch_correction'],
  'Amp Sim': ['saturation'],
  'Multi-FX': [],
  Modulation: [],
  Creative: [],
  Utility: [],
  'Vocal Effects': [],
  Instrument: [],
  Other: [],
};

/** Category list for the custom-plugin form — catalog categories plus 'Other'. */
export const CUSTOM_CATEGORIES: string[] = Array.from(
  new Set([...CATEGORIES, ...Object.keys(CAPABILITIES_FOR_CATEGORY)]),
).sort((a, b) => a.localeCompare(b));

const CATALOG_BY_KEY = new Map<string, CatalogPlugin>(
  PLUGIN_CATALOG.map((p) => [`${p.manufacturer.toLowerCase()}::${p.name.toLowerCase()}`, p]),
);
const CATALOG_BY_NAME = new Map<string, CatalogPlugin>();
for (const plugin of PLUGIN_CATALOG) {
  const key = plugin.name.toLowerCase();
  if (!CATALOG_BY_NAME.has(key)) CATALOG_BY_NAME.set(key, plugin);
}

/** Look a plugin up by name (and manufacturer, if we have it). */
export function findCatalogPlugin(
  name: string,
  manufacturer?: string | null,
): CatalogPlugin | undefined {
  if (typeof name !== 'string') return undefined;
  const lowered = name.trim().toLowerCase();
  if (lowered === '') return undefined;
  if (manufacturer) {
    const exact = CATALOG_BY_KEY.get(`${manufacturer.trim().toLowerCase()}::${lowered}`);
    if (exact) return exact;
  }
  return CATALOG_BY_NAME.get(lowered);
}

/**
 * Best guess at what a plugin can do, for rows that arrive without tags — the
 * `/plugins` API predates the capability vocabulary and stores name, category
 * and manufacturer only.
 */
export function inferCapabilities(
  name: string,
  category: string,
  manufacturer?: string | null,
): string[] {
  const known = findCatalogPlugin(name, manufacturer);
  if (known) return known.capabilities;
  return normaliseCapabilities(CAPABILITIES_FOR_CATEGORY[category] ?? []);
}

/* ------------------------------------------------------------------ */
/* Quick-start presets                                                 */
/* ------------------------------------------------------------------ */

export interface VaultPreset {
  id: string;
  label: string;
  detail: string;
  /** Empty for the stock preset, which clears the vault instead of adding. */
  plugins: CatalogPlugin[];
  clears?: boolean;
}

function byManufacturer(manufacturer: string, exclude: string[] = []): CatalogPlugin[] {
  const skip = new Set(exclude.map((n) => n.toLowerCase()));
  return PLUGIN_CATALOG.filter(
    (p) => p.manufacturer === manufacturer && !skip.has(p.name.toLowerCase()),
  );
}

function byNames(names: string[]): CatalogPlugin[] {
  return names
    .map((n) => findCatalogPlugin(n))
    .filter((p): p is CatalogPlugin => p !== undefined);
}

export const VAULT_PRESETS: VaultPreset[] = [
  {
    id: 'stock',
    label: 'Stock DAW only',
    detail: 'Clears the list. Advice falls back to the stock set every DAW ships with.',
    plugins: [],
    clears: true,
  },
  {
    id: 'fabfilter',
    label: 'FabFilter bundle',
    detail: 'Total Bundle, minus the synth.',
    plugins: byManufacturer('FabFilter', ['Twin 3']),
  },
  {
    id: 'waves-mercury',
    label: 'Waves Mercury',
    detail: 'The Waves catalog in this list.',
    plugins: byManufacturer('Waves'),
  },
  {
    id: 'izotope',
    label: 'iZotope suite',
    detail: 'Ozone, Neutron, Nectar, RX and the meters.',
    plugins: byManufacturer('iZotope'),
  },
  {
    id: 'uad',
    label: 'UAD',
    detail: 'The Universal Audio classics.',
    plugins: byManufacturer('Universal Audio'),
  },
  {
    id: 'free',
    label: 'Free essentials',
    detail: 'The no-cost tools most producers already have installed.',
    plugins: byNames([
      'TDR Nova',
      'TDR Kotelnikov',
      'TDR VOS SlickEQ',
      'SPAN',
      'Youlean Loudness Meter 2',
      'ValhallaDSP Supermassive',
      'MEqualizer',
      'MCompressor',
      'IVGI2',
      'Ozone Imager 2',
      'OTT',
    ]),
  },
];

/* ------------------------------------------------------------------ */
/* Back-compat                                                         */
/* ------------------------------------------------------------------ */

/**
 * The old `{ name, category }` shape, still consumed by SettingsModal. New code
 * should use PLUGIN_CATALOG — this keeps the account-based settings panel
 * compiling while it is migrated.
 */
export const PLUGIN_DATABASE: Record<string, { name: string; category: string }[]> =
  Object.fromEntries(
    Object.entries(CATALOG_BY_MANUFACTURER).map(([manufacturer, entries]) => [
      manufacturer,
      entries.map(({ name, category }) => ({ name, category })),
    ]),
  );

/** Manufacturers with an all-you-can-eat subscription — worth a "select all" hint. */
export const SUBSCRIPTION_MANUFACTURERS: string[] = [
  'Waves',
  'Plugin Alliance',
  'Slate Digital',
  'Universal Audio',
];
