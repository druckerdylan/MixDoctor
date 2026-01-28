export interface AudioMetrics {
  integrated_lufs: number;
  short_term_lufs_max: number;
  true_peak_dbfs: number;
  loudness_range: number;
  spectrum: Record<string, number>;
  stereo_correlation: number;
  stereo_width: number;
  crest_factor_db: number;
  rms_db: number;
  dynamic_range_db: number;
  onset_density: number;
  estimated_tempo: number;
  duration_seconds: number;
  sample_rate: number;
  // Clipping
  clipped_samples: number;
  clip_percentage: number;
  max_consecutive_clips: number;
  has_clipping: boolean;
  clipping_severity: 'good' | 'warning' | 'critical';
  // Phase
  mono_compatible: boolean;
  phase_status: 'good' | 'acceptable' | 'warning' | 'critical';
  mono_energy_loss_db: number;
}

export interface Diagnosis {
  category: string;
  severity: 'critical' | 'warning' | 'info';
  priority: 1 | 2 | 3;  // 1=CRITICAL (must fix), 2=IMPORTANT (improves quality), 3=OPTIONAL (polish)
  issue: string;
  suggestion: string;
  affected_frequencies?: string;
  // Engineering-focused fields
  what_it_affects?: string;
  likely_causes?: string[];
  dont_do?: string;
  done_when?: string;
  confidence?: 'high' | 'medium' | 'low';
}

export const PRIORITY_LABELS: Record<1 | 2 | 3, { label: string; description: string }> = {
  1: { label: 'CRITICAL', description: 'Must fix' },
  2: { label: 'IMPORTANT', description: 'Improves quality' },
  3: { label: 'OPTIONAL', description: 'Polish' },
};

export interface QuickWin {
  action: string;
  impact: 'high' | 'medium' | 'low';
  plugin_suggestions?: string[];
}

export interface ReferenceComparison {
  loudness_delta: number;
  dynamics_delta: number;
  stereo_width_delta: number;
  spectrum_comparison: Record<string, number>;
  overall_similarity: number;
  key_differences: string[];
}

export interface AnalysisResult {
  engineer_summary: string;  // Conversational summary like a real engineer would give
  summary: string;
  health_score: number;
  diagnoses: Diagnosis[];
  quick_wins: QuickWin[];
  mastering_ready: boolean;
  mastering_notes: string;
  metrics: AudioMetrics;
  reference_comparison?: ReferenceComparison;
  next_session_focus?: string[];  // Clear next actions for the user
}

export type AnalysisStatus = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error';

export type AnalysisMode = 'full_mix' | 'stems';

export type StemType =
  | 'kick'
  | 'snare_hats'
  | 'percussion'
  | 'bass'
  | 'vocals'
  | 'lead'
  | 'pads'
  | 'guitars'
  | 'fx'
  | 'other';

export const STEM_LABELS: Record<StemType, string> = {
  kick: 'Kick',
  snare_hats: 'Snare/Hats',
  percussion: 'Percussion',
  bass: 'Bass',
  vocals: 'Vocals',
  lead: 'Lead Synths/Instruments',
  pads: 'Pads/Chords',
  guitars: 'Guitars',
  fx: 'FX/Ambience',
  other: 'Other',
};

export interface StemAnalysis {
  stem_type: string;
  metrics: AudioMetrics;
  issues: Diagnosis[];
}

export interface StemInteraction {
  stem_a: string;
  stem_b: string;
  interaction_type: string;
  severity: 'critical' | 'warning' | 'info';
  description: string;
  suggestion: string;
  affected_frequencies?: string;
}

export interface StemsAnalysisResult {
  engineer_summary: string;  // Conversational summary like a real engineer would give
  summary: string;
  overall_health_score: number;
  stem_analyses: StemAnalysis[];
  interactions: StemInteraction[];
  quick_wins: QuickWin[];
  mastering_ready: boolean;
  mastering_notes: string;
}

// User and Auth types
export interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
}

// Plugin types
export interface Plugin {
  id: number;
  name: string;
  category: string;
  manufacturer: string | null;
  created_at: string;
}

export interface PluginCreate {
  name: string;
  category: string;
  manufacturer?: string;
}

// Analysis History types
export interface AnalysisHistoryItem {
  id: number;
  filename: string;
  genre: string;
  health_score: number;
  summary: string | null;
  created_at: string;
}

// Plugin categories
export const PLUGIN_CATEGORIES = [
  'EQ',
  'Compressor',
  'Limiter',
  'Reverb',
  'Delay',
  'Saturator',
  'Chorus',
  'Phaser',
  'Flanger',
  'Gate',
  'De-esser',
  'Multiband',
  'Exciter',
  'Stereo Imager',
  'Analyzer',
  'Other',
] as const;
