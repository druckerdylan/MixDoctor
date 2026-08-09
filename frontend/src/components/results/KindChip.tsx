/**
 * Defect or deviation — the line the whole report turns on.
 *
 * A **defect** is wrong regardless of genre, intent, artist or decade: a
 * squared-off waveform, an inverted channel, a mix that vanishes in mono.
 * Nobody chose it and nobody would, so it keeps the alarm colour.
 *
 * A **deviation** is a measured difference from the genre reference. It may be
 * deliberate and it may be the best decision on the record — a beat leaving the
 * mid-range open for a topline reads as a deviation and is exactly right. It
 * therefore gets a neutral chip and never the red one. Presenting somebody's
 * choice back to them as damage is the failure this component exists to stop.
 *
 * The colour is deliberately *not* the severity colour. Severity is magnitude,
 * kind is category; sharing a hue would collapse two independent axes into one
 * and make a small defect look like a large deviation.
 */

import type { FindingKind } from '../../types/analysis';

interface KindSpec {
  word: string;
  glyph: string;
  cls: string;
  hint: string;
}

const KIND: Record<FindingKind, KindSpec> = {
  defect: {
    word: 'Defect',
    glyph: '✕',
    cls: 'sev-critical',
    hint: 'Wrong regardless of genre, intent or taste. Nobody chose this.',
  },
  deviation: {
    word: 'Differs from reference',
    glyph: '≠',
    // Neutral ink rather than a severity class: this chip states a category,
    // not an alarm, and the critical red is reserved for genuine faults.
    cls: 'text-ink-muted',
    hint: 'A measured difference from the genre reference. It may well be deliberate — the report says what it costs and what it buys.',
  },
};

export interface KindChipProps {
  /**
   * Optional because a payload from an older server has no `kind`. Rather than
   * guess a category we render nothing, which leaves the card exactly as it was
   * before this distinction existed.
   */
  kind?: FindingKind | null;
  className?: string;
}

export default function KindChip({ kind, className = '' }: KindChipProps) {
  const spec = kind ? KIND[kind] : undefined;
  if (!spec) return null;

  return (
    <span className={`sev-chip ${spec.cls} ${className}`.trim()} title={spec.hint}>
      <span aria-hidden="true">{spec.glyph}</span>
      {spec.word}
    </span>
  );
}
