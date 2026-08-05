import { useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  DIMENSIONS,
  DIMENSION_LABELS,
  SEVERITY_VAR,
  type Dimension,
  type DimensionScore,
  type Finding,
  type Severity,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const SEV_ORDER: Severity[] = ['critical', 'major', 'minor', 'clean'];

const SEV_CLASS: Record<Severity, string> = {
  critical: 'sev-critical',
  major: 'sev-major',
  minor: 'sev-minor',
  clean: 'sev-clean',
};

const SEV_WORD: Record<Severity, string> = {
  critical: 'Critical',
  major: 'Major',
  minor: 'Minor',
  clean: 'Clean',
};

/** Shape carries the severity too — colour is never the only signal. */
const SEV_GLYPH: Record<Severity, string> = {
  critical: '▲',
  major: '●',
  minor: '◆',
  clean: '✓',
};

function clamp100(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(100, Math.max(0, v));
}

/** Problems are loud; clean dimensions recede. */
function cardChrome(sev: Severity, isSelected: boolean): {
  borderColor: string;
  boxShadow: string | undefined;
  background: string | undefined;
} {
  if (isSelected) {
    return {
      borderColor: 'rgba(82,242,196,0.55)',
      boxShadow: '0 0 0 1px rgba(82,242,196,0.28), 0 24px 60px -34px rgba(0,0,0,0.95)',
      background: 'linear-gradient(180deg, rgba(82,242,196,0.06), transparent 55%)',
    };
  }
  const c = SEVERITY_VAR[sev];
  switch (sev) {
    case 'critical':
      return {
        borderColor: `color-mix(in srgb, ${c} 44%, transparent)`,
        boxShadow: `0 0 30px -20px ${c}, 0 20px 50px -30px rgba(0,0,0,0.9)`,
        background: `linear-gradient(180deg, color-mix(in srgb, ${c} 9%, transparent), transparent 60%)`,
      };
    case 'major':
      return {
        borderColor: `color-mix(in srgb, ${c} 30%, transparent)`,
        boxShadow: `0 0 26px -22px ${c}, 0 20px 50px -32px rgba(0,0,0,0.9)`,
        background: `linear-gradient(180deg, color-mix(in srgb, ${c} 6%, transparent), transparent 60%)`,
      };
    case 'minor':
      return { borderColor: '#1E1E29', boxShadow: undefined, background: undefined };
    default:
      return { borderColor: '#16161F', boxShadow: undefined, background: undefined };
  }
}

export interface DimensionGridProps {
  dimensions: DimensionScore[];
  findings: Finding[];
  onSelect: (d: Dimension) => void;
  selected: Dimension | null;
}

export default function DimensionGrid({
  dimensions,
  findings,
  onSelect,
  selected,
}: DimensionGridProps) {
  const reduce = useReducedMotion() ?? false;

  const byDim = useMemo(() => {
    const m = new Map<Dimension, DimensionScore>();
    for (const d of dimensions ?? []) m.set(d.dimension, d);
    return m;
  }, [dimensions]);

  const findingCounts = useMemo(() => {
    const m = new Map<Dimension, number>();
    for (const f of findings ?? []) m.set(f.dimension, (m.get(f.dimension) ?? 0) + 1);
    return m;
  }, [findings]);

  // Diagnostic order, not alphabetical: what breaks a mix before what leaves it unfinished.
  const ordered = useMemo(
    () => DIMENSIONS.map((d) => byDim.get(d)).filter((d): d is DimensionScore => Boolean(d)),
    [byDim],
  );

  const tally = useMemo(() => {
    const m: Record<Severity, number> = { critical: 0, major: 0, minor: 0, clean: 0 };
    for (const d of ordered) m[d.severity] += 1;
    return m;
  }, [ordered]);

  // SEV_ORDER is already worst-first.
  const worstFirst = useMemo(() => SEV_ORDER.filter((s) => tally[s] > 0), [tally]);

  if (!ordered.length) {
    return (
      <div className="panel flex min-h-[140px] items-center justify-center p-8">
        <p className="eyebrow">No dimension scores in this analysis</p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-x-5 gap-y-3">
        <p className="eyebrow text-ink-muted">{ordered.length} dimensions · diagnostic order</p>
        <div className="flex flex-wrap items-center gap-2">
          {worstFirst.map((s) => (
            <span key={s} className={`sev-chip ${SEV_CLASS[s]}`}>
              <span aria-hidden="true">{SEV_GLYPH[s]}</span>
              <span className="tabular-nums">{tally[s]}</span>
              <span className="text-ink-muted">{SEV_WORD[s]}</span>
            </span>
          ))}
        </div>
      </div>

      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {ordered.map((d, i) => {
          const sev = d.severity;
          const score = clamp100(d.score);
          const isSelected = selected === d.dimension;
          const isClean = sev === 'clean' && !isSelected;
          const count = findingCounts.get(d.dimension) ?? 0;
          const chrome = cardChrome(sev, isSelected);
          const color = SEVERITY_VAR[sev];

          return (
            <motion.li
              key={d.dimension}
              initial={reduce ? { opacity: 0 } : { opacity: 0, y: 14 }}
              animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
              transition={{
                duration: reduce ? 0.2 : 0.55,
                ease: EASE,
                delay: reduce ? 0 : 0.04 * i,
              }}
            >
              <button
                type="button"
                onClick={() => onSelect(d.dimension)}
                aria-pressed={isSelected}
                aria-label={`${DIMENSION_LABELS[d.dimension]}: ${Math.round(score)} out of 100, ${SEV_WORD[sev]}${
                  count > 0 ? `, ${count} finding${count === 1 ? '' : 's'}` : ''
                }`}
                className={`group relative flex h-full w-full flex-col rounded-xl2 border bg-void-panel p-4 text-left
                            transition-all duration-300 ease-cine hover:-translate-y-0.5 hover:bg-void-raised
                            ${isClean ? 'opacity-[0.55] hover:opacity-100' : ''}`}
                style={{
                  borderColor: chrome.borderColor,
                  boxShadow: chrome.boxShadow,
                  backgroundImage: chrome.background,
                }}
              >
                {/* Selected marker: a solid rail, not just a hue shift. */}
                {isSelected && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-y-3 left-0 w-[2px] rounded-full bg-signal"
                  />
                )}

                <div className="flex items-start justify-between gap-3">
                  <span
                    className={`font-display text-[13px] font-semibold leading-tight tracking-tight ${
                      isClean ? 'text-ink-dim' : 'text-ink'
                    }`}
                  >
                    {DIMENSION_LABELS[d.dimension]}
                  </span>
                  <span
                    className="stat shrink-0 text-[17px] leading-none"
                    style={{ color: isClean ? '#70707E' : color }}
                  >
                    {Math.round(score)}
                  </span>
                </div>

                <div
                  className="mt-3 h-[3px] w-full overflow-hidden rounded-full bg-void-line/60"
                  aria-hidden="true"
                >
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      transformOrigin: 'left center',
                      background: `linear-gradient(90deg, color-mix(in srgb, ${color} 55%, transparent), ${color})`,
                      boxShadow: isClean ? undefined : `0 0 10px -2px ${color}`,
                      width: '100%',
                    }}
                    initial={{ scaleX: reduce ? score / 100 : 0 }}
                    animate={{ scaleX: score / 100 }}
                    transition={{
                      duration: reduce ? 0 : 0.85,
                      ease: EASE,
                      delay: reduce ? 0 : 0.12 + 0.04 * i,
                    }}
                  />
                </div>

                <p
                  className={`mt-3 line-clamp-2 min-h-[2.15rem] text-[12.5px] leading-snug ${
                    isClean ? 'text-ink-muted' : 'text-ink-dim'
                  }`}
                >
                  {d.headline || '—'}
                </p>

                <div className="mt-3 flex items-center justify-between gap-2 pt-1">
                  <span className={`sev-chip ${SEV_CLASS[sev]}`}>
                    <span aria-hidden="true">{SEV_GLYPH[sev]}</span>
                    {SEV_WORD[sev]}
                  </span>
                  {count > 0 ? (
                    <span className="stat text-micro uppercase text-ink-faint">
                      {count} finding{count === 1 ? '' : 's'}
                    </span>
                  ) : (
                    <span className="stat text-micro uppercase text-ink-faint/60">no findings</span>
                  )}
                </div>
              </button>
            </motion.li>
          );
        })}
      </ul>
    </div>
  );
}
