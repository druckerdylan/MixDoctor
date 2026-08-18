/**
 * DimensionGrid — fourteen scores, worst first.
 *
 * A score is a verdict, not an explanation. Selecting a dimension that has
 * findings therefore opens the same teaching panel the fix stack uses, right
 * under the grid: a card here used to be a dead end unless you knew to scroll
 * to the plan and find the matching row.
 */

import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import useKnowledge from '../../hooks/useKnowledge';
import Explainer, { Chevron } from './Explainer';
import KindChip from './KindChip';
import AcknowledgedChip from './AcknowledgedChip';
import {
  DIMENSIONS,
  DIMENSION_LABELS,
  SEVERITY_RANK,
  SEVERITY_VAR,
  type Dimension,
  type DimensionScore,
  type Finding,
  type FindingKind,
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

/** Shape carries the severity too — color is never the only signal. */
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
  const { has } = useKnowledge();

  /** Which finding inside the open dimension is being taught. */
  const [activeId, setActiveId] = useState<string | null>(null);
  /** The panel is closable without losing the selection the rest of the page uses. */
  const [dismissed, setDismissed] = useState<Dimension | null>(null);

  const byDim = useMemo(() => {
    const m = new Map<Dimension, DimensionScore>();
    for (const d of dimensions ?? []) m.set(d.dimension, d);
    return m;
  }, [dimensions]);

  /**
   * Worst first, so the panel opens on the thing most worth reading about —
   * and everything the producer has confirmed as deliberate after everything
   * that is still open, however severe it was measured to be. A confirmed
   * choice is not the headline of a dimension.
   */
  const byDimFindings = useMemo(() => {
    const m = new Map<Dimension, Finding[]>();
    for (const f of findings ?? []) {
      const list = m.get(f.dimension);
      if (list) list.push(f);
      else m.set(f.dimension, [f]);
    }
    for (const list of m.values()) {
      list.sort((a, b) => {
        const ack = Number(Boolean(a.acknowledged)) - Number(Boolean(b.acknowledged));
        if (ack !== 0) return ack;
        const s = SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity];
        return s !== 0 ? s : (b.impact ?? 0) - (a.impact ?? 0);
      });
    }
    return m;
  }, [findings]);

  /** Two counts per dimension: what is still open, and what has been answered. */
  const { findingCounts, acknowledgedCounts } = useMemo(() => {
    const live = new Map<Dimension, number>();
    const ack = new Map<Dimension, number>();
    for (const [dimension, list] of byDimFindings) {
      live.set(dimension, list.filter((f) => !f.acknowledged).length);
      ack.set(dimension, list.filter((f) => f.acknowledged).length);
    }
    return { findingCounts: live, acknowledgedCounts: ack };
  }, [byDimFindings]);

  /**
   * One kind per card. A defect anywhere in the dimension makes the whole card
   * a defect card — a genuine fault is not softened by the deviations sitting
   * next to it — and a dimension holding nothing but deviations says so, so a
   * low score there reads as "different from the reference" rather than damage.
   *
   * Read off what is still open, since that is what the score on the card is
   * now made of; a dimension whose findings have all been confirmed falls back
   * to the full set rather than losing its chip.
   *
   * Dimensions whose findings carry no `kind` at all (an older payload) are
   * left out of the map entirely, and `KindChip` renders nothing for them.
   */
  const kindByDim = useMemo(() => {
    const m = new Map<Dimension, FindingKind>();
    for (const [dimension, list] of byDimFindings) {
      const live = list.filter((f) => !f.acknowledged);
      const known = (live.length ? live : list).filter(
        (f) => f.kind === 'defect' || f.kind === 'deviation',
      );
      if (!known.length) continue;
      m.set(dimension, known.some((f) => f.kind === 'defect') ? 'defect' : 'deviation');
    }
    return m;
  }, [byDimFindings]);

  // Only findings we can actually teach get a panel; the rest keep the score
  // and the headline, which is all we honestly have for them.
  const teachable = useMemo(() => {
    if (!selected || selected === dismissed) return [];
    return (byDimFindings.get(selected) ?? []).filter((f) => has(f.id));
  }, [selected, dismissed, byDimFindings, has]);

  const active = useMemo(
    () => teachable.find((f) => f.id === activeId) ?? teachable[0] ?? null,
    [teachable, activeId],
  );

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

  /** Selecting always re-opens the panel, including after a close. */
  const openDimension = (d: Dimension) => {
    setDismissed(null);
    setActiveId(null);
    onSelect(d);
  };

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
          const ackCount = acknowledgedCounts.get(d.dimension) ?? 0;
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
                onClick={() => openDimension(d.dimension)}
                aria-pressed={isSelected}
                aria-label={`${DIMENSION_LABELS[d.dimension]}: ${Math.round(score)} out of 100, ${SEV_WORD[sev]}${
                  count > 0
                    ? `, ${count} finding${count === 1 ? '' : 's'}`
                    : ''
                }${
                  ackCount > 0
                    ? `, ${ackCount} confirmed as deliberate and no longer counted`
                    : ''
                }${count + ackCount > 0 ? ' — opens the explanation' : ''}`}
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

                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 pt-1">
                  <span className="flex flex-wrap items-center gap-1.5">
                    <span className={`sev-chip ${SEV_CLASS[sev]}`}>
                      <span aria-hidden="true">{SEV_GLYPH[sev]}</span>
                      {SEV_WORD[sev]}
                    </span>
                    <KindChip kind={kindByDim.get(d.dimension)} />
                    {/* Set aside, and said so on the card: a dimension that
                        scores well because the producer confirmed the thing
                        dragging it down should never look like it scored well
                        on its own. */}
                    {ackCount > 0 ? <AcknowledgedChip count={ackCount} /> : null}
                  </span>
                  {count + ackCount > 0 ? (
                    // The chevron is the promise that something opens. Without
                    // it a count reads as a statistic and the card as a dead end.
                    <span className="flex items-center gap-1.5">
                      <span className="stat text-micro uppercase text-ink-faint transition-colors duration-300 group-hover:text-signal">
                        {/* Says where it went, now that selecting actually travels. */}
                        {isSelected && active
                          ? 'showing the fix'
                          : count > 0
                            ? `${count} finding${count === 1 ? '' : 's'} · explain`
                            : 'explain'}
                      </span>
                      <Chevron
                        open={Boolean(isSelected && active)}
                        className="text-ink-faint transition-colors duration-300 group-hover:text-signal"
                      />
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

      {/* The teaching, in the same section as the score that prompted it. */}
      <AnimatePresence initial={false}>
        {active && selected ? (
          <motion.div
            key={selected}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: -8 }}
            transition={{ duration: reduce ? 0.2 : 0.45, ease: EASE }}
            className="panel mt-4 overflow-hidden"
          >
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-void-line/70 px-4 py-3.5 sm:px-5">
              {active.acknowledged ? (
                <AcknowledgedChip />
              ) : (
                <span className={`sev-chip ${SEV_CLASS[active.severity]}`}>
                  <span aria-hidden="true">{SEV_GLYPH[active.severity]}</span>
                  {SEV_WORD[active.severity]}
                </span>
              )}
              {/* The open finding's own kind, not the card's roll-up: a
                  dimension can hold a defect and a deviation at once, and the
                  panel is teaching exactly one of them. */}
              <KindChip kind={active.kind} />
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                {DIMENSION_LABELS[selected]}
              </span>
              <span className="hairline hidden flex-1 sm:block" aria-hidden="true" />
              <button
                type="button"
                onClick={() => setDismissed(selected)}
                className="btn-ghost px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.13em]"
              >
                Close
              </button>
            </div>

            {teachable.length > 1 ? (
              <div
                className="flex flex-wrap gap-2 border-b border-void-line/70 px-4 py-3 sm:px-5"
                aria-label={`Findings in ${DIMENSION_LABELS[selected]}`}
              >
                {teachable.map((f) => {
                  const on = f.id === active.id;
                  return (
                    <button
                      key={f.id}
                      type="button"
                      aria-pressed={on}
                      onClick={() => setActiveId(f.id)}
                      className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-left text-[12px] leading-snug transition-colors duration-300 ease-cine ${
                        on
                          ? 'border-signal-dim/60 bg-signal/10 text-ink'
                          : 'border-void-line text-ink-muted hover:border-ink-faint hover:text-ink-dim'
                      } ${f.acknowledged && !on ? 'opacity-70' : ''}`}
                    >
                      {/* A tick rather than a hidden tab: the confirmed ones
                          stay reachable, they just stop looking like work. */}
                      {f.acknowledged ? (
                        <span
                          aria-hidden="true"
                          className="text-[10px] text-sev-clean"
                          title="You confirmed this was intentional"
                        >
                          ✓
                        </span>
                      ) : null}
                      {f.title}
                    </button>
                  );
                })}
              </div>
            ) : null}

            <div className="p-4 sm:p-5">
              {teachable.length === 1 ? (
                <p className="mb-4 max-w-[62ch] font-display text-[14.5px] font-semibold leading-snug tracking-tight text-ink">
                  {active.title}
                </p>
              ) : null}
              <Explainer findingId={active.id} showTime />
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
