/**
 * FixStack — the session plan. This is the part the product is actually sold on.
 *
 * Prescriptions are ordered by the points they buy back, numbered like a shot
 * list, and written to be read with a DAW open: target, action, tool, settings.
 * Ticking one off moves the projected score toward the ceiling, which is the
 * whole loop — you watch the number climb as you work.
 *
 * If the AI layer failed, `engineer` is null. The deterministic findings are
 * always there, so we fall back to those rather than showing an empty page.
 *
 * An expanded card is layered worst-to-best-understood, not most-to-least
 * precise:
 *
 *   1. the explainer — what the problem is and how to fix it, in general;
 *   2. the prescription, when the AI layer landed — the same problem, but for
 *      this arrangement, with settings;
 *   3. the measurements, folded away — the proof, for anyone who wants to
 *      check the working.
 *
 * The numbers used to be the whole card. They are correct and they are the
 * evidence, but a reading is not an explanation, and leading with one leaves
 * anyone who does not already know the concept exactly where they started.
 */

import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';
import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from 'framer-motion';
import type { EngineerStatus } from '../../hooks/useAnalysis';
import useKnowledge from '../../hooks/useKnowledge';
import Explainer, { Chevron, Disclosure } from './Explainer';
import KindChip from './KindChip';
import AcknowledgedChip from './AcknowledgedChip';
import {
  DIMENSION_LABELS,
  SEVERITY_RANK,
  SEVERITY_VAR,
  formatHz,
  formatTime,
  severityFromScore,
  type Evidence,
  type Finding,
  type MixAnalysis,
  type Move,
  type Prescription,
  type Severity,
  type Verdict,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

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

type Difficulty = Prescription['difficulty'];

const DIFFICULTY: Record<Difficulty, { word: string; cls: string; glyph: string }> = {
  easy: { word: 'Easy', cls: 'sev-clean', glyph: '○' },
  moderate: { word: 'Moderate', cls: 'sev-minor', glyph: '◐' },
  advanced: { word: 'Advanced', cls: 'sev-major', glyph: '●' },
};

/* ------------------------------------------------------------------ utils */

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function clamp100(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(100, Math.max(0, v));
}

function pad2(n: number): string {
  return n.toString().padStart(2, '0');
}

function minutesLabel(m: number): string {
  const v = Math.max(0, Math.round(m));
  if (v < 60) return `${v} min`;
  const h = Math.floor(v / 60);
  const rem = v % 60;
  return rem ? `${h} h ${rem} min` : `${h} h`;
}

function humanKey(k: string): string {
  return k.replace(/_/g, ' ');
}

/**
 * Evidence values arrive as anything from 0.68 to 1842. Print the precision
 * that is actually there rather than forcing decimals onto a sample count.
 */
function evValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  const r = Math.round(v * 100) / 100;
  return Math.abs(r) >= 10000 ? r.toLocaleString('en-US') : String(r);
}

/* ---------------------------------------------------------------- pieces */

function CheckBox({
  checked,
  onToggle,
  label,
}: {
  checked: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={label}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      title={label}
      className="grid h-[30px] w-[30px] shrink-0 place-items-center rounded-lg border transition-all duration-300 ease-cine hover:border-signal-dim"
      style={{
        borderColor: checked ? 'rgba(82,242,196,0.65)' : '#33334A',
        background: checked ? 'rgba(82,242,196,0.14)' : 'rgba(255,255,255,0.025)',
        boxShadow: checked ? '0 0 18px -6px rgba(82,242,196,0.75)' : undefined,
      }}
    >
      <svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true">
        <motion.path
          d="M2.5 7.4L5.6 10.4L11.5 3.9"
          fill="none"
          stroke={checked ? '#52F2C4' : '#70707E'}
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={false}
          animate={{ pathLength: checked ? 1 : 0.999, opacity: checked ? 1 : 0.3 }}
          transition={{ duration: 0.32, ease: EASE }}
        />
      </svg>
    </button>
  );
}

function EvidenceRows({ evidence }: { evidence: Evidence[] }) {
  if (!evidence?.length) return null;
  return (
    <ul className="divide-y divide-void-lineSoft">
      {evidence.map((e, i) => {
        const sev = VERDICT_SEV[e.verdict] ?? 'minor';
        const target =
          e.target_range && e.target_range.length === 2
            ? `${e.target_range[0]} to ${e.target_range[1]}`
            : typeof e.target === 'number' && Number.isFinite(e.target)
              ? `${e.target}`
              : null;
        return (
          <li
            key={`${e.label}-${i}`}
            className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-1 py-2.5"
          >
            <span className="min-w-0 text-[12.5px] leading-snug text-ink-dim">{e.label}</span>
            <span className="flex items-baseline gap-2 whitespace-nowrap">
              {target ? (
                <span className="stat text-[10.5px] text-ink-faint">target {target}</span>
              ) : null}
              <span className="stat text-[13px]" style={{ color: SEVERITY_VAR[sev] }}>
                {evValue(finite(e.value))}
                {e.unit ? <span className="ml-1 text-[10px] text-ink-faint">{e.unit}</span> : null}
              </span>
            </span>
            {e.detail ? (
              <span className="col-span-2 text-[11.5px] leading-snug text-ink-muted">{e.detail}</span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * The proof. Kept whole — the prose reading, every evidence row, the impact
 * and confidence — but demoted behind a fold, because it answers "how do you
 * know" rather than "what do I do".
 */
function Measurements({ finding }: { finding: Finding }) {
  const confidence = Math.round(clamp100(finite(finding.confidence) * 100));

  return (
    <div className="rounded-xl border border-void-line/60 bg-void/50 px-4 py-4 sm:px-5">
      {finding.detail ? (
        <p className="max-w-[70ch] text-pretty text-[13px] leading-relaxed text-ink-dim">
          {finding.detail}
        </p>
      ) : null}

      {finding.evidence?.length ? (
        <div className={finding.detail ? 'mt-3' : undefined}>
          <EvidenceRows evidence={finding.evidence} />
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-void-lineSoft pt-3">
        <span className="stat text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">
          impact {finite(finding.impact).toFixed(1)}
        </span>
        <span className="stat text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">
          confidence {confidence}%
        </span>
        {finding.moments?.length ? (
          <span className="stat text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">
            {finding.moments.length} moment{finding.moments.length === 1 ? '' : 's'} · first{' '}
            {formatTime(finite(finding.moments[0]?.t_start))}
          </span>
        ) : null}
      </div>
    </div>
  );
}

/** How much is behind the fold, so it is worth opening or worth skipping. */
function measurementCount(finding: Finding | null): string | undefined {
  const rows = finding?.evidence?.length ?? 0;
  return rows ? `${rows} reading${rows === 1 ? '' : 's'}` : undefined;
}

/**
 * The affordance. These cards always opened; nobody realised they did, because
 * a severity chip and a headline read as a summary rather than as a door. So a
 * collapsed card now says what is inside it, in words, and lights up on hover.
 */
function OpenRow({
  onOpen,
  hasTeaching,
  acknowledged = false,
}: {
  onOpen: () => void;
  hasTeaching: boolean;
  /** Nothing is being fixed here, so the door must not promise a fix. */
  acknowledged?: boolean;
}) {
  return (
    <button
      type="button"
      tabIndex={-1}
      onClick={(e) => {
        e.stopPropagation();
        onOpen();
      }}
      className="group/open mt-5 flex w-full items-center gap-3 text-left"
    >
      <span className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-muted transition-colors duration-300 ease-cine group-hover/open:text-signal">
        {acknowledged
          ? 'What was measured here'
          : hasTeaching
            ? 'What this means & how to fix it'
            : 'Open the detail'}
      </span>
      <span className="hairline flex-1" aria-hidden="true" />
      <Chevron
        open={false}
        className="text-ink-faint transition-colors duration-300 group-hover/open:text-signal"
      />
    </button>
  );
}

/** Said once, on the first card, because the interaction has to be learned once. */
function FirstCardHint() {
  return (
    <p className="mt-4 flex items-start gap-2 text-[12px] leading-relaxed text-ink-muted">
      <span aria-hidden="true" className="mt-[5px] text-[9px] text-signal-dim">
        ▾
      </span>
      <span>
        Every card opens like this. Click any one of them for the explanation, the steps, and the
        numbers behind it.
      </span>
    </p>
  );
}

function MoveBlock({ move, index }: { move: Move; index: number }) {
  const settings = Object.entries(move.settings ?? {}).filter(([, v]) => typeof v === 'string' && v !== '');

  return (
    <li className="rounded-xl border border-void-line/70 bg-void-deep/60 p-4 sm:p-5">
      <div className="mb-4 flex items-center gap-3">
        <span className="stat shrink-0 text-[11px] tracking-[0.12em] text-signal-dim">
          MOVE {pad2(index)}
        </span>
        <span className="hairline flex-1" aria-hidden="true" />
      </div>

      <dl className="grid gap-x-5 gap-y-3 sm:grid-cols-[74px_minmax(0,1fr)]">
        {move.target ? (
          <>
            <dt className="eyebrow sm:pt-[3px]">Target</dt>
            <dd className="text-[13px] font-medium leading-snug text-ink">{move.target}</dd>
          </>
        ) : null}

        {move.action ? (
          <>
            <dt className="eyebrow sm:pt-[4px]">Action</dt>
            <dd className="text-[13.5px] leading-relaxed text-ink">{move.action}</dd>
          </>
        ) : null}

        {move.tool ? (
          <>
            <dt className="eyebrow sm:pt-[3px]">Tool</dt>
            <dd className="font-mono text-[12.5px] leading-snug text-ink-dim">{move.tool}</dd>
          </>
        ) : null}

        {settings.length ? (
          <>
            <dt className="eyebrow sm:pt-[9px]">Settings</dt>
            <dd>
              <div className="rounded-lg border border-void-line bg-void px-3.5 py-3">
                <dl className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] gap-x-6 gap-y-2">
                  {settings.map(([k, v]) => (
                    <Fragment key={k}>
                      <dt className="whitespace-nowrap font-mono text-[10px] uppercase tracking-[0.13em] text-ink-faint">
                        {humanKey(k)}
                      </dt>
                      <dd className="stat break-words text-[12.5px] leading-snug text-ink">{v}</dd>
                    </Fragment>
                  ))}
                </dl>
              </div>
            </dd>
          </>
        ) : null}
      </dl>

      {move.why ? (
        <p className="mt-3.5 text-[12px] leading-relaxed text-ink-muted sm:pl-[94px]">{move.why}</p>
      ) : null}
    </li>
  );
}

/* ---------------------------------------------------------- the progress */

function ProgressHead({
  base,
  ceiling,
  projected,
  doneCount,
  total,
  pointsLeft,
  minutesLeft,
  reduce,
  split = false,
}: {
  base: number;
  ceiling: number;
  projected: number;
  doneCount: number;
  total: number;
  pointsLeft: number;
  minutesLeft: number;
  reduce: boolean;
  /**
   * Whether the report carries a `ScoreCard`. Every figure in this panel comes
   * off the legacy composite, and where the top of the page is showing
   * `Technical 100 · A+`, an unlabelled "Projected score 54 / 84" underneath it
   * reads as a contradiction rather than as a second measure. It is not one —
   * the composite mixes defects and genre distance, which is exactly why the
   * split exists — so when the split is present this says which number it is.
   */
  split?: boolean;
}) {
  const mv = useMotionValue(base);

  useEffect(() => {
    if (reduce) {
      mv.set(projected);
      return;
    }
    const controls = animate(mv, projected, { duration: 0.8, ease: EASE });
    return () => controls.stop();
  }, [projected, reduce, mv]);

  const text = useTransform(mv, (v) => Math.round(v).toString());
  const sev = severityFromScore(projected);
  const gain = Math.max(0, projected - base);
  const basePct = clamp100(base);
  const projPct = clamp100(projected);
  const ceilPct = clamp100(ceiling);

  return (
    <div className="panel-raised relative overflow-hidden p-5 sm:p-7">
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(130% 100% at 0% 0%, rgba(82,242,196,0.10), transparent 58%)',
        }}
      />

      <div className="relative flex flex-wrap items-end justify-between gap-x-8 gap-y-6">
        <div className="min-w-0">
          <p className="eyebrow">
            {split ? 'Projected composite score' : 'Projected score'}
          </p>
          <div className="mt-3 flex items-baseline gap-3">
            <motion.span
              className="stat text-[clamp(2.6rem,7vw,4.2rem)] font-semibold leading-[0.85]"
              style={{ color: SEVERITY_VAR[sev] }}
            >
              {text}
            </motion.span>
            <span className="stat text-[15px] leading-none text-ink-faint">
              / {Math.round(ceiling)}
            </span>
            {gain > 0.5 ? (
              <motion.span
                key={Math.round(gain)}
                initial={reduce ? { opacity: 0 } : { opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: reduce ? 0.2 : 0.45, ease: EASE }}
                className="stat text-[15px] leading-none text-signal"
              >
                +{Math.round(gain)}
              </motion.span>
            ) : null}
          </div>
          <p className="mt-3 text-[12.5px] leading-snug text-ink-muted">
            Starting at <span className="stat text-ink-dim">{Math.round(base)}</span>. The ceiling is
            what this mix can reach with the moves below — not what a different arrangement could.
            {split
              ? ' This is the older combined score, which mixes defects and distance from the genre together — it is not the technical grade above, and it is here because it is the one that moves as you work.'
              : ''}
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <div>
            <p className="eyebrow">Ticked off</p>
            <p className="stat mt-2.5 text-[20px] leading-none text-ink">
              {pad2(doneCount)}
              <span className="text-ink-faint"> / {pad2(total)}</span>
            </p>
          </div>
          <div>
            <p className="eyebrow">Points left</p>
            <p
              className="stat mt-2.5 text-[20px] leading-none"
              style={{ color: pointsLeft > 0 ? '#52F2C4' : '#70707E' }}
            >
              +{Math.round(pointsLeft)}
            </p>
          </div>
          <div>
            <p className="eyebrow">Time left</p>
            <p className="stat mt-2.5 whitespace-nowrap text-[20px] leading-none text-ink-dim">
              {minutesLeft > 0 ? minutesLabel(minutesLeft) : '—'}
            </p>
          </div>
        </div>
      </div>

      {/* The climb */}
      <div className="relative mt-7">
        <div
          className="relative h-[10px] w-full overflow-hidden rounded-full border border-void-line bg-void"
          role="img"
          aria-label={`${
            split ? 'Composite score' : 'Health score'
          } ${Math.round(base)} of 100. With the ticked fixes applied, ${Math.round(
            projected,
          )}. Ceiling ${Math.round(ceiling)}.`}
        >
          {/* Ceiling shadow */}
          <div
            className="absolute inset-y-0 left-0 bg-signal/10"
            style={{ width: `${ceilPct}%` }}
            aria-hidden="true"
          />
          {/* Where you started */}
          <div
            className="absolute inset-y-0 left-0 rounded-r-full"
            style={{
              width: `${basePct}%`,
              background: `linear-gradient(90deg, color-mix(in srgb, ${SEVERITY_VAR[severityFromScore(base)]} 40%, transparent), ${SEVERITY_VAR[severityFromScore(base)]})`,
            }}
            aria-hidden="true"
          />
          {/* What the ticked fixes buy */}
          <motion.div
            className="absolute inset-y-0"
            style={{
              left: `${basePct}%`,
              background: 'linear-gradient(90deg, rgba(82,242,196,0.55), #52F2C4)',
              boxShadow: '0 0 16px -2px rgba(82,242,196,0.8)',
            }}
            initial={false}
            animate={{ width: `${Math.max(0, projPct - basePct)}%` }}
            transition={{ duration: reduce ? 0 : 0.8, ease: EASE }}
            aria-hidden="true"
          />
        </div>

        {/* Ceiling tick */}
        <div
          className="absolute -top-1 h-[18px] w-px bg-signal/70"
          style={{ left: `${ceilPct}%` }}
          aria-hidden="true"
        />
        <span
          className="stat absolute top-[20px] text-[10px] text-signal-dim"
          style={{ left: `${ceilPct}%`, transform: 'translateX(-50%)' }}
          aria-hidden="true"
        >
          ceiling
        </span>
        <div className="mt-[34px] flex justify-between">
          <span className="stat text-[10px] leading-none text-ink-faint">0</span>
          <span className="stat text-[10px] leading-none text-ink-faint">100</span>
        </div>
      </div>
    </div>
  );
}

/**
 * The heading over the confirmed choices.
 *
 * Set aside, not swept away. The difference matters: a producer who told the
 * report their intro was written thin should be able to see that the report
 * heard them, and still see the numbers underneath it. A finding that vanished
 * would just look like the analyzer had changed its mind.
 */
function SetAsideHeader({ count }: { count: number }) {
  const one = count === 1;
  return (
    <div className="mb-5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <p className="eyebrow text-signal-dim">Set aside — your call</p>
        <span className="hairline hidden flex-1 sm:block" aria-hidden="true" />
        <p className="eyebrow">
          {count} confirmed as deliberate
        </p>
      </div>
      <p className="mt-3 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">
        You said {one ? 'this was' : 'these were'} intentional, so {one ? 'it is' : 'they are'} out
        of the running order and out of the score. Nothing was deleted — the measurements are still
        here, and changing your answer at the top of the report puts {one ? 'it' : 'them'} back.
      </p>
    </div>
  );
}

/* ---------------------------------------------------------------- a card */

function PrescriptionCard({
  p,
  index,
  finding,
  done,
  expanded,
  first,
  acknowledged = false,
  onToggleDone,
  onToggleExpand,
  onFocus,
  reduce,
}: {
  p: Prescription;
  index: number;
  finding: Finding | null;
  done: boolean;
  expanded: boolean;
  /** The first card carries the one-time note that these things open. */
  first: boolean;
  /**
   * The producer confirmed this was deliberate. The card keeps everything it
   * had — the reading, the moves, the numbers — and stops being a task: no tick
   * box, no points, no place in the running order. It is here to be looked at,
   * not to be closed.
   */
  acknowledged?: boolean;
  onToggleDone: () => void;
  onToggleExpand: () => void;
  onFocus: () => void;
  reduce: boolean;
}) {
  const { has } = useKnowledge();
  const diff = DIFFICULTY[p.difficulty] ?? DIFFICULTY.moderate;
  const gain = Math.max(0, Math.round(finite(p.expected_gain)));
  const mins = Math.max(0, Math.round(finite(p.minutes)));
  const moves = useMemo(
    () => [...(p.moves ?? [])].sort((a, b) => finite(a.order) - finite(b.order)),
    [p.moves],
  );
  const sev = finding?.severity ?? 'major';
  const bodyId = `fix-body-${p.finding_id}-${index}`;
  const teaches = has(p.finding_id);
  const hasNumbers = Boolean(finding && (finding.detail || finding.evidence?.length));
  const hasTrackLayer = Boolean(
    p.diagnosis || p.root_cause || moves.length || p.alternative || p.do_not || p.done_when,
  );

  // The header is the handle. Clicking it opens the card — it used to jump to
  // the timeline, which is a fine thing to want and now has its own button,
  // but it made the obvious gesture do the non-obvious thing and left the
  // explanation hidden. The body is deliberately not a toggle: collapsing the
  // thing someone is halfway through reading because they clicked a paragraph
  // is worse than any discoverability it would buy.
  const handleHeaderClick = () => {
    const sel = typeof window !== 'undefined' ? window.getSelection() : null;
    if (sel && sel.toString().length > 0) return;
    onToggleExpand();
  };

  return (
    <motion.article
      id={`fix-${p.finding_id}`}
      data-finding-id={p.finding_id}
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18 }}
      whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.08 }}
      transition={{ duration: reduce ? 0.25 : 0.6, ease: EASE }}
      className="panel group/card relative scroll-mt-28 overflow-hidden transition-all duration-500 ease-cine hover:border-void-line/90 hover:bg-void-raised/40"
      style={{
        opacity: done || acknowledged ? 0.62 : 1,
        borderColor: done || acknowledged ? 'rgba(82,242,196,0.28)' : undefined,
      }}
    >
      {/* Severity rail — colour plus position, never colour alone. */}
      <span
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: done || acknowledged ? SEVERITY_VAR.clean : SEVERITY_VAR[sev] }}
      />

      <div className="p-5 pl-6 sm:p-7 sm:pl-8">
        <div onClick={handleHeaderClick} className="flex cursor-pointer items-start gap-4 sm:gap-5">
          <span
            aria-hidden="true"
            className="stat mt-[2px] shrink-0 text-[clamp(1.5rem,3.4vw,2.1rem)] font-semibold leading-none tracking-[-0.04em] transition-colors duration-500"
            style={{ color: done ? SEVERITY_VAR.clean : '#4A4A57' }}
          >
            {acknowledged ? '·' : pad2(index + 1)}
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              {acknowledged ? (
                <AcknowledgedChip />
              ) : (
                <span className={`sev-chip ${SEV_CLASS[sev]}`}>
                  <span aria-hidden="true">{SEV_GLYPH[sev]}</span>
                  {SEV_WORD[sev]}
                </span>
              )}
              {/* Severity says how far; this says whether "far" is even the
                  right frame. A prescription against a deviation is an option,
                  not a repair, and the chip is what makes that visible. */}
              <KindChip kind={finding?.kind} />
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                {DIMENSION_LABELS[p.dimension] ?? humanKey(p.dimension)}
              </span>
            </div>

            <h4 className="mt-3">
              <button
                type="button"
                aria-expanded={expanded}
                aria-controls={bodyId}
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleExpand();
                }}
                className="display max-w-3xl text-balance text-left text-[clamp(1.05rem,2.1vw,1.5rem)] leading-[1.16] tracking-[-0.03em] text-ink transition-colors duration-300 ease-cine hover:text-signal group-hover/card:text-signal"
              >
                {p.headline || 'Fix'}
              </button>
            </h4>

            <div className="mt-4 flex flex-wrap items-center gap-x-2.5 gap-y-2">
              {/* No points on an acknowledged card. There is nothing to buy
                  back — the finding stopped counting the moment it was
                  confirmed, and offering a score for it would be a lie. */}
              {acknowledged ? null : (
                <span className="sev-chip sev-clean">
                  <span aria-hidden="true">↑</span>+{gain} pts
                </span>
              )}
              <span className="sev-chip text-ink-muted">
                <span aria-hidden="true">◷</span>
                {mins} min
              </span>
              <span className={`sev-chip ${diff.cls}`}>
                <span aria-hidden="true">{diff.glyph}</span>
                {diff.word}
              </span>
              {moves.length ? (
                <span className="sev-chip text-ink-muted">
                  {moves.length} move{moves.length === 1 ? '' : 's'}
                </span>
              ) : null}
            </div>
          </div>

          <div className="flex shrink-0 flex-col items-center gap-2">
            {acknowledged ? null : (
              <>
                <span
                  aria-hidden="true"
                  className="font-mono text-[9px] uppercase tracking-[0.16em] transition-colors duration-300"
                  style={{ color: done ? '#52F2C4' : '#4A4A57' }}
                >
                  {done ? 'done' : 'do'}
                </span>
                <CheckBox
                  checked={done}
                  onToggle={onToggleDone}
                  label={`Mark "${p.headline || 'this fix'}" as done — worth ${gain} points`}
                />
              </>
            )}
            <button
              type="button"
              aria-expanded={expanded}
              aria-controls={bodyId}
              aria-label={
                expanded
                  ? `Collapse "${p.headline || 'this fix'}"`
                  : `Open "${p.headline || 'this fix'}" — what it means and how to fix it`
              }
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand();
              }}
              className="grid h-[26px] w-[26px] place-items-center rounded-lg border border-void-line text-ink-muted transition-colors duration-300 ease-cine hover:border-signal-dim hover:text-signal group-hover/card:border-ink-faint group-hover/card:text-ink-dim"
            >
              <Chevron open={expanded} />
            </button>
          </div>
        </div>

        {expanded ? (
          <motion.div
            id={bodyId}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: reduce ? 0.15 : 0.45, ease: EASE }}
            className="mt-6 space-y-7"
          >
            {first ? <FirstCardHint /> : null}

            {/* The teaching first: what the problem is, before what to type. */}
            <Explainer findingId={p.finding_id} />

            {/* Then the layer that only applies to this arrangement. */}
            {hasTrackLayer ? (
              <section className="rounded-xl2 border border-void-line/70 bg-void-panel/50 p-5 sm:p-6">
                <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-2">
                  <p className="eyebrow text-signal-dim">For this track specifically</p>
                  <span className="hairline hidden flex-1 sm:block" aria-hidden="true" />
                  <span className="stat text-[10px] uppercase tracking-[0.12em] text-ink-faint">
                    {mins} min · {diff.word}
                  </span>
                </div>

                {p.diagnosis ? (
                  <p className="max-w-[70ch] text-pretty text-[14px] leading-relaxed text-ink-dim">
                    {p.diagnosis}
                  </p>
                ) : null}

                {p.root_cause ? (
                  <div className="mt-5 flex gap-3.5">
                    <span className="eyebrow mt-[3px] shrink-0">Cause</span>
                    <p className="max-w-2xl text-[13px] leading-relaxed text-ink-muted">
                      {p.root_cause}
                    </p>
                  </div>
                ) : null}

                {moves.length ? (
                  <div className="mt-7">
                    <p className="eyebrow mb-4">Do this</p>
                    <ol className="space-y-4">
                      {moves.map((mv, i) => (
                        <MoveBlock key={`${p.finding_id}-move-${i}`} move={mv} index={i + 1} />
                      ))}
                    </ol>
                  </div>
                ) : null}

                {p.alternative ? (
                  <div className="mt-5 rounded-xl border border-void-line/70 px-4 py-3.5">
                    <p className="eyebrow">If you don’t have that</p>
                    <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">{p.alternative}</p>
                  </div>
                ) : null}

                {/* The two fields that make the advice trustworthy. */}
                {p.do_not || p.done_when ? (
                  <div className="mt-6 grid gap-4 lg:grid-cols-2">
                    {p.do_not ? (
                      <div
                        className="relative overflow-hidden rounded-xl2 border p-4 sm:p-5"
                        style={{
                          borderColor: 'rgba(255,159,28,0.34)',
                          background:
                            'linear-gradient(120deg, rgba(255,159,28,0.10), rgba(255,159,28,0.02) 60%, transparent)',
                        }}
                      >
                        <span aria-hidden="true" className="absolute inset-y-0 left-0 w-[3px] bg-sev-major" />
                        <div className="relative flex items-center gap-2.5">
                          <span className="text-[11px] text-sev-major" aria-hidden="true">
                            ▲
                          </span>
                          <p className="eyebrow text-sev-major">Do not</p>
                        </div>
                        <p className="relative mt-3 text-[13.5px] leading-relaxed text-ink">{p.do_not}</p>
                      </div>
                    ) : null}

                    {p.done_when ? (
                      <div
                        className="relative overflow-hidden rounded-xl2 border p-4 sm:p-5"
                        style={{
                          borderColor: 'rgba(82,242,196,0.30)',
                          background:
                            'linear-gradient(120deg, rgba(82,242,196,0.10), rgba(82,242,196,0.02) 60%, transparent)',
                        }}
                      >
                        <span aria-hidden="true" className="absolute inset-y-0 left-0 w-[3px] bg-signal" />
                        <div className="relative flex items-center gap-2.5">
                          <span className="text-[11px] text-signal" aria-hidden="true">
                            ✓
                          </span>
                          <p className="eyebrow text-signal-dim">Done when</p>
                        </div>
                        <p className="relative mt-3 text-[13.5px] leading-relaxed text-ink">
                          {p.done_when}
                        </p>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>
            ) : null}

            {/* And last, the proof. Kept, folded, no longer the answer. */}
            {finding && hasNumbers ? (
              teaches ? (
                <Disclosure label="The measurements" count={measurementCount(finding)}>
                  <Measurements finding={finding} />
                </Disclosure>
              ) : (
                <div>
                  <p className="eyebrow mb-3">The measurements</p>
                  <Measurements finding={finding} />
                </div>
              )
            ) : null}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onFocus();
                }}
                className="btn-ghost px-4 py-2 font-mono text-[11px] uppercase tracking-[0.13em]"
                aria-label={`Show "${p.headline || 'this fix'}" in the timeline`}
              >
                Hear it in the timeline
                <span aria-hidden="true">↗</span>
              </button>
              {acknowledged ? (
                <span className="text-[12px] leading-snug text-ink-muted">
                  Kept because you said it was deliberate. Change that answer at the top of the
                  report and it comes back into the plan.
                </span>
              ) : (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleDone();
                  }}
                  className={done ? 'btn-ghost px-4 py-2 text-[11px]' : 'btn-primary px-4 py-2 text-[11px]'}
                >
                  <span className="font-mono uppercase tracking-[0.13em]">
                    {done ? 'Reopen' : `Mark done · +${gain}`}
                  </span>
                </button>
              )}
            </div>
          </motion.div>
        ) : (
          <OpenRow onOpen={onToggleExpand} hasTeaching={teaches} acknowledged={acknowledged} />
        )}
      </div>
    </motion.article>
  );
}

/* --------------------------------------------------------------- fallback */

function FindingCard({
  f,
  index,
  expanded,
  first,
  onToggleExpand,
  onFocus,
  reduce,
}: {
  f: Finding;
  index: number;
  expanded: boolean;
  first: boolean;
  onToggleExpand: () => void;
  onFocus: () => void;
  reduce: boolean;
}) {
  const { has } = useKnowledge();
  const sev = f.severity ?? 'minor';
  const teaches = has(f.id);
  const hasNumbers = Boolean(f.detail || f.evidence?.length);
  const bodyId = `finding-body-${f.id}`;
  const acknowledged = Boolean(f.acknowledged);

  return (
    <motion.article
      id={`fix-${f.id}`}
      data-finding-id={f.id}
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 16 }}
      whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.1 }}
      transition={{ duration: reduce ? 0.25 : 0.55, ease: EASE }}
      className="panel group/card relative scroll-mt-28 overflow-hidden transition-all duration-500 ease-cine hover:border-void-line/90 hover:bg-void-raised/40"
      style={{
        opacity: acknowledged ? 0.62 : 1,
        borderColor: acknowledged ? 'rgba(82,242,196,0.28)' : undefined,
      }}
    >
      <span
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: acknowledged ? SEVERITY_VAR.clean : SEVERITY_VAR[sev] }}
      />
      <div className="p-5 pl-6 sm:p-7 sm:pl-8">
        <div className="flex items-start gap-4 sm:gap-5">
          <span
            aria-hidden="true"
            className="stat mt-[2px] shrink-0 text-[clamp(1.4rem,3vw,1.9rem)] font-semibold leading-none tracking-[-0.04em] text-ink-faint"
          >
            {acknowledged ? '·' : pad2(index + 1)}
          </span>
          <div className="min-w-0 flex-1">
            {/* Only the header toggles: a click on the prose below must not
                fold away the thing being read. */}
            <div
              className="cursor-pointer"
              onClick={() => {
                const sel = typeof window !== 'undefined' ? window.getSelection() : null;
                if (sel && sel.toString().length > 0) return;
                onToggleExpand();
              }}
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                {acknowledged ? (
                  <AcknowledgedChip />
                ) : (
                  <span className={`sev-chip ${SEV_CLASS[sev]}`}>
                    <span aria-hidden="true">{SEV_GLYPH[sev]}</span>
                    {SEV_WORD[sev]}
                  </span>
                )}
                <KindChip kind={f.kind} />
                <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                  {DIMENSION_LABELS[f.dimension] ?? humanKey(f.dimension)}
                </span>
                {f.band_hz && f.band_hz.length === 2 ? (
                  <span className="stat text-[10px] text-ink-faint">
                    {formatHz(f.band_hz[0])}–{formatHz(f.band_hz[1])} Hz
                  </span>
                ) : null}
              </div>

              <h4 className="mt-3">
                <button
                  type="button"
                  aria-expanded={expanded}
                  aria-controls={bodyId}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleExpand();
                  }}
                  className="display max-w-3xl text-balance text-left text-[clamp(1rem,2vw,1.35rem)] leading-[1.18] tracking-[-0.03em] text-ink transition-colors duration-300 ease-cine hover:text-signal group-hover/card:text-signal"
                >
                  {f.title}
                </button>
              </h4>
            </div>

            {expanded ? (
              <motion.div
                id={bodyId}
                initial={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: reduce ? 0.15 : 0.45, ease: EASE }}
                className="mt-5 space-y-6"
              >
                {first ? <FirstCardHint /> : null}

                <Explainer findingId={f.id} showTime />

                {hasNumbers ? (
                  teaches ? (
                    <Disclosure label="The measurements" count={measurementCount(f)}>
                      <Measurements finding={f} />
                    </Disclosure>
                  ) : (
                    <div>
                      <p className="eyebrow mb-3">The measurements</p>
                      <Measurements finding={f} />
                    </div>
                  )
                ) : null}

                {f.moments?.length ? (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onFocus();
                    }}
                    className="btn-ghost px-4 py-2 font-mono text-[11px] uppercase tracking-[0.13em]"
                    aria-label={`Show "${f.title}" in the timeline`}
                  >
                    Hear it in the timeline
                    <span aria-hidden="true">↗</span>
                  </button>
                ) : null}
              </motion.div>
            ) : (
              <OpenRow onOpen={onToggleExpand} hasTeaching={teaches} acknowledged={acknowledged} />
            )}
          </div>

          <button
            type="button"
            aria-expanded={expanded}
            aria-controls={bodyId}
            aria-label={
              expanded ? `Collapse "${f.title}"` : `Open "${f.title}" — what it means and how to fix it`
            }
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand();
            }}
            className="mt-[2px] grid h-[26px] w-[26px] shrink-0 place-items-center rounded-lg border border-void-line text-ink-muted transition-colors duration-300 ease-cine hover:border-signal-dim hover:text-signal group-hover/card:border-ink-faint group-hover/card:text-ink-dim"
          >
            <Chevron open={expanded} />
          </button>
        </div>
      </div>
    </motion.article>
  );
}

/* ------------------------------------------------------------------ props */

/**
 * The state of the second request, stated plainly.
 *
 * "Consulting" is a real wait — generation is output-token bound and runs for
 * minutes — so this says what is happening and makes clear the findings below
 * are already final, rather than implying the page is half-loaded.
 */
function ConsultBanner({
  status,
  error,
  onRetry,
}: {
  status: EngineerStatus;
  error: string | null;
  onRetry?: () => void;
}) {
  if (status === 'consulting') {
    return (
      <div className="shimmer flex items-start gap-3 rounded-xl border border-signal-dim/30 bg-signal/[0.04] px-4 py-3.5">
        <span
          className="mt-[3px] h-2 w-2 shrink-0 animate-breathe rounded-full bg-signal"
          aria-hidden="true"
        />
        <p className="text-[12.5px] leading-relaxed text-ink-muted" role="status" aria-live="polite">
          <span className="text-ink-dim">The engineer is writing your fix plan.</span> It reads every
          measurement before it prescribes anything, so this takes a couple of minutes. The findings
          below are already final — start on them now if you like.
        </p>
      </div>
    );
  }

  const failed = status === 'failed' || status === 'unavailable';
  return (
    <div className="flex items-start gap-3 rounded-xl border border-void-line/70 px-4 py-3.5">
      <span className="mt-[2px] shrink-0 text-[11px] text-sev-major" aria-hidden="true">
        ●
      </span>
      <div className="flex-1">
        <p className="text-[12.5px] leading-relaxed text-ink-muted">
          <span className="text-ink-dim">
            {status === 'unavailable'
              ? 'The written plan is switched off on this server.'
              : 'The written plan is unavailable for this run.'}
          </span>{' '}
          These are the measured findings straight from the analyzer — always computed locally,
          worst first. Every number below came from your file.
        </p>
        {failed && error ? (
          <p className="mt-1.5 font-mono text-micro uppercase text-ink-faint">{error}</p>
        ) : null}
      </div>
      {status === 'failed' && onRetry ? (
        <button type="button" onClick={onRetry} className="btn-ghost shrink-0 px-3 py-1.5 text-xs">
          Retry
        </button>
      ) : null}
    </div>
  );
}

export interface FixStackProps {
  analysis: MixAnalysis;
  onFocusFinding: (id: string) => void;
  /**
   * The write-up arrives on a second request, minutes after the measurements.
   * Until it lands this section shows the measured findings, so the page is
   * useful immediately rather than parking a spinner where the value is.
   */
  engineerStatus?: EngineerStatus;
  engineerError?: string | null;
  onRetryEngineer?: () => void;
}

export default function FixStack({
  analysis,
  onFocusFinding,
  engineerStatus = 'idle',
  engineerError = null,
  onRetryEngineer,
}: FixStackProps) {
  const reduce = useReducedMotion() ?? false;

  const findingById = useMemo(() => {
    const m = new Map<string, Finding>();
    for (const f of analysis.findings ?? []) m.set(f.id, f);
    return m;
  }, [analysis.findings]);

  /** Highest expected gain first — the plan is ordered by what it buys back. */
  const prescriptions = useMemo<Prescription[]>(() => {
    const list = analysis.engineer?.prescriptions ?? [];
    return [...list].sort((a, b) => {
      const g = finite(b.expected_gain) - finite(a.expected_gain);
      if (Math.abs(g) > 0.001) return g;
      const sa = findingById.get(a.finding_id)?.severity ?? 'minor';
      const sb = findingById.get(b.finding_id)?.severity ?? 'minor';
      return SEVERITY_RANK[sa] - SEVERITY_RANK[sb];
    });
  }, [analysis.engineer, findingById]);

  /**
   * A prescription against a finding the producer has confirmed is not a fix,
   * because there is nothing wrong. It comes out of the running order, out of
   * the projected score and out of the time estimate — and is kept below, with
   * everything it said, because a decision is still worth being able to read.
   */
  const { livePrescriptions, setAside } = useMemo(() => {
    const live: Prescription[] = [];
    const kept: Prescription[] = [];
    for (const p of prescriptions) {
      if (findingById.get(p.finding_id)?.acknowledged) kept.push(p);
      else live.push(p);
    }
    return { livePrescriptions: live, setAside: kept };
  }, [prescriptions, findingById]);

  const hasPlan = prescriptions.length > 0;

  const [done, setDone] = useState<Record<string, boolean>>({});
  const [expandOverride, setExpandOverride] = useState<Record<string, boolean>>({});

  const toggleDone = useCallback((key: string) => {
    setDone((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  /**
   * `defaultOpen` is what the card shows before anyone touches it, so the first
   * click on an already-open card closes it rather than appearing to do nothing.
   */
  const toggleExpand = useCallback((key: string, defaultOpen = false) => {
    setExpandOverride((prev) => ({ ...prev, [key]: !(prev[key] ?? defaultOpen) }));
  }, []);

  const base = clamp100(finite(analysis.health_score));
  const ceiling = Math.max(base, clamp100(finite(analysis.ceiling_score, base)));

  const { gainDone, pointsLeft, minutesLeft, doneCount } = useMemo(() => {
    let g = 0;
    let left = 0;
    let mins = 0;
    let count = 0;
    livePrescriptions.forEach((p, i) => {
      const key = `${p.finding_id}-${i}`;
      const val = Math.max(0, finite(p.expected_gain));
      if (done[key]) {
        g += val;
        count += 1;
      } else {
        left += val;
        mins += Math.max(0, finite(p.minutes));
      }
    });
    return { gainDone: g, pointsLeft: left, minutesLeft: mins, doneCount: count };
  }, [livePrescriptions, done]);

  const projected = Math.min(ceiling, base + gainDone);

  /** Same split as the plan, for the no-write-up path. */
  const { fallbackFindings, fallbackSetAside } = useMemo(() => {
    if (hasPlan) return { fallbackFindings: [] as Finding[], fallbackSetAside: [] as Finding[] };
    const ranked = [...(analysis.findings ?? [])]
      .filter((f) => f.severity !== 'clean')
      .sort((a, b) => {
        const s = SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity];
        if (s !== 0) return s;
        return finite(b.impact) - finite(a.impact);
      });
    return {
      fallbackFindings: ranked.filter((f) => !f.acknowledged),
      fallbackSetAside: ranked.filter((f) => f.acknowledged),
    };
  }, [hasPlan, analysis.findings]);

  const sessionPlan = analysis.engineer?.session_plan ?? [];

  const totalMinutes = useMemo(
    () => livePrescriptions.reduce((acc, p) => acc + Math.max(0, finite(p.minutes)), 0),
    [livePrescriptions],
  );

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 14 },
    whileInView: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.15 },
    transition: { duration: reduce ? 0.25 : 0.6, ease: EASE, delay: reduce ? 0 : delay },
  });

  /* ------------------------------------------------------------ fallback */

  if (!hasPlan) {
    if (!fallbackFindings.length && !fallbackSetAside.length) {
      return (
        <div className="panel flex min-h-[160px] flex-col items-center justify-center gap-3 p-8 text-center">
          <span className="text-[15px] text-sev-clean" aria-hidden="true">
            ✓
          </span>
          <p className="display text-[16px] tracking-[-0.02em] text-ink">Nothing to fix.</p>
          <p className="max-w-sm text-[13px] leading-relaxed text-ink-muted">
            No dimension came back worse than clean. Print the master.
          </p>
        </div>
      );
    }

    return (
      <div>
        <motion.div {...rise(0)} className="mb-6">
          <ConsultBanner
            status={engineerStatus}
            error={engineerError}
            onRetry={onRetryEngineer}
          />
        </motion.div>

        {fallbackFindings.length ? (
          <div className="space-y-4">
            {fallbackFindings.map((f, i) => {
              const key = `finding-${f.id}`;
              // The first one is open so the shape of a card is visible without
              // anyone having to guess there is more inside it.
              return (
                <FindingCard
                  key={f.id}
                  f={f}
                  index={i}
                  expanded={expandOverride[key] ?? i === 0}
                  first={i === 0}
                  onToggleExpand={() => toggleExpand(key, i === 0)}
                  reduce={reduce}
                  onFocus={() => onFocusFinding(f.id)}
                />
              );
            })}
          </div>
        ) : (
          <div className="panel flex min-h-[140px] flex-col items-center justify-center gap-3 p-8 text-center">
            <span className="text-[15px] text-sev-clean" aria-hidden="true">
              ✓
            </span>
            <p className="display text-[16px] tracking-[-0.02em] text-ink">Nothing left to fix.</p>
            <p className="max-w-sm text-[13px] leading-relaxed text-ink-muted">
              Everything measured on this mix was something you confirmed you meant.
            </p>
          </div>
        )}

        {fallbackSetAside.length ? (
          <div className="mt-10">
            <SetAsideHeader count={fallbackSetAside.length} />
            <div className="space-y-4">
              {fallbackSetAside.map((f, i) => {
                const key = `finding-ack-${f.id}`;
                return (
                  <FindingCard
                    key={f.id}
                    f={f}
                    index={i}
                    expanded={expandOverride[key] ?? false}
                    first={false}
                    onToggleExpand={() => toggleExpand(key, false)}
                    reduce={reduce}
                    onFocus={() => onFocusFinding(f.id)}
                  />
                );
              })}
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  /* ---------------------------------------------------------------- plan */

  return (
    <div>
      {livePrescriptions.length ? (
        <motion.div {...rise(0)}>
          <ProgressHead
            base={base}
            ceiling={ceiling}
            projected={projected}
            doneCount={doneCount}
            total={livePrescriptions.length}
            pointsLeft={pointsLeft}
            minutesLeft={minutesLeft}
            reduce={reduce}
            split={Boolean(analysis.scores)}
          />
        </motion.div>
      ) : (
        <motion.div {...rise(0)} className="panel flex flex-col items-center gap-3 p-8 text-center">
          <span className="text-[15px] text-sev-clean" aria-hidden="true">
            ✓
          </span>
          <p className="display text-[16px] tracking-[-0.02em] text-ink">Nothing left to do.</p>
          <p className="max-w-sm text-[13px] leading-relaxed text-ink-muted">
            Every prescription on this report is against something you confirmed you meant.
          </p>
        </motion.div>
      )}

      {sessionPlan.length && livePrescriptions.length ? (
        <motion.div {...rise(0.08)} className="panel mt-4 p-5 sm:p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="eyebrow">Running order</p>
            <p className="eyebrow">
              {livePrescriptions.length} fix{livePrescriptions.length === 1 ? '' : 'es'} ·{' '}
              {minutesLabel(totalMinutes)} of work
            </p>
          </div>
          <ol className="mt-4 grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
            {sessionPlan.map((step, i) => (
              <li key={`${step}-${i}`} className="flex gap-3">
                <span className="stat mt-[2px] shrink-0 text-[11px] text-signal-dim">
                  {pad2(i + 1)}
                </span>
                <span className="text-[13px] leading-snug text-ink-dim">{step}</span>
              </li>
            ))}
          </ol>
        </motion.div>
      ) : null}

      <div className="mt-6 space-y-4">
        {livePrescriptions.map((p, i) => {
          const key = `${p.finding_id}-${i}`;
          const isDone = Boolean(done[key]);
          // The top fix is open — it is the one to start on, and it shows what
          // a card contains. The rest stay shut so the plan reads as a plan
          // rather than a wall; finishing one folds it away either way.
          const defaultOpen = i === 0 && !isDone;
          const expanded = expandOverride[key] ?? defaultOpen;
          return (
            <PrescriptionCard
              key={key}
              p={p}
              index={i}
              finding={findingById.get(p.finding_id) ?? null}
              done={isDone}
              expanded={expanded}
              first={i === 0}
              reduce={reduce}
              onToggleDone={() => {
                toggleDone(key);
                setExpandOverride((prev) => {
                  const next = { ...prev };
                  delete next[key];
                  return next;
                });
              }}
              onToggleExpand={() => toggleExpand(key, defaultOpen)}
              onFocus={() => onFocusFinding(p.finding_id)}
            />
          );
        })}
      </div>

      {setAside.length ? (
        <div className="mt-10">
          <SetAsideHeader count={setAside.length} />
          <div className="space-y-4">
            {setAside.map((p, i) => {
              const key = `ack-${p.finding_id}-${i}`;
              return (
                <PrescriptionCard
                  key={key}
                  p={p}
                  index={i}
                  finding={findingById.get(p.finding_id) ?? null}
                  done={false}
                  expanded={expandOverride[key] ?? false}
                  first={false}
                  acknowledged
                  reduce={reduce}
                  onToggleDone={() => undefined}
                  onToggleExpand={() => toggleExpand(key, false)}
                  onFocus={() => onFocusFinding(p.finding_id)}
                />
              );
            })}
          </div>
        </div>
      ) : null}

      {doneCount === livePrescriptions.length && livePrescriptions.length > 0 ? (
        <motion.div
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduce ? 0.25 : 0.6, ease: EASE }}
          className="panel-raised mt-6 flex flex-col items-center gap-3 p-8 text-center"
          style={{ borderColor: 'rgba(82,242,196,0.3)' }}
        >
          <span className="text-[16px] text-signal" aria-hidden="true">
            ✓
          </span>
          <p className="display text-[clamp(1.1rem,2.4vw,1.6rem)] leading-tight tracking-[-0.03em] text-ink">
            Whole plan ticked off. Projected {Math.round(projected)}.
          </p>
          <p className="max-w-md text-[13px] leading-relaxed text-ink-muted">
            Bounce the revision and run it through again — the measurement is the only thing that
            settles whether the moves landed.
          </p>
        </motion.div>
      ) : null}
    </div>
  );
}
