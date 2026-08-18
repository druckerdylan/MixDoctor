/**
 * Clarify — the questions the report asks instead of guessing.
 *
 * "The dominant issue is low end steps back in intro" was the analyzer reading
 * an arrangement as a fault. The measurement was right and the conclusion was
 * invented: an intro written thin and a low end that went missing produce the
 * same number, and nothing in the file separates them.
 *
 * So the report asks. Four questions at most, each answerable from memory of
 * the session rather than from the session, each one carrying what a yes will
 * do before it is given. A yes moves the finding out of the score and leaves it
 * in the report as a description of the record; a no leaves it as a finding.
 *
 * It sits between the verdict and the dimension grid on purpose — early enough
 * that the answers change how everything below reads, late enough that the
 * reader knows what is being asked about. If nothing is ambiguous it renders
 * nothing at all: a manufactured question is worse than no question, because
 * the second one it asks is the one that teaches people to stop reading them.
 */

import { motion, useReducedMotion } from 'framer-motion';
import { useRef } from 'react';

import useClarifications from '../../hooks/useClarifications';
import { DIMENSION_LABELS, type Finding, type MixAnalysis } from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const SIGNAL_BORDER = 'rgba(82,242,196,0.55)';
const SIGNAL_FILL = 'rgba(82,242,196,0.12)';
const MAJOR_BORDER = 'rgba(255,159,28,0.42)';
const MAJOR_FILL = 'rgba(255,159,28,0.10)';

function pad2(n: number): string {
  return n.toString().padStart(2, '0');
}

function finite(v: number | undefined | null): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/* ----------------------------------------------------------------- pieces */

function Tick() {
  return (
    <svg viewBox="0 0 14 14" width="12" height="12" aria-hidden="true" fill="none">
      <path
        d="M2.4 7.3 5.5 10.3 11.6 3.8"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * One of the two answers. Both stay on screen after either is picked — that is
 * the whole of "reversible", and it costs nothing but the width it already has.
 */
function Choice({
  pressed,
  tone,
  label,
  hint,
  onClick,
}: {
  pressed: boolean;
  tone: 'yes' | 'no';
  label: string;
  hint: string;
  onClick: () => void;
}) {
  const yes = tone === 'yes';
  return (
    <button
      type="button"
      aria-pressed={pressed}
      aria-label={hint}
      onClick={onClick}
      className="btn-ghost px-4 py-2 text-[12.5px] font-medium tracking-normal transition-all duration-300 ease-cine"
      style={
        pressed
          ? {
              borderColor: yes ? SIGNAL_BORDER : MAJOR_BORDER,
              background: yes ? SIGNAL_FILL : MAJOR_FILL,
              color: '#F4F4F7',
            }
          : undefined
      }
    >
      <span
        aria-hidden="true"
        className="grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border transition-colors duration-300"
        style={{
          borderColor: pressed ? (yes ? '#52F2C4' : '#FF9F1C') : '#4A4A57',
          color: yes ? '#52F2C4' : '#FF9F1C',
        }}
      >
        {pressed ? <Tick /> : null}
      </span>
      {label}
    </button>
  );
}

/**
 * What the answer did, in the report's own words.
 *
 * `if_intended` and `if_not` are written by the backend next to the question
 * precisely so a promise made before the click is the one kept after it. This
 * shows whichever one came true, and never paraphrases either.
 */
function Outcome({ intended, finding, reduce }: { intended: boolean; finding: Finding; reduce: boolean }) {
  const c = finding.clarification;
  if (!c) return null;

  return (
    <motion.div
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0.15 : 0.4, ease: EASE }}
      className="mt-4 overflow-hidden rounded-xl border px-4 py-3.5 lg:max-w-[78ch]"
      style={{
        borderColor: intended ? 'rgba(82,242,196,0.28)' : 'rgba(255,159,28,0.26)',
        background: intended
          ? 'linear-gradient(120deg, rgba(82,242,196,0.08), transparent 62%)'
          : 'linear-gradient(120deg, rgba(255,159,28,0.07), transparent 62%)',
      }}
    >
      <p
        className="eyebrow flex items-center gap-2"
        style={{ color: intended ? '#52F2C4' : '#FF9F1C' }}
      >
        <span aria-hidden="true">{intended ? '✓' : '●'}</span>
        {intended ? 'Moved out of the score' : 'Kept as a finding'}
      </p>
      <p className="mt-2 max-w-[72ch] text-pretty text-[12.5px] leading-relaxed text-ink-dim">
        {intended ? c.if_intended : c.if_not}
      </p>
    </motion.div>
  );
}

function QuestionCard({
  finding,
  index,
  answered,
  onAnswer,
  reduce,
}: {
  finding: Finding;
  index: number;
  /** `undefined` means unanswered, which is a third state and not a no. */
  answered: boolean | undefined;
  onAnswer: (intended: boolean) => void;
  reduce: boolean;
}) {
  const c = finding.clarification;
  if (!c) return null;

  const isAnswered = answered !== undefined;
  const questionId = `clarify-q-${finding.id}`;

  return (
    <motion.li
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 14 }}
      animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0.2 : 0.5, ease: EASE, delay: reduce ? 0 : 0.05 * index }}
      className="panel relative overflow-hidden p-5 transition-colors duration-500 ease-cine sm:p-6"
      style={{
        borderColor: answered === true ? 'rgba(82,242,196,0.3)' : undefined,
      }}
    >
      <span
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-[2px] transition-colors duration-500"
        style={{
          background:
            answered === true ? '#52F2C4' : answered === false ? '#FF9F1C' : 'rgba(255,255,255,0.06)',
        }}
      />

      <div className="flex items-start gap-4 sm:gap-5">
        <span
          aria-hidden="true"
          className="stat mt-[1px] shrink-0 text-[15px] leading-none tracking-[-0.03em] text-ink-faint"
        >
          {pad2(index + 1)}
        </span>

        <div className="min-w-0 flex-1">
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
            {DIMENSION_LABELS[finding.dimension] ?? finding.dimension}
          </p>

          <h4
            id={questionId}
            className="display mt-2.5 text-balance text-[clamp(1rem,1.7vw,1.22rem)] leading-[1.25] tracking-[-0.02em] text-ink"
          >
            {c.question}
          </h4>

          <p className="mt-3 text-pretty text-[12.5px] leading-relaxed text-ink-muted">
            {c.context}
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-2.5" role="group" aria-labelledby={questionId}>
            <Choice
              pressed={answered === true}
              tone="yes"
              label="Yes, on purpose"
              hint={`Yes, on purpose — ${c.question}`}
              onClick={() => onAnswer(true)}
            />
            <Choice
              pressed={answered === false}
              tone="no"
              label="No, that’s a problem"
              hint={`No, that is a problem — ${c.question}`}
              onClick={() => onAnswer(false)}
            />
            {!isAnswered ? (
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint/70">
                unanswered · counts against the score
              </span>
            ) : null}
          </div>

          {isAnswered ? <Outcome intended={answered} finding={finding} reduce={reduce} /> : null}
        </div>
      </div>
    </motion.li>
  );
}

/* ------------------------------------------------------------------ props */

export interface ClarifyProps {
  analysis: MixAnalysis;
  /**
   * Where the re-scored report goes. Without it the questions would be a
   * survey — asked, answered, and changing nothing — so the section renders
   * nothing at all rather than pretend.
   */
  onAnalysisChange?: (next: MixAnalysis) => void;
}

export default function Clarify({ analysis, onAnalysisChange }: ClarifyProps) {
  const reduce = useReducedMotion() ?? false;
  const { questions, answers, answeredCount, status, error, answer, acknowledgeAll, undoAll } =
    useClarifications(analysis, onAnalysisChange);

  /**
   * Where the reference match stood before any of this was answered, so the
   * section can show the movement rather than only the new number. Captured on
   * the first render of a report and never updated — it is a starting line.
   */
  const baseline = useRef<number | null>(finite(analysis.scores?.reference_match));

  if (!onAnalysisChange || questions.length === 0) return null;

  const total = questions.length;
  const outstanding = total - answeredCount;
  const match = finite(analysis.scores?.reference_match);
  const gained =
    baseline.current !== null && match !== null ? Math.round(match - baseline.current) : 0;

  return (
    <motion.section
      id="results-clarify"
      aria-labelledby="clarify-heading"
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18 }}
      animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0.3 : 0.7, ease: EASE, delay: reduce ? 0 : 0.1 }}
      className="mt-12 scroll-mt-32 sm:mt-16"
    >
      <div className="hairline mb-8 sm:mb-10" />

      <div className="mb-6 flex flex-wrap items-end justify-between gap-x-10 gap-y-4">
        <div className="min-w-0">
          <p className="eyebrow">
            <span className="text-ink-muted">Before the rest</span>
            <span className="mx-2 text-ink-faint/60" aria-hidden="true">
              /
            </span>
            {total} question{total === 1 ? '' : 's'}
          </p>
          <h2
            id="clarify-heading"
            className="display mt-3.5 max-w-[24ch] text-balance text-[clamp(1.4rem,3vw,2.1rem)] leading-[1.04] tracking-[-0.035em] text-ink"
          >
            Some of this might be on purpose.
          </h2>
          <p className="mt-3.5 max-w-2xl text-pretty text-[13.5px] leading-relaxed text-ink-muted">
            A decision and a mistake can measure exactly the same. Rather than call these faults,
            the report is asking. A yes leaves the finding in as a description of the record and
            stops it counting against the score; nothing you say here can excuse a real defect, and
            you can change any answer at any time.
          </p>
        </div>

        <div className="flex shrink-0 flex-col items-start gap-3 sm:items-end">
          <p className="stat text-[11px] uppercase tracking-[0.12em] text-ink-faint" aria-live="polite">
            {answeredCount} of {total} answered
            {gained > 0 ? (
              <span className="ml-2 text-signal">reference match +{gained}</span>
            ) : null}
          </p>
          {outstanding > 0 ? (
            <button
              type="button"
              onClick={acknowledgeAll}
              className="btn-ghost px-4 py-2 font-mono text-[11px] uppercase tracking-[0.13em]"
            >
              Nothing here is a mistake
            </button>
          ) : undoAll ? (
            <button
              type="button"
              onClick={undoAll}
              className="btn-ghost px-4 py-2 font-mono text-[11px] uppercase tracking-[0.13em]"
            >
              Undo — ask me again
            </button>
          ) : null}
        </div>
      </div>

      {/* A reading column rather than the full page width. Four questions laid
          across 1300px reads as a form to complete; this reads as being asked. */}
      <ol className="max-w-[62rem] space-y-3.5">
        {questions.map((finding, i) => (
          <QuestionCard
            key={finding.id}
            finding={finding}
            index={i}
            answered={finding.id in answers ? answers[finding.id] : undefined}
            onAnswer={(intended) => answer(finding.id, intended)}
            reduce={reduce}
          />
        ))}
      </ol>

      <div className="mt-4 flex min-h-[18px] flex-wrap items-center gap-x-3 gap-y-1" aria-live="polite">
        {status === 'saving' ? (
          <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
            <span className="h-1.5 w-1.5 animate-breathe rounded-full bg-signal" aria-hidden="true" />
            Re-scoring the report
          </span>
        ) : null}
        {status === 'error' && error ? (
          <span className="flex items-center gap-2 text-[12px] leading-snug text-sev-major">
            <span aria-hidden="true">▲</span>
            {error} Your answer was not saved — try it again.
          </span>
        ) : null}
        {status === 'idle' && answeredCount === total ? (
          <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
            All answered — the report below reads with your answers applied
          </span>
        ) : null}
      </div>
    </motion.section>
  );
}
