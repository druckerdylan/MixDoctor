import { useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import ScoreRing from './ScoreRing';
import type { EngineerStatus } from '../../hooks/useAnalysis';
import {
  DIMENSION_LABELS,
  SEVERITY_VAR,
  TRACK_INTENT_LABELS,
  TRACK_INTENT_SHORT,
  formatTime,
  severityFromScore,
  type DimensionScore,
  type Finding,
  type MixAnalysis,
  type ScoreCard,
  type Severity,
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
  major: 'Needs work',
  minor: 'Nearly there',
  clean: 'Clean',
};

/** Lower-case a title for use mid-sentence, unless it opens with an acronym. */
function decap(s: string): string {
  const trimmed = s.trim();
  if (!trimmed) return trimmed;
  const first = trimmed.split(/\s+/)[0] ?? '';
  if (first.length > 1 && first === first.toUpperCase()) return trimmed;
  return trimmed.charAt(0).toLowerCase() + trimmed.slice(1);
}

function stripPeriod(s: string): string {
  return s.trim().replace(/[.]+$/, '');
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

/** Everything still counting. An acknowledged finding is not an issue. */
function liveFindings(a: MixAnalysis): Finding[] {
  return [...(a.findings ?? [])]
    .filter((f) => !f.acknowledged)
    .sort((x, y) => y.impact - x.impact);
}

/**
 * The mastering sentence, which is the same either way.
 */
function masteringSentence(a: MixAnalysis): string {
  if (a.mastering_ready) return 'It is ready for mastering as it stands.';
  const blockers = a.mastering_blockers?.length ?? 0;
  return blockers > 0
    ? `${plural(blockers, 'thing')} still block${blockers === 1 ? 's' : ''} mastering.`
    : 'It is not yet ready for mastering.';
}

/**
 * Deterministic stand-in for the engineer's verdict when the AI layer is
 * unavailable. Built only from measured numbers, so it is never wrong.
 *
 * The old version opened with the composite and then named the highest-impact
 * finding as "the dominant issue" — which is how an intro written thin got
 * reported as the main problem with a record. It now separates the two
 * questions the way the score does: a defect is an issue, a deviation is a
 * difference, and something the producer has confirmed is neither.
 */
function buildScoredVerdict(a: MixAnalysis, scores: ScoreCard): string {
  const live = liveFindings(a);
  const defects = live.filter((f) => f.kind === 'defect');
  const deviations = live.filter((f) => f.kind !== 'defect');
  const out: string[] = [];

  out.push(
    defects.length === 0
      ? `Technically it reads ${scores.technical_grade} — nothing measurably wrong with the render.`
      : `Technically it reads ${scores.technical_grade}, with ${plural(defects.length, 'defect')} measured.`,
  );

  if (defects.length) {
    out.push(`Clear ${decap(stripPeriod(defects[0].title))} first.`);
  } else if (deviations.length) {
    out.push(
      `The largest difference from the ${a.genre || 'genre'} reference is ` +
        `${decap(stripPeriod(deviations[0].title))} — a difference, not a fault.`,
    );
  } else {
    out.push('Nothing measured outside its window.');
  }

  if (scores.acknowledged > 0) {
    out.push(
      `${plural(scores.acknowledged, 'finding')} you confirmed as deliberate ` +
        `${scores.acknowledged === 1 ? 'is' : 'are'} held aside.`,
    );
  }

  out.push(masteringSentence(a));
  return out.join(' ');
}

/** The pre-split version, kept intact for a payload with no `scores`. */
function buildLegacyVerdict(a: MixAnalysis): string {
  const findings = [...(a.findings ?? [])].sort((x, y) => y.impact - x.impact);
  const dims = [...(a.dimensions ?? [])].sort((x, y) => x.score - y.score);
  const weakest = dims.length ? dims[0] : undefined;
  const score = Math.round(a.health_score);
  const ceiling = Math.round(a.ceiling_score);
  const out: string[] = [];

  out.push(`This mix lands at ${score} out of 100 — grade ${a.grade || '—'}.`);

  if (findings.length >= 2) {
    out.push(
      `The dominant issue is ${decap(stripPeriod(findings[0].title))}, with ` +
        `${decap(stripPeriod(findings[1].title))} close behind.`,
    );
  } else if (findings.length === 1) {
    out.push(`The one flagged issue is ${decap(stripPeriod(findings[0].title))}.`);
  } else {
    out.push('Nothing crossed the threshold for a flagged issue.');
  }

  if (weakest && weakest.severity !== 'clean') {
    out.push(
      `${DIMENSION_LABELS[weakest.dimension]} is the weakest link at ${Math.round(weakest.score)}.`,
    );
  } else if (dims.length) {
    out.push('Every dimension reads clean.');
  }

  if (ceiling > score + 1) {
    // No article before the number — "a 84" and "an 84" are both traps.
    out.push(`Work the prescriptions and this becomes ${ceiling}.`);
  }

  out.push(masteringSentence(a));
  return out.join(' ');
}

/**
 * Split off the opening sentence so it can be set brighter than the rest.
 * The terminator must follow a word character, a grade suffix ("C+.") or a
 * closing bracket, and be followed by whitespace and a capital — which keeps
 * decimals like "4.2 dB" and "-6.8 LUFS" from being mistaken for full stops.
 */
function splitLead(text: string): [string, string] {
  const m = /^([\s\S]*?[A-Za-z0-9)"'’”+%][.!?])\s+(?=["'“(‘]?[A-Z0-9])([\s\S]+)$/.exec(
    text.trim(),
  );
  if (!m) return [text.trim(), ''];
  return [m[1], m[2]];
}

function CheckMark() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" fill="none">
      <circle cx="8" cy="8" r="6.4" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M5.2 8.2 7.1 10.1l3.8-4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function AlertMark() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" fill="none">
      <path
        d="M8 2.2 14.6 13.4H1.4L8 2.2Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M8 6.3v3.1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="8" cy="11.3" r="0.75" fill="currentColor" />
    </svg>
  );
}

/**
 * How close this is to the genre reference — and nothing more than that.
 *
 * Deliberately the calmest thing on the page. No grade, no severity colour, no
 * red: a low number here means the record does not sound like the reference,
 * which is a description and is often the entire point. The composite that
 * this replaces let exactly this figure drag a defect-free, master-ready mix
 * down to a D-, which is how a score stops meaning anything.
 */
function ReferenceMatch({ scores, genre }: { scores: ScoreCard; genre: string }) {
  const match = Math.max(0, Math.min(100, scores.reference_match));
  const outstanding = Math.max(0, scores.deviations - scores.acknowledged);

  return (
    <div className="panel w-full p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="eyebrow text-ink-muted">Reference match</p>
        <p className="eyebrow text-ink-faint/70">not a grade</p>
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="stat text-[26px] leading-none tracking-[-0.03em] text-ink-dim">
          {Math.round(match)}
        </span>
        <span className="stat text-[11px] leading-none text-ink-faint">/100</span>
      </div>

      <div
        className="mt-3 h-[3px] w-full overflow-hidden rounded-full bg-void-line/60"
        role="img"
        aria-label={`Reference match ${Math.round(match)} out of 100. ${scores.reference_label}. Different is not worse.`}
      >
        <div
          className="h-full rounded-full bg-ink-faint/70"
          style={{ width: `${match}%` }}
          aria-hidden="true"
        />
      </div>

      <p className="mt-3.5 text-[13px] leading-snug text-ink-dim">{scores.reference_label}</p>
      <p className="mt-2 text-[12px] leading-relaxed text-ink-muted">
        Different is not worse. This says how close the record sits to what {genre || 'the genre'}{' '}
        usually measures, so you know which one you are doing — it is never counted as damage.
      </p>

      {scores.deviations > 0 ? (
        <p className="mt-3 border-t border-void-lineSoft pt-3 font-mono text-[10px] uppercase tracking-[0.13em] text-ink-faint">
          {outstanding} difference{outstanding === 1 ? '' : 's'} measured
          {scores.acknowledged > 0 ? (
            <span className="text-sev-clean"> · {scores.acknowledged} confirmed as yours</span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

export interface VerdictProps {
  analysis: MixAnalysis;
  /** So the caption can say "still writing" rather than "unavailable". */
  engineerStatus?: EngineerStatus;
}

export default function Verdict({ analysis, engineerStatus = 'idle' }: VerdictProps) {
  const reduce = useReducedMotion() ?? false;
  const engineer = analysis.engineer ?? null;

  const rise = (delay: number) => ({
    initial: reduce ? { opacity: 0 } : { opacity: 0, y: 20 },
    animate: reduce ? { opacity: 1 } : { opacity: 1, y: 0 },
    transition: { duration: reduce ? 0.25 : 0.75, ease: EASE, delay: reduce ? 0 : delay },
  });

  const scores = analysis.scores ?? null;

  /**
   * Acknowledged findings are out. "The one thing" is the next thing to do, and
   * a decision the producer has already confirmed is not a thing to do.
   */
  const topFindings = useMemo<Finding[]>(() => liveFindings(analysis), [analysis]);

  const cleanDims = useMemo<DimensionScore[]>(
    () =>
      [...(analysis.dimensions ?? [])]
        .filter((d) => d.severity === 'clean')
        .sort((x, y) => y.score - x.score),
    [analysis.dimensions],
  );

  const verdictText = engineer?.verdict?.trim()
    ? engineer.verdict.trim()
    : scores
      ? buildScoredVerdict(analysis, scores)
      : buildLegacyVerdict(analysis);
  const usingFallback = !engineer?.verdict?.trim();
  const [lead, rest] = splitLead(verdictText);

  const oneThing =
    engineer?.the_one_thing?.trim() ||
    (topFindings.length ? stripPeriod(topFindings[0].title) : null);
  const oneThingSource = engineer?.the_one_thing?.trim()
    ? 'Engineer’s call'
    : topFindings.length
      ? `Highest impact · ${DIMENSION_LABELS[topFindings[0].dimension]}`
      : null;

  const strengths =
    engineer?.strengths?.length
      ? engineer.strengths
      : cleanDims.slice(0, 4).map((d) => DIMENSION_LABELS[d.dimension]);

  const blockers = analysis.mastering_blockers ?? [];
  const ready = analysis.mastering_ready;
  // The legacy composite's severity. Only ever used on a payload with no
  // `scores` — with one, the technical grade is the honest version of this.
  const sev = severityFromScore(analysis.health_score);
  const warnings = analysis.warnings ?? [];

  const technicalSev = scores ? severityFromScore(scores.technical) : sev;
  const technicalNote = scores
    ? scores.defects === 0
      ? 'Nothing measurably wrong with the render.'
      : `${plural(scores.defects, 'defect')} — wrong regardless of genre, intent or taste.`
    : '';

  const durationLabel = Number.isFinite(analysis.measurements?.duration_seconds)
    ? formatTime(analysis.measurements.duration_seconds)
    : null;

  return (
    <section className="relative">
      {/* Meta line */}
      <motion.div
        {...rise(0.05)}
        className="flex flex-wrap items-center gap-x-3 gap-y-2 text-ink-faint"
      >
        <span className="max-w-[16rem] truncate font-mono text-eyebrow uppercase text-ink-dim sm:max-w-[26rem]">
          {analysis.filename || 'Untitled mix'}
        </span>
        {analysis.genre ? (
          <>
            <span aria-hidden="true" className="text-ink-faint/60">
              /
            </span>
            <span className="eyebrow">{analysis.genre}</span>
          </>
        ) : null}
        {/* What the analyser assumed the file was. Next to the genre because
            the two together are the whole frame the report was written in —
            and because a reader who sees "judged as a full mix" over a beat
            now knows the fix is one radio button, not an argument. */}
        {analysis.intent ? (
          <>
            <span aria-hidden="true" className="text-ink-faint/60">
              /
            </span>
            <span className="eyebrow" title={TRACK_INTENT_LABELS[analysis.intent]}>
              <span className="text-ink-faint/70">judged as</span>{' '}
              {TRACK_INTENT_SHORT[analysis.intent]}
            </span>
          </>
        ) : null}
        {durationLabel ? (
          <>
            <span aria-hidden="true" className="text-ink-faint/60">
              /
            </span>
            <span className="stat text-eyebrow uppercase text-ink-faint">{durationLabel}</span>
          </>
        ) : null}
        <span aria-hidden="true" className="text-ink-faint/60">
          /
        </span>
        <span className="stat text-eyebrow uppercase text-ink-faint">
          analysed in {Math.round(analysis.analysis_ms)}ms
        </span>
      </motion.div>

      {/* Two columns from lg: the words on the left, the instrument on the right.
          With `scores` the headline takes the first row and the ring drops
          beside it, so the narrow-screen reading order is headline → technical
          health → reference match → the engineer's read → the one thing →
          strengths → readiness. Without it, the pre-split layout is unchanged. */}
      <div className="mt-7 grid grid-cols-1 items-start gap-x-14 gap-y-8 lg:mt-9 lg:grid-cols-[minmax(0,1fr)_minmax(300px,380px)]">
        {/* The answer to "how did I do". Everything else on this page is a
            qualification of this sentence, so nothing may be set larger. */}
        {scores ? (
          <motion.div {...rise(0.08)} className="order-1 min-w-0 lg:col-start-1 lg:row-start-1">
            <h2 className="display max-w-[20ch] text-balance text-[clamp(1.75rem,4.4vw,3.1rem)] leading-[0.98] tracking-[-0.04em] text-ink">
              {scores.headline}
            </h2>

            <div className="mt-6">
              <p className="eyebrow text-ink-muted">Is anything actually wrong?</p>
              <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
                <span className={`sev-chip ${SEV_CLASS[technicalSev]}`}>
                  <span
                    aria-hidden="true"
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: SEVERITY_VAR[technicalSev] }}
                  />
                  Technical health {scores.technical_grade}
                </span>
                <span className="stat text-[12px] text-ink-faint">
                  {Math.round(scores.technical)}/100
                </span>
                <span className="text-[13px] leading-snug text-ink-dim">{technicalNote}</span>
              </div>
            </div>
          </motion.div>
        ) : null}

        <motion.div
          {...rise(0.1)}
          className={`flex flex-col items-center gap-5 lg:items-end lg:pt-1 ${
            scores
              ? 'order-2 lg:col-start-2 lg:row-span-2 lg:row-start-1'
              : 'order-1 lg:col-start-2 lg:row-start-1'
          }`}
        >
          {scores ? (
            <ScoreRing
              score={scores.technical}
              grade={scores.technical_grade}
              label="TECHNICAL"
              caption={scores.defects === 0 ? 'NO DEFECTS' : `${scores.defects} DEFECT${scores.defects === 1 ? '' : 'S'}`}
              size={320}
            />
          ) : (
            <ScoreRing
              score={analysis.health_score}
              ceiling={analysis.ceiling_score}
              grade={analysis.grade}
              size={320}
            />
          )}

          {scores ? <ReferenceMatch scores={scores} genre={analysis.genre} /> : null}
        </motion.div>

        <div
          className={`min-w-0 ${
            scores ? 'order-3 lg:col-start-1 lg:row-start-2' : 'order-2 lg:col-start-1 lg:row-start-1'
          }`}
        >
          {/* The verdict */}
          <motion.div {...rise(0.32)}>
            <div className="mb-4 flex flex-wrap items-center gap-3">
              {/* The composite's severity chip only survives where the split
                  score does not. With `scores` on the payload, the technical
                  grade above already answers this and answers it honestly. */}
              {scores ? null : (
                <span className={`sev-chip ${SEV_CLASS[sev]}`}>
                  <span
                    aria-hidden="true"
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: SEVERITY_VAR[sev] }}
                  />
                  {SEV_WORD[sev]}
                </span>
              )}
              <span className="eyebrow">
                {usingFallback ? 'Automated summary' : 'The engineer’s read'}
              </span>
            </div>

            <p
              className={`text-pretty font-display font-medium leading-[1.16] tracking-[-0.03em] text-ink ${
                scores
                  ? 'text-[clamp(1.05rem,2vw,1.45rem)]'
                  : 'text-[clamp(1.3rem,2.7vw,2.1rem)]'
              }`}
            >
              {lead}
              {rest ? <span className="text-ink-dim"> {rest}</span> : null}
            </p>

            {usingFallback ? (
              <p className="mt-3 flex items-center gap-2 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                {engineerStatus === 'consulting' ? (
                  <>
                    <span
                      className="h-1.5 w-1.5 animate-breathe rounded-full bg-signal"
                      aria-hidden="true"
                    />
                    <span role="status" aria-live="polite">
                      Measured summary — the engineer&rsquo;s read is being written
                    </span>
                  </>
                ) : (
                  'Engineer notes unavailable — summary generated from the measurements'
                )}
              </p>
            ) : null}
          </motion.div>

          {/* THE ONE THING */}
          {oneThing ? (
            <motion.div
              {...rise(0.48)}
              className="relative mt-7 overflow-hidden rounded-xl2 border bg-void-panel p-5 shadow-panel sm:p-6"
              style={{ borderColor: 'rgba(82,242,196,0.26)' }}
            >
              <span
                aria-hidden="true"
                className="absolute inset-y-0 left-0 w-[3px] bg-grade-signal"
              />
              <span
                aria-hidden="true"
                className="pointer-events-none absolute inset-0"
                style={{
                  background:
                    'radial-gradient(115% 90% at 0% 0%, rgba(82,242,196,0.09), transparent 62%)',
                }}
              />
              <div className="relative flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                <p className="eyebrow text-signal-dim">The one thing</p>
                {oneThingSource ? (
                  <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                    {oneThingSource}
                  </p>
                ) : null}
              </div>
              <p className="display relative mt-3 text-pretty text-[clamp(1.05rem,2vw,1.5rem)] leading-[1.22] tracking-[-0.02em] text-ink">
                {oneThing}
              </p>
            </motion.div>
          ) : null}

          {/* Strengths */}
          {strengths.length ? (
            <motion.div {...rise(0.6)} className="mt-7">
              <p className="eyebrow text-ink-muted">What is already working</p>
              <ul className="mt-3 flex flex-wrap gap-2">
                {strengths.map((s, i) => (
                  <li key={`${s}-${i}`}>
                    <span className="sev-chip sev-clean max-w-full normal-case">
                      <span className="shrink-0 text-sev-clean">
                        <CheckMark />
                      </span>
                      <span className="truncate font-sans text-[12px] tracking-normal text-ink-dim">
                        {s}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ) : null}
        </div>

        {/* Mastering readiness — sits under the ring on wide screens. */}
        <motion.div
          {...rise(0.72)}
          className={`panel p-5 sm:p-6 lg:col-span-2 ${
            scores ? 'order-4 lg:row-start-3' : 'order-3 lg:row-start-2'
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="eyebrow text-ink-muted">Mastering readiness</p>
              <span className={`sev-chip ${ready ? 'sev-clean' : 'sev-major'}`}>
                <span className="shrink-0">{ready ? <CheckMark /> : <AlertMark />}</span>
                {ready ? 'Ready' : 'Not ready'}
              </span>
            </div>

            {engineer?.mastering_verdict?.trim() ? (
              <p className="mt-4 text-[13.5px] leading-relaxed text-ink-dim">
                {engineer.mastering_verdict.trim()}
              </p>
            ) : (
              <p className="mt-4 text-[13.5px] leading-relaxed text-ink-dim">
                {ready
                  ? 'No blocking problems were measured. Print the master.'
                  : 'Clear the blockers below before this goes to a mastering chain.'}
              </p>
            )}

            {blockers.length ? (
              <>
                <div className="hairline my-4" />
                <p className="eyebrow">
                  {blockers.length} blocker{blockers.length === 1 ? '' : 's'}
                </p>
                <ol className="mt-3 grid gap-x-8 gap-y-2.5 sm:grid-cols-2 lg:grid-cols-3">
                  {blockers.map((b, i) => (
                    <li key={`${b}-${i}`} className="flex gap-3">
                      <span className="stat mt-[3px] shrink-0 text-micro text-sev-major">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span className="text-[13px] leading-snug text-ink-dim">{b}</span>
                    </li>
                  ))}
                </ol>
              </>
            ) : null}
        </motion.div>

        {/* Analyser warnings — honest about the limits of the measurement. */}
        {warnings.length ? (
          <motion.div
            {...rise(0.84)}
            className={`flex items-start gap-2.5 rounded-xl border border-void-line/70 px-4 py-3 lg:col-span-2 ${
              scores ? 'order-5 lg:row-start-4' : 'order-4 lg:row-start-3'
            }`}
          >
            <span className="mt-[3px] shrink-0 text-sev-major">
              <AlertMark />
            </span>
            <p className="text-[12.5px] leading-snug text-ink-muted">
              {warnings[0]}
              {warnings.length > 1 ? (
                <span className="stat text-ink-faint"> (+{warnings.length - 1} more)</span>
              ) : null}
            </p>
          </motion.div>
        ) : null}
      </div>
    </section>
  );
}
