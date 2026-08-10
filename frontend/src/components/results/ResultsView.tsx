/**
 * ResultsView — the report, ordered as an argument rather than a dashboard.
 *
 * Verdict states the case. The map and the grid show where the damage is. The
 * timeline proves it against the audio, and the section map proves it one level
 * up, across the arrangement. If deep analysis ran, the stem panel makes the
 * distinct claim that only separation can make — this element, on its own. The
 * spectrum shows the shape of it. The fix stack is what you do about it. The
 * scope, the delivery targets and the full measurement set are the receipts.
 *
 * Two cross-links hold it together: a shared `selected` dimension that the map,
 * the grid and the timeline all key off, and a scroll bridge so a fix can point
 * at the moment in the audio it came from, and a seek can point back at the fix.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';

import { DonatePanel } from '../Donate';
import Verdict from './Verdict';
import Clarify from './Clarify';
import MixMap from './MixMap';
import DimensionGrid from './DimensionGrid';
import Timeline from './Timeline';
import SectionMap from './SectionMap';
import StemBalance from './StemBalance';
import SpectrumCurve from './SpectrumCurve';
import FixStack from './FixStack';
import type { EngineerStatus } from '../../hooks/useAnalysis';
import StereoScope from './StereoScope';
import LoudnessTargets from './LoudnessTargets';
import MetricsTable from './MetricsTable';
import ReportDownload from './ReportDownload';
import { usePluginVault } from '../../hooks/usePluginVault';

import {
  SEVERITY_RANK,
  SEVERITY_VAR,
  severityFromScore,
  type Dimension,
  type Finding,
  type MixAnalysis,
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

const FLASH_MS = 1500;
const FLASH_OUTLINE = '1px solid rgba(82,242,196,0.55)';

function finite(v: number | undefined | null, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/* --------------------------------------------------------------- section */

interface SectionProps {
  id: string;
  index: string;
  eyebrow: string;
  title: string;
  blurb?: string;
  aside?: ReactNode;
  children: ReactNode;
  reduce: boolean;
  /** First section skips the leading rule — the verdict already opened the page. */
  rule?: boolean;
  innerRef?: React.RefObject<HTMLElement>;
}

function Section({
  id,
  index,
  eyebrow,
  title,
  blurb,
  aside,
  children,
  reduce,
  rule = true,
  innerRef,
}: SectionProps) {
  return (
    <motion.section
      id={id}
      ref={innerRef}
      className="scroll-mt-32 rounded-xl2 transition-shadow duration-500 ease-cine"
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 24 }}
      whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.06 }}
      transition={{ duration: reduce ? 0.3 : 0.75, ease: EASE }}
    >
      {rule ? <div className="hairline mb-10 sm:mb-14" /> : null}

      <header className="mb-6 flex flex-wrap items-end justify-between gap-x-10 gap-y-4 sm:mb-8">
        <div className="min-w-0">
          <p className="eyebrow">
            <span className="text-ink-muted">{index}</span>
            <span className="mx-2 text-ink-faint/60" aria-hidden="true">
              /
            </span>
            {eyebrow}
          </p>
          <h2 className="display mt-3.5 text-[clamp(1.6rem,3.4vw,2.5rem)] leading-[0.98] tracking-[-0.035em] text-ink">
            {title}
          </h2>
          {blurb ? (
            <p className="mt-3.5 max-w-2xl text-pretty text-[13.5px] leading-relaxed text-ink-muted">
              {blurb}
            </p>
          ) : null}
        </div>
        {aside ? <div className="shrink-0">{aside}</div> : null}
      </header>

      {children}
    </motion.section>
  );
}

/* ------------------------------------------------------------------ props */

export interface ResultsViewProps {
  analysis: MixAnalysis;
  audioUrl: string | null;
  onReset: () => void;
  /** The write-up is fetched separately and lands after the measurements. */
  engineerStatus?: EngineerStatus;
  engineerError?: string | null;
  onRetryEngineer?: () => void;
  /**
   * Where a re-scored report goes when the producer answers a question about
   * what was deliberate. Without it the question section renders nothing — an
   * answer that changes nothing is not worth asking for.
   */
  onAnalysisChange?: (next: MixAnalysis) => void;
}

export default function ResultsView({
  analysis,
  audioUrl,
  onReset,
  engineerStatus = 'idle',
  engineerError = null,
  onRetryEngineer,
  onAnalysisChange,
}: ResultsViewProps) {
  const reduce = useReducedMotion() ?? false;

  // The document resolves its fix steps against what this producer owns, the
  // same way the fix stack on screen does.
  const { plugins } = usePluginVault();

  const [selected, setSelected] = useState<Dimension | null>(null);

  const timelineRef = useRef<HTMLElement>(null);
  const fixStackRef = useRef<HTMLElement>(null);
  const lastSeekMatch = useRef<string | null>(null);
  const flashTimer = useRef<number | null>(null);
  const flashTarget = useRef<HTMLElement | null>(null);

  const behavior: ScrollBehavior = reduce ? 'auto' : 'smooth';

  const clearFlash = useCallback(() => {
    if (flashTimer.current !== null) {
      window.clearTimeout(flashTimer.current);
      flashTimer.current = null;
    }
    const prev = flashTarget.current;
    if (prev) {
      prev.style.outline = '';
      prev.style.outlineOffset = '';
      flashTarget.current = null;
    }
  }, []);

  useEffect(() => clearFlash, [clearFlash]);

  /**
   * A short outline pulse so a programmatic scroll lands somewhere obvious.
   * Inline styles rather than classes: outline sits outside box-shadow, so it
   * can never fight the panel's own lighting.
   */
  const flash = useCallback(
    (el: HTMLElement) => {
      clearFlash();
      flashTarget.current = el;
      el.style.outline = FLASH_OUTLINE;
      el.style.outlineOffset = '4px';
      flashTimer.current = window.setTimeout(() => {
        el.style.outline = '';
        el.style.outlineOffset = '';
        flashTarget.current = null;
        flashTimer.current = null;
      }, FLASH_MS);
    },
    [clearFlash],
  );

  const findingById = useMemo(() => {
    const m = new Map<string, Finding>();
    for (const f of analysis.findings ?? []) m.set(f.id, f);
    return m;
  }, [analysis.findings]);

  /**
   * Selecting a dimension has to *go* somewhere.
   *
   * This used to only set state, so a card that said "explained below" did
   * nothing visible — the explanation was six screens down and the user was
   * left looking at the same grid. Selecting now travels to the first finding
   * in that dimension and flashes it, and falls back to the fix section when a
   * clean dimension has no finding to land on.
   */
  const handleSelect = useCallback(
    (d: Dimension) => {
      const next = selected === d ? null : d;
      setSelected(next);
      if (!next) return;

      const target = (analysis.findings ?? [])
        .filter((f) => f.dimension === d)
        .sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity])[0];

      // Let the expansion render before measuring where to scroll to.
      window.requestAnimationFrame(() => {
        let el: HTMLElement | null = null;
        if (target) {
          const cards = fixStackRef.current?.querySelectorAll<HTMLElement>('[data-finding-id]');
          for (const c of Array.from(cards ?? [])) {
            if (c.dataset.findingId === target.id) {
              el = c;
              break;
            }
          }
        }
        el = el ?? fixStackRef.current;
        if (!el) return;
        el.scrollIntoView({ behavior, block: target ? 'center' : 'start' });
        flash(el);
      });
    },
    [selected, analysis.findings, behavior, flash],
  );

  /** A fix says "here is where this lives in the track". */
  const handleFocusFinding = useCallback(
    (id: string) => {
      const finding = findingById.get(id);
      if (finding) setSelected(finding.dimension);
      lastSeekMatch.current = id;

      const el = timelineRef.current;
      if (!el) return;
      el.scrollIntoView({ behavior, block: 'start' });
      flash(el);
    },
    [findingById, behavior, flash],
  );

  /** …and a deliberate seek answers back with the fix for what you just heard. */
  const handleSeek = useCallback(
    (t: number) => {
      let best: Finding | null = null;
      for (const f of analysis.findings ?? []) {
        for (const m of f.moments ?? []) {
          const s = finite(m.t_start);
          const e = Math.max(s, finite(m.t_end, s));
          if (t >= s - 0.3 && t <= e + 0.3) {
            if (!best || finite(f.impact) > finite(best.impact)) best = f;
            break;
          }
        }
      }
      if (!best) return;

      setSelected(best.dimension);

      // Only travel once per finding — scrubbing must not yank the page around.
      if (lastSeekMatch.current === best.id) return;
      lastSeekMatch.current = best.id;

      // Scoped to the fix section and matched on the dataset rather than a
      // selector, so a finding id containing punctuation can never break it.
      let card: HTMLElement | null = null;
      const cards = fixStackRef.current?.querySelectorAll<HTMLElement>('[data-finding-id]');
      if (cards) {
        for (const c of Array.from(cards)) {
          if (c.dataset.findingId === best.id) {
            card = c;
            break;
          }
        }
      }
      if (!card) return;

      card.scrollIntoView({ behavior, block: 'center' });
      flash(card);
    },
    [analysis.findings, behavior, flash],
  );

  /* ------------------------------------------------------------- summary */

  /**
   * The bar follows the verdict: with a `ScoreCard` it pins the technical
   * score, because that is the one that answers "is anything wrong" and the
   * only one that has earned a grade. Reference match is deliberately absent —
   * it is not a number to be reminded of every time you scroll.
   */
  const scores = analysis.scores ?? null;
  const barValue = finite(scores ? scores.technical : analysis.health_score);
  const score = Math.round(barValue);
  const sev = severityFromScore(barValue);
  const grade = scores
    ? scores.technical_grade?.trim() || '—'
    : analysis.grade?.trim()
      ? analysis.grade.trim()
      : '—';

  const tally = useMemo(() => {
    const m: Record<Severity, number> = { critical: 0, major: 0, minor: 0, clean: 0 };
    for (const d of analysis.dimensions ?? []) m[d.severity] += 1;
    return m;
  }, [analysis.dimensions]);

  const worstFirst = useMemo(
    () =>
      (['critical', 'major', 'minor'] as Severity[])
        .filter((s) => tally[s] > 0)
        .sort((a, b) => SEVERITY_RANK[a] - SEVERITY_RANK[b]),
    [tally],
  );

  /**
   * What is still work. A prescription against a finding the producer has
   * confirmed was deliberate is not a fix, so it must not be counted as one
   * here — the header would otherwise contradict the plan directly below it.
   */
  const acknowledgedIds = useMemo(
    () => new Set((analysis.findings ?? []).filter((f) => f.acknowledged).map((f) => f.id)),
    [analysis.findings],
  );
  const prescriptionCount = (analysis.engineer?.prescriptions ?? []).filter(
    (p) => !acknowledgedIds.has(p.finding_id),
  ).length;
  const findingCount = (analysis.findings ?? []).filter(
    (f) => f.severity !== 'clean' && !f.acknowledged,
  ).length;
  const fixCount = prescriptionCount || findingCount;

  const momentCount = useMemo(
    () => (analysis.findings ?? []).reduce((acc, f) => acc + (f.moments?.length ?? 0), 0),
    [analysis.findings],
  );

  const dimensionCount = (analysis.dimensions ?? []).length;

  /* Both optional stages. They are always present on the wire, so the gate is
     `.available` rather than a null check — and each component re-checks it,
     so a partially-populated payload can only ever hide itself. */
  const sections = analysis.measurements?.sections ?? null;
  const stems = analysis.measurements?.stems ?? null;
  const sectionCount = sections?.sections?.length ?? 0;
  const hasSections = Boolean(sections?.available) && sectionCount >= 2;
  const hasStems = Boolean(stems?.available);
  const stemCount = (stems?.stems ?? []).filter((s) => s?.present).length;
  const maskingCount = (stems?.masking_pairs ?? []).length;

  const hasSpectral = Boolean(analysis.measurements?.spectral);
  const hasStereo = Boolean(analysis.measurements?.stereo && analysis.measurements?.phase);
  const hasDelivery =
    (analysis.platform_targets ?? []).length > 0 && Boolean(analysis.measurements?.loudness);
  const extraWarnings = (analysis.warnings ?? []).slice(1);

  return (
    <div className="relative pb-10">
      {/* ------------------------------------------------------- verdict */}
      <div className="mx-auto w-full max-w-[1400px] px-4 pb-12 pt-10 sm:px-6 sm:pb-16 sm:pt-14 lg:px-10">
        <Verdict analysis={analysis} engineerStatus={engineerStatus} />

        {/* Between the verdict and the diagnosis, because the answers change
            how everything below it reads — and because by here the reader
            knows what is being asked about. Renders nothing, including its own
            spacing, when nothing on the report is ambiguous. */}
        <Clarify analysis={analysis} onAnalysisChange={onAnalysisChange} />
      </div>

      {/* Pins itself the moment the verdict has scrolled away. */}
      <div className="sticky top-14 z-30 border-y border-void-line/70 bg-void/80 backdrop-blur-xl sm:top-16">
        <div className="mx-auto flex w-full max-w-[1400px] items-center gap-3 px-4 py-2.5 sm:gap-5 sm:px-6 lg:px-10">
          <div className={`flex shrink-0 items-baseline gap-2 ${SEV_CLASS[sev]}`}>
            <span className="stat text-[19px] font-semibold leading-none tracking-[-0.03em]">
              {score}
            </span>
            <span className="stat text-[10px] leading-none text-ink-faint">/100</span>
            <span
              aria-hidden="true"
              className="ml-1 h-1.5 w-1.5 rounded-full"
              style={{ background: SEVERITY_VAR[sev] }}
            />
            <span className="hidden font-mono text-[10px] uppercase tracking-[0.14em] text-ink-muted sm:inline">
              {scores ? `${SEV_WORD[sev]} · technical` : SEV_WORD[sev]}
            </span>
          </div>

          <span aria-hidden="true" className="h-4 w-px shrink-0 bg-void-line" />

          <span className="min-w-0 flex-1 truncate font-mono text-[11px] uppercase tracking-[0.1em] text-ink-dim">
            {analysis.filename || 'Untitled mix'}
          </span>

          <span className="hidden shrink-0 items-center gap-2 md:flex">
            <span className="sev-chip text-ink-muted">
              {scores ? 'technical' : 'grade'} {grade}
            </span>
            {worstFirst.map((s) => (
              <span key={s} className={`sev-chip ${SEV_CLASS[s]}`}>
                <span className="tabular-nums">{tally[s]}</span>
                {s}
              </span>
            ))}
          </span>

          <ReportDownload analysis={analysis} plugins={plugins} variant="compact" />

          <button
            type="button"
            onClick={onReset}
            className="btn-ghost shrink-0 px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-[0.13em]"
          >
            <span className="hidden sm:inline">Analyse another</span>
            <span className="sm:hidden">New</span>
          </button>
        </div>
      </div>

      <div className="mx-auto w-full max-w-[1400px] space-y-14 px-4 pt-14 sm:space-y-20 sm:px-6 sm:pt-20 lg:px-10">
        {/* ------------------------------------------------ where it hurts */}
        <Section
          id="results-map"
          index="01"
          eyebrow="Diagnosis"
          title="Where it hurts"
          blurb="Fourteen dimensions, each scored against what this genre is measured to do. Pick one and it stays selected across the map, the grid and the timeline."
          reduce={reduce}
          rule={false}
          aside={
            selected ? (
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="btn-ghost px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-[0.13em]"
              >
                Clear selection
              </button>
            ) : (
              <p className="eyebrow">{dimensionCount} scored</p>
            )
          }
        >
          {/* items-start so the map panel hugs the radar instead of stretching
              to the height of the grid beside it. */}
          <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,400px)_minmax(0,1fr)] xl:gap-8">
            {/* MixMap renders its own panel when it has nothing to draw. */}
            <div className={dimensionCount ? 'panel p-4 sm:p-6' : ''}>
              <MixMap dimensions={analysis.dimensions ?? []} onSelect={handleSelect} selected={selected} />
            </div>
            <div className="min-w-0">
              <DimensionGrid
                dimensions={analysis.dimensions ?? []}
                findings={analysis.findings ?? []}
                onSelect={handleSelect}
                selected={selected}
              />
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------ timeline */}
        <Section
          id="results-timeline"
          index="02"
          eyebrow="Evidence"
          title="Hear it for yourself"
          blurb="Every finding is anchored to the seconds it happens in. Scrub the waveform, jump between flagged moments, and the matching fix comes to you."
          reduce={reduce}
          innerRef={timelineRef}
          aside={
            <p className="eyebrow">
              {momentCount} flagged moment{momentCount === 1 ? '' : 's'}
              {hasSections ? ` · ${sectionCount} sections` : ''}
            </p>
          }
        >
          <div className="space-y-5">
            <Timeline
              analysis={analysis}
              audioUrl={audioUrl}
              selected={selected}
              onSeek={handleSeek}
            />
            {/* Same evidence, one level up: the arrangement rather than the bar. */}
            {hasSections && sections ? (
              <SectionMap
                sections={sections}
                duration={finite(analysis.measurements?.duration_seconds)}
                onSeek={handleSeek}
              />
            ) : null}
          </div>
        </Section>

        {/* -------------------------------------------------------- stems */}
        {hasStems && stems ? (
          <Section
            id="results-stems"
            index="03"
            eyebrow="Deep analysis"
            title="Element by element"
            blurb="The mix was separated into its sources and each one measured on its own. Level against the full mix, compression on that element alone, and which source is burying which — none of which a two-track can tell you."
            reduce={reduce}
            aside={
              <p className="eyebrow">
                {stemCount} source{stemCount === 1 ? '' : 's'}
                {maskingCount ? ` · ${maskingCount} masking pair${maskingCount === 1 ? '' : 's'}` : ''}
              </p>
            }
          >
            <StemBalance stems={stems} genre={analysis.genre} />
          </Section>
        ) : null}

        {/* ------------------------------------------------------ spectrum */}
        {hasSpectral ? (
          <Section
            id="results-spectrum"
            index="04"
            eyebrow="Tonality"
            title="The shape of the record"
            blurb="Third-octave analysis against the target curve for this genre. The shaded area is the difference — that is the EQ move, drawn."
            reduce={reduce}
            aside={analysis.reference ? <p className="eyebrow">reference loaded</p> : null}
          >
            <SpectrumCurve
              spectral={analysis.measurements.spectral}
              reference={analysis.reference ?? null}
            />
          </Section>
        ) : null}

        {/* ----------------------------------------------------- fix stack */}
        <Section
          id="results-fixes"
          index="05"
          eyebrow="The work"
          title="What to do about it"
          blurb="Ordered by the points each fix buys back. Tick them off as you go and watch the projected score climb toward the ceiling."
          reduce={reduce}
          innerRef={fixStackRef}
          aside={
            fixCount ? (
              <p className="eyebrow">
                {fixCount} {prescriptionCount ? 'prescription' : 'finding'}
                {fixCount === 1 ? '' : 's'}
              </p>
            ) : null
          }
        >
          <FixStack
            analysis={analysis}
            onFocusFinding={handleFocusFinding}
            engineerStatus={engineerStatus}
            engineerError={engineerError}
            onRetryEngineer={onRetryEngineer}
          />
        </Section>

        {/* -------------------------------------------- scope + delivery */}
        {hasStereo || hasDelivery ? (
          <Section
            id="results-delivery"
            index="06"
            eyebrow="Image & delivery"
            title="How it travels"
            blurb="What survives a mono fold-down, and what each platform will do to the level once it leaves your room."
            reduce={reduce}
          >
            <div className="space-y-5">
              {hasStereo ? (
                <StereoScope
                  stereo={analysis.measurements.stereo}
                  phase={analysis.measurements.phase}
                />
              ) : null}
              {hasDelivery ? (
                <LoudnessTargets
                  targets={analysis.platform_targets ?? []}
                  loudness={analysis.measurements.loudness}
                />
              ) : null}
            </div>
          </Section>
        ) : null}

        {/* -------------------------------------------------------- tables */}
        {analysis.measurements ? (
          <Section
            id="results-metrics"
            index="07"
            eyebrow="Receipts"
            title="Every number"
            blurb="Nothing on this page is a vibe. This is the full measured dataset behind it, units and all."
            reduce={reduce}
          >
            <MetricsTable measurements={analysis.measurements} />
          </Section>
        ) : null}

        {/* The document comes before the tip jar: the last thing the report
            offers should be the report itself, not the ask. */}
        <div className="mt-16 sm:mt-20">
          <ReportDownload analysis={analysis} plugins={plugins} />
        </div>

        {/* The ask goes after the whole report — value first, then the jar. */}
        <div className="mt-12 sm:mt-16">
          <DonatePanel />
        </div>

        {/* -------------------------------------------------------- outro */}
        <motion.section
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: 24 }}
          whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: reduce ? 0.3 : 0.7, ease: EASE }}
        >
          <div className="hairline mb-10 sm:mb-14" />

          {extraWarnings.length ? (
            <div className="mb-8 rounded-xl border border-void-line/70 p-5">
              <p className="eyebrow">Analyser notes</p>
              <ul className="mt-3 space-y-2">
                {extraWarnings.map((w, i) => (
                  <li key={`${w}-${i}`} className="flex gap-3">
                    <span className="mt-[3px] shrink-0 text-[10px] text-sev-major" aria-hidden="true">
                      ●
                    </span>
                    <span className="text-[12.5px] leading-snug text-ink-muted">{w}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="panel-raised relative overflow-hidden p-7 sm:p-10">
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-0"
              style={{
                background: 'radial-gradient(120% 100% at 100% 0%, rgba(82,242,196,0.09), transparent 60%)',
              }}
            />
            <div className="relative flex flex-wrap items-end justify-between gap-x-10 gap-y-6">
              <div className="min-w-0">
                <p className="eyebrow">Next</p>
                <h2 className="display mt-3.5 max-w-xl text-balance text-[clamp(1.4rem,3vw,2.1rem)] leading-[1.02] tracking-[-0.035em] text-ink">
                  Do the work, bounce it, measure it again.
                </h2>
                <p className="mt-3.5 max-w-lg text-[13.5px] leading-relaxed text-ink-muted">
                  A revision that measures better is the only proof that the moves landed. Bring the
                  new bounce back through and compare.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button type="button" onClick={onReset} className="btn-primary px-6 py-3 text-sm">
                  Analyse another mix
                </button>
              </div>
            </div>
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-5 gap-y-2">
            <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
              {analysis.filename || 'Untitled mix'}
            </p>
            {analysis.genre ? (
              <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                {analysis.genre}
              </p>
            ) : null}
            <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
              analysed in {Math.round(finite(analysis.analysis_ms))} ms
            </p>
            <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
              schema v{analysis.schema_version}
            </p>
          </div>
        </motion.section>
      </div>
    </div>
  );
}
