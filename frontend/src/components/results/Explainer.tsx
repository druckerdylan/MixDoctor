/**
 * Explainer — the part that teaches, rather than restates.
 *
 * A finding already knows that 150-400 Hz is carrying more than the target
 * window allows. That is the proof, not the answer. This panel is the answer:
 * what the thing being measured actually is, what it sounds like, why it
 * matters once the track leaves your room, the steps that fix it with the
 * plugins you own, and how you know when to stop.
 *
 * Everything here is prose to be read, so it is set like prose — sentence
 * case, a ~68ch measure, real paragraph spacing. The mono/uppercase treatment
 * belongs to the metrics; using it here would turn an explanation back into a
 * readout, which is the complaint this panel exists to answer.
 *
 * Order is deliberate and matches the order someone actually needs: the
 * consequence, the concept, the symptom, the stakes, the fix, the stop
 * condition. The underlying theory is last and folded away — present for
 * someone who wants to learn it, out of the way for someone mid-session.
 */

import { useMemo, useState, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import useKnowledge, { type ResolvedExplainer } from '../../hooks/useKnowledge';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

/* ------------------------------------------------------------------ */
/* Shared pieces                                                       */
/* ------------------------------------------------------------------ */

export function Chevron({ open, className = '' }: { open: boolean; className?: string }) {
  return (
    <svg
      viewBox="0 0 12 12"
      width="11"
      height="11"
      aria-hidden="true"
      className={`shrink-0 transition-transform duration-300 ease-cine ${open ? 'rotate-180' : ''} ${className}`}
    >
      <path
        d="M2.5 4.25L6 7.75L9.5 4.25"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * A labelled fold. Used for the theory here and for the measurements in the
 * fix stack — same control, so "there is more under this" reads the same way
 * wherever it appears.
 */
export function Disclosure({
  label,
  openLabel,
  children,
  defaultOpen = false,
  count,
}: {
  label: string;
  /** Label once open, when the closed one is a question. Defaults to `label`. */
  openLabel?: string;
  children: ReactNode;
  defaultOpen?: boolean;
  /** Shown beside the label — say how much is hidden before it is opened. */
  count?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const reduce = useReducedMotion() ?? false;

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="group flex w-full items-center gap-2.5 rounded-lg py-1.5 text-left transition-colors duration-300 ease-cine"
      >
        <span className="eyebrow transition-colors duration-300 group-hover:text-ink-dim">
          {open ? (openLabel ?? label) : label}
        </span>
        {count ? <span className="stat text-[10px] text-ink-faint">{count}</span> : null}
        <span className="hairline flex-1" aria-hidden="true" />
        <Chevron open={open} className="text-ink-faint group-hover:text-ink-dim" />
      </button>

      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: -4 }}
            transition={{ duration: reduce ? 0.15 : 0.35, ease: EASE }}
            className="pt-3"
          >
            {children}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

/** Sentence case, sans, sitting above prose rather than above a number. */
function Head({ children }: { children: ReactNode }) {
  return (
    <h5 className="mb-2 flex items-center gap-2.5 font-display text-[13.5px] font-semibold tracking-tight text-ink">
      <span aria-hidden="true" className="h-[11px] w-[2px] rounded-full bg-signal-dim/70" />
      {children}
    </h5>
  );
}

/**
 * The server writes real paragraphs and separates them with a blank line —
 * most of the long prose fields, and some fix-step details, are
 * multi-paragraph. HTML collapses that whitespace, so rendering the string
 * into a single `<p>` silently turns a structured explanation into a wall,
 * which is precisely the failure this panel exists to avoid.
 */
function paragraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p !== '');
}

/** A paragraph run at one measure and one rhythm, at whatever scale is asked for. */
function Paragraphs({ text, className }: { text: string; className: string }) {
  const parts = paragraphs(text);
  if (parts.length <= 1) return <p className={className}>{parts[0] ?? text}</p>;
  return (
    <div className="space-y-3">
      {parts.map((p, i) => (
        <p key={i} className={className}>
          {p}
        </p>
      ))}
    </div>
  );
}

/** One measure, one rhythm. Every body paragraph in the panel goes through here. */
function Prose({ children }: { children: ReactNode }) {
  const cls = 'max-w-[68ch] text-pretty text-[14.5px] leading-[1.75] text-ink-dim';
  if (typeof children !== 'string') return <p className={cls}>{children}</p>;
  return <Paragraphs text={children} className={cls} />;
}

/* ------------------------------------------------------------------ */
/* Sections                                                            */
/* ------------------------------------------------------------------ */

function Causes({ causes }: { causes: string[] }) {
  if (!causes.length) return null;
  return (
    <section>
      <Head>Usually caused by</Head>
      <ul className="max-w-[68ch] space-y-1.5">
        {causes.map((cause, i) => (
          <li key={i} className="flex gap-2.5 text-[14px] leading-[1.65] text-ink-dim">
            <span aria-hidden="true" className="mt-[9px] h-[3px] w-[3px] shrink-0 rounded-full bg-ink-faint" />
            <span>{cause}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Steps({ ex }: { ex: ResolvedExplainer }) {
  /**
   * A substituted step names the tool that would have replaced it — once per
   * capability, so a panel with three workarounds around the same missing box
   * mentions it a single time. It is a footnote about why the instruction
   * looks the way it does, not a suggestion to go shopping.
   */
  const notes = useMemo(() => {
    const seen = new Set<string>();
    return ex.steps.map((step) => {
      if (!step.substitutedFor || seen.has(step.substitutedFor)) return null;
      seen.add(step.substitutedFor);
      return step.substitutedFor;
    });
  }, [ex.steps]);

  if (!ex.steps.length) return null;

  return (
    <section>
      <Head>How to fix it</Head>
      <ol className="mt-3 space-y-4">
        {ex.steps.map((step, i) => (
          <li key={i} className="grid grid-cols-[22px_minmax(0,1fr)] gap-x-3">
            <span
              aria-hidden="true"
              className="stat pt-[3px] text-[12px] leading-none text-signal-dim"
            >
              {i + 1}
            </span>
            <div className="min-w-0">
              <p className="max-w-[66ch] text-[14.5px] font-medium leading-[1.6] text-ink">
                {step.action}
              </p>
              {step.detail ? (
                <div className="mt-1.5">
                  <Paragraphs
                    text={step.detail}
                    className="max-w-[66ch] text-[13.5px] leading-[1.7] text-ink-muted"
                  />
                </div>
              ) : null}
              {notes[i] ? (
                <p className="mt-2 max-w-[66ch] text-[12.5px] leading-[1.6] text-ink-faint">
                  Written for what is in your vault. {notes[i]} would make this a single step.
                </p>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Verify({ text }: { text: string }) {
  return (
    <section
      className="relative overflow-hidden rounded-xl2 border p-4 sm:p-5"
      style={{
        borderColor: 'rgba(82,242,196,0.30)',
        background:
          'linear-gradient(120deg, rgba(82,242,196,0.09), rgba(82,242,196,0.02) 60%, transparent)',
      }}
    >
      <span aria-hidden="true" className="absolute inset-y-0 left-0 w-[3px] bg-signal" />
      <h5 className="flex items-center gap-2.5 font-display text-[13.5px] font-semibold tracking-tight text-ink">
        <span className="text-[11px] text-signal" aria-hidden="true">
          ✓
        </span>
        You’ll know it’s fixed when
      </h5>
      <div className="mt-2.5">
        <Paragraphs
          text={text}
          className="max-w-[68ch] text-pretty text-[14.5px] leading-[1.75] text-ink"
        />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* The panel                                                           */
/* ------------------------------------------------------------------ */

export interface ExplainerProps {
  /** The finding to teach. Resolution falls back to the dimension server-side rules. */
  findingId: string;
  /**
   * Show the rough time the fix takes. Off where a prescription already states
   * its own estimate, so the card never quotes two different numbers.
   */
  showTime?: boolean;
  className?: string;
}

/**
 * Renders nothing when there is no explainer for this finding — the caller
 * keeps whatever it had. Never a spinner: this arrives in one request at page
 * load and a placeholder inside an expanded card would flash and go.
 */
export default function Explainer({ findingId, showTime = false, className = '' }: ExplainerProps) {
  const { explain } = useKnowledge();
  const ex = explain(findingId);
  if (!ex) return null;

  const hasBody =
    ex.whatItIs || ex.whatYouHear || ex.whyItMatters || ex.steps.length || ex.howToVerify;
  if (!ex.headline && !hasBody) return null;

  return (
    <div
      className={`rounded-xl2 border border-void-line/70 bg-void-deep/40 p-5 sm:p-6 ${className}`}
    >
      {ex.headline ? (
        <p className="display max-w-[62ch] text-balance text-[clamp(1rem,1.7vw,1.28rem)] leading-[1.32] tracking-[-0.02em] text-ink">
          {ex.headline}
        </p>
      ) : null}

      {showTime && ex.minutes > 0 ? (
        <p className="mt-3 text-[12.5px] text-ink-faint">
          Roughly {ex.minutes} minutes of work, once you know what you are looking at.
        </p>
      ) : null}

      <div className="mt-6 space-y-6">
        {ex.whatItIs ? (
          <section>
            <Head>What this actually means</Head>
            <Prose>{ex.whatItIs}</Prose>
          </section>
        ) : null}

        {ex.whatYouHear ? (
          <section>
            <Head>What you’re hearing</Head>
            <Prose>{ex.whatYouHear}</Prose>
          </section>
        ) : null}

        {ex.whyItMatters ? (
          <section>
            <Head>Why it matters</Head>
            <Prose>{ex.whyItMatters}</Prose>
          </section>
        ) : null}

        <Causes causes={ex.commonCauses} />

        <Steps ex={ex} />

        {ex.howToVerify ? <Verify text={ex.howToVerify} /> : null}

        {ex.learnMore ? (
          <Disclosure label="Why this happens" openLabel="The concept">
            <div className="rounded-xl border border-void-line/60 bg-void/60 p-4 sm:p-5">
              <Prose>{ex.learnMore}</Prose>
            </div>
          </Disclosure>
        ) : null}
      </div>
    </div>
  );
}
