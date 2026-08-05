/**
 * Tip jar.
 *
 * Two rules, both deliberate:
 *
 * 1. Nothing renders unless a destination is configured. An unset env var
 *    means no dead button and no link to a 404.
 * 2. It stays understated. This sits alongside a tool people may pay a
 *    subscription for, and a loud coffee button reads as "hobby project" —
 *    which is exactly the wrong signal next to a paid tier. It asks once,
 *    quietly, after the work has already been delivered.
 */

import { motion, useReducedMotion } from 'framer-motion';

import { DONATE_IS_KOFI, DONATE_URL } from '../config';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

/** A cup, drawn rather than imported, so there is no icon dependency. */
function CupGlyph({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      <path d="M3.5 7h10v5.5a3.5 3.5 0 0 1-3.5 3.5H7a3.5 3.5 0 0 1-3.5-3.5V7Z" />
      <path d="M13.5 8.5h1.75a2.25 2.25 0 0 1 0 4.5H13.5" />
      <path d="M6 4.2c0-.7.8-.9.8-1.7M9 4.2c0-.7.8-.9.8-1.7" opacity="0.7" />
    </svg>
  );
}

/**
 * Footer variant — a text link, weighted like the rest of the footer meta.
 * This is the one that is always present.
 */
export function DonateLink({ className = '' }: { className?: string }) {
  if (!DONATE_URL) return null;

  return (
    <a
      href={DONATE_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center gap-1.5 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint transition-colors duration-200 hover:text-signal ${className}`}
    >
      <CupGlyph />
      {DONATE_IS_KOFI ? 'Buy me a coffee' : 'Support this'}
    </a>
  );
}

/**
 * Post-report variant — shown once, at the bottom of a finished analysis.
 *
 * The ask is placed here on purpose: the user has just been handed a full
 * diagnosis and a fix plan, so it lands after value rather than in front of
 * it. It is a panel rather than a modal because a tip jar that interrupts is
 * a tip jar people resent.
 */
export function DonatePanel() {
  const reduce = useReducedMotion() ?? false;
  if (!DONATE_URL) return null;

  return (
    <motion.aside
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 16 }}
      whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: reduce ? 0.25 : 0.7, ease: EASE }}
      className="panel flex flex-col gap-5 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-7"
    >
      <div className="max-w-[62ch]">
        <p className="eyebrow mb-2 text-ink-faint">Free to use</p>
        <p className="display text-[17px] leading-snug tracking-[-0.02em] text-ink">
          MixDoctor doesn&rsquo;t charge for this report.
        </p>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">
          No account, no upload limit per day, no watermark on the findings. If it saved you an
          hour of squinting at an analyser, you can put something in the jar.
        </p>
      </div>

      <a
        href={DONATE_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="btn-ghost shrink-0 gap-2 px-5 py-2.5 text-[13px] hover:border-signal-dim hover:text-signal"
      >
        <CupGlyph />
        {DONATE_IS_KOFI ? 'Buy me a coffee' : 'Support this'}
      </a>
    </motion.aside>
  );
}

export default DonatePanel;
