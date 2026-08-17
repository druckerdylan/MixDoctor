import type { ReactNode } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { DonateLink } from './Donate';
import { PluginVaultHost, PluginVaultTrigger } from './PluginVault';

interface ShellProps {
  children: ReactNode;
  /** Right-hand header slot — reset buttons, grade chips, whatever the view needs. */
  actions?: ReactNode;
  /** When provided the wordmark becomes a button back to the start. */
  onHome?: (() => void) | null;
}

/**
 * Bar levels as a fraction of full height. The bars are drawn full-height and
 * scaled from the baseline rather than having their `height` attribute
 * animated: framer-motion resolves keyframed SVG *attributes* a frame late,
 * which emits `<rect> attribute height: Expected length, "undefined"` on mount.
 * A transform has no such gap, and it composites on the GPU.
 */
const BAR_REST = [0.35, 0.65, 0.9, 0.55, 0.4];
/** Peak each bar drifts toward. Prime-ish periods keep them out of lockstep. */
const BAR_PEAK = [0.7, 0.95, 0.45, 0.85, 0.75];
const BAR_PERIOD = [1.5, 1.9, 1.15, 2.3, 1.7];

/**
 * A five-bar level meter beside the wordmark. It is decorative — the product is
 * a meter, so the logo behaves like one — but it stays still under
 * prefers-reduced-motion and is hidden from assistive tech.
 */
function LevelGlyph() {
  const reduce = useReducedMotion();

  return (
    <svg
      viewBox="0 0 22 20"
      width="18"
      height="16"
      aria-hidden="true"
      className="shrink-0 overflow-visible text-signal"
    >
      {BAR_REST.map((rest, i) => {
        const peak = BAR_PEAK[i] ?? rest;
        return (
          <motion.rect
            key={i}
            x={i * 4.4}
            y={0}
            width={2.6}
            height={20}
            rx={1.3}
            fill="currentColor"
            opacity={i === 2 ? 1 : 0.42 + i * 0.05}
            // Scale up from the baseline, so 0..1 maps to an empty..full bar.
            style={{ transformOrigin: '50% 100%', transformBox: 'fill-box' }}
            initial={{ scaleY: rest }}
            animate={reduce ? { scaleY: rest } : { scaleY: [rest, peak, rest] }}
            transition={
              reduce
                ? { duration: 0 }
                : {
                    duration: BAR_PERIOD[i] ?? 1.6,
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: i * 0.11,
                  }
            }
          />
        );
      })}
    </svg>
  );
}

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <LevelGlyph />
      <span className="display text-[0.95rem] leading-none tracking-[-0.045em] sm:text-base">
        <span className="text-ink">MIX</span>
        <span className="text-ink-muted">DIAGNOSTIC</span>
      </span>
    </span>
  );
}

export default function Shell({ children, actions, onHome }: ShellProps) {
  return (
    <div className="grain relative min-h-screen overflow-x-clip">
      {/* Key light from above. Fixed so it does not scroll away with the hero. */}
      <div className="vignette-top pointer-events-none fixed inset-x-0 top-0 z-0 h-[60vh]" aria-hidden="true" />

      <header className="fixed inset-x-0 top-0 z-40">
        <div className="border-b border-void-line/70 bg-void/70 backdrop-blur-xl">
          <div className="mx-auto flex h-14 max-w-[1800px] items-center justify-between gap-4 px-4 sm:h-16 sm:px-6 lg:px-10">
            {onHome ? (
              <button
                type="button"
                onClick={onHome}
                className="rounded-md transition-opacity duration-300 ease-cine hover:opacity-80"
                aria-label="Mix Diagnostic — back to start"
              >
                <Wordmark />
              </button>
            ) : (
              <Wordmark />
            )}

            <div className="flex min-w-0 items-center gap-2 sm:gap-3">
              {/* What the producer owns changes every prescription, so the way
                  in is one click from anywhere rather than buried in settings. */}
              <PluginVaultTrigger className="btn-ghost shrink-0 px-3 py-2 text-xs" />
              {actions}
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 pt-14 sm:pt-16">{children}</main>

      <footer className="relative z-10 mt-24">
        <div className="hairline" />
        <div className="mx-auto flex max-w-[1800px] flex-col gap-2 px-4 py-8 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-10">
          <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
            MIX DIAGNOSTIC — measurement first, opinions second
          </p>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
              LUFS &amp; TRUE PEAK PER ITU-R BS.1770-4 · 1/3-OCTAVE ANALYSIS
            </p>
            <DonateLink />
          </div>
        </div>
      </footer>

      {/* One drawer for the whole app — the header button and the intake form
          both raise this instance. */}
      <PluginVaultHost />
    </div>
  );
}
