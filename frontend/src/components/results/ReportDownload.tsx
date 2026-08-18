/**
 * "Download report" — the whole analysis as a document you can keep.
 *
 * Rendered twice on the results page, and the two placements want different
 * things. In the sticky header it is a control you should be able to find
 * without reading anything, so it is compact and unlabeled beyond the verb. At
 * the foot of the report it is an offer, so it explains what the document
 * actually contains — which is the only place a first-time user learns that
 * this is eight sections and a short course rather than a screenshot.
 *
 * Both share one `useReport` instance per mount, so a spinner in the header
 * does not spin the footer button as well.
 */

import { useReport } from '../../hooks/useReport';
import type { MixAnalysis, OwnedPlugin } from '../../types/analysis';

export interface ReportDownloadProps {
  analysis: MixAnalysis;
  /** Owned plugins, so the document's fix steps resolve against real tools. */
  plugins?: readonly OwnedPlugin[];
  variant?: 'compact' | 'full';
  className?: string;
}

/** A three-dot pulse rather than a spinner — no rotation to fight reduced motion. */
function Working({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="sr-only">{label}</span>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          aria-hidden="true"
          className="h-1 w-1 animate-pulse rounded-full bg-current"
          style={{ animationDelay: `${i * 160}ms`, animationDuration: '1s' }}
        />
      ))}
    </span>
  );
}

export default function ReportDownload({
  analysis,
  plugins = [],
  variant = 'full',
  className = '',
}: ReportDownloadProps) {
  const { pending, error, downloadMarkdown, openPrintable, dismissError } = useReport(
    analysis,
    plugins,
  );

  const busy = pending !== null;

  /* ------------------------------------------------------------ compact */

  if (variant === 'compact') {
    return (
      <div className={`flex shrink-0 items-center gap-2 ${className}`.trim()}>
        <button
          type="button"
          onClick={downloadMarkdown}
          disabled={busy}
          title={
            error ??
            'Download the full write-up as a Markdown document — every finding, every fix, every number.'
          }
          className="btn-ghost shrink-0 px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-[0.13em]"
        >
          {pending === 'markdown' ? (
            <Working label="Building your report" />
          ) : (
            <>
              <span className="hidden sm:inline">Download report</span>
              <span className="sm:hidden">Report</span>
            </>
          )}
        </button>
        {/* The header has no room for an error message, so the failure shows
            as a dot with the reason in its tooltip. The full control at the
            foot of the page states it properly. */}
        {error ? (
          <button
            type="button"
            onClick={dismissError}
            title={`${error} (click to dismiss)`}
            aria-label={error}
            className="sev-chip sev-major shrink-0"
          >
            !
          </button>
        ) : null}
      </div>
    );
  }

  /* --------------------------------------------------------------- full */

  return (
    <section className={`panel-raised p-7 sm:p-9 ${className}`.trim()}>
      <div className="flex flex-wrap items-start justify-between gap-x-10 gap-y-6">
        <div className="min-w-0 max-w-xl">
          <p className="eyebrow">Take it with you</p>
          <h2 className="display mt-3.5 text-balance text-[clamp(1.3rem,2.6vw,1.9rem)] leading-[1.05] tracking-[-0.035em] text-ink">
            Download the full report
          </h2>
          <p className="mt-3.5 text-[13.5px] leading-relaxed text-ink-muted">
            Everything on this page as a document you can read away from the browser: the three
            moves that matter most, what is genuinely wrong and how to fix it step by step, where
            this differs from the {analysis.genre || 'genre'} reference and what those choices buy
            you, an ordered session plan, and every measurement with its target.
          </p>
          <p className="mt-2.5 text-[13.5px] leading-relaxed text-ink-muted">
            It also carries the theory behind everything that fired on{' '}
            <em>this</em> track — a short course written from your own mix rather than a manual.
          </p>
        </div>

        <div className="flex min-w-0 flex-col gap-3">
          <button
            type="button"
            onClick={downloadMarkdown}
            disabled={busy}
            className="btn-primary px-6 py-3 text-sm"
          >
            {pending === 'markdown' ? <Working label="Building your report" /> : 'Download Markdown'}
          </button>
          <button
            type="button"
            onClick={openPrintable}
            disabled={busy}
            className="btn-ghost px-6 py-3 text-sm"
          >
            {pending === 'html' ? <Working label="Building your report" /> : 'Print / Save as PDF'}
          </button>
          <p className="max-w-[15rem] font-mono text-micro uppercase tracking-[0.13em] text-ink-faint">
            Opens in a new tab and prints. Choose "Save as PDF" as the destination.
          </p>
        </div>
      </div>

      {error ? (
        <div
          role="alert"
          className="mt-6 flex flex-wrap items-start justify-between gap-x-6 gap-y-3 rounded-xl border border-sev-major/40 bg-sev-major/5 p-4"
        >
          <p className="min-w-0 flex-1 text-[13px] leading-snug text-ink-muted">
            <span className="sev-major font-medium">Could not build the document. </span>
            {error} Nothing on this page is affected — the analysis above is complete.
          </p>
          <button
            type="button"
            onClick={dismissError}
            className="btn-ghost shrink-0 px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-[0.13em]"
          >
            Dismiss
          </button>
        </div>
      ) : null}
    </section>
  );
}
