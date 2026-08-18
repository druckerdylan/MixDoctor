/**
 * Google Analytics 4, loaded only when it is configured.
 *
 * Env-gated the same way lilbeats does it: with `VITE_GA_ID` unset, no script
 * is injected, no cookie is set, and no request leaves the browser. That keeps
 * local development and preview deployments out of the numbers — measuring
 * yourself is how an analytics property becomes useless in its first week.
 *
 * Use a **separate GA property** from lilbeats. Two products in one property
 * gives you one blended funnel that describes neither.
 */

const GA_ID = (import.meta.env.VITE_GA_ID ?? '').trim();

/** Whether analytics is configured for this build. */
export const analyticsEnabled = /^G-[A-Z0-9]+$/i.test(GA_ID);

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

let loaded = false;

/**
 * Inject gtag.js once.
 *
 * Called from main.tsx rather than a component, so a re-render can never
 * double-load it and React Strict Mode's deliberate double-invoke in dev
 * cannot either.
 */
export function initAnalytics(): void {
  if (!analyticsEnabled || loaded || typeof document === 'undefined') return;
  loaded = true;

  const script = document.createElement('script');
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA_ID)}`;
  document.head.appendChild(script);

  window.dataLayer = window.dataLayer || [];
  function gtag(...args: unknown[]) {
    window.dataLayer?.push(args);
  }
  window.gtag = gtag;

  gtag('js', new Date());
  // The app is a single page with no router, so GA's automatic page_view is
  // the only one there is — nothing here needs manual page tracking.
  gtag('config', GA_ID, { anonymize_ip: true });
}

/**
 * Record something a producer did.
 *
 * Deliberately a thin, typed surface rather than exposing gtag directly: the
 * events worth having are few, and an untyped call site is how you end up with
 * four spellings of the same event name and a funnel you cannot read.
 */
export type AnalyticsEvent =
  | 'analysis_started'
  | 'analysis_completed'
  | 'analysis_failed'
  | 'report_downloaded'
  | 'engineer_requested'
  | 'clarification_answered'
  | 'plugin_vault_opened'
  // A learn-more link opened. The only signal on whether anyone wants the
  // teaching resources at all — a section nobody clicks should be cut, and
  // without this the answer is a guess.
  | 'resource_clicked'
  | 'donate_clicked';

export function track(event: AnalyticsEvent, params: Record<string, string | number | boolean> = {}): void {
  if (!analyticsEnabled || typeof window === 'undefined' || !window.gtag) return;
  window.gtag('event', event, params);
}
