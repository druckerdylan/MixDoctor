/// <reference types="vite/client" />

/**
 * Runtime configuration.
 *
 * The API base used to be hard-coded to production, which meant `npm run dev`
 * always talked to Railway and local backend work was impossible. Resolution
 * order is now:
 *   1. VITE_API_BASE, if set (any environment — this is the override)
 *   2. localhost:8000 in dev
 *   3. the Railway deployment in a production build
 */

const PRODUCTION_API = 'https://mixdoctor-production.up.railway.app';
const LOCAL_API = 'http://localhost:8000';

function resolveApiBase(): string {
  const override = import.meta.env.VITE_API_BASE;
  if (typeof override === 'string' && override.trim() !== '') {
    // Trailing slashes make `${API_BASE}/analyze` a double-slash 404 on some hosts.
    return override.trim().replace(/\/+$/, '');
  }
  return import.meta.env.DEV ? LOCAL_API : PRODUCTION_API;
}

export const API_BASE = resolveApiBase();

/**
 * Tip jar.
 *
 * Ko-fi's free plan takes 0% of one-time tips (you pay only Stripe/PayPal
 * processing), so it is the default framing. Any of the other platforms work
 * by setting VITE_DONATE_URL to a full link instead.
 *
 * Defaults to the Ko-fi account above. Set VITE_KOFI_USERNAME to point it
 * somewhere else, or VITE_DONATE_URL for a non-Ko-fi platform; both override
 * the default without a code change. Setting KOFI_DEFAULT to an empty string
 * restores the old behavior of hiding every donate surface.
 */
// The tip jar's default home. Hardcoded on purpose: this is a public URL, not
// a secret, and making it an env var bought nothing but a deploy loop — a
// Vite build bakes VITE_* values in, so a redeploy that reuses the build cache
// silently ships without them and the whole donate surface vanishes with no
// error anywhere. A committed default cannot fail that way.
const KOFI_DEFAULT = 'lilbeats';

const KOFI_USERNAME = (import.meta.env.VITE_KOFI_USERNAME ?? '').trim() || KOFI_DEFAULT;
const DONATE_URL_OVERRIDE = (import.meta.env.VITE_DONATE_URL ?? '').trim();

export const DONATE_URL: string | null =
  DONATE_URL_OVERRIDE || (KOFI_USERNAME ? `https://ko-fi.com/${KOFI_USERNAME}` : null);

/** Ko-fi says "coffee"; a bare link says nothing. Label follows the platform. */
export const DONATE_IS_KOFI = !DONATE_URL_OVERRIDE && Boolean(KOFI_USERNAME);

/** Hard ceiling on upload size, enforced client-side before we waste a round trip. */
export const MAX_UPLOAD_BYTES = 250 * 1024 * 1024;

/** Extensions the backend can decode. Lower-case, leading dot. */
export const ACCEPTED_EXTENSIONS: string[] = [
  '.wav',
  '.mp3',
  '.flac',
  '.aiff',
  '.aif',
  '.m4a',
  '.ogg',
];

/** `accept` attribute for the file inputs — extensions plus the broad MIME wildcard. */
export const ACCEPT_ATTR = [...ACCEPTED_EXTENSIONS, 'audio/*'].join(',');

/**
 * Above this size we skip the client-side waveform preview: decoding a
 * quarter-gigabyte WAV to float32 PCM in the tab is a good way to kill it.
 * The upload itself is unaffected.
 */
export const PREVIEW_MAX_BYTES = 96 * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  const mb = bytes / (1024 * 1024);
  return mb >= 100 ? `${mb.toFixed(0)} MB` : `${mb.toFixed(1)} MB`;
}
