/**
 * The teaching layer: what a finding *means*, fetched once and shared.
 *
 * The measurement layer already tells someone that 150-400 Hz sits 34.7 dB over
 * 1-3 kHz. That sentence is correct and useless unless you already know what it
 * describes. `/knowledge` carries the other half — what it sounds like, what
 * causes it, what to do — written once on the server and identical for every
 * track, which is exactly why it can be cached in module scope and never
 * refetched.
 *
 * Two things this hook does beyond fetching:
 *
 * 1. **Resolution.** `explain()` mirrors `knowledge.explain()` on the backend:
 *    exact finding id first, then the `_hot`/`_thin` variant for the dimension,
 *    then the dimension itself. A per-band finding inherits the general
 *    teaching rather than showing nothing.
 * 2. **Tool tiering.** Fix steps come back with the capability each one needs.
 *    A step the producer cannot perform is swapped for its `without` variant,
 *    or dropped when there isn't one — mirroring `applicable_steps()`. Showing
 *    someone an instruction for a plugin they do not own is how a tool starts
 *    feeling generic.
 *
 * If the endpoint is missing — an older server, an offline tab — every lookup
 * returns null and the caller falls back to the numbers it already has. The
 * teaching is an upgrade, never a dependency.
 */

import { useCallback, useEffect, useMemo, useSyncExternalStore } from 'react';

import { API_BASE } from '../config';
import { CAPABILITY_LABELS } from '../data/plugins';
import type { Resource, ResourceKind } from '../types/analysis';
import { usePluginVault } from './usePluginVault';

/* ------------------------------------------------------------------ */
/* Shape                                                               */
/* ------------------------------------------------------------------ */

/** One step, already decided against the producer's vault. */
export interface ResolvedFixStep {
  action: string;
  detail: string;
  /** Capability slug this step uses, when the producer has it. '' otherwise. */
  requires: string;
  /**
   * Set when this is the `without` variant: the human label of the capability
   * that would have made the step easier. Null on a step rendered as written.
   */
  substitutedFor: string | null;
}

export interface ResolvedExplainer {
  /** The finding id that was asked for. */
  findingId: string;
  /** The key it actually resolved to — the dimension, on a fallback. */
  sourceId: string;
  /** False when this is the dimension's general teaching, not the exact finding. */
  exact: boolean;
  headline: string;
  whatItIs: string;
  whatYouHear: string;
  whyItMatters: string;
  commonCauses: string[];
  /** Fix steps, resolved. May be shorter than the server's list. */
  steps: ResolvedFixStep[];
  howToVerify: string;
  learnMore: string;
  minutes: number;
  /** Where to go and learn this properly. Empty until someone writes links for it. */
  resources: Resource[];
}

/** Straight off the wire, before any vault resolution. */
interface RawFixStep {
  action: string;
  detail: string;
  needs: string;
  without: string;
}

interface RawExplainer {
  headline: string;
  what_it_is: string;
  what_you_hear: string;
  why_it_matters: string;
  common_causes: string[];
  how_to_fix: RawFixStep[];
  how_to_verify: string;
  learn_more: string;
  minutes: number;
  resources: Resource[];
}

type Phase = 'idle' | 'loading' | 'ready' | 'unavailable';

interface KnowledgeState {
  phase: Phase;
  index: Record<string, RawExplainer>;
  /**
   * What the server says advice may assume of everyone. `data/plugins.ts`
   * carries the same list, but taking the server's copy means a stock
   * capability added after this build shipped still resolves as owned rather
   * than quietly demoting every step behind it to its `without` variant.
   */
  stock: string[];
}

/* ------------------------------------------------------------------ */
/* Wire parsing                                                        */
/* ------------------------------------------------------------------ */

/**
 * Everything below treats the response as untrusted. A field that is missing,
 * null or the wrong type must degrade to an empty section rather than throw —
 * a server one deploy ahead of this build should cost the user a paragraph,
 * not the page.
 */

function str(v: unknown): string {
  return typeof v === 'string' ? v.trim() : '';
}

function strList(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.map(str).filter((s) => s !== '');
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function parseStep(raw: unknown): RawFixStep | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const action = str(r.action);
  if (action === '') return null;
  return {
    action,
    detail: str(r.detail),
    // `applicable_steps()` calls the same field `requires` once resolved, so
    // accept both spellings rather than depending on which one is serialised.
    needs: str(r.needs) || str(r.requires),
    without: str(r.without),
  };
}

const RESOURCE_KINDS: readonly ResourceKind[] = ['search', 'reference'];

/**
 * The one field parsed strictly rather than leniently, because it is the one
 * field that becomes something the user clicks under our name.
 *
 * `https` only: an `href` is where a string off the wire turns executable —
 * `javascript:` in an anchor runs on click — and no real resource is anything
 * else anyway. And never an uncredited link: `source` is required on the
 * server, but if one ever arrives without it the host stands in, which is
 * still a true statement about whose page this is.
 */
function parseResource(raw: unknown): Resource | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Record<string, unknown>;

  const kind = str(r.kind) as ResourceKind;
  if (!RESOURCE_KINDS.includes(kind)) return null;

  const label = str(r.label);
  const url = str(r.url);
  if (label === '' || url === '') return null;

  let host = '';
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:') return null;
    host = parsed.hostname.replace(/^www\./, '');
  } catch {
    return null;
  }

  return { kind, label, url, note: str(r.note), source: str(r.source) || host };
}

function parseExplainer(raw: unknown): RawExplainer | null {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null;
  const r = raw as Record<string, unknown>;

  const headline = str(r.headline);
  const whatItIs = str(r.what_it_is);
  // An entry with neither is a stub or a stray scalar in the envelope. Either
  // way there is nothing to teach, so it must not suppress the numbers below.
  if (headline === '' && whatItIs === '') return null;

  const steps: RawFixStep[] = [];
  const fixes = Array.isArray(r.how_to_fix) ? r.how_to_fix : Array.isArray(r.steps) ? r.steps : [];
  for (const item of fixes) {
    const step = parseStep(item);
    if (step) steps.push(step);
  }

  const resources: Resource[] = [];
  if (Array.isArray(r.resources)) {
    for (const item of r.resources) {
      const resource = parseResource(item);
      if (resource) resources.push(resource);
    }
  }

  return {
    headline,
    what_it_is: whatItIs,
    what_you_hear: str(r.what_you_hear),
    why_it_matters: str(r.why_it_matters),
    common_causes: strList(r.common_causes),
    how_to_fix: steps,
    how_to_verify: str(r.how_to_verify),
    learn_more: str(r.learn_more),
    minutes: Math.max(0, Math.round(num(r.minutes, 0))),
    resources,
  };
}

function parseMap(source: Record<string, unknown>): Record<string, RawExplainer> {
  const out: Record<string, RawExplainer> = {};
  for (const [id, value] of Object.entries(source)) {
    const parsed = parseExplainer(value);
    if (parsed) out[id] = parsed;
  }
  return out;
}

/**
 * The explainers may arrive under a key or as the whole body. Try the likely
 * envelopes in order and take the first that yields anything; scalar siblings
 * like `count` fall out on their own because they do not parse.
 */
function readIndex(body: unknown): Record<string, RawExplainer> {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) return {};
  const root = body as Record<string, unknown>;
  for (const candidate of [root.explainers, root.knowledge, root.findings, root]) {
    if (typeof candidate !== 'object' || candidate === null || Array.isArray(candidate)) continue;
    const parsed = parseMap(candidate as Record<string, unknown>);
    if (Object.keys(parsed).length > 0) return parsed;
  }
  return {};
}

/**
 * Labels for capability slugs this build has not heard of. The local table in
 * `data/plugins.ts` is authoritative for the 34 known slugs; this only covers
 * a server that has added a thirty-fifth.
 */
const serverLabels: Record<string, string> = {};

function harvestLabels(body: unknown): void {
  if (typeof body !== 'object' || body === null) return;
  const caps = (body as Record<string, unknown>).capabilities;
  if (typeof caps !== 'object' || caps === null || Array.isArray(caps)) return;
  for (const [slug, value] of Object.entries(caps as Record<string, unknown>)) {
    if (typeof value === 'string') {
      serverLabels[slug] = value;
    } else if (Array.isArray(value) && typeof value[0] === 'string') {
      serverLabels[slug] = value[0];
    } else if (typeof value === 'object' && value !== null) {
      const label = str((value as Record<string, unknown>).label);
      if (label) serverLabels[slug] = label;
    }
  }
}

/** Human name for a capability slug, however unfamiliar. */
export function capabilityLabel(slug: string): string {
  return CAPABILITY_LABELS[slug] ?? serverLabels[slug] ?? slug.replace(/_/g, ' ');
}

/* ------------------------------------------------------------------ */
/* Store — one fetch per page load, shared by every mount              */
/* ------------------------------------------------------------------ */

const EMPTY_INDEX: Record<string, RawExplainer> = {};
const NO_STOCK: string[] = [];

let state: KnowledgeState = { phase: 'idle', index: EMPTY_INDEX, stock: NO_STOCK };
const listeners = new Set<() => void>();

function setState(next: KnowledgeState): void {
  state = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): KnowledgeState {
  return state;
}

const SERVER_STATE: KnowledgeState = { phase: 'idle', index: EMPTY_INDEX, stock: NO_STOCK };
function getServerSnapshot(): KnowledgeState {
  return SERVER_STATE;
}

/**
 * Fetch once. A failure is recorded as `unavailable` and not retried: the
 * content is static, so a second attempt on the next card expansion would
 * cost a round trip to learn the same thing.
 */
function ensureLoaded(): void {
  if (state.phase !== 'idle') return;
  setState({ phase: 'loading', index: EMPTY_INDEX, stock: NO_STOCK });

  void (async () => {
    try {
      const response = await fetch(`${API_BASE}/knowledge`);
      if (!response.ok) throw new Error(String(response.status));
      const body: unknown = await response.json();
      harvestLabels(body);
      const index = readIndex(body);
      const stock = strList((body as Record<string, unknown>)?.stock_capabilities);
      setState(
        Object.keys(index).length > 0
          ? { phase: 'ready', index, stock }
          : { phase: 'unavailable', index: EMPTY_INDEX, stock: NO_STOCK },
      );
    } catch {
      // No /knowledge on this server, or no network. The numeric detail on
      // every finding still renders; the page loses a panel, not its meaning.
      setState({ phase: 'unavailable', index: EMPTY_INDEX, stock: NO_STOCK });
    }
  })();
}

/** Drop the cache and fetch again. Only useful after a deploy mid-session. */
export function reloadKnowledge(): void {
  state = { phase: 'idle', index: EMPTY_INDEX, stock: NO_STOCK };
  ensureLoaded();
}

/* ------------------------------------------------------------------ */
/* Resolution                                                          */
/* ------------------------------------------------------------------ */

/**
 * Mirrors `knowledge.explain()`: exact id, then the dimension's `_hot`/`_thin`
 * variant, then the bare dimension. Kept in step with the backend so the same
 * finding never teaches one thing here and another in an export.
 */
function lookup(
  index: Record<string, RawExplainer>,
  findingId: string,
): { raw: RawExplainer; sourceId: string } | null {
  const direct = index[findingId];
  if (direct) return { raw: direct, sourceId: findingId };

  const dot = findingId.indexOf('.');
  const dimension = dot === -1 ? findingId : findingId.slice(0, dot);

  for (const suffix of ['_hot', '_thin']) {
    if (!findingId.endsWith(suffix)) continue;
    const generic = index[`${dimension}${suffix}`];
    if (generic) return { raw: generic, sourceId: `${dimension}${suffix}` };
  }

  const byDimension = index[dimension];
  return byDimension ? { raw: byDimension, sourceId: dimension } : null;
}

/**
 * Mirrors `applicable_steps()`. A step whose capability the producer owns
 * renders as written; one they cannot do renders its `without` variant, and is
 * dropped when there isn't one. The missing capability is carried through so
 * the panel can say what would have made it easier.
 */
function resolveSteps(raw: RawExplainer, owned: ReadonlySet<string>): ResolvedFixStep[] {
  const out: ResolvedFixStep[] = [];
  for (const step of raw.how_to_fix) {
    if (step.needs === '' || owned.has(step.needs)) {
      out.push({
        action: step.action,
        detail: step.detail,
        requires: step.needs,
        substitutedFor: null,
      });
    } else if (step.without !== '') {
      out.push({
        action: step.without,
        detail: '',
        requires: '',
        substitutedFor: capabilityLabel(step.needs),
      });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Hook                                                                */
/* ------------------------------------------------------------------ */

export interface Knowledge {
  /** False while the first fetch is in flight. */
  ready: boolean;
  /** True once explainers are loaded. False on an older or unreachable server. */
  available: boolean;
  /** How many explainers the server carries. */
  count: number;
  /** Teaching for a finding, steps already resolved. Null when we have nothing. */
  explain: (findingId: string) => ResolvedExplainer | null;
  /** Cheap existence check, for deciding whether to offer the affordance. */
  has: (findingId: string) => boolean;
}

export function useKnowledge(): Knowledge {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const { capabilities } = usePluginVault();

  useEffect(ensureLoaded, []);

  const { index, phase, stock } = snapshot;
  const owned = useMemo(() => new Set([...capabilities, ...stock]), [capabilities, stock]);

  const explain = useCallback(
    (findingId: string): ResolvedExplainer | null => {
      if (!findingId) return null;
      const hit = lookup(index, findingId);
      if (!hit) return null;
      const { raw, sourceId } = hit;
      return {
        findingId,
        sourceId,
        exact: sourceId === findingId,
        headline: raw.headline,
        whatItIs: raw.what_it_is,
        whatYouHear: raw.what_you_hear,
        whyItMatters: raw.why_it_matters,
        commonCauses: raw.common_causes,
        steps: resolveSteps(raw, owned),
        howToVerify: raw.how_to_verify,
        learnMore: raw.learn_more,
        minutes: raw.minutes,
        resources: raw.resources,
      };
    },
    [index, owned],
  );

  const has = useCallback(
    (findingId: string) => Boolean(findingId) && lookup(index, findingId) !== null,
    [index],
  );

  return {
    ready: phase === 'ready' || phase === 'unavailable',
    available: phase === 'ready',
    count: Object.keys(index).length,
    explain,
    has,
  };
}

export default useKnowledge;
