/**
 * Which questions to ask, and where the answers are remembered.
 *
 * Mirrors `backend/analysis/clarify.py` — `ASK_LIMIT`, `_ask_rank`, `pending`
 * and `select`. The backend's own docstring says the set of questions a user is
 * shown is decided once so it cannot drift between the API and the UI; that
 * helper is Python and never crosses the wire, so this is the copy, and it has
 * to stay a copy. If the ranking changes there, change it here in the same
 * commit.
 */

import type { ClarificationAnswer, Finding } from '../types/analysis';

/**
 * How many questions a report is allowed to actually ask.
 *
 * Fifteen is a form. Four is a conversation, and the difference decides whether
 * anybody answers any of them.
 */
export const ASK_LIMIT = 4;

function finite(v: number | undefined | null): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** Highest impact first, then the largest miss, then the id for stability. */
function byAskRank(a: Finding, b: Finding): number {
  const impact = finite(b.impact) - finite(a.impact);
  if (Math.abs(impact) > 1e-9) return impact;
  const miss = finite(b.miss_ratio) - finite(a.miss_ratio);
  if (Math.abs(miss) > 1e-9) return miss;
  return a.id.localeCompare(b.id);
}

export interface SelectOptions {
  limit?: number;
  /**
   * Keep answered questions in the set. The backend's `pending()` drops them,
   * because it is choosing what to ask next; the UI is choosing what to *show*,
   * and a card that vanishes the moment you answer it cannot be un-answered.
   *
   * The two agree where it matters: a report arrives with nothing acknowledged,
   * so this selects exactly what `select()` would. And because `impact`,
   * `miss_ratio` and `dimension` are untouched by an answer, the set stays put
   * as the user works through it instead of reshuffling under their hands.
   */
  includeAnswered?: boolean;
}

/** Every finding carrying a question, most worth asking first. */
export function pendingClarifications(
  findings: readonly Finding[],
  includeAnswered = false,
): Finding[] {
  return findings
    .filter((f) => Boolean(f.clarification) && (includeAnswered || !f.acknowledged))
    .sort(byAskRank);
}

/**
 * The questions to actually put in front of somebody. At most `limit`.
 *
 * One per dimension first, then fill: two questions about the low end in a set
 * of four reads as the analyzer labouring a point, and spreading them means the
 * four cover four different parts of the record.
 */
export function selectClarifications(
  findings: readonly Finding[],
  { limit = ASK_LIMIT, includeAnswered = false }: SelectOptions = {},
): Finding[] {
  if (limit <= 0) return [];

  const ranked = pendingClarifications(findings, includeAnswered);
  const primary: Finding[] = [];
  const extra: Finding[] = [];
  const seen = new Set<string>();

  for (const finding of ranked) {
    if (seen.has(finding.dimension)) extra.push(finding);
    else {
      seen.add(finding.dimension);
      primary.push(finding);
    }
  }

  const chosen = primary.slice(0, limit);
  for (const finding of extra) {
    if (chosen.length >= limit) break;
    chosen.push(finding);
  }
  return chosen.sort(byAskRank);
}

/** `{id: intended}` — the shape both the UI and localStorage hold answers in. */
export type AnswerMap = Record<string, boolean>;

export function toAnswerList(answers: AnswerMap): ClarificationAnswer[] {
  return Object.entries(answers).map(([finding_id, intended]) => ({ finding_id, intended }));
}

/* ------------------------------------------------------------------ memory */

/**
 * Answers are remembered per filename, so bouncing a revision and running it
 * through again does not ask the same four questions about the same deliberate
 * decisions. Versioned because the shape is stored, not derived — a v2 that
 * meets a v1 payload ignores it rather than mis-reading it.
 */
const STORAGE_PREFIX = 'mixdoctor.clarify.v1:';

export function storageKeyFor(filename: string): string | null {
  const name = filename.trim();
  return name ? `${STORAGE_PREFIX}${name}` : null;
}

/** Never throws: Safari in private mode makes every one of these calls a risk. */
export function readAnswers(key: string | null): AnswerMap {
  if (!key || typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return {};
    const out: AnswerMap = {};
    for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === 'boolean') out[id] = value;
    }
    return out;
  } catch {
    return {};
  }
}

export function writeAnswers(key: string | null, answers: AnswerMap): void {
  if (!key || typeof window === 'undefined') return;
  try {
    if (Object.keys(answers).length === 0) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, JSON.stringify(answers));
  } catch {
    // A full or disabled store costs the memory, not the answer — the answer
    // itself already went to the server and is on screen.
  }
}
