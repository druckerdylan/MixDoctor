/**
 * The answer side of the question flow.
 *
 * `POST /reassess` is stateless in exactly the way `/engineer` is: the client
 * posts back the analysis it already holds, plus what the producer said about
 * it, and gets a re-scored report. No audio moves and no DSP runs, so this is
 * milliseconds — which is what lets an answer feel like a toggle rather than a
 * submission.
 *
 * Two rules the UI depends on and the server enforces:
 *   - a defect is never answerable, so no yes can excuse a clipped render;
 *   - `technical` and `mastering_ready` never move, whatever is answered.
 *
 * Everything that *does* move — the dimensions, `reference_match`, the headline
 * — comes back on the response. Nothing here recomputes a score locally.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { API_BASE } from '../config';
import {
  readAnswers,
  selectClarifications,
  storageKeyFor,
  toAnswerList,
  writeAnswers,
  type AnswerMap,
} from '../lib/clarify';
import type { Finding, MixAnalysis } from '../types/analysis';

export type ClarifyStatus = 'idle' | 'saving' | 'error';

export interface UseClarificationsReturn {
  /** At most four, already ordered. Empty means there is nothing to ask. */
  questions: Finding[];
  /** `{findingId: intended}` for every question answered so far. */
  answers: AnswerMap;
  answeredCount: number;
  status: ClarifyStatus;
  error: string | null;
  answer: (findingId: string, intended: boolean) => void;
  /** "Nothing here is a mistake" — one round trip, every open question at once. */
  acknowledgeAll: () => void;
  /** Available only straight after `acknowledgeAll`, and only until the next answer. */
  undoAll: (() => void) | null;
}

export function useClarifications(
  analysis: MixAnalysis,
  onAnalysisChange?: (next: MixAnalysis) => void,
): UseClarificationsReturn {
  const filename = analysis.filename ?? '';
  const findings = useMemo<Finding[]>(() => analysis.findings ?? [], [analysis.findings]);

  /**
   * The set is chosen once per report and then held still. Selecting over the
   * unanswered findings — which is what the backend does — would drop a card
   * the instant it was answered and promote a fifth question in its place,
   * turning a four-question conversation into an interrogation that refills.
   */
  const questions = useMemo(
    () => selectClarifications(findings, { includeAnswered: true }),
    [findings],
  );

  const [answers, setAnswers] = useState<AnswerMap>({});
  const [status, setStatus] = useState<ClarifyStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [bulkUndo, setBulkUndo] = useState<AnswerMap | null>(null);

  // Refs so a POST always sees the current report, the current handler and the
  // current answers without the callbacks having to be rebuilt on every render.
  const analysisRef = useRef(analysis);
  analysisRef.current = analysis;
  const onChangeRef = useRef(onAnalysisChange);
  onChangeRef.current = onAnalysisChange;
  const answersRef = useRef(answers);
  answersRef.current = answers;

  const runRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const restoredFor = useRef<string | null>(null);
  /**
   * Whether the remembered answers have actually made it to the server.
   *
   * Separate from `restoredFor` because the two can disagree: unmounting aborts
   * whatever is in flight, and React's StrictMode mounts, unmounts and mounts
   * again in development — which aborted the restore mid-request and, with a
   * single "have I restored yet" flag, left the answers on screen but never
   * applied to the report. The retry is the whole point of the second flag.
   */
  const restoreDone = useRef(false);

  // In flight when the component goes away: drop it. Letting it land would
  // hand a stale report to a page that has since moved on to another mix.
  useEffect(
    () => () => {
      abortRef.current?.abort();
      abortRef.current = null;
    },
    [],
  );

  const storageKey = useMemo(() => storageKeyFor(filename), [filename]);

  /**
   * Send the whole answer set, not a delta.
   *
   * `apply_answers` treats each answer as an assignment and leaves unmentioned
   * findings alone, so posting the full map means the server's state converges
   * on ours no matter which request wins the race or which one got dropped.
   */
  const post = useCallback(
    async (next: AnswerMap, previous: AnswerMap) => {
      const run = ++runRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus('saving');
      setError(null);

      try {
        const response = await fetch(`${API_BASE}/reassess`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            analysis: analysisRef.current,
            answers: toAnswerList(next),
          }),
          signal: controller.signal,
        });
        if (runRef.current !== run) return;

        if (!response.ok) throw new Error(`The report could not be re-scored (HTTP ${response.status}).`);

        const updated = (await response.json()) as MixAnalysis;
        if (runRef.current !== run) return;

        onChangeRef.current?.(updated);
        setStatus('idle');
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (runRef.current !== run) return;
        // Put the answer back the way it was. A toggle that silently failed
        // would leave the user believing the report had changed when it had
        // not, which is worse than the failure.
        answersRef.current = previous;
        setAnswers(previous);
        writeAnswers(storageKey, previous);
        setBulkUndo(null);
        setError(
          err instanceof TypeError
            ? `Could not reach ${API_BASE} to update the report.`
            : err instanceof Error
              ? err.message
              : 'The report could not be re-scored.',
        );
        setStatus('error');
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [storageKey],
  );

  /**
   * A new file: load what this track was told last time.
   *
   * Keyed on the filename and guarded by a ref, so the analysis coming back
   * from `/reassess` — a new object every time — cannot re-trigger the restore
   * and start a loop.
   */
  useEffect(() => {
    const sameTrack = restoredFor.current === filename;
    if (sameTrack && restoreDone.current) return;

    if (!sameTrack) {
      restoredFor.current = filename;
      restoreDone.current = false;
      setStatus('idle');
      setError(null);
      setBulkUndo(null);
    }

    const askable = new Set(questions.map((q) => q.id));

    // What the payload already says, first: a report that arrives already
    // reassessed must show its answers rather than asking again.
    const fromPayload: AnswerMap = {};
    for (const q of questions) if (q.acknowledged) fromPayload[q.id] = true;

    // Then what this track was told last time, which is more recent. Answers
    // about questions this report does not ask are dropped rather than posted.
    const stored = readAnswers(storageKeyFor(filename));
    const restored: AnswerMap = { ...fromPayload };
    for (const [id, intended] of Object.entries(stored)) {
      if (askable.has(id)) restored[id] = intended;
    }

    answersRef.current = restored;
    setAnswers(restored);

    // Nothing to send once the report already agrees with what is remembered —
    // which is also how this settles after the round trip succeeds.
    const needsSync = questions.some((q) => Boolean(restored[q.id]) !== Boolean(q.acknowledged));
    if (!needsSync) {
      restoreDone.current = true;
      return;
    }
    void post(restored, fromPayload);
  }, [filename, questions, post]);

  /**
   * Optimistic by design. The answer lands on screen on the click and the
   * server catches up; a question that greys out for a round trip stops being
   * a conversation. A failure puts it back — see the catch in `post`.
   */
  const commit = useCallback(
    (next: AnswerMap) => {
      const previous = answersRef.current;
      answersRef.current = next;
      setAnswers(next);
      writeAnswers(storageKey, next);
      void post(next, previous);
    },
    [post, storageKey],
  );

  const answer = useCallback(
    (findingId: string, intended: boolean) => {
      setBulkUndo(null);
      commit({ ...answersRef.current, [findingId]: intended });
    },
    [commit],
  );

  const acknowledgeAll = useCallback(() => {
    const previous = answersRef.current;
    const next: AnswerMap = { ...previous };
    for (const q of questions) next[q.id] = true;
    setBulkUndo(previous);
    commit(next);
  }, [commit, questions]);

  const undoAll = useCallback(() => {
    if (!bulkUndo) return;
    const previous = bulkUndo;
    setBulkUndo(null);
    commit(previous);
  }, [bulkUndo, commit]);

  const answeredCount = useMemo(
    () => questions.filter((q) => q.id in answers).length,
    [questions, answers],
  );

  return {
    questions,
    answers,
    answeredCount,
    status,
    error,
    answer,
    acknowledgeAll,
    undoAll: bulkUndo ? undoAll : null,
  };
}

export default useClarifications;
