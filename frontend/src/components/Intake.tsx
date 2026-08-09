import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import DropZone from './DropZone';
import { PluginVaultTrigger } from './PluginVault';
import { usePluginVault } from '../hooks/usePluginVault';
import { useCapabilities } from '../hooks/useCapabilities';
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_BYTES, formatBytes } from '../config';
import { TRACK_INTENT_SHORT, type TrackIntent } from '../types/analysis';

const EASE = [0.16, 1, 0.3, 1] as const;
const NOTES_LIMIT = 500;

interface GenreGroup {
  label: string;
  genres: string[];
}

interface IntentOption {
  value: TrackIntent;
  title: string;
  /** One line, plain. What the file is — not what the analyser will do with it. */
  line: string;
}

/**
 * What the file is, asked before genre because it changes more.
 *
 * Genre picks the reference a mix is measured against. This picks which
 * questions are worth asking of the file at all — and getting it wrong is how
 * a beat with the hook tucked under the drums gets told its vocal is buried and
 * its hi-hats are sibilance.
 */
const INTENT_OPTIONS: IntentOption[] = [
  { value: 'full_mix', title: 'Full mix', line: 'A finished song with a lead vocal.' },
  {
    value: 'beat',
    title: 'Beat / instrumental for a topline',
    line: 'Someone will rap or sing over this.',
  },
  { value: 'instrumental', title: 'Instrumental', line: 'Finished, no vocal expected.' },
  { value: 'stem', title: 'Single stem', line: 'One element on its own.' },
  {
    value: 'reference',
    title: 'Reference track',
    line: "Someone else's record, measuring it to learn.",
  },
  { value: 'demo', title: 'Rough demo', line: 'Early, not polished yet.' },
];

/**
 * The promise attached to each choice, shown the moment it is made.
 *
 * Every sentence here names something the analyser actually stops doing or
 * starts doing — these are the intent gates in `detectors.py`, in plain words.
 * If a gate changes there, change the sentence here: an unkept promise on this
 * screen is worse than no promise at all.
 */
/** Reads inside "Ready. Read as …" on the submit line. Spelled out rather than
 *  built from a short label and an article, which would produce "a
 *  instrumental" the first time somebody picked it. */
const INTENT_AS: Record<TrackIntent, string> = {
  full_mix: 'a full mix',
  beat: 'a beat for a topline',
  instrumental: 'an instrumental',
  stem: 'a single stem',
  reference: 'a reference track',
  demo: 'a rough demo',
};

const INTENT_PROMISE: Record<TrackIntent, string> = {
  full_mix:
    'Everything is measured, the lead vocal’s level against the music among it. This is the strictest reading — a finished song is held to the whole checklist.',
  beat:
    'The lead is meant to be absent or tucked, so nothing here will call that a fault. Bursty top end is read as hi-hats, shakers and rim clicks rather than vocal sibilance. Mids sitting light is the pocket you left for the topline, not a hole in the mix. And nobody tells you it hasn’t been mastered — that headroom is the room the vocalist needs.',
  instrumental:
    'No lead is expected, so vocal balance is not scored and the top end is read as percussion rather than consonants. Everything else is judged as a finished record.',
  stem:
    'Judged as one element, not as a mix. Mud, clarity, frequency balance and vocal balance stand down — a bass stem is supposed to be all low end, and a vocal stem has no kick to collide with. Every number still appears under Details.',
  reference:
    'Measured, not marked. You get every figure and no prescription, because nobody needs to be told to fix a record that already came out.',
  demo:
    'Nothing is judged as a master you haven’t attempted yet: “too quiet to compete”, “not mastered” and the limiter’s behaviour all stand down. Being louder than the genre is still reported — a rough that’s already slammed is still slammed. Anything genuinely broken still surfaces too: a rough with an inverted channel is still an inverted channel.',
};

/**
 * Grouped rather than an alphabetical list, because the grouping *is* the
 * information: everything inside a family shares a target curve and differs
 * only in the scalar windows.
 */
const GENRE_GROUPS: GenreGroup[] = [
  { label: 'Pop & urban', genres: ['Pop', 'Hip-Hop', 'Trap', 'R&B/Soul'] },
  { label: 'Electronic', genres: ['EDM/Electronic', 'House', 'Techno', 'Drum & Bass'] },
  { label: 'Guitar', genres: ['Rock', 'Metal', 'Punk', 'Indie/Alternative'] },
  { label: 'Roots', genres: ['Country', 'Acoustic', 'Folk'] },
  { label: 'Composed', genres: ['Jazz', 'Classical', 'Orchestral', 'Cinematic'] },
  { label: 'Texture', genres: ['Ambient', 'Lo-Fi', 'Other'] },
];

export interface IntakeSubmission {
  file: File;
  /**
   * What the file is. Decides which findings are worth asking for at all, which
   * is why it is collected before genre and defaulted rather than left blank.
   */
  intent: TrackIntent;
  genre: string;
  referenceFile: File | null;
  notes: string | null;
  /** Preview peaks for the main file, handed on so the wait can draw them. */
  peaks: number[] | null;
  /** Opt-in source separation. Off by default — it costs real minutes. */
  separateStems: boolean;
}

export interface IntakeProps {
  onSubmit: (submission: IntakeSubmission) => void;
  /** Server-side failure from the last attempt. */
  error?: string | null;
  busy?: boolean;
  /**
   * Repopulates the form after a failed attempt. Without it, a failure would
   * make the user re-pick a file that may have taken a minute to upload.
   */
  restore?: IntakeSubmission | null;
}

function SectionHead({ index, label, note }: { index: string; label: string; note?: string }) {
  return (
    <div className="mb-4 flex items-baseline gap-3">
      <span className="stat text-micro text-signal-dim">{index}</span>
      <span className="eyebrow text-ink-dim">{label}</span>
      {note && <span className="font-mono text-micro uppercase text-ink-faint">{note}</span>}
    </div>
  );
}

export default function Intake({
  onSubmit,
  error = null,
  busy = false,
  restore = null,
}: IntakeProps) {
  const caps = useCapabilities();
  const reduce = useReducedMotion();

  const [file, setFile] = useState<File | null>(restore?.file ?? null);
  const [peaks, setPeaks] = useState<number[] | null>(restore?.peaks ?? null);
  // Defaulted, not blank: the commonest case should cost nobody a click, and an
  // unanswered "what is this?" would have to fall back to full_mix anyway.
  const [intent, setIntent] = useState<TrackIntent>(restore?.intent ?? 'full_mix');
  const [genre, setGenre] = useState<string>(restore?.genre ?? '');
  const [referenceFile, setReferenceFile] = useState<File | null>(restore?.referenceFile ?? null);
  const [notes, setNotes] = useState(restore?.notes ?? '');
  const [separateStems, setSeparateStems] = useState(restore?.separateStems ?? false);
  const [showRequirements, setShowRequirements] = useState(false);

  /** Read-only here — the picker itself lives in the drawer. */
  const { count: pluginCount, capabilities: pluginCapabilities } = usePluginVault();

  const blocker = useMemo(() => {
    if (!file) return 'Load a mix to analyse.';
    if (!genre) return 'Choose a genre — the targets depend on it.';
    return null;
  }, [file, genre]);

  const ready = blocker === null;

  return (
    <section id="intake" className="relative mx-auto max-w-[1800px] px-4 pb-28 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-5xl">
        <div className="hairline mb-14" />

        <p className="eyebrow text-signal-dim">Start an analysis</p>
        <h2 className="display mt-4 text-display-md text-ink">Load a bounce.</h2>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-ink-dim">
          Stereo master or rough mix, pre-master or post. Nothing needs to be finished — the
          report tells you how far from finished it is.
        </p>

        <form
          className="panel mt-10 p-5 sm:p-8"
          onSubmit={(event) => {
            event.preventDefault();
            if (!file || !genre || busy) {
              setShowRequirements(true);
              return;
            }
            onSubmit({
              file,
              intent,
              genre,
              referenceFile,
              notes: notes.trim() === '' ? null : notes.trim(),
              peaks,
              separateStems,
            });
          }}
        >
          {/* 01 — Source ------------------------------------------------ */}
          <div>
            <SectionHead index="01" label="Source" />
            <DropZone
              id="intake-source"
              file={file}
              onFile={(next) => {
                setFile(next);
                if (next) setShowRequirements(false);
              }}
              onPeaks={setPeaks}
              initialPeaks={restore?.peaks ?? null}
              disabled={busy}
              title="The mix"
              hint={`${ACCEPTED_EXTENSIONS.map((e) => e.replace('.', '').toUpperCase()).join(' · ')} — up to ${formatBytes(MAX_UPLOAD_BYTES)}. Highest-quality bounce you have; MP3 hides the top octave.`}
            />
          </div>

          {/* 02 — What is this? ------------------------------------------
              Ahead of genre on purpose, and full width rather than tucked in a
              column, because it changes more about the report than genre does.
              Genre picks the reference; this picks which questions get asked. */}
          <div className="mt-10">
            <SectionHead index="02" label="What is this?" note="Changes the most" />

            <p className="mb-5 max-w-2xl text-xs leading-relaxed text-ink-muted">
              Genre sets the reference this gets measured against. This sets which questions are
              worth asking of it at all. A beat with the hook tucked under the drums is doing its
              job — say so here and nothing in the report will pretend otherwise.
            </p>

            <div
              role="radiogroup"
              aria-label="What is this?"
              className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3"
            >
              {INTENT_OPTIONS.map((option) => {
                const active = intent === option.value;
                return (
                  <label key={option.value} className="block h-full cursor-pointer">
                    <input
                      type="radio"
                      name="intent"
                      value={option.value}
                      checked={active}
                      disabled={busy}
                      onChange={() => setIntent(option.value)}
                      className="peer sr-only"
                    />
                    <span
                      className={[
                        'flex h-full items-start gap-2.5 rounded-xl border p-3.5',
                        'transition-all duration-200 ease-cine',
                        'peer-focus-visible:outline peer-focus-visible:outline-2',
                        'peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal',
                        active
                          ? 'border-signal bg-signal/10'
                          : 'border-void-line hover:border-ink-faint hover:bg-void-raised',
                      ].join(' ')}
                    >
                      {/* Shape, not just colour, marks the selection. */}
                      <span
                        aria-hidden="true"
                        className={[
                          'mt-[3px] h-2 w-2 shrink-0 rounded-full',
                          active ? 'bg-signal' : 'border border-void-line',
                        ].join(' ')}
                      />
                      <span className="min-w-0">
                        <span
                          className={`block font-display text-[13px] font-semibold leading-tight tracking-tight ${
                            active ? 'text-signal' : 'text-ink'
                          }`}
                        >
                          {option.title}
                        </span>
                        <span className="mt-1 block text-[12px] leading-snug text-ink-muted">
                          {option.line}
                        </span>
                      </span>
                    </span>
                  </label>
                );
              })}
            </div>

            {/* The consequence, stated the moment the choice is made. Without
                this the control is six radio buttons and a shrug; with it the
                promise is on screen before the upload ever runs. */}
            <div className="mt-4 rounded-xl border border-void-line/70 bg-void-deep/40 p-4">
              <p className="eyebrow mb-2 text-signal-dim">
                What that changes · {TRACK_INTENT_SHORT[intent]}
              </p>
              <AnimatePresence mode="wait" initial={false}>
                <motion.p
                  key={intent}
                  initial={reduce ? { opacity: 0 } : { opacity: 0, y: 4 }}
                  animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3, ease: EASE }}
                  aria-live="polite"
                  className="max-w-3xl text-xs leading-relaxed text-ink-dim"
                >
                  {INTENT_PROMISE[intent]}
                </motion.p>
              </AnimatePresence>
            </div>
          </div>

          <div className="mt-10 grid grid-cols-1 gap-10 lg:grid-cols-[1.15fr_1fr] lg:gap-12">
            {/* 03 — Genre ---------------------------------------------- */}
            <div>
              <SectionHead index="03" label="Genre" note="Required" />

              <p className="mb-5 max-w-md text-xs leading-relaxed text-ink-muted">
                A trap master is measured against a different loudness window, low-end curve and
                dynamic-range floor than a folk record — holding both to one average is how generic
                tools get it wrong. Genre is a reference, though, not a rulebook: departing from it
                comes back as a difference with its cost and its upside, never as a fault.
              </p>

              <div
                role="radiogroup"
                aria-label="Genre"
                aria-required="true"
                className="space-y-4"
              >
                {GENRE_GROUPS.map((group) => (
                  <div key={group.label}>
                    <p className="mb-2 font-mono text-micro uppercase text-ink-faint">
                      {group.label}
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {group.genres.map((option) => {
                        const active = genre === option;
                        return (
                          <label key={option} className="cursor-pointer">
                            <input
                              type="radio"
                              name="genre"
                              value={option}
                              checked={active}
                              disabled={busy}
                              onChange={() => {
                                setGenre(option);
                                setShowRequirements(false);
                              }}
                              className="peer sr-only"
                            />
                            <span
                              className={[
                                'flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs',
                                'transition-all duration-200 ease-cine',
                                'peer-focus-visible:outline peer-focus-visible:outline-2',
                                'peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal',
                                active
                                  ? 'border-signal bg-signal/10 font-medium text-signal'
                                  : 'border-void-line text-ink-dim hover:border-ink-faint hover:bg-void-raised hover:text-ink',
                              ].join(' ')}
                            >
                              {/* Shape, not just colour, marks the selection. */}
                              <span
                                aria-hidden="true"
                                className={
                                  active
                                    ? 'h-1.5 w-1.5 rounded-full bg-signal'
                                    : 'h-1.5 w-1.5 rounded-full border border-void-line'
                                }
                              />
                              {option}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 04 / 05 — Reference and notes ---------------------------- */}
            <div className="space-y-10">
              <div>
                <SectionHead index="04" label="Reference" note="Optional" />
                <DropZone
                  id="intake-reference"
                  file={referenceFile}
                  onFile={setReferenceFile}
                  disabled={busy}
                  compact
                  title="Reference track"
                  hint="A released record you want this to sit next to. Adds a spectrum, width, dynamics and loudness delta against it."
                />
              </div>

              <div>
                <SectionHead index="05" label="Notes" note="Optional" />
                <label htmlFor="intake-notes" className="sr-only">
                  Notes for the engineer
                </label>
                <textarea
                  id="intake-notes"
                  rows={4}
                  value={notes}
                  disabled={busy}
                  maxLength={NOTES_LIMIT}
                  onChange={(event) => setNotes(event.target.value)}
                  placeholder="What are you worried about? e.g. “vocal disappears in the chorus”, “low end falls apart in the car”, “heading to mastering Friday”."
                  className="input-field resize-none text-sm leading-relaxed"
                />
                <div className="mt-2 flex items-center justify-between">
                  <p className="font-mono text-micro uppercase text-ink-faint">
                    Aims the report at your actual problem
                  </p>
                  <p className="stat text-micro text-ink-faint">
                    {notes.length}/{NOTES_LIMIT}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* 06 — Plugins ----------------------------------------------- */}
          <div className="mt-10">
            <SectionHead index="06" label="Your plugins" note="Optional" />
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <p className="max-w-xl text-xs leading-relaxed text-ink-muted">
                Fixes are written against the tools you actually have. With soothe2 on the list a
                resonance becomes “set the depth and let it track”; without it the same finding
                becomes a static notch you have to automate. Stock EQ, compressor and limiter are
                always assumed, so this only ever makes the advice more specific.
              </p>
              <div className="shrink-0 sm:text-right">
                <PluginVaultTrigger className="btn-ghost px-4 py-2.5 text-xs" />
                {pluginCount > 0 && (
                  <p className="mt-2 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                    {pluginCapabilities.length} capabilities available
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* 07 — Deep analysis. Hidden entirely when the server cannot run
              separation, rather than offered and then silently ignored. */}
          {caps.stems ? (
          <div className="mt-10">
            <SectionHead index="07" label="Deep analysis" note="Optional · slower" />
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="max-w-xl">
                <p className="text-xs leading-relaxed text-ink-muted">
                  Separating the mix into vocals, drums, bass and music takes a few extra minutes,
                  and in exchange every per-element number stops being a guess: the vocal level is
                  measured against the actual mix, compression is read on the element that has it,
                  and “the music is burying the vocal” becomes a figure in dB instead of an
                  inference from a busy band.
                </p>
                <p
                  id="intake-stems-tradeoff"
                  className="mt-2 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint"
                >
                  Honest tradeoff: minutes instead of seconds. Everything else in the report is
                  unchanged.
                </p>
              </div>

              <div className="shrink-0">
                <label className="inline-flex cursor-pointer items-center gap-3">
                  <input
                    type="checkbox"
                    checked={separateStems}
                    disabled={busy}
                    aria-describedby="intake-stems-tradeoff"
                    onChange={(event) => setSeparateStems(event.target.checked)}
                    className="peer sr-only"
                  />
                  {/* Track and knob are driven off state rather than
                      peer-checked: the knob is a child of the track, not a
                      sibling of the input, so the peer variant cannot reach it. */}
                  <span
                    className={[
                      'relative h-6 w-11 shrink-0 rounded-full border transition-colors duration-300 ease-cine',
                      'peer-focus-visible:outline peer-focus-visible:outline-2',
                      'peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal',
                      separateStems ? 'border-signal bg-signal/20' : 'border-void-line bg-void-deep',
                      busy ? 'opacity-40' : '',
                    ].join(' ')}
                  >
                    <span
                      aria-hidden="true"
                      className={[
                        'absolute top-[3px] h-[18px] w-[18px] rounded-full transition-all duration-300 ease-cine',
                        separateStems ? 'left-[23px] bg-signal' : 'left-[3px] bg-ink-faint',
                      ].join(' ')}
                      style={
                        separateStems
                          ? { boxShadow: '0 0 14px -2px rgba(82,242,196,0.85)' }
                          : undefined
                      }
                    />
                  </span>
                  <span className="text-xs text-ink-dim">
                    Separate into stems
                    <span className="ml-2 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                      {separateStems ? 'on' : 'off'}
                    </span>
                  </span>
                </label>
              </div>
            </div>
          </div>
          ) : null}

          <div className="hairline my-8" />

          {/* Submit ---------------------------------------------------- */}
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <AnimatePresence mode="wait" initial={false}>
                <motion.p
                  key={error ? 'error' : blocker ?? 'ready'}
                  initial={reduce ? { opacity: 0 } : { opacity: 0, y: 4 }}
                  animate={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3, ease: EASE }}
                  className={[
                    'text-xs leading-relaxed',
                    error
                      ? 'text-sev-critical'
                      : ready
                        ? 'text-ink-muted'
                        : showRequirements
                          ? 'text-sev-major'
                          : 'text-ink-faint',
                  ].join(' ')}
                  {...(error ? { role: 'alert' as const } : {})}
                >
                  {error ??
                    blocker ??
                    `Ready. Read as ${INTENT_AS[intent]} against ${genre} targets, ${
                      referenceFile ? 'with' : 'no'
                    } reference${
                      separateStems ? ', stem separation on — expect a few minutes' : ''
                    }.`}
                </motion.p>
              </AnimatePresence>
            </div>

            <button
              type="submit"
              className="btn-primary shrink-0"
              disabled={busy}
              aria-disabled={!ready || busy}
            >
              {busy ? 'Analysing…' : 'Run analysis'}
              {!busy && (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path
                    d="M5 12h13m0 0-5-5m5 5-5 5"
                    stroke="currentColor"
                    strokeWidth={1.8}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
