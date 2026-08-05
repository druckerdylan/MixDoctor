import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import DropZone from './DropZone';
import { PluginVaultTrigger } from './PluginVault';
import { usePluginVault } from '../hooks/usePluginVault';
import { useCapabilities } from '../hooks/useCapabilities';
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_BYTES, formatBytes } from '../config';

const EASE = [0.16, 1, 0.3, 1] as const;
const NOTES_LIMIT = 500;

interface GenreGroup {
  label: string;
  genres: string[];
}

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

          <div className="mt-10 grid grid-cols-1 gap-10 lg:grid-cols-[1.15fr_1fr] lg:gap-12">
            {/* 02 — Genre ---------------------------------------------- */}
            <div>
              <SectionHead index="02" label="Genre" note="Required" />

              <p className="mb-5 max-w-md text-xs leading-relaxed text-ink-muted">
                Targets are genre-specific. A trap master is graded against a different loudness
                window, low-end curve and dynamic-range floor than a folk record — scoring both
                against one average is how generic tools get it wrong.
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

            {/* 03 / 04 — Reference and notes ---------------------------- */}
            <div className="space-y-10">
              <div>
                <SectionHead index="03" label="Reference" note="Optional" />
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
                <SectionHead index="04" label="Notes" note="Optional" />
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

          {/* 05 — Plugins ----------------------------------------------- */}
          <div className="mt-10">
            <SectionHead index="05" label="Your plugins" note="Optional" />
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

          {/* 06 — Deep analysis. Hidden entirely when the server cannot run
              separation, rather than offered and then silently ignored. */}
          {caps.stems ? (
          <div className="mt-10">
            <SectionHead index="06" label="Deep analysis" note="Optional · slower" />
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
                    `Ready. ${genre} targets, ${referenceFile ? 'with' : 'no'} reference${
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
