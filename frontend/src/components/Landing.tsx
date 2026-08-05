import { useEffect, useRef } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

import { DIMENSIONS, DIMENSION_LABELS } from '../types/analysis';
import { formatBytes, MAX_UPLOAD_BYTES } from '../config';

const EASE = [0.16, 1, 0.3, 1] as const;

/* ------------------------------------------------------------------ */
/* Animated spectrum field                                             */
/* ------------------------------------------------------------------ */

/** Heat ramp, sub -> air. Same stops as `bg-grade-heat` in the Tailwind config. */
const HEAT_STOPS: Array<[number, string]> = [
  [0.0, '#5B4BFF'],
  [0.22, '#2E7BFF'],
  [0.45, '#4CC9F0'],
  [0.7, '#52F2C4'],
  [1.0, '#C8FF6B'],
];

/** Deterministic per-bar phase offset — same field on every load, no Math.random. */
function seedFor(i: number): number {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  return (s - Math.floor(s)) * Math.PI * 2;
}

/**
 * Rough shape of a full-band musical spectrum: sub rolls in below ~40 Hz, a
 * broadly pink downward tilt, a small presence bump around 3 kHz. It is not a
 * measurement — it is a plausible one, which is the point.
 */
function envelope(u: number): number {
  const subRoll = Math.min(1, Math.pow(u / 0.06, 1.5));
  const tilt = Math.pow(1 - u, 0.55);
  const presence = 1 + 0.3 * Math.exp(-Math.pow((u - 0.62) / 0.13, 2));
  return Math.max(0.05, subRoll * (0.35 + 0.65 * tilt) * presence);
}

function SpectrumField() {
  const reduce = useReducedMotion();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let bars = 0;
    let gradient: CanvasGradient | null = null;
    let hold = new Float32Array(0);

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = wrap.clientWidth;
      h = wrap.clientHeight;
      if (w <= 0 || h <= 0) return;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      bars = Math.max(24, Math.min(150, Math.floor(w / 12)));
      hold = new Float32Array(bars);

      gradient = ctx.createLinearGradient(0, 0, w, 0);
      for (const [stop, color] of HEAT_STOPS) gradient.addColorStop(stop, color);
    };

    const draw = (t: number, dt: number) => {
      if (w <= 0 || h <= 0 || !gradient) return;
      ctx.clearRect(0, 0, w, h);

      const slot = w / bars;
      const barW = Math.max(1.5, slot * 0.52);
      const maxH = h * 0.72;
      const decay = maxH * 0.45 * dt;

      // Bars — one gradient shared by every rect, so colour tracks frequency
      // position for free and we allocate nothing per frame.
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = gradient;
      for (let i = 0; i < bars; i += 1) {
        const u = bars === 1 ? 0 : i / (bars - 1);
        const seed = seedFor(i);
        const mod =
          0.54 +
          0.27 * Math.sin(t * 0.62 + i * 0.2 + seed) +
          0.16 * Math.sin(t * 1.27 - i * 0.09 + seed * 2.1) +
          0.13 * Math.sin(t * 0.21 + i * 0.045);
        const bh = Math.max(2, Math.min(1.05, envelope(u) * Math.max(0.04, mod)) * maxH);

        const prev = hold[i] ?? 0;
        hold[i] = bh > prev ? bh : Math.max(bh, prev - decay);

        ctx.fillRect(i * slot + (slot - barW) / 2, h - bh, barW, bh);
      }

      // Peak-hold trace above the bars — the bright analyser line.
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < bars; i += 1) {
        const x = i * slot + slot / 2;
        const y = h - (hold[i] ?? 0) - 1.5;
        if (i === 0) ctx.moveTo(x, y);
        else {
          const px = (i - 1) * slot + slot / 2;
          const py = h - (hold[i - 1] ?? 0) - 1.5;
          ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2);
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    resize();

    if (reduce) {
      // One representative frame. No loop, no motion.
      draw(0.8, 1);
      const ro = new ResizeObserver(() => {
        resize();
        draw(0.8, 1);
      });
      ro.observe(wrap);
      return () => ro.disconnect();
    }

    let raf = 0;
    let last = performance.now();
    let clock = 0;

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      clock += dt;
      draw(clock, dt);
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      if (raf) return;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    };
    const onVisibility = () => (document.visibilityState === 'visible' ? start() : stop());

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    document.addEventListener('visibilitychange', onVisibility);
    if (document.visibilityState === 'visible') start();

    return () => {
      stop();
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [reduce]);

  return (
    <div
      ref={wrapRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 bottom-0 h-[62%] select-none"
      style={{
        WebkitMaskImage: 'linear-gradient(to top, #000 0%, #000 26%, transparent 96%)',
        maskImage: 'linear-gradient(to top, #000 0%, #000 26%, transparent 96%)',
      }}
    >
      <canvas ref={canvasRef} className="block h-full w-full opacity-70" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* What gets measured                                                  */
/* ------------------------------------------------------------------ */

/** One line of real substance per dimension — what the pass actually computes. */
const DIMENSION_BLURB: Record<(typeof DIMENSIONS)[number], string> = {
  clipping: 'Sample & inter-sample overs, flat-topped runs',
  phase: 'Correlation, polarity, mono-sum loss per band',
  loudness: 'Integrated LUFS, loudness range, short-term peaks',
  limiter: 'Ceiling behaviour, PSR, signs of over-limiting',
  dynamic_range: 'Crest factor, DR value, macro movement',
  compression: 'Pumping rate & depth, gain-reduction estimate',
  frequency_balance: '1/3-octave curve against the genre target',
  mud: '200–500 Hz build-up measured against the midrange',
  harshness: '2–5 kHz bite, sharpness in acum, sibilance',
  low_end: 'Kick fundamental, 808 collision, sub energy',
  vocal_balance: 'Vocal-to-instrument dB, intelligibility, masking',
  stereo_width: 'Width and correlation, band by band',
  transients: 'Attack time, punch index, transient smearing',
  clarity: 'Masking index, spectral contrast, definition',
};

function MeasuredStrip() {
  const reduce = useReducedMotion();

  return (
    <section id="measured" className="relative mx-auto max-w-[1800px] px-4 py-20 sm:px-6 lg:px-10 lg:py-28">
      <motion.div
        initial={reduce ? { opacity: 0 } : { opacity: 0, y: 20 }}
        whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.8, ease: EASE }}
        className="max-w-3xl"
      >
        <p className="eyebrow text-signal-dim">What gets measured</p>
        <h2 className="display mt-4 text-display-md text-ink">
          Fourteen dimensions, each scored 0–100 against your genre&rsquo;s targets.
        </h2>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-ink-dim">
          Every score is derived from the signal, not from a language model reading a
          description of it. The report shows the number, the target it was compared to, and
          the seconds of your track where the problem is worst.
        </p>
      </motion.div>

      <div className="mt-12 grid grid-cols-1 gap-x-8 gap-y-7 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-7">
        {DIMENSIONS.map((dimension, i) => (
          <motion.div
            key={dimension}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 14 }}
            whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.6, ease: EASE, delay: Math.min(i, 8) * 0.035 }}
            className="border-t border-void-line pt-4"
          >
            <div className="flex items-baseline gap-2.5">
              <span className="stat text-micro text-ink-faint">
                {(i + 1).toString().padStart(2, '0')}
              </span>
              <h3 className="font-display text-sm font-semibold tracking-tight text-ink">
                {DIMENSION_LABELS[dimension]}
              </h3>
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
              {DIMENSION_BLURB[dimension]}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Landing                                                             */
/* ------------------------------------------------------------------ */

const UNITS = ['LUFS', 'dBTP', 'LRA', 'PSR', 'CORRELATION', '1/3-OCTAVE', 'CREST'];

export interface LandingProps {
  /** Scrolls to / focuses the intake form. */
  onStart: () => void;
}

export default function Landing({ onStart }: LandingProps) {
  const reduce = useReducedMotion();

  const container = {
    hidden: {},
    show: { transition: { staggerChildren: 0.085, delayChildren: 0.12 } },
  };
  const item = {
    hidden: reduce ? { opacity: 0 } : { opacity: 0, y: 22 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.9, ease: EASE },
    },
  };

  return (
    <>
      <section className="relative flex min-h-[calc(100svh-3.5rem)] flex-col justify-center overflow-hidden sm:min-h-[calc(100svh-4rem)]">
        <SpectrumField />

        {/*
          Title-card scrim. The analyser runs behind the copy, and at the bottom
          of the hero the bars are dense enough to eat the small format line.
          A left-weighted wash keeps the text column readable while leaving the
          right side of the field fully visible — the same trick a title
          sequence uses to hold type over footage.
        */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-[5]"
          style={{
            background:
              'linear-gradient(100deg, rgba(6,6,10,0.94) 0%, rgba(6,6,10,0.86) 32%, rgba(6,6,10,0.45) 58%, rgba(6,6,10,0) 82%)',
          }}
        />

        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="relative z-10 mx-auto w-full max-w-[1800px] px-4 py-20 sm:px-6 lg:px-10"
        >
          <motion.div variants={item} className="flex items-center gap-3">
            <span className="h-1.5 w-1.5 rounded-full bg-signal shadow-glow" aria-hidden="true" />
            <p className="eyebrow text-ink-dim">Mix diagnostics · 14 dimensions</p>
          </motion.div>

          <motion.h1 variants={item} className="display mt-7 max-w-[16ch] text-display-xl text-ink">
            Every flaw,
            <br />
            <span className="text-gradient">timestamped.</span>
          </motion.h1>

          <motion.p
            variants={item}
            className="mt-8 max-w-[58ch] text-base leading-relaxed text-ink-dim sm:text-lg"
          >
            Upload a bounce. Fourteen measurement passes — loudness, true peak, phase, masking,
            kick-versus-808 collision, transient smear — scored against your genre&rsquo;s targets,
            with the exact moves that fix them and the second of the track where it goes wrong.
          </motion.p>

          <motion.ul
            variants={item}
            className="no-scrollbar mask-fade-r mt-8 flex gap-2 overflow-x-auto pb-1"
          >
            {UNITS.map((unit) => (
              <li
                key={unit}
                className="shrink-0 rounded-full border border-void-line bg-void-panel/60 px-3 py-1.5 font-mono text-micro uppercase text-ink-muted"
              >
                {unit}
              </li>
            ))}
          </motion.ul>

          <motion.div variants={item} className="mt-10 flex flex-wrap items-center gap-3">
            <button type="button" onClick={onStart} className="btn-primary">
              Analyse a mix
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M5 12h13m0 0-5-5m5 5-5 5"
                  stroke="currentColor"
                  strokeWidth={1.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            <a href="#measured" className="btn-ghost">
              What gets measured
            </a>
          </motion.div>

          <motion.p variants={item} className="mt-6 font-mono text-micro uppercase text-ink-faint">
            WAV · MP3 · FLAC · AIFF · M4A · OGG — up to {formatBytes(MAX_UPLOAD_BYTES)} ·
            {' '}typical analysis 10–40s
          </motion.p>
        </motion.div>
      </section>

      <MeasuredStrip />
    </>
  );
}
