import { useEffect, useId } from 'react';
import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from 'framer-motion';
import { SEVERITY_VAR, severityFromScore, type Severity } from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

/** viewBox is a fixed 100x100 square so every dimension below is a percentage
 *  of the ring and the whole thing scales with its container. */
const C = 50;
const R = 39;
const W = 5.4;
const CIRC = 2 * Math.PI * R;

const SEV_CLASS: Record<Severity, string> = {
  critical: 'sev-critical',
  major: 'sev-major',
  minor: 'sev-minor',
  clean: 'sev-clean',
};

const SEV_WORD: Record<Severity, string> = {
  critical: 'Critical',
  major: 'Needs work',
  minor: 'Nearly there',
  clean: 'Clean',
};

function clamp100(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(100, Math.max(0, v));
}

/** Point on the ring for a 0-100 value, 0 at twelve o'clock, clockwise. */
function polar(radius: number, value: number): { x: number; y: number } {
  const a = (value / 100) * Math.PI * 2 - Math.PI / 2;
  return { x: C + radius * Math.cos(a), y: C + radius * Math.sin(a) };
}

export interface ScoreRingProps {
  score: number;
  ceiling: number;
  grade: string;
  /** Rendered edge length in px. Shrinks to fit narrower containers. */
  size?: number;
}

export default function ScoreRing({ score, ceiling, grade, size = 260 }: ScoreRingProps) {
  const reduce = useReducedMotion() ?? false;
  const uid = useId().replace(/:/g, '');
  const glowId = `ring-glow-${uid}`;

  const target = clamp100(score);
  const cap = Math.max(target, clamp100(ceiling));
  const headroom = Math.round(cap - target);
  const sev = severityFromScore(target);
  const sevColor = SEVERITY_VAR[sev];
  const gradeText = grade && grade.trim() ? grade.trim() : '—';

  const scoreMV = useMotionValue(reduce ? target : 0);
  const ceilMV = useMotionValue(reduce ? cap : 0);

  // With motion off there is nothing to draw on, so land the final value during
  // render — an effect would leave one frame of empty ring.
  if (reduce) {
    if (scoreMV.get() !== target) scoreMV.set(target);
    if (ceilMV.get() !== cap) ceilMV.set(cap);
  }

  useEffect(() => {
    if (reduce) {
      scoreMV.set(target);
      ceilMV.set(cap);
      return;
    }
    const arc = animate(scoreMV, target, { duration: 1.15, ease: EASE, delay: 0.2 });
    // Draw to the score, hold, then extend to the ceiling — the extension IS the pitch.
    const reach = animate(ceilMV, [0, target, target, cap], {
      duration: 2.15,
      times: [0, 0.53, 0.66, 1],
      ease: EASE,
      delay: 0.2,
    });
    return () => {
      arc.stop();
      reach.stop();
    };
  }, [reduce, target, cap, scoreMV, ceilMV]);

  const scoreOffset = useTransform(scoreMV, (v) => CIRC * (1 - v / 100));
  const scoreText = useTransform(scoreMV, (v) => Math.round(v).toString());
  // The headroom arc begins exactly where the score arc ends.
  const gapArray = useTransform(() => {
    const len = (Math.max(0, ceilMV.get() - scoreMV.get()) / 100) * CIRC;
    return `${len.toFixed(2)} ${CIRC.toFixed(2)}`;
  });
  const gapOffset = useTransform(scoreMV, (v) => -(v / 100) * CIRC);

  const tickIn = R + W / 2 + 1.3;
  const ticks = Array.from({ length: 20 }, (_, i) => {
    const value = i * 5;
    const major = i % 5 === 0;
    const a = polar(tickIn, value);
    const b = polar(tickIn + (major ? 3.4 : 2), value);
    return { key: value, major, a, b };
  });

  const capIn = polar(R - W / 2 - 0.8, cap);
  const capOut = polar(R + W / 2 + 2.6, cap);

  const label =
    `Health score ${Math.round(target)} out of 100, grade ${gradeText}, ${SEV_WORD[sev]}.` +
    (headroom > 0
      ? ` Ceiling ${Math.round(cap)} — ${headroom} points available from the prescribed fixes.`
      : ' Already at its ceiling.');

  return (
    <div
      className={`relative ${SEV_CLASS[sev]}`}
      style={{ width: size, maxWidth: '100%', aspectRatio: '1 / 1' }}
    >
      {/* Bloom keyed to severity via --glow-color, which .sev-* sets. */}
      <motion.div
        aria-hidden="true"
        className="pointer-events-none absolute inset-[18%] rounded-full shadow-glow"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.85 }}
        transition={{ duration: reduce ? 0 : 1.4, ease: EASE, delay: reduce ? 0 : 0.25 }}
      />

      <svg
        viewBox="0 0 100 100"
        width="100%"
        height="100%"
        role="img"
        aria-label={label}
        className="relative block overflow-visible"
      >
        <defs>
          {/* Bloom, dialled back with a linear alpha ramp so the arc keeps a
              hard edge instead of reading as neon. */}
          <filter id={glowId} x="-45%" y="-45%" width="190%" height="190%">
            <feGaussianBlur stdDeviation="2.4" result="b" />
            <feComponentTransfer in="b" result="bd">
              <feFuncA type="linear" slope="0.42" />
            </feComponentTransfer>
            <feMerge>
              <feMergeNode in="bd" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Scale ticks — 5-point increments, longer every 25. */}
        <g>
          {ticks.map((t) => (
            <line
              key={t.key}
              x1={t.a.x}
              y1={t.a.y}
              x2={t.b.x}
              y2={t.b.y}
              stroke={t.major ? '#2A2A38' : '#1A1A24'}
              strokeWidth={t.major ? 0.7 : 0.45}
              strokeLinecap="round"
            />
          ))}
        </g>

        {/* Track */}
        <circle cx={C} cy={C} r={R} fill="none" stroke="#16161F" strokeWidth={W} />
        <circle
          cx={C}
          cy={C}
          r={R - W / 2 - 1.1}
          fill="none"
          stroke="#101018"
          strokeWidth={0.5}
        />

        {/* Headroom: what the score becomes if every prescription lands. */}
        <motion.circle
          cx={C}
          cy={C}
          r={R}
          fill="none"
          stroke="var(--sev-clean)"
          strokeWidth={W * 0.86}
          strokeLinecap="butt"
          opacity={0.34}
          transform={`rotate(-90 ${C} ${C})`}
          style={{ strokeDasharray: gapArray, strokeDashoffset: gapOffset }}
        />

        {/* The score itself. */}
        <motion.circle
          cx={C}
          cy={C}
          r={R}
          fill="none"
          stroke={sevColor}
          strokeWidth={W}
          strokeLinecap="round"
          strokeDasharray={CIRC}
          transform={`rotate(-90 ${C} ${C})`}
          filter={`url(#${glowId})`}
          style={{ strokeDashoffset: scoreOffset }}
        />

        {/* Ceiling marker */}
        {headroom > 0 && (
          <motion.line
            x1={capIn.x}
            y1={capIn.y}
            x2={capOut.x}
            y2={capOut.y}
            stroke="var(--sev-clean)"
            strokeWidth={0.9}
            strokeLinecap="round"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.9 }}
            transition={{ duration: reduce ? 0 : 0.6, ease: EASE, delay: reduce ? 0 : 2.15 }}
          />
        )}

        {/* Centre readout — SVG text so it scales exactly with the ring. */}
        <text
          x={C}
          y={33}
          textAnchor="middle"
          className="font-mono"
          fontSize={3.3}
          letterSpacing={1.05}
          fill="#5C5C6A"
        >
          HEALTH
        </text>

        <text
          x={C}
          y={57}
          textAnchor="middle"
          className="font-display"
          fontSize={24}
          fontWeight={600}
          letterSpacing={-0.9}
          fill={sevColor}
        >
          {gradeText}
        </text>

        <text
          x={C}
          y={68.5}
          textAnchor="middle"
          className="font-mono"
          fontSize={8}
          fill="#F4F4F7"
        >
          <motion.tspan>{scoreText}</motion.tspan>
          <tspan dx={1.3} fontSize={4.4} fill="#4A4A57">
            /100
          </tspan>
        </text>

        <line x1={40} y1={73} x2={60} y2={73} stroke="#1E1E29" strokeWidth={0.5} />

        <motion.text
          x={C}
          y={79.2}
          textAnchor="middle"
          className="font-mono"
          fontSize={3.4}
          letterSpacing={0.6}
          fill={headroom > 0 ? 'var(--sev-clean)' : '#4A4A57'}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: reduce ? 0 : 0.6, ease: EASE, delay: reduce ? 0 : 2.2 }}
        >
          {headroom > 0 ? `CEILING ${Math.round(cap)} · +${headroom}` : 'AT CEILING'}
        </motion.text>
      </svg>
    </div>
  );
}
