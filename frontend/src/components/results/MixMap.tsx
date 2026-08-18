import { useId, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  DIMENSIONS,
  DIMENSION_LABELS,
  DIMENSION_SHORT,
  SEVERITY_VAR,
  type Dimension,
  type DimensionScore,
  type Severity,
} from '../../types/analysis';

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1];

const VB = 400;
const CX = VB / 2;
const CY = VB / 2;
const R_MAX = 124;
const R_LABEL = 157;
const RINGS = [25, 50, 75, 100];

const SEV_CLASS: Record<Severity, string> = {
  critical: 'sev-critical',
  major: 'sev-major',
  minor: 'sev-minor',
  clean: 'sev-clean',
};

const SEV_WORD: Record<Severity, string> = {
  critical: 'Critical',
  major: 'Major',
  minor: 'Minor',
  clean: 'Clean',
};

/** Shape as well as color, so severity is never carried by hue alone. */
const SEV_GLYPH: Record<Severity, string> = {
  critical: '▲',
  major: '●',
  minor: '◆',
  clean: '✓',
};

interface Axis {
  dim: Dimension;
  d: DimensionScore;
  score: number;
  sev: Severity;
  /** Vertex of the data polygon. */
  px: number;
  py: number;
  /** Outer end of the spoke, at 100. */
  ox: number;
  oy: number;
  cos: number;
  sin: number;
}

function clamp100(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(100, Math.max(0, v));
}

function ringPoints(level: number, n: number): string {
  const r = (R_MAX * level) / 100;
  const pts: string[] = [];
  for (let i = 0; i < n; i += 1) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    pts.push(`${(CX + r * Math.cos(a)).toFixed(2)},${(CY + r * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(' ');
}

export interface MixMapProps {
  dimensions: DimensionScore[];
  onSelect: (d: Dimension) => void;
  selected: Dimension | null;
}

export default function MixMap({ dimensions, onSelect, selected }: MixMapProps) {
  const reduce = useReducedMotion() ?? false;
  const uid = useId().replace(/:/g, '');
  const fillId = `mixmap-fill-${uid}`;
  const maskId = `mixmap-reveal-${uid}`;
  const [hovered, setHovered] = useState<Dimension | null>(null);

  const axes = useMemo<Axis[]>(() => {
    const byDim = new Map<Dimension, DimensionScore>();
    for (const d of dimensions ?? []) byDim.set(d.dimension, d);
    const present = DIMENSIONS.filter((d) => byDim.has(d));
    const n = present.length;
    return present.map((dim, i) => {
      // Non-null assertion avoided: `present` was filtered on `has`, but re-read safely.
      const d = byDim.get(dim);
      const score = clamp100(d ? d.score : 0);
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      const cos = Math.cos(a);
      const sin = Math.sin(a);
      const r = (R_MAX * score) / 100;
      return {
        dim,
        d: d ?? { dimension: dim, label: DIMENSION_LABELS[dim], score, severity: 'clean', headline: '', finding_ids: [] },
        score,
        sev: d ? d.severity : 'clean',
        px: CX + r * cos,
        py: CY + r * sin,
        ox: CX + R_MAX * cos,
        oy: CY + R_MAX * sin,
        cos,
        sin,
      };
    });
  }, [dimensions]);

  const n = axes.length;
  const polygon = useMemo(
    () => axes.map((a) => `${a.px.toFixed(2)},${a.py.toFixed(2)}`).join(' '),
    [axes],
  );
  const mean = useMemo(() => {
    if (!axes.length) return 0;
    return Math.round(axes.reduce((sum, a) => sum + a.score, 0) / axes.length);
  }, [axes]);

  const active = hovered ?? selected;
  const activeAxis = active ? (axes.find((a) => a.dim === active) ?? null) : null;

  if (!n) {
    return (
      <div className="panel flex min-h-[200px] items-center justify-center p-8">
        <p className="eyebrow">No dimension scores in this analysis</p>
      </div>
    );
  }

  const centreValue = activeAxis ? Math.round(activeAxis.score) : mean;
  const centreLabel = activeAxis ? DIMENSION_SHORT[activeAxis.dim].toUpperCase() : 'MEAN SCORE';
  const centreColor = activeAxis ? SEVERITY_VAR[activeAxis.sev] : '#52F2C4';

  return (
    <div className="w-full">
      <div className="relative mx-auto aspect-square w-full max-w-[620px]">
        <svg
          viewBox={`0 0 ${VB} ${VB}`}
          width="100%"
          height="100%"
          aria-hidden="true"
          className="absolute inset-0 h-full w-full"
        >
          <defs>
            <radialGradient id={fillId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#52F2C4" stopOpacity="0.30" />
              <stop offset="55%" stopColor="#4CC9F0" stopOpacity="0.16" />
              <stop offset="100%" stopColor="#5B4BFF" stopOpacity="0.06" />
            </radialGradient>
            <mask id={maskId}>
              <rect x="0" y="0" width={VB} height={VB} fill="black" />
              <motion.circle
                cx={CX}
                cy={CY}
                fill="white"
                initial={{ r: reduce ? R_MAX + 6 : 0 }}
                animate={{ r: R_MAX + 6 }}
                transition={{ duration: reduce ? 0 : 1.15, ease: EASE, delay: reduce ? 0 : 0.25 }}
              />
            </mask>
          </defs>

          {/* Grid: concentric n-gons at 25 / 50 / 75 / 100 */}
          <motion.g
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: reduce ? 0 : 0.8, ease: EASE, delay: reduce ? 0 : 0.1 }}
          >
            {RINGS.map((level) => (
              <polygon
                key={level}
                points={ringPoints(level, n)}
                fill="none"
                stroke={level === 100 ? '#23232F' : '#191922'}
                strokeWidth={level === 100 ? 1 : 0.8}
              />
            ))}
            {axes.map((a) => (
              <line
                key={a.dim}
                x1={CX}
                y1={CY}
                x2={a.ox}
                y2={a.oy}
                stroke="#17171F"
                strokeWidth={0.8}
              />
            ))}
          </motion.g>

          {/* Highlight spoke, under the shape so it never cuts across it. */}
          {activeAxis && (
            <line
              x1={CX}
              y1={CY}
              x2={activeAxis.ox}
              y2={activeAxis.oy}
              stroke={selected === activeAxis.dim ? '#52F2C4' : '#4A8579'}
              strokeWidth={1}
              opacity={selected === activeAxis.dim ? 0.55 : 0.4}
              strokeDasharray={selected === activeAxis.dim ? undefined : '4 4'}
            />
          )}

          {/* The mix shape, revealed outward from the center */}
          <g mask={`url(#${maskId})`}>
            <polygon points={polygon} fill={`url(#${fillId})`} />
            <polygon
              points={polygon}
              fill="none"
              stroke="#52F2C4"
              strokeWidth={1.8}
              strokeLinejoin="round"
            />
          </g>

          {/* Vertices, colored by that dimension's severity */}
          {axes.map((a, i) => {
            const isActive = active === a.dim;
            const isSel = selected === a.dim;
            const base = a.sev === 'clean' ? 3 : 4.2;
            return (
              <g key={a.dim}>
                {(isActive || isSel) && (
                  <circle
                    cx={a.px}
                    cy={a.py}
                    r={base + 5}
                    fill="none"
                    stroke={SEVERITY_VAR[a.sev]}
                    strokeWidth={1}
                    opacity={0.55}
                  />
                )}
                <motion.circle
                  initial={{
                    cx: reduce ? a.px : CX,
                    cy: reduce ? a.py : CY,
                    opacity: reduce ? 1 : 0,
                  }}
                  animate={{ cx: a.px, cy: a.py, opacity: 1 }}
                  transition={{
                    duration: reduce ? 0 : 0.85,
                    ease: EASE,
                    delay: reduce ? 0 : 0.45 + i * 0.03,
                  }}
                  r={isActive ? base + 1.6 : base}
                  fill={SEVERITY_VAR[a.sev]}
                  stroke="#06060A"
                  strokeWidth={1.2}
                />
                {/* Generous invisible hit target for pointers. */}
                <circle
                  cx={a.px}
                  cy={a.py}
                  r={16}
                  fill="transparent"
                  style={{ cursor: 'pointer' }}
                  onPointerEnter={() => setHovered(a.dim)}
                  onPointerLeave={() => setHovered((h) => (h === a.dim ? null : h))}
                  onClick={() => onSelect(a.dim)}
                />
              </g>
            );
          })}

          {/* Center readout */}
          <circle cx={CX} cy={CY} r={34} fill="#06060A" opacity={0.9} />
          <circle cx={CX} cy={CY} r={34} fill="none" stroke="#16161F" strokeWidth={0.8} />
          <text
            x={CX}
            y={CY - 8}
            textAnchor="middle"
            className="font-mono"
            fontSize={7.5}
            letterSpacing={1.3}
            fill="#4A4A57"
          >
            {centreLabel}
          </text>
          <text
            x={CX}
            y={CY + 16}
            textAnchor="middle"
            className="font-mono"
            fontSize={22}
            fill={centreColor}
          >
            {centreValue}
          </text>

          {/* Ring scale, drawn last so the shape and the center disc never bury it. */}
          <motion.g
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: reduce ? 0 : 0.8, ease: EASE, delay: reduce ? 0 : 0.1 }}
          >
            {RINGS.map((level) => (
              <text
                key={`lbl-${level}`}
                x={CX + 6}
                y={CY - (R_MAX * level) / 100 - 4}
                className="font-mono"
                fontSize={9}
                fill="#40404E"
              >
                {level}
              </text>
            ))}
          </motion.g>
        </svg>

        {/* Real buttons for the axes: keyboard reachable, visibly focusable,
            and HTML text so labels stay legible at every width. */}
        {axes.map((a) => {
          const isSel = selected === a.dim;
          const isActive = active === a.dim;
          const loud = a.sev === 'critical' || a.sev === 'major';
          return (
            <button
              key={a.dim}
              type="button"
              onClick={() => onSelect(a.dim)}
              onPointerEnter={() => setHovered(a.dim)}
              onPointerLeave={() => setHovered((h) => (h === a.dim ? null : h))}
              onFocus={() => setHovered(a.dim)}
              onBlur={() => setHovered((h) => (h === a.dim ? null : h))}
              aria-pressed={isSel}
              aria-label={`${DIMENSION_LABELS[a.dim]}: ${Math.round(a.score)} out of 100, ${SEV_WORD[a.sev]}`}
              className={`absolute z-10 whitespace-nowrap rounded-md px-1.5 py-0.5 text-center leading-none transition-colors duration-200 ease-cine ${
                isSel ? 'bg-signal/10 ring-1 ring-signal/50' : 'hover:bg-void-raised/70'
              }`}
              style={{
                left: `${((CX + R_LABEL * a.cos) / VB) * 100}%`,
                top: `${((CY + R_LABEL * a.sin) / VB) * 100}%`,
                transform: 'translate(-50%, -50%)',
              }}
            >
              <span
                className="block font-display text-[9px] font-semibold tracking-tight sm:text-[10.5px] lg:text-[11.5px]"
                style={{
                  color:
                    isActive || isSel
                      ? '#F4F4F7'
                      : loud
                        ? SEVERITY_VAR[a.sev]
                        : a.sev === 'minor'
                          ? '#8A8A98'
                          : '#5F5F6C',
                }}
              >
                {DIMENSION_SHORT[a.dim]}
              </span>
              {loud && (
                <span
                  className="stat mt-0.5 block text-[8px] sm:text-[9px]"
                  style={{ color: SEVERITY_VAR[a.sev] }}
                >
                  {Math.round(a.score)}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Readout strip — fixed height so nothing reflows on hover. */}
      <div
        aria-live="polite"
        className="panel mt-5 flex min-h-[76px] items-start gap-4 p-4 sm:items-center"
      >
        {activeAxis ? (
          <>
            <div className="flex shrink-0 flex-col gap-1.5">
              <span className="eyebrow text-ink-muted">{DIMENSION_LABELS[activeAxis.dim]}</span>
              <div className="flex items-baseline gap-2">
                <span
                  className="stat text-2xl leading-none"
                  style={{ color: SEVERITY_VAR[activeAxis.sev] }}
                >
                  {Math.round(activeAxis.score)}
                </span>
                <span className="stat text-micro text-ink-faint">/100</span>
              </div>
            </div>
            <div className="hidden h-10 w-px shrink-0 bg-void-line sm:block" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <span className={`sev-chip ${SEV_CLASS[activeAxis.sev]}`}>
                <span aria-hidden="true">{SEV_GLYPH[activeAxis.sev]}</span>
                {SEV_WORD[activeAxis.sev]}
              </span>
              <p className="mt-2 text-[13px] leading-snug text-ink-dim">
                {activeAxis.d.headline || 'No headline reported for this dimension.'}
              </p>
            </div>
          </>
        ) : (
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="eyebrow text-ink-muted">Mix map · {n} dimensions</span>
            <p className="text-[13px] leading-snug text-ink-muted">
              Radius is the dimension score, 0 at the center to 100 at the outer ring. Hover or
              focus an axis to inspect it; select one to pin it across the report.
            </p>
          </div>
        )}
      </div>

      <p className="sr-only">
        Mean dimension score {mean} out of 100 across {n} dimensions.
      </p>
    </div>
  );
}
