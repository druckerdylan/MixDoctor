/**
 * The plugin vault picker.
 *
 * This is the one screen that makes the advice personal, so it is built as a
 * proper tool rather than a settings checkbox: search, browse by maker, and —
 * the part that earns its place — a plain reading of what the producer *cannot*
 * do, because that is what changes the shape of every prescription.
 *
 * Open state lives in a module store so the header button and the intake form
 * can both raise the same drawer without App threading props through.
 */

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import type { ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import type { Dimension, OwnedPlugin } from '../types/analysis';
import { DIMENSION_SHORT } from '../types/analysis';
import {
  ALL_CAPABILITIES,
  CAPABILITIES_FOR_CATEGORY,
  CAPABILITY_BLURBS,
  CAPABILITY_GROUPS,
  CAPABILITY_LABELS,
  CATEGORIES,
  CUSTOM_CATEGORIES,
  MANUFACTURERS,
  PLUGIN_CATALOG,
  STOCK_CAPABILITIES,
  VAULT_PRESETS,
  orderCapabilities,
} from '../data/plugins';
import type { CatalogPlugin } from '../data/plugins';
import { usePluginVault } from '../hooks/usePluginVault';

const EASE = [0.16, 1, 0.3, 1] as const;

/* ------------------------------------------------------------------ */
/* Open state — one drawer, several triggers                           */
/* ------------------------------------------------------------------ */

let vaultOpen = false;
const openListeners = new Set<() => void>();

function emitOpen(): void {
  for (const listener of openListeners) listener();
}

function subscribeOpen(listener: () => void): () => void {
  openListeners.add(listener);
  return () => {
    openListeners.delete(listener);
  };
}

export function openPluginVault(): void {
  if (vaultOpen) return;
  vaultOpen = true;
  emitOpen();
}

export function closePluginVault(): void {
  if (!vaultOpen) return;
  vaultOpen = false;
  emitOpen();
}

function useVaultOpen(): boolean {
  return useSyncExternalStore(
    subscribeOpen,
    () => vaultOpen,
    () => false,
  );
}

/* ------------------------------------------------------------------ */
/* Gaps                                                                */
/* ------------------------------------------------------------------ */

interface GapRule {
  id: string;
  /** Owning any one of these closes the gap. */
  needs: string[];
  title: string;
  consequence: string;
  dimensions: Dimension[];
}

/**
 * Derived from CAPABILITY_FOR_DIMENSION, but written out rather than generated:
 * the useful sentence is not "you are missing eq_dynamic", it is what the fix
 * degrades into when you are. Ordered by how much of the report it touches.
 */
export const GAP_RULES: GapRule[] = [
  {
    id: 'reactive',
    needs: ['eq_dynamic', 'resonance_suppressor'],
    title: 'Nothing that reacts',
    consequence:
      'No dynamic EQ and no resonance suppressor. A problem that comes and goes — mud that only builds in the chorus, an edge that only bites on the loud lines — has to be treated with a static cut and then automated off in the bars where it is not there.',
    dimensions: ['mud', 'harshness', 'clarity'],
  },
  {
    id: 'truepeak',
    needs: ['limiter_truepeak'],
    title: 'No true-peak ceiling',
    consequence:
      'Your limiter measures sample peaks, so a master that reads -1.0 dBFS can still clip once a platform encodes it to AAC. Set the ceiling to -1.5 dB and accept the lost headroom.',
    dimensions: ['clipping', 'limiter', 'loudness'],
  },
  {
    id: 'sidechain',
    needs: ['comp_sidechain_ext'],
    title: 'No external sidechain',
    consequence:
      'Kick and 808 cannot duck one another. Separation has to come from the arrangement, a static carve at the collision, or volume automation on the sub.',
    dimensions: ['low_end'],
  },
  {
    id: 'multiband',
    needs: ['comp_multiband', 'eq_dynamic'],
    title: 'No band-independent dynamics',
    consequence:
      'Every compressor you own moves the whole spectrum at once, so a loud low end pulls the vocal down with it. Congestion has to be fixed with balance and EQ rather than dynamics.',
    dimensions: ['compression', 'clarity', 'mud'],
  },
  {
    id: 'mono',
    needs: ['mono_maker', 'eq_mid_side'],
    title: 'No way to mono the low end',
    consequence:
      'Nothing you own can collapse the bottom to the center. Low-end phase smear has to be fixed at source — re-center the parts, or narrow the sub in the arrangement.',
    dimensions: ['phase', 'low_end'],
  },
  {
    id: 'punch',
    needs: ['transient_shaper', 'clipper'],
    title: 'No way to put punch back',
    consequence:
      'If compression has flattened the drums, the only lever available is less compression — attack cannot be raised independently of level.',
    dimensions: ['transients', 'dynamic_range'],
  },
  {
    id: 'sibilance',
    needs: ['deesser', 'eq_dynamic', 'resonance_suppressor'],
    title: 'No de-esser',
    consequence:
      'Sibilance has to be handled by hand: clip-gain each S, or ride the level. A static shelf at 7 kHz will dull the whole vocal to fix six syllables.',
    dimensions: ['harshness', 'vocal_balance'],
  },
  {
    id: 'width',
    needs: ['imager', 'eq_mid_side'],
    title: 'No width control',
    consequence:
      'Width can only be changed by re-panning and re-balancing the parts — there is no way to narrow or spread the finished stereo image.',
    dimensions: ['stereo_width'],
  },
  {
    id: 'loudness-meter',
    needs: ['meter_loudness'],
    title: 'No loudness meter',
    consequence:
      'You cannot verify LUFS or true peak while you work, so this report is your only reading. Re-measure here after every loudness change.',
    dimensions: ['loudness'],
  },
  {
    id: 'reference',
    needs: ['reference_matching', 'eq_match'],
    title: 'No reference tool',
    consequence:
      'Matching a commercial record means A/B by ear at matched level, which is honest but slow. Drop a reference into the analysis instead and read the deltas.',
    dimensions: ['frequency_balance'],
  },
  {
    id: 'correlation',
    needs: ['meter_correlation'],
    title: 'No correlation meter',
    consequence:
      'Phase trouble stays invisible until you sum to mono and listen for what disappears.',
    dimensions: ['phase'],
  },
];

export function findGaps(capabilities: string[]): GapRule[] {
  const owned = new Set(capabilities);
  return GAP_RULES.filter((rule) => !rule.needs.some((need) => owned.has(need)));
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

/** "pro q 3" must find "Pro-Q 3", and "soothe 2" must find "soothe2". */
function squash(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}

interface Indexed {
  plugin: CatalogPlugin;
  haystack: string;
  squashed: string;
}

const SEARCH_INDEX: Indexed[] = PLUGIN_CATALOG.map((plugin) => {
  const labels = plugin.capabilities.map((c) => CAPABILITY_LABELS[c] ?? c).join(' ');
  const haystack = `${plugin.name} ${plugin.manufacturer} ${plugin.category} ${labels}`.toLowerCase();
  return { plugin, haystack, squashed: squash(haystack) };
});

export function searchCatalog(query: string, category: string): CatalogPlugin[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const squashedQuery = squash(query);
  return SEARCH_INDEX.filter(({ plugin, haystack, squashed }) => {
    if (category !== 'all' && plugin.category !== category) return false;
    if (terms.length === 0) return true;
    if (terms.every((term) => haystack.includes(term))) return true;
    return squashedQuery.length > 2 && squashed.includes(squashedQuery);
  }).map((entry) => entry.plugin);
}

/* ------------------------------------------------------------------ */
/* Small pieces                                                        */
/* ------------------------------------------------------------------ */

function CapabilityChip({
  slug,
  tone = 'quiet',
}: {
  slug: string;
  tone?: 'quiet' | 'signal' | 'stock';
}) {
  const label = CAPABILITY_LABELS[slug] ?? slug;
  const colour =
    tone === 'signal' ? 'text-signal' : tone === 'stock' ? 'text-ink-faint' : 'text-ink-muted';
  return (
    <span className={`sev-chip ${colour}`} title={CAPABILITY_BLURBS[slug] ?? label}>
      {label}
    </span>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={`shrink-0 transition-transform duration-200 ease-cine ${open ? 'rotate-90' : ''}`}
    >
      <path d="m9 6 6 6-6 6" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Tick({ on }: { on: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={[
        'flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors duration-200',
        on ? 'border-signal bg-signal text-void-deep' : 'border-void-line bg-void-deep text-transparent',
      ].join(' ')}
    >
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
        <path d="m5 13 4 4L19 7" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

function PluginRow({
  name,
  manufacturer,
  capabilities,
  owned,
  onToggle,
}: {
  name: string;
  manufacturer: string | null;
  capabilities: string[];
  owned: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={owned}
      className={[
        'flex w-full items-start gap-3 rounded-xl border px-3 py-2.5 text-left',
        'transition-colors duration-200 ease-cine',
        owned
          ? 'border-signal/40 bg-signal/[0.06]'
          : 'border-transparent hover:border-void-line hover:bg-void-raised',
      ].join(' ')}
    >
      <span className="pt-0.5">
        <Tick on={owned} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-baseline gap-x-2">
          <span className={`text-sm ${owned ? 'text-ink' : 'text-ink-dim'}`}>{name}</span>
          {manufacturer && (
            <span className="font-mono text-micro uppercase text-ink-faint">{manufacturer}</span>
          )}
        </span>
        {capabilities.length > 0 ? (
          <span className="mt-1.5 flex flex-wrap gap-1">
            {capabilities.map((slug) => (
              <CapabilityChip key={slug} slug={slug} tone={owned ? 'signal' : 'quiet'} />
            ))}
          </span>
        ) : (
          <span className="mt-1.5 block font-mono text-micro uppercase text-ink-faint">
            No mix-shaping capability
          </span>
        )}
      </span>
    </button>
  );
}

function Group({
  label,
  count,
  open,
  onToggle,
  action,
  children,
}: {
  label: string;
  count: string;
  open: boolean;
  onToggle: () => void;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="border-b border-void-lineSoft last:border-b-0">
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 px-1 py-3 text-left text-ink-dim transition-colors duration-200 hover:text-ink"
        >
          <Chevron open={open} />
          <span className="flex-1 truncate text-sm">{label}</span>
          <span className="stat text-micro text-ink-faint">{count}</span>
        </button>
        {action}
      </div>
      {open && <div className="space-y-0.5 pb-3">{children}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Summary panel                                                       */
/* ------------------------------------------------------------------ */

function SummaryPanel({ capabilities, count }: { capabilities: string[]; count: number }) {
  const owned = useMemo(() => new Set(capabilities), [capabilities]);
  const stock = useMemo(() => new Set(STOCK_CAPABILITIES), []);
  const gaps = useMemo(() => findGaps(capabilities), [capabilities]);
  const beyondStock = capabilities.filter((c) => !stock.has(c)).length;

  const [showAllGaps, setShowAllGaps] = useState(false);
  const shownGaps = showAllGaps ? gaps : gaps.slice(0, 4);

  return (
    <div className="space-y-6">
      <div>
        <p className="eyebrow text-signal-dim">What you can do</p>
        <p className="mt-3 text-xs leading-relaxed text-ink-muted">
          {count === 0
            ? 'Nothing added, so advice assumes the stock set every DAW ships with. That is enough for a complete plan — it just cannot name your boxes.'
            : `${beyondStock} capabilit${beyondStock === 1 ? 'y' : 'ies'} beyond stock, from ${count} plugin${count === 1 ? '' : 's'}. Faded chips are assumed for everyone.`}
        </p>

        <div className="mt-4 space-y-3">
          {CAPABILITY_GROUPS.map((group) => {
            const present = group.capabilities.filter((c) => owned.has(c));
            if (present.length === 0) return null;
            return (
              <div key={group.label}>
                <p className="mb-1.5 font-mono text-micro uppercase text-ink-faint">{group.label}</p>
                <div className="flex flex-wrap gap-1">
                  {present.map((slug) => (
                    <CapabilityChip key={slug} slug={slug} tone={stock.has(slug) ? 'stock' : 'signal'} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="hairline" />

      <div>
        <p className="eyebrow text-ink-dim">Notable gaps</p>
        {gaps.length === 0 ? (
          <p className="mt-3 text-xs leading-relaxed text-ink-muted">
            Nothing missing that would change the shape of a fix. Every dimension in the report has a
            tool behind it.
          </p>
        ) : (
          <>
            <p className="mt-3 text-xs leading-relaxed text-ink-muted">
              Not a shopping list — this is how the advice will be written instead.
            </p>
            <ul className="mt-4 space-y-4">
              {shownGaps.map((gap) => (
                <li key={gap.id}>
                  <div className="flex items-baseline justify-between gap-2">
                    <p className="text-xs font-medium text-sev-major">{gap.title}</p>
                    <div className="flex shrink-0 flex-wrap justify-end gap-1">
                      {gap.dimensions.map((dimension) => (
                        <span
                          key={dimension}
                          className="font-mono text-micro uppercase text-ink-faint"
                        >
                          {DIMENSION_SHORT[dimension]}
                        </span>
                      ))}
                    </div>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-ink-muted">{gap.consequence}</p>
                </li>
              ))}
            </ul>
            {gaps.length > 4 && (
              <button
                type="button"
                onClick={() => setShowAllGaps((value) => !value)}
                className="mt-4 font-mono text-micro uppercase tracking-[0.14em] text-signal-dim hover:text-signal"
              >
                {showAllGaps ? 'Show fewer' : `Show ${gaps.length - 4} more`}
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Custom plugin form                                                  */
/* ------------------------------------------------------------------ */

function CustomForm({ onAdd }: { onAdd: (plugin: OwnedPlugin) => void }) {
  const fieldId = useId();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [manufacturer, setManufacturer] = useState('');
  const [category, setCategory] = useState('EQ');
  const [selected, setSelected] = useState<string[]>(CAPABILITIES_FOR_CATEGORY['EQ'] ?? []);

  const chooseCategory = (next: string) => {
    setCategory(next);
    setSelected(CAPABILITIES_FOR_CATEGORY[next] ?? []);
  };

  const toggleCapability = (slug: string) => {
    setSelected((current) =>
      current.includes(slug) ? current.filter((c) => c !== slug) : [...current, slug],
    );
  };

  const submit = () => {
    const trimmed = name.trim();
    if (trimmed === '') return;
    onAdd({
      name: trimmed,
      manufacturer: manufacturer.trim() === '' ? null : manufacturer.trim(),
      category,
      capabilities: orderCapabilities(selected),
    });
    setName('');
    setManufacturer('');
    setOpen(false);
  };

  return (
    <div className="border-t border-void-line pt-4">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 text-left text-ink-dim transition-colors duration-200 hover:text-ink"
      >
        <Chevron open={open} />
        <span className="text-sm">Add something not in the list</span>
      </button>

      {open && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label htmlFor={`${fieldId}-name`} className="eyebrow mb-1.5 block text-ink-faint">
                Name
              </label>
              <input
                id={`${fieldId}-name`}
                value={name}
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    submit();
                  }
                }}
                placeholder="e.g. Kirchhoff-EQ"
                className="input-field px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label htmlFor={`${fieldId}-maker`} className="eyebrow mb-1.5 block text-ink-faint">
                Manufacturer
              </label>
              <input
                id={`${fieldId}-maker`}
                value={manufacturer}
                onChange={(event) => setManufacturer(event.target.value)}
                placeholder="Optional"
                className="input-field px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div>
            <label htmlFor={`${fieldId}-category`} className="eyebrow mb-1.5 block text-ink-faint">
              Category
            </label>
            <select
              id={`${fieldId}-category`}
              value={category}
              onChange={(event) => chooseCategory(event.target.value)}
              className="input-field px-3 py-2 text-sm"
            >
              {CUSTOM_CATEGORIES.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          <div>
            <p className="eyebrow mb-2 text-ink-faint">
              What it can do — {selected.length} selected
            </p>
            <div className="flex flex-wrap gap-1.5">
              {ALL_CAPABILITIES.map((slug) => {
                const on = selected.includes(slug);
                return (
                  <button
                    key={slug}
                    type="button"
                    onClick={() => toggleCapability(slug)}
                    aria-pressed={on}
                    title={CAPABILITY_BLURBS[slug]}
                    className={[
                      'rounded-full border px-2.5 py-1 font-mono text-micro uppercase transition-colors duration-200',
                      on
                        ? 'border-signal/45 bg-signal/10 text-signal'
                        : 'border-void-line text-ink-faint hover:border-ink-faint hover:text-ink-dim',
                    ].join(' ')}
                  >
                    {CAPABILITY_LABELS[slug]}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            type="button"
            onClick={submit}
            disabled={name.trim() === ''}
            className="btn-primary px-5 py-2 text-xs"
          >
            Add plugin
          </button>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Drawer                                                              */
/* ------------------------------------------------------------------ */

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export interface PluginVaultProps {
  open: boolean;
  onClose: () => void;
}

export default function PluginVault({ open, onClose }: PluginVaultProps) {
  const reduce = useReducedMotion();
  const { plugins, add, remove, toggle, has, clear, capabilities, count } = usePluginVault();

  const panelRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['__owned__']));

  const filtering = query.trim() !== '' || category !== 'all';

  const results = useMemo(() => searchCatalog(query, category), [query, category]);

  /** Catalog results grouped by maker, makers with no hits dropped. */
  const grouped = useMemo(() => {
    const map = new Map<string, CatalogPlugin[]>();
    for (const plugin of results) {
      const list = map.get(plugin.manufacturer);
      if (list) list.push(plugin);
      else map.set(plugin.manufacturer, [plugin]);
    }
    return MANUFACTURERS.filter((maker) => map.has(maker)).map((maker) => ({
      manufacturer: maker,
      plugins: map.get(maker) ?? [],
    }));
  }, [results]);

  /** Owned entries that survive the current filter, catalog or custom. */
  const ownedVisible = useMemo(() => {
    const inResults = new Set(results.map((p) => `${p.manufacturer}::${p.name}`));
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return plugins.filter((plugin) => {
      if (inResults.has(`${plugin.manufacturer ?? ''}::${plugin.name}`)) return true;
      // Custom entries are not in the search index; filter them by hand.
      if (category !== 'all' && plugin.category !== category) return false;
      if (terms.length === 0) return true;
      const haystack =
        `${plugin.name} ${plugin.manufacturer ?? ''} ${plugin.category} ${plugin.capabilities
          .map((c) => CAPABILITY_LABELS[c] ?? c)
          .join(' ')}`.toLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  }, [plugins, results, query, category]);

  const toggleGroup = useCallback((key: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  /**
   * Searching opens the makers that matched; clearing the search closes them
   * again. Expansion stays real state rather than being forced by `filtering`,
   * so a group header is never a button that does nothing.
   */
  useEffect(() => {
    setExpanded((current) => {
      const next = new Set<string>(['__owned__']);
      if (filtering) for (const plugin of results) next.add(plugin.manufacturer);
      if (next.size === current.size && [...next].every((key) => current.has(key))) return current;
      return next;
    });
  }, [results, filtering]);

  const applyPreset = useCallback(
    (id: string) => {
      const preset = VAULT_PRESETS.find((p) => p.id === id);
      if (!preset) return;
      if (preset.clears) {
        clear();
        return;
      }
      add(
        preset.plugins.map((plugin) => ({
          name: plugin.name,
          manufacturer: plugin.manufacturer,
          category: plugin.category,
          capabilities: plugin.capabilities,
        })),
      );
    },
    [add, clear],
  );

  /* Escape, focus trap, scroll lock, focus return -------------------- */
  useEffect(() => {
    if (!open) return undefined;

    const opener = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const focusTimer = window.setTimeout(() => searchRef.current?.focus(), 60);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== 'Tab') return;
      const panel = panelRef.current;
      if (!panel) return;
      const focusables = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (element) => element.offsetParent !== null,
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      if (!panel.contains(active)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener('keydown', onKeyDown, true);
      document.body.style.overflow = previousOverflow;
      // Put the caret back where the producer left it.
      if (opener && typeof opener.focus === 'function') opener.focus();
    };
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        // Keyed motion element, not a plain div: AnimatePresence only waits for
        // its own children, so a bare wrapper would swallow the exit animation.
        <motion.div key="plugin-vault" className="fixed inset-0 z-[70]" role="presentation">
          <motion.div
            className="scrim z-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: EASE }}
            onClick={onClose}
            aria-hidden="true"
          />

          <motion.div
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-label="Your plugins"
            initial={reduce ? { opacity: 0 } : { opacity: 0, x: 48 }}
            animate={reduce ? { opacity: 1 } : { opacity: 1, x: 0 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, x: 48 }}
            transition={{ duration: 0.42, ease: EASE }}
            className="absolute inset-y-0 right-0 z-10 flex w-full max-w-[64rem] flex-col border-l border-void-line bg-void shadow-lift"
          >
            {/* Header ------------------------------------------------ */}
            <div className="shrink-0 border-b border-void-line px-5 py-4 sm:px-7">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="eyebrow text-signal-dim">Your plugins</p>
                  <h2 className="display mt-2 text-xl text-ink">
                    What you actually have to work with
                  </h2>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="stat rounded-full border border-void-line px-3 py-1.5 text-xs text-ink-dim">
                    {count} owned
                  </span>
                  <button
                    type="button"
                    onClick={onClose}
                    aria-label="Close plugin list"
                    className="rounded-full border border-void-line p-2 text-ink-muted transition-colors duration-200 hover:border-ink-faint hover:text-ink"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path
                        d="M6 6l12 12M18 6L6 18"
                        stroke="currentColor"
                        strokeWidth={1.8}
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                </div>
              </div>

              <p className="mt-3 max-w-2xl text-xs leading-relaxed text-ink-muted">
                Prescriptions are written against capabilities, not brand names. Tell us you own
                soothe2 and the fix becomes “set depth, let it track the resonance”; tell us nothing
                and it becomes “notch 318 Hz and automate it off in the verses”. Both work — one is
                faster.
              </p>
            </div>

            {/* Body -------------------------------------------------- */}
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="grid grid-cols-1 gap-8 px-5 py-6 sm:px-7 lg:grid-cols-[1.35fr_1fr] lg:gap-10">
                {/* Browse ------------------------------------------- */}
                <div className="min-w-0">
                  {/* Presets */}
                  <div>
                    <p className="eyebrow mb-2.5 text-ink-faint">Quick start</p>
                    <div className="flex flex-wrap gap-1.5">
                      {VAULT_PRESETS.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => applyPreset(preset.id)}
                          title={preset.detail}
                          className="rounded-full border border-void-line px-3 py-1.5 text-xs text-ink-dim transition-colors duration-200 ease-cine hover:border-signal-dim hover:bg-signal/[0.06] hover:text-signal"
                        >
                          {preset.label}
                          {!preset.clears && (
                            <span className="stat ml-1.5 text-micro text-ink-faint">
                              {preset.plugins.length}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Search + category */}
                  <div className="mt-6 flex flex-col gap-2 sm:flex-row">
                    <div className="relative flex-1">
                      <label htmlFor="plugin-vault-search" className="sr-only">
                        Search plugins
                      </label>
                      <input
                        id="plugin-vault-search"
                        ref={searchRef}
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        type="search"
                        placeholder="Search name, maker or capability — “dynamic EQ”, “fabfilter”, “soothe”"
                        className="input-field py-2.5 pl-9 text-sm"
                      />
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        aria-hidden="true"
                        className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint"
                      >
                        <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth={1.8} />
                        <path d="m20 20-3.5-3.5" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
                      </svg>
                    </div>
                    <div className="sm:w-44">
                      <label htmlFor="plugin-vault-category" className="sr-only">
                        Filter by category
                      </label>
                      <select
                        id="plugin-vault-category"
                        value={category}
                        onChange={(event) => setCategory(event.target.value)}
                        className="input-field py-2.5 text-sm"
                      >
                        <option value="all">All categories</option>
                        {CATEGORIES.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <p className="mt-2 font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                    {results.length} of {PLUGIN_CATALOG.length} in the catalog
                    {filtering ? ' match' : ''}
                  </p>

                  {/* Empty state */}
                  {count === 0 && !filtering && (
                    <div className="panel mt-5 p-4">
                      <p className="text-sm text-ink">Nothing here yet — the report still works.</p>
                      <p className="mt-2 text-xs leading-relaxed text-ink-muted">
                        Every plan assumes the stock set: a static EQ, a compressor, a limiter, a
                        gate, reverb, delay, saturation and an analyzer. Adding what you own only
                        changes which of the several right answers you get told about. Two clicks on
                        a bundle above is usually enough.
                      </p>
                    </div>
                  )}

                  {/* Lists */}
                  <div className="mt-5">
                    {ownedVisible.length > 0 && (
                      <Group
                        label="Owned"
                        count={`${ownedVisible.length}${
                          ownedVisible.length !== count ? ` / ${count}` : ''
                        }`}
                        open={expanded.has('__owned__')}
                        onToggle={() => toggleGroup('__owned__')}
                      >
                        {ownedVisible.map((plugin) => (
                          <PluginRow
                            key={`owned-${plugin.manufacturer ?? ''}-${plugin.name}`}
                            name={plugin.name}
                            manufacturer={plugin.manufacturer ?? null}
                            capabilities={plugin.capabilities}
                            owned
                            onToggle={() => remove(plugin)}
                          />
                        ))}
                      </Group>
                    )}

                    {grouped.map(({ manufacturer, plugins: makerPlugins }) => {
                      const ownedHere = makerPlugins.filter((plugin) => has(plugin)).length;
                      const allOwned = ownedHere === makerPlugins.length && makerPlugins.length > 0;
                      return (
                        <Group
                          key={manufacturer}
                          label={manufacturer}
                          count={`${ownedHere > 0 ? `${ownedHere} / ` : ''}${makerPlugins.length}`}
                          open={expanded.has(manufacturer)}
                          onToggle={() => toggleGroup(manufacturer)}
                          action={
                            <button
                              type="button"
                              onClick={() => {
                                if (allOwned) {
                                  // By object, not by name: matches() falls back to a
                                  // bare-name compare for strings, which would also
                                  // delete a user's custom entry that happens to share
                                  // the name. pluginKey is maker-scoped.
                                  makerPlugins.forEach((plugin) =>
                                    remove({
                                      name: plugin.name,
                                      manufacturer: plugin.manufacturer,
                                      category: plugin.category,
                                      capabilities: plugin.capabilities,
                                    }),
                                  );
                                } else {
                                  add(
                                    makerPlugins
                                      .filter((plugin) => !has(plugin))
                                      .map((plugin) => ({
                                        name: plugin.name,
                                        manufacturer: plugin.manufacturer,
                                        category: plugin.category,
                                        capabilities: plugin.capabilities,
                                      })),
                                  );
                                }
                              }}
                              aria-label={`${allOwned ? 'Remove all' : 'Add all'} ${manufacturer} plugins`}
                              className="shrink-0 rounded-lg border border-void-line px-2.5 py-1 font-mono text-micro uppercase tracking-[0.1em] text-ink-faint transition-colors hover:border-ink-faint hover:text-ink"
                            >
                              {allOwned ? 'Remove all' : 'Add all'}
                            </button>
                          }
                        >
                          {makerPlugins.map((plugin) => (
                            <PluginRow
                              key={`${manufacturer}-${plugin.name}`}
                              name={plugin.name}
                              manufacturer={null}
                              capabilities={plugin.capabilities}
                              owned={has(plugin)}
                              onToggle={() =>
                                toggle({
                                  name: plugin.name,
                                  manufacturer: plugin.manufacturer,
                                  category: plugin.category,
                                  capabilities: plugin.capabilities,
                                })
                              }
                            />
                          ))}
                        </Group>
                      );
                    })}

                    {grouped.length === 0 && ownedVisible.length === 0 && (
                      <p className="py-8 text-center text-xs text-ink-muted">
                        Nothing matches “{query}”. Add it by hand below — the capabilities are what
                        matter, not the name.
                      </p>
                    )}
                  </div>

                  <div className="mt-6">
                    <CustomForm onAdd={add} />
                  </div>
                </div>

                {/* Summary ------------------------------------------ */}
                <aside className="min-w-0 lg:sticky lg:top-0 lg:self-start">
                  <div className="panel p-5">
                    <SummaryPanel capabilities={capabilities} count={count} />
                  </div>
                </aside>
              </div>
            </div>

            {/* Footer ------------------------------------------------ */}
            <div className="shrink-0 border-t border-void-line px-5 py-3.5 sm:px-7">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="font-mono text-micro uppercase tracking-[0.14em] text-ink-faint">
                  Saved in this browser · no account needed
                </p>
                <div className="flex items-center gap-2">
                  {count > 0 && (
                    <button
                      type="button"
                      onClick={clear}
                      className="btn-ghost px-4 py-2 text-xs"
                    >
                      Clear all
                    </button>
                  )}
                  <button type="button" onClick={onClose} className="btn-primary px-5 py-2 text-xs">
                    Done
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ------------------------------------------------------------------ */
/* Host + trigger                                                      */
/* ------------------------------------------------------------------ */

/** Mount once, near the app root. Every trigger raises this instance. */
export function PluginVaultHost() {
  const open = useVaultOpen();
  return <PluginVault open={open} onClose={closePluginVault} />;
}

export interface PluginVaultTriggerProps {
  className?: string;
  /** Overrides the "Plugins · N" label; the count is still appended. */
  label?: string;
}

/** "Plugins · N" — the count is the whole point, so it is always shown. */
export function PluginVaultTrigger({ className, label = 'Plugins' }: PluginVaultTriggerProps) {
  const { count } = usePluginVault();
  return (
    <button
      type="button"
      onClick={openPluginVault}
      className={className ?? 'btn-ghost px-4 py-2 text-xs'}
      aria-haspopup="dialog"
    >
      {label}
      <span className="stat text-ink-faint" aria-hidden="true">
        ·
      </span>
      <span className="stat">{count}</span>
      <span className="sr-only">{count === 1 ? '1 plugin saved' : `${count} plugins saved`}</span>
    </button>
  );
}
