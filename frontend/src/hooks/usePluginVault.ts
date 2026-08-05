/**
 * The plugin vault: what the producer owns, and therefore what advice is
 * allowed to assume.
 *
 * Two rules shape this file.
 *
 * 1. **No account required.** A first-time visitor must be able to fill this in
 *    and get personalised advice immediately, so localStorage is the source of
 *    truth. The server copy is a convenience for people who happen to be logged
 *    in, never a prerequisite, and the UI never waits on it.
 * 2. **One store, many mounts.** The header count and the picker are separate
 *    components that must agree instantly. A module-level store read through
 *    `useSyncExternalStore` gives every mount the same array identity, which
 *    per-component `useState` cannot.
 *
 * The `/plugins` API predates the capability vocabulary and stores only
 * name/category/manufacturer, so rows pulled from it are re-tagged locally via
 * `inferCapabilities()`.
 */

import { useCallback, useEffect, useMemo, useSyncExternalStore } from 'react';

import type { OwnedPlugin } from '../types/analysis';
import { API_BASE } from '../config';
import {
  ALL_CAPABILITIES,
  STOCK_CAPABILITIES,
  inferCapabilities,
  normaliseCapabilities,
} from '../data/plugins';

const STORAGE_KEY = 'mixdoctor.plugins.v1';
const TOKEN_KEY = 'mixdoctor_token';

/* ------------------------------------------------------------------ */
/* Identity                                                            */
/* ------------------------------------------------------------------ */

/**
 * Manufacturer + name, because two different boxes are called "Tape". Falls
 * back to the bare name so a custom entry and its catalogue twin still collide.
 */
export function pluginKey(plugin: OwnedPlugin): string {
  const maker = (plugin.manufacturer ?? '').trim().toLowerCase();
  const name = plugin.name.trim().toLowerCase();
  return maker ? `${maker}::${name}` : name;
}

function matches(plugin: OwnedPlugin, target: OwnedPlugin | string): boolean {
  if (typeof target === 'string') {
    const wanted = target.trim().toLowerCase();
    return plugin.name.trim().toLowerCase() === wanted || pluginKey(plugin) === wanted;
  }
  return pluginKey(plugin) === pluginKey(target);
}

/* ------------------------------------------------------------------ */
/* Persistence                                                         */
/* ------------------------------------------------------------------ */

/**
 * localStorage is attacker-adjacent in the sense that matters here: another
 * tab, an older build, a hand-edited value or a quota failure can all put
 * nonsense in it. Anything that is not a well-formed plugin is dropped rather
 * than thrown, because a corrupt vault must never stop the app from loading.
 */
export function sanitisePlugin(raw: unknown): OwnedPlugin | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const value = raw as Record<string, unknown>;

  const name = typeof value.name === 'string' ? value.name.trim() : '';
  if (name === '' || name.length > 120) return null;

  const category = typeof value.category === 'string' && value.category.trim() !== ''
    ? value.category.trim()
    : 'Other';

  const manufacturer =
    typeof value.manufacturer === 'string' && value.manufacturer.trim() !== ''
      ? value.manufacturer.trim()
      : null;

  const capabilities = normaliseCapabilities(
    Array.isArray(value.capabilities) ? (value.capabilities as string[]) : [],
  );

  return { name, manufacturer, category, capabilities };
}

function dedupe(plugins: OwnedPlugin[]): OwnedPlugin[] {
  const seen = new Set<string>();
  const out: OwnedPlugin[] = [];
  for (const plugin of plugins) {
    const key = pluginKey(plugin);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(plugin);
  }
  return out;
}

function load(): OwnedPlugin[] {
  if (typeof window === 'undefined') return [];
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return []; // Private mode / storage disabled.
  }
  if (!stored) return [];

  try {
    const parsed: unknown = JSON.parse(stored);
    // Accept both the bare array and a `{ plugins: [...] }` envelope, so a
    // future wrapper does not orphan everyone's existing list.
    const list = Array.isArray(parsed)
      ? parsed
      : typeof parsed === 'object' && parsed !== null && Array.isArray((parsed as { plugins?: unknown }).plugins)
        ? ((parsed as { plugins: unknown[] }).plugins)
        : [];
    return dedupe(list.map(sanitisePlugin).filter((p): p is OwnedPlugin => p !== null));
  } catch {
    return [];
  }
}

function persist(plugins: OwnedPlugin[]): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(plugins));
  } catch {
    // Quota or private mode. The in-memory store still works for this session.
  }
}

/* ------------------------------------------------------------------ */
/* Store                                                               */
/* ------------------------------------------------------------------ */

let state: OwnedPlugin[] = load();
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
  scheduleSync();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): OwnedPlugin[] {
  return state;
}

const EMPTY: OwnedPlugin[] = [];
function getServerSnapshot(): OwnedPlugin[] {
  return EMPTY;
}

/** Replace the list, persist, notify. No-op when nothing actually changed. */
function setPlugins(next: OwnedPlugin[]): void {
  const deduped = dedupe(next);
  if (
    deduped.length === state.length &&
    deduped.every((plugin, i) => pluginKey(plugin) === pluginKey(state[i]))
  ) {
    return;
  }
  state = deduped;
  persist(state);
  emit();
}

// Another tab edited the vault. Adopt it rather than fighting over the key.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key !== null && event.key !== STORAGE_KEY) return;
    const next = load();
    state = next;
    for (const listener of listeners) listener();
  });
}

/* ------------------------------------------------------------------ */
/* Optional server sync                                                */
/* ------------------------------------------------------------------ */

/** name-key -> server row id, for the DELETE path. */
const serverIds = new Map<string, number>();
let syncToken: string | null = null;
let syncing = false;
let syncQueued = false;

interface ServerPlugin {
  id: number;
  name: string;
  category: string;
  manufacturer: string | null;
}

function readServerRows(data: unknown): ServerPlugin[] {
  if (!Array.isArray(data)) return [];
  const rows: ServerPlugin[] = [];
  for (const raw of data) {
    if (typeof raw !== 'object' || raw === null) continue;
    const row = raw as Record<string, unknown>;
    if (typeof row.id !== 'number' || typeof row.name !== 'string') continue;
    rows.push({
      id: row.id,
      name: row.name,
      category: typeof row.category === 'string' ? row.category : 'Other',
      manufacturer: typeof row.manufacturer === 'string' ? row.manufacturer : null,
    });
  }
  return rows;
}

/**
 * Fold the account's list into the local one. Deliberately a union rather than
 * a replacement: signing in must never wipe the list someone spent five minutes
 * building anonymously, and the machine they are on is as likely to be right as
 * the account is. Deletion is an explicit act, so only `push` ever removes.
 */
async function pull(token: string): Promise<void> {
  const response = await fetch(`${API_BASE}/plugins`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) return;

  const rows = readServerRows(await response.json());
  const existing = new Set(state.map(pluginKey));
  const additions: OwnedPlugin[] = [];

  for (const row of rows) {
    const plugin: OwnedPlugin = {
      name: row.name,
      manufacturer: row.manufacturer,
      category: row.category,
      capabilities: inferCapabilities(row.name, row.category, row.manufacturer),
    };
    serverIds.set(pluginKey(plugin), row.id);
    if (!existing.has(pluginKey(plugin))) additions.push(plugin);
  }

  if (additions.length > 0) setPlugins([...state, ...additions]);
}

/** Make the server look like the local list. Failures are per-row and ignored. */
async function push(token: string): Promise<void> {
  const wanted = new Map(state.map((plugin) => [pluginKey(plugin), plugin]));

  for (const [key, plugin] of wanted) {
    if (serverIds.has(key)) continue;
    try {
      const response = await fetch(`${API_BASE}/plugins`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          name: plugin.name,
          category: plugin.category,
          manufacturer: plugin.manufacturer ?? undefined,
        }),
      });
      if (response.ok) {
        const created = (await response.json()) as { id?: unknown };
        if (typeof created.id === 'number') serverIds.set(key, created.id);
      } else if (response.status === 400) {
        // Already there under a different key shape; nothing to do.
        serverIds.set(key, -1);
      }
    } catch {
      // Offline. The local list is unaffected; we will retry on the next change.
    }
  }

  for (const [key, id] of Array.from(serverIds)) {
    if (wanted.has(key) || id < 0) continue;
    try {
      const response = await fetch(`${API_BASE}/plugins/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok || response.status === 404) serverIds.delete(key);
    } catch {
      // Offline; leave the mapping in place and retry later.
    }
  }
}

async function runSync(): Promise<void> {
  const token = syncToken;
  if (!token || syncing) {
    syncQueued = syncQueued || Boolean(token);
    return;
  }
  syncing = true;
  try {
    await push(token);
  } catch {
    // Network errors must never surface here — the vault is already saved.
  } finally {
    syncing = false;
    if (syncQueued) {
      syncQueued = false;
      void runSync();
    }
  }
}

/** Coalesce a burst of adds (a preset is 30-odd) into one sync pass. */
let syncTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleSync(): void {
  if (typeof window === 'undefined') return;
  if (!syncToken) {
    // Logging in on this tab writes the token without firing a storage event,
    // so look again before concluding this is an anonymous session. An
    // explicitly-set token is never second-guessed.
    const token = readToken();
    if (!token) return;
    syncVaultWithAccount(token);
    return;
  }
  if (syncTimer !== null) clearTimeout(syncTimer);
  syncTimer = setTimeout(() => {
    syncTimer = null;
    void runSync();
  }, 400);
}

/**
 * Point the vault at an account, or at nothing. Signing in adopts whatever was
 * collected anonymously; signing out simply stops syncing and leaves the local
 * list alone. Returns immediately — all the work is scheduled.
 */
export function syncVaultWithAccount(token: string | null): void {
  if (token === syncToken) return;
  syncToken = token;
  serverIds.clear();
  if (!token) return;
  void (async () => {
    try {
      await pull(token);
    } catch {
      // Signed in but offline, or the token expired. Local list stands.
    }
    void runSync();
  })();
}

/** The auth context is optional here — the vault must work signed out. */
function readToken(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Store API (React-free, so it can be driven and tested on its own)   */
/* ------------------------------------------------------------------ */

export {
  subscribe as subscribeVault,
  getSnapshot as getVaultPlugins,
  setPlugins as setVaultPlugins,
};

/**
 * Every capability the producer can reach, stock included, in canonical order.
 * Mirrors `owned_capabilities()` in capabilities.py.
 */
export function vaultCapabilities(plugins: readonly OwnedPlugin[]): string[] {
  const owned = new Set<string>(STOCK_CAPABILITIES);
  for (const plugin of plugins) {
    for (const capability of plugin.capabilities) owned.add(capability);
  }
  // Canonical order, so the chip row does not reshuffle as things are added.
  return ALL_CAPABILITIES.filter((capability) => owned.has(capability));
}

/** Add one or many. Anything malformed is dropped; duplicates collapse. */
export function addToVault(plugin: OwnedPlugin | OwnedPlugin[]): void {
  const incoming = (Array.isArray(plugin) ? plugin : [plugin])
    .map(sanitisePlugin)
    .filter((p): p is OwnedPlugin => p !== null);
  if (incoming.length === 0) return;
  setPlugins([...state, ...incoming]);
}

export function removeFromVault(target: OwnedPlugin | string): void {
  setPlugins(state.filter((plugin) => !matches(plugin, target)));
}

export function toggleInVault(plugin: OwnedPlugin): void {
  if (state.some((owned) => matches(owned, plugin))) {
    removeFromVault(plugin);
    return;
  }
  addToVault(plugin);
}

export function clearVault(): void {
  setPlugins([]);
}

export function vaultHas(target: OwnedPlugin | string): boolean {
  return state.some((plugin) => matches(plugin, target));
}

/* ------------------------------------------------------------------ */
/* Hook                                                                */
/* ------------------------------------------------------------------ */

export interface PluginVault {
  plugins: OwnedPlugin[];
  add: (plugin: OwnedPlugin | OwnedPlugin[]) => void;
  remove: (target: OwnedPlugin | string) => void;
  toggle: (plugin: OwnedPlugin) => void;
  has: (target: OwnedPlugin | string) => boolean;
  clear: () => void;
  /** Owned capabilities plus the stock set — mirrors `owned_capabilities()`. */
  capabilities: string[];
  count: number;
}

export function usePluginVault(): PluginVault {
  const plugins = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Signing in adopts whatever was collected anonymously; signing out keeps it
  // local. Either way the effect only ever schedules work — it never blocks.
  useEffect(() => {
    const recheck = () => syncVaultWithAccount(readToken());
    recheck();
    const onStorage = (event: StorageEvent) => {
      if (event.key === TOKEN_KEY || event.key === null) recheck();
    };
    // `focus` catches the case where the producer signed in on this tab while
    // these components stayed mounted. Both handlers are idempotent.
    window.addEventListener('storage', onStorage);
    window.addEventListener('focus', recheck);
    return () => {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('focus', recheck);
    };
  }, []);

  const add = useCallback((plugin: OwnedPlugin | OwnedPlugin[]) => addToVault(plugin), []);
  const remove = useCallback((target: OwnedPlugin | string) => removeFromVault(target), []);
  const toggle = useCallback((plugin: OwnedPlugin) => toggleInVault(plugin), []);
  const clear = useCallback(() => clearVault(), []);

  // Depends on the snapshot rather than the module state so that a row
  // re-renders the moment it is toggled.
  const has = useCallback(
    (target: OwnedPlugin | string) => plugins.some((plugin) => matches(plugin, target)),
    [plugins],
  );

  const capabilities = useMemo(() => vaultCapabilities(plugins), [plugins]);

  return { plugins, add, remove, toggle, has, clear, capabilities, count: plugins.length };
}

export default usePluginVault;
