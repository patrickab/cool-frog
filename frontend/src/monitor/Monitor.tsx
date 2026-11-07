import { useEffect, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import { Plus } from 'lucide-react';
import { addHost, fetchHosts, fetchState, fetchTable, figureUrl, removeHost } from './api';
import type { Gpu, HostInfo, MonitorState, ProgressState, TableData, TableInfo } from './api';
import { errorMessage, formatCount } from '../lib/format';
import { ErrorPanel, HEADING, Layout, Panel, useTheme } from '../lib/ui';

const POLL_MILLISECONDS = 2000;
const SELECTED_HOST_KEY = 'coolfrog-monitor-host';
const MIB_PER_GIB = 1024;
const TABLE_TAIL = 500; // rows a table panel shows, the per-design validation table grows by ~900 per validation
const SHOWN_ELSEWHERE = new Set(['title', 'progress', 'tables', 'gpus']); // state keys with their own sidebar block

type Monitored = { state?: MonitorState; error?: string };

/** State of the training on `host` every `POLL_MILLISECONDS`, keeps the last state while the host is unreachable. */
function useMonitor(host: string | undefined): Monitored {
  const [result, setResult] = useState<Monitored>({});
  useEffect(() => {
    setResult({});
    if (!host) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        setResult({ state: await fetchState(host, controller.signal) });
      } catch (error) {
        if (controller.signal.aborted) return;
        setResult((previous) => ({ ...previous, error: errorMessage(error) }));
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, POLL_MILLISECONDS);
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [host]);
  return result;
}

/** The hub's matplotlib figure of one table, reloaded when rows were added, hidden when the table has nothing to plot. */
function TableFigure({ host, name, rows }: { host: string; name: string; rows: number }) {
  const { theme } = useTheme();
  const url = figureUrl(host, name, theme === 'dark', rows);
  const [failedUrl, setFailedUrl] = useState<string>();
  if (failedUrl === url) return null;
  return <Panel className="space-y-2 p-4">
    <h2 className={HEADING}>{name}</h2>
    <img src={url} alt={`${name} curves`} onError={() => setFailedUrl(url)} className="h-auto w-full" />
  </Panel>;
}

/** Hairline meter under a GPU or progress row, as gigachad-bot's usage meters. */
function ThinMeter({ label, percent }: { label: string; percent: number }) {
  return <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(percent)} className="h-0.5 overflow-hidden rounded-full bg-divider">
    <div className="h-full rounded-full bg-ink-faint" style={{ width: `${Math.min(100, percent)}%` }} />
  </div>;
}

const gib = (mib: number) => (mib / MIB_PER_GIB).toFixed(1);

/** One line of load, memory and temperature over the load meter, a click shows the model. */
function GpuRow({ gpu }: { gpu: Gpu }) {
  const [open, setOpen] = useState(false);
  return <li>
    <button onClick={() => setOpen(!open)} aria-expanded={open} title={`Memory ${gib(gpu.memory_used_mib)} of ${gib(gpu.memory_total_mib)} GiB`}
      className="flex w-full items-baseline gap-2 rounded-md px-2 py-1 text-left text-xs transition hover:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
      <span className="text-ink-muted">GPU {gpu.index}</span>
      <span className="ml-auto tabular-nums text-ink-muted">
        {Math.round(gpu.utilization_percent)}% · {gib(gpu.memory_used_mib)} GiB · {Math.round(gpu.temperature_celsius)} °C
      </span>
    </button>
    <div className="mx-2 mb-1"><ThinMeter label={`GPU ${gpu.index} load`} percent={gpu.utilization_percent} /></div>
    {open && <p className="mb-1 truncate px-2 text-[11px] text-ink-muted" title={gpu.name}>{gpu.name}, {gib(gpu.memory_total_mib)} GiB</p>}
  </li>;
}

/** Bottom sidebar block: mean load in the header, one line per GPU below. */
function GpuSection({ gpus }: { gpus: Gpu[] }) {
  const [open, setOpen] = useState(true);
  const meanUtilization = gpus.reduce((total, gpu) => total + gpu.utilization_percent, 0) / Math.max(gpus.length, 1);
  return <section aria-label="GPUs" className="mx-2 border-t border-divider-soft py-2">
    <button onClick={() => setOpen(!open)} aria-expanded={open}
      className="flex h-6 w-full items-center justify-between gap-2 px-2 text-xs text-ink-muted transition hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
      <span className="font-medium"><span aria-hidden className={`mr-1.5 inline-block transition ${open ? 'rotate-90' : ''}`}>›</span>GPUs</span>
      <span className="tabular-nums">{gpus.length ? `${Math.round(meanUtilization)}% mean load` : 'none'}</span>
    </button>
    {open && (gpus.length
      ? <ul className="mt-1 space-y-1">{gpus.map((gpu) => <GpuRow key={gpu.index} gpu={gpu} />)}</ul>
      : <p className="px-2 pt-1 text-xs text-ink-muted">No nvidia-smi on the training host</p>)}
  </section>;
}

/** Chevron group of the sidebar tree, its rows indented below. */
function SidebarSection({ label, count, defaultOpen, children }: { label: string; count?: number; defaultOpen: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return <section>
    <button onClick={() => setOpen(!open)} aria-expanded={open}
      className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-sm font-medium transition hover:bg-hover hover:text-ink">
      <span><span aria-hidden className={`mr-1.5 inline-block text-ink-muted transition ${open ? 'rotate-90' : ''}`}>›</span>{label}</span>
      {count !== undefined && <span className="text-xs font-normal text-ink-muted tabular-nums">{count}</span>}
    </button>
    {open && <div className="space-y-3 pb-2 pl-5 pr-2 pt-1">{children}</div>}
  </section>;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);

/** Numbers to 4 significant digits without trailing zeros (`5e-5`, `0.4823`), strings as they are, the rest as JSON. */
function formatValue(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4).replace(/(\.\d*?)0+(?=e|$)/, '$1').replace(/\.(?=e|$)/, '');
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/** Collapsed table, its newest `TABLE_TAIL` rows first, fetched while open and again when rows were added. */
function TablePanel({ host, name, info }: { host: string; name: string; info: TableInfo }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<TableData>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    fetchTable(host, name, controller.signal, TABLE_TAIL).then(
      (next) => { setData(next); setError(undefined); },
      (failure) => { if (!controller.signal.aborted) setError(errorMessage(failure)); },
    );
    return () => controller.abort();
  }, [open, host, name, info.rows]);
  const shown = info.rows > TABLE_TAIL ? ` · newest ${formatCount(TABLE_TAIL)} shown` : '';
  return <Panel>
    <button onClick={() => setOpen(!open)} aria-expanded={open}
      className="flex w-full items-center justify-between px-4 py-3 text-xs text-ink-muted transition hover:text-ink">
      <span><span aria-hidden className={`mr-1.5 inline-block transition ${open ? 'rotate-90' : ''}`}>›</span>{name}</span>
      <span className="tabular-nums">{formatCount(info.rows)} rows{open && shown}</span>
    </button>
    {open && error && <p role="alert" className="border-t border-divider-soft px-4 py-3 text-xs text-danger">{error}</p>}
    {open && data && <div tabIndex={0} role="region" aria-label={`${name} rows`} className="max-h-96 overflow-auto border-t border-divider-soft outline-none focus-visible:ring-2 focus-visible:ring-divider [scrollbar-width:thin]">
      <table className="w-full text-right text-xs tabular-nums">
        <caption className="sr-only">{name}, newest row first</caption>
        <thead className="sticky top-0 bg-surface text-ink-muted shadow-[inset_0_-1px_0_var(--divider-soft)]">
          <tr>{data.columns.map((column) => <th key={column} scope="col" className="whitespace-nowrap px-3 py-2 font-normal">{column}</th>)}</tr>
        </thead>
        <tbody>{[...data.rows].reverse().map((row, index) => <tr key={index} className="hover:bg-hover">
          {row.map((cell, column) => <td key={data.columns[column]} className={`whitespace-nowrap px-3 py-1 ${cell === null || column === 0 ? 'text-ink-muted' : ''}`}>{cell === null ? '-' : formatValue(cell)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>}
  </Panel>;
}

function duration(seconds: number): string {
  const total = Math.round(seconds);
  return `${Math.floor(total / 3600)}:${String(Math.floor(total / 60) % 60).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

/** `0:12:03 elapsed · ~1:02:00 left`, `done in 0:04:05` once complete. */
function progressTimes(progress: ProgressState): string {
  if (progress.completed >= progress.total) return `done in ${duration(progress.elapsed_seconds)}`;
  const remaining = progress.remaining_seconds === null ? undefined : `~${duration(progress.remaining_seconds)} left`;
  return [`${duration(progress.elapsed_seconds)} elapsed`, remaining, progress.paused && 'paused'].filter(Boolean).join(' · ');
}

/** Count over the hairline meter, elapsed and remaining time below on one line. */
function ProgressRow({ label, progress }: { label: string; progress: ProgressState }) {
  const times = progressTimes(progress);
  return <li>
    <div className="flex items-baseline gap-2 px-2 py-1 text-xs">
      <span className="text-ink-muted">{label}</span>
      <span className="ml-auto tabular-nums text-ink-muted">{formatCount(progress.completed)} / {formatCount(progress.total)}</span>
    </div>
    <div className="mx-2"><ThinMeter label={`${label} progress`} percent={(100 * progress.completed) / Math.max(progress.total, 1)} /></div>
    <p className="truncate whitespace-nowrap px-2 pt-1 text-[11px] tabular-nums text-ink-muted" title={times}>{times}</p>
  </li>;
}

/** One state dict as a folded group: its plain values as `key value` rows, nested dicts as nested groups, e.g. `configs`. */
function StateGroup({ label, value }: { label: string; value: Record<string, unknown> }) {
  const entries = Object.entries(value);
  const leaves = entries.filter(([, inner]) => !isRecord(inner));
  return <SidebarSection label={label} count={entries.length} defaultOpen={false}>
    {leaves.length > 0 && <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[11px]">
      {leaves.map(([key, inner]) => <div key={key} className="contents">
        <dt className="text-ink-muted">{key}</dt>
        <dd className="truncate text-right tabular-nums" title={formatValue(inner)}>{formatValue(inner)}</dd>
      </div>)}
    </dl>}
    {entries.filter(([, inner]) => isRecord(inner)).map(([key, inner]) => <StateGroup key={key} label={key} value={inner as Record<string, unknown>} />)}
  </SidebarSection>;
}

/** Free-form state keys of the run: nested dicts as groups, top-level plain values together in one `state` group. */
function StateSection({ state }: { state: MonitorState }) {
  const entries = Object.entries(state).filter(([key]) => !SHOWN_ELSEWHERE.has(key));
  const leaves = Object.fromEntries(entries.filter(([, value]) => !isRecord(value)));
  if (!entries.length) return null;
  return <section aria-label="Run state" className="mx-2 border-t border-divider-soft py-2">
    {entries.filter(([, value]) => isRecord(value)).map(([key, value]) => <StateGroup key={key} label={key} value={value as Record<string, unknown>} />)}
    {Object.keys(leaves).length > 0 && <StateGroup label="state" value={leaves} />}
  </section>;
}

/** Configured SSH hosts: pick the watched one, add one (address, optional password) or remove one. */
function HostsSection({ selected, onSelect }: { selected: string | undefined; onSelect: (address: string | undefined) => void }) {
  const [hosts, setHosts] = useState<HostInfo[]>();
  const [error, setError] = useState<string>();
  const [adding, setAdding] = useState(false);
  const [address, setAddress] = useState('');
  const [password, setPassword] = useState('');
  const update = (request: Promise<HostInfo[]>) => request.then(
    (next) => { setHosts(next); setError(undefined); return true; },
    (failure) => { setError(errorMessage(failure)); return false; },
  );
  useEffect(() => { void update(fetchHosts()); }, []);
  // Keep the selection on a configured host, e.g. after removing the selected one
  useEffect(() => {
    if (hosts && !hosts.some((host) => host.address === selected)) onSelect(hosts[0]?.address);
  }, [hosts, selected]);

  const add = async () => {
    const added = address.trim();
    if (added && await update(addHost(added, password || undefined))) {
      onSelect(added);
      cancel();
    }
  };
  const cancel = () => {
    setAdding(false);
    setAddress('');
    setPassword('');
  };
  // Enter in either field adds the host, Escape discards the entry
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') void add();
    if (event.key === 'Escape') cancel();
  };
  const inputClass = 'w-full rounded-lg bg-surface px-2.5 py-1.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-divider';

  return <SidebarSection label="Hosts" count={hosts?.length} defaultOpen={false}>
    {hosts?.length === 0 && <p className="text-xs text-ink-muted">No hosts in monitor-hosts.json yet</p>}
    {hosts && hosts.length > 0 && <ul className="space-y-0.5 text-xs">
      {hosts.map((host) => <li key={host.address} className="flex items-center gap-1">
        <button aria-pressed={host.address === selected} onClick={() => onSelect(host.address)}
          className={`min-w-0 flex-1 truncate rounded-lg px-2.5 py-1.5 text-left transition hover:bg-hover hover:text-ink ${host.address === selected ? 'bg-hover-strong text-ink' : 'text-ink-muted'}`}>
          {host.address}{host.has_password && <span className="text-ink-muted"> · password</span>}
        </button>
        <button aria-label={`Remove ${host.address}`} onClick={() => void update(removeHost(host.address))}
          className="shrink-0 rounded-lg px-2 py-1.5 text-[11px] text-ink-muted transition hover:bg-hover hover:text-ink">Remove</button>
      </li>)}
    </ul>}
    {adding
      ? <div className="space-y-2">
        <input autoFocus value={address} onChange={(event) => setAddress(event.target.value)} onKeyDown={onKeyDown} aria-label="SSH address"
          placeholder="user@host, Tailscale name or ssh alias" autoComplete="off" spellCheck={false} className={inputClass} />
        <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} onKeyDown={onKeyDown} aria-label="SSH password, optional"
          placeholder="Password, empty = SSH keys · Enter adds" autoComplete="new-password" className={inputClass} />
      </div>
      : <button aria-label="Add host" onClick={() => setAdding(true)}
        className="grid size-7 place-items-center rounded-lg text-ink-muted transition hover:bg-hover hover:text-ink focus-visible:outline-2 focus-visible:outline-divider">
        <Plus className="size-4" />
      </button>}
    {error && <p role="alert" className="text-xs text-danger">{error}</p>}
  </SidebarSection>;
}

export default function Monitor() {
  const [host, setHost] = useState<string | undefined>(() => {
    try { return localStorage.getItem(SELECTED_HOST_KEY) ?? undefined; } catch { return undefined; }
  });
  useEffect(() => {
    try { if (host) localStorage.setItem(SELECTED_HOST_KEY, host); } catch { /* Keep the in-memory selection when storage is unavailable. */ }
  }, [host]);
  const { state, error } = useMonitor(host);
  useEffect(() => { document.title = state?.title ? `Coolfrog · ${state.title}` : 'Coolfrog · Training monitor'; }, [state?.title]);
  const progress = Object.entries(state?.progress ?? {});
  const tables = Object.entries(state?.tables ?? {}); // order of first append, `train` before `validation`

  const sidebar = <Panel className="py-2">
    <nav aria-label="Hosts" className="px-2 pb-2"><HostsSection selected={host} onSelect={setHost} /></nav>
    {progress.length > 0 && <section aria-label="Progress" className="mx-2 border-t border-divider-soft py-2">
      <ul className="space-y-2">{progress.map(([label, bar]) => <ProgressRow key={label} label={label} progress={bar} />)}</ul>
    </section>}
    {state && <StateSection state={state} />}
    {state && <GpuSection gpus={state.gpus} />}
  </Panel>;

  return <Layout nav={null} sidebar={sidebar}>
    {error && <ErrorPanel message={error} />}
    {state && host
      ? <>
        <div className="grid gap-5 lg:grid-cols-2">{tables.map(([name, info]) => <TableFigure key={name} host={host} name={name} rows={info.rows} />)}</div>
        {tables.map(([name, info]) => <TablePanel key={name} host={host} name={name} info={info} />)}
      </>
      : !error && <Panel className={`grid h-40 place-items-center text-xs text-ink-muted ${host ? 'animate-pulse' : ''}`}>
        <span>{host ? `Connecting to ${host}…` : 'Open Hosts in the sidebar and add an SSH host to watch its training'}</span>
      </Panel>}
  </Layout>;
}
