import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { addHost, fetchHosts, fetchMonitor, removeHost } from './api';
import type { Chart, Gpu, HostInfo, MonitorState, Progress } from './api';
import { errorMessage, formatNumber, formatTick } from '../lib/format';
import { HEADING, Layout, Panel } from '../lib/ui';

const POLL_MILLISECONDS = 2000;
const SELECTED_HOST_KEY = 'coolfrog-monitor-host';
const MIB_PER_GIB = 1024;

/** Latest state of the training on `host`, polled every `POLL_MILLISECONDS`. Keeps the last state while it is unreachable. */
function useMonitor(host: string | undefined): { state?: MonitorState; error?: string } {
  const [result, setResult] = useState<{ state?: MonitorState; error?: string }>({});
  useEffect(() => {
    setResult({});
    if (!host) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const state = await fetchMonitor(host, controller.signal);
        setResult({ state });
      } catch (error) {
        if (controller.signal.aborted) return;
        setResult((previous) => ({ state: previous.state, error: errorMessage(error) }));
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, POLL_MILLISECONDS);
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [host]);
  return result;
}

// ---------------------------------------------------------------------------
// GPUs and progress
// ---------------------------------------------------------------------------

/** Usage bar, ported from gigachad-bot `ProviderUsage`. */
function Meter({ label, percent }: { label: string; percent: number }) {
  return <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(percent)} className="h-1 overflow-hidden rounded-full bg-divider-soft">
    <div className="h-full rounded-full bg-ink-subtle" style={{ width: `${Math.min(100, percent)}%` }} />
  </div>;
}

function GpuRow({ gpu }: { gpu: Gpu }) {
  const memoryPercent = (100 * gpu.memory_used_mib) / gpu.memory_total_mib;
  const gib = (mib: number) => (mib / MIB_PER_GIB).toFixed(1);
  return <li className="space-y-1.5 text-xs">
    <div className="flex items-baseline gap-2">
      <span className="min-w-0 truncate">GPU {gpu.index} · {gpu.name}</span>
      <span className="ml-auto shrink-0 tabular-nums text-ink-muted">{Math.round(gpu.utilization_percent)}%</span>
    </div>
    <Meter label={`GPU ${gpu.index} utilization`} percent={gpu.utilization_percent} />
    <div className="flex items-baseline gap-2 text-ink-muted">
      <span className="tabular-nums">Memory {gib(gpu.memory_used_mib)} / {gib(gpu.memory_total_mib)} GiB · {Math.round(gpu.temperature_celsius)} °C</span>
      <span className="ml-auto shrink-0 tabular-nums">{Math.round(memoryPercent)}%</span>
    </div>
    <Meter label={`GPU ${gpu.index} memory`} percent={memoryPercent} />
  </li>;
}

function ProgressRow({ progress }: { progress: Progress }) {
  return <li className="space-y-1.5 text-xs">
    <div className="flex items-baseline gap-2">
      <span>{progress.label}</span>
      <span className="ml-auto tabular-nums text-ink-muted">{progress.completed.toLocaleString('en')} / {progress.total.toLocaleString('en')}</span>
    </div>
    <Meter label={`${progress.label} progress`} percent={(100 * progress.completed) / Math.max(progress.total, 1)} />
    <p className="tabular-nums text-ink-muted">{progress.times}</p>
  </li>;
}

// ---------------------------------------------------------------------------
// Line chart
// ---------------------------------------------------------------------------

// Close to the rendered size of a chart in the two-column grid, so 11 px labels stay about 11 px
const WIDTH = 480;
const HEIGHT = 220;
const MARGIN = { top: 10, right: 48, bottom: 24, left: 48 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;
const MAX_TICKS = 5;
const MAX_DOTTED_POINTS = 40; // sparse series such as validations also mark each point
const MIN_LOG_DECADES = 1;

type Point = [step: number, value: number];
type Axis = { y: (value: number) => number; ticks: number[] };

/** At most `MAX_TICKS` round 1-2-5 ticks within [low, high]. */
function linearTicks(low: number, high: number): number[] {
  const rough = (high - low) / MAX_TICKS;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = magnitude * ([1, 2, 5, 10].find((factor) => magnitude * factor >= rough) ?? 10);
  const ticks = [];
  for (let multiple = Math.ceil(low / step); multiple * step <= high; multiple++) ticks.push(multiple * step);
  return ticks;
}

/** 1-2-5 ticks per decade within log10 range [low, high], thinned to `MAX_TICKS`. */
function logTicks(low: number, high: number): number[] {
  const ticks = [];
  for (let exponent = Math.floor(low); exponent <= Math.ceil(high); exponent++) {
    for (const mantissa of [1, 2, 5]) {
      const tick = mantissa * 10 ** exponent;
      if (Math.log10(tick) >= low && Math.log10(tick) <= high) ticks.push(tick);
    }
  }
  const stride = Math.ceil(ticks.length / MAX_TICKS);
  return ticks.filter((_, index) => index % stride === 0);
}

/** Y axis over `values`, log10 spanning at least `MIN_LOG_DECADES`, undefined without values. */
function axisOf(values: number[], log: boolean): Axis | undefined {
  if (!values.length) return undefined;
  const transformed = values.map(log ? Math.log10 : (value) => value);
  let [low, high] = [Math.min(...transformed), Math.max(...transformed)];
  const minSpan = log ? MIN_LOG_DECADES : Math.abs(low) * 0.1 || 1;
  if (high - low < minSpan) [low, high] = [(low + high - minSpan) / 2, (low + high + minSpan) / 2];
  const y = (value: number) => MARGIN.top + (1 - ((log ? Math.log10(value) : value) - low) / (high - low)) * PLOT_HEIGHT;
  const ticks = log ? logTicks(low, high) : linearTicks(low, high);
  return { y, ticks };
}

function LineChart({ chart, lastStep }: { chart: Chart; lastStep: number }) {
  // Non-positive values are dropped on a log axis
  const lines = chart.series.map((series) => ({
    series,
    points: series.steps.map((step, index): Point => [step, series.values[index]])
      .filter(([, value]) => Number.isFinite(value) && (!chart.log_y || value > 0)),
  }));
  const [left, right] = (['left', 'right'] as const).map((side) =>
    axisOf(lines.filter(({ series }) => series.side === side).flatMap(({ points }) => points.map(([, value]) => value)), chart.log_y));
  const stepMax = Math.max(lastStep, 1);
  const x = (step: number) => MARGIN.left + (step / stepMax) * PLOT_WIDTH;
  const hasRight = right !== undefined;

  return <Panel className="space-y-3 p-4">
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <h2 className={HEADING}>{chart.title}</h2>
      <ul className="flex flex-wrap gap-x-3 text-xs text-ink-muted">
        {chart.series.map((series) => <li key={series.label} className="flex items-center gap-1.5">
          <span className="size-2 rounded-full" style={{ backgroundColor: series.color }} />
          {series.label}{hasRight && series.side === 'right' && ' (right axis)'}
        </li>)}
      </ul>
    </div>
    {!left && !right
      ? <p className="grid place-items-center text-xs text-ink-muted" style={{ aspectRatio: `${WIDTH} / ${HEIGHT}` }}>Nothing logged yet</p>
      : <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={chart.title} className="block h-auto w-full text-[11px]">
        {left?.ticks.map((tick) => <g key={`l${tick}`}>
          <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={left.y(tick)} y2={left.y(tick)} stroke="var(--divider-soft)" />
          <text x={MARGIN.left - 6} y={left.y(tick)} textAnchor="end" dominantBaseline="middle" className="fill-ink-muted tabular-nums">{formatTick(tick)}</text>
        </g>)}
        {right?.ticks.map((tick) => <text key={`r${tick}`} x={WIDTH - MARGIN.right + 6} y={right.y(tick)} dominantBaseline="middle" className="fill-ink-muted tabular-nums">{formatTick(tick)}</text>)}
        {linearTicks(0, stepMax).filter(Number.isInteger).map((step) => <text key={`x${step}`} x={x(step)} y={HEIGHT - 6} textAnchor="middle" className="fill-ink-muted tabular-nums">{formatNumber(step)}</text>)}
        <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={MARGIN.top + PLOT_HEIGHT} y2={MARGIN.top + PLOT_HEIGHT} stroke="var(--divider)" />
        {lines.map(({ series, points }) => {
          const axis = series.side === 'left' ? left : right;
          if (!axis || !points.length) return null;
          const coordinates = points.map(([step, value]) => [x(step), axis.y(value)]);
          return <g key={series.label} stroke={series.color} fill={series.color}>
            <path d={`M${coordinates.map(([px, py]) => `${px},${py}`).join('L')}`} fill="none" strokeWidth="1.5" strokeLinejoin="round" />
            {points.length <= MAX_DOTTED_POINTS && coordinates.map(([px, py]) => <circle key={px} cx={px} cy={py} r="2.5" stroke="none" />)}
          </g>;
        })}
      </svg>}
  </Panel>;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/** Configured SSH hosts: pick the watched one, add one (address, optional password) or remove one. */
function HostsPanel({ selected, onSelect }: { selected: string | undefined; onSelect: (address: string | undefined) => void }) {
  const [hosts, setHosts] = useState<HostInfo[]>();
  const [error, setError] = useState<string>();
  const [address, setAddress] = useState('');
  const [password, setPassword] = useState('');
  const update = (request: (signal: AbortSignal) => Promise<HostInfo[]>) => request(new AbortController().signal).then(
    (next) => { setHosts(next); setError(undefined); return true; },
    (failure) => { setError(errorMessage(failure)); return false; },
  );
  useEffect(() => { void update(fetchHosts); }, []);
  // Keep the selection on a configured host, e.g. after removing the selected one
  useEffect(() => {
    if (hosts && !hosts.some((host) => host.address === selected)) onSelect(hosts[0]?.address);
  }, [hosts, selected]);

  const add = async (event: FormEvent) => {
    event.preventDefault();
    const added = address.trim();
    if (await update((signal) => addHost(added, password || undefined, signal))) {
      onSelect(added);
      setAddress('');
      setPassword('');
    }
  };
  const inputClass = 'w-full rounded-lg bg-paper px-2.5 py-1.5 text-xs outline-none focus-visible:ring-2 focus-visible:ring-divider';

  return <Panel className="space-y-3 p-4">
    <h2 className={HEADING}>Hosts</h2>
    {hosts?.length === 0 && <p className="text-xs text-ink-muted">No hosts in monitor-hosts.json yet</p>}
    {hosts && hosts.length > 0 && <ul className="space-y-1">
      {hosts.map((host) => <li key={host.address} className="flex items-center gap-2">
        <button aria-pressed={host.address === selected} onClick={() => onSelect(host.address)}
          className={`min-w-0 flex-1 truncate rounded-lg px-2.5 py-1 text-left text-xs transition hover:text-ink ${host.address === selected ? 'bg-surface-elevated text-ink shadow-[var(--shadow-sm)]' : 'text-ink-muted'}`}>
          {host.address}{host.has_password && <span className="text-ink-muted"> · password</span>}
        </button>
        <button aria-label={`Remove ${host.address}`} onClick={() => void update((signal) => removeHost(host.address, signal))}
          className="shrink-0 text-[11px] text-ink-muted transition hover:text-ink">Remove</button>
      </li>)}
    </ul>}
    <form onSubmit={(event) => void add(event)} className="space-y-2">
      <input value={address} onChange={(event) => setAddress(event.target.value)} aria-label="SSH address" placeholder="user@host, Tailscale name or ssh alias"
        autoComplete="off" spellCheck={false} className={inputClass} />
      <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} aria-label="SSH password, optional"
        placeholder="Password, empty = SSH keys" autoComplete="new-password" className={inputClass} />
      <button type="submit" disabled={!address.trim()}
        className="w-full rounded-xl border border-divider-soft py-1.5 text-xs text-ink-muted transition hover:border-divider hover:text-ink disabled:opacity-40">Add host</button>
    </form>
    {error && <p role="alert" className="text-xs text-danger">{error}</p>}
  </Panel>;
}

export default function Monitor() {
  const [host, setHost] = useState<string | undefined>(() => {
    try { return localStorage.getItem(SELECTED_HOST_KEY) ?? undefined; } catch { return undefined; }
  });
  useEffect(() => {
    try { if (host) localStorage.setItem(SELECTED_HOST_KEY, host); } catch { /* Keep the in-memory selection when storage is unavailable. */ }
  }, [host]);
  const { state, error } = useMonitor(host);
  useEffect(() => { document.title = state ? `Coolfrog · ${state.title}` : 'Coolfrog · Training monitor'; }, [state?.title]);
  const nav = <p className="mt-1 text-sm text-ink-muted">Training monitor{state && ` · ${state.title}`}</p>;
  const status = state && [host, state.last_step !== null && `${state.step_header} ${formatNumber(state.last_step)}`, `polled every ${POLL_MILLISECONDS / 1000} s`].filter(Boolean).join(' · ');

  const sidebar = <>
    <HostsPanel selected={host} onSelect={setHost} />
    {state && state.progress.length > 0 && <Panel className="space-y-4 p-4">
      <h2 className={HEADING}>Progress</h2>
      <ul className="space-y-4">{state.progress.map((progress) => <ProgressRow key={progress.label} progress={progress} />)}</ul>
      {state.summary && <p className="text-xs leading-5 text-ink-muted">{state.summary}</p>}
    </Panel>}
    {state && <Panel className="space-y-4 p-4">
      <h2 className={HEADING}>GPUs</h2>
      {state.gpus.length
        ? <ul className="space-y-4">{state.gpus.map((gpu) => <GpuRow key={gpu.index} gpu={gpu} />)}</ul>
        : <p className="text-xs text-ink-muted">No nvidia-smi on the training host</p>}
    </Panel>}
  </>;

  return <Layout nav={nav} sidebar={sidebar}>
    <div className="flex flex-wrap items-center gap-3">
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      <span className="ml-auto text-xs text-ink-muted" role="status">{status}</span>
    </div>
    {state
      ? <div className="grid gap-5 lg:grid-cols-2">{state.charts.map((chart) => <LineChart key={chart.title} chart={chart} lastStep={state.last_step ?? 0} />)}</div>
      : !error && <Panel className={`grid h-40 place-items-center text-xs text-ink-muted ${host ? 'animate-pulse' : ''}`}>
        <span>{host ? `Connecting to ${host}…` : 'Add an SSH host to watch its training'}</span>
      </Panel>}
  </Layout>;
}
