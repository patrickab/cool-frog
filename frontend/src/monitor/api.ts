import { fetchJson } from '../lib/http';

/** One progress bar of `Tracker.snapshot()["progress"]`, times in seconds, `remaining_seconds` null once done or before the first item. */
export type ProgressState = { completed: number; total: number; elapsed_seconds: number; remaining_seconds: number | null; paused: boolean };
/** Row count, column types (`int`, `float`, `str`, ...) and hub figure count of one tracker table. */
export type TableInfo = { rows: number; columns: Record<string, string>; figures: number };
export type Gpu = {
  index: number; name: string; utilization_percent: number;
  memory_used_mib: number; memory_total_mib: number; temperature_celsius: number;
};
/** `coolfrog.lib.monitor` `/api/state`: the tracker snapshot plus `gpus`, other keys are free-form run state, e.g. `configs`. */
export type MonitorState = {
  title?: string; progress: Record<string, ProgressState>; tables: Record<string, TableInfo>; gpus: Gpu[]; [key: string]: unknown;
};
export type Cell = number | string | boolean | null;
/** `/api/tables/{name}`: column names and rows, non-finite floats as null. */
export type TableData = { columns: string[]; rows: Cell[][] };

/** Configured SSH host of `coolfrog.api.monitor_hub`, its password never leaves the hub. */
export type HostInfo = { address: string; has_password: boolean };

export const fetchHosts = () => fetchJson<HostInfo[]>('/api/hosts');
/** Add or replace the host at `address`, no `password` = plain `ssh`. */
export const addHost = (address: string, password: string | undefined) => fetchJson<HostInfo[]>('/api/hosts', { body: { address, password } });
export const removeHost = (address: string) => fetchJson<HostInfo[]>('/api/hosts/remove', { body: { address } });

const monitorUrl = (address: string, path: string, query = '') => `/api/hosts/monitor/${path}?address=${encodeURIComponent(address)}${query}`;
/** Training state on `address`, through the hub's SSH tunnel. */
export const fetchState = (address: string, signal: AbortSignal) => fetchJson<MonitorState>(monitorUrl(address, 'state'), { signal });
/** One table of the training on `address`, its last `tail` rows when given. */
export const fetchTable = (address: string, name: string, signal: AbortSignal, tail?: number) =>
  fetchJson<TableData>(monitorUrl(address, `tables/${encodeURIComponent(name)}`, tail ? `&tail=${tail}` : ''), { signal });
/** SVG of the hub's matplotlib figure `index` of one table, `rows` only busts the browser cache once the table grew. */
export const figureUrl = (address: string, name: string, index: number, dark: boolean, rows: number) =>
  monitorUrl(address, `figures/${encodeURIComponent(name)}.svg`, `&index=${index}&dark=${Number(dark)}&rows=${rows}`);
