import { fetchJson } from '../lib/http';

/** Contract of `coolfrog.lib.monitor.MonitorState`, served by the training process, passed through by the hub. */
export type Series = { label: string; color: string; side: 'left' | 'right'; steps: number[]; values: number[] };
export type Chart = { title: string; log_y: boolean; series: Series[] };
/** `times` as in the terminal dashboard: elapsed, remaining, total. */
export type Progress = { label: string; completed: number; total: number; times: string };
export type Gpu = {
  index: number; name: string; utilization_percent: number;
  memory_used_mib: number; memory_total_mib: number; temperature_celsius: number;
};
export type MonitorState = {
  title: string; summary: string; step_header: string; last_step: number | null;
  progress: Progress[]; charts: Chart[]; gpus: Gpu[];
};

/** Configured SSH host of `coolfrog.api.monitor_hub`, its password never leaves the hub. */
export type HostInfo = { address: string; has_password: boolean };

export const fetchHosts = (signal: AbortSignal) => fetchJson<HostInfo[]>('/api/hosts', signal);
/** Add or replace the host at `address`, no `password` = plain `ssh`. */
export const addHost = (address: string, password: string | undefined, signal: AbortSignal) =>
  fetchJson<HostInfo[]>('/api/hosts', signal, { address, password });
export const removeHost = (address: string, signal: AbortSignal) => fetchJson<HostInfo[]>('/api/hosts/remove', signal, { address });
/** Training state on `address`, through the hub's SSH tunnel. */
export const fetchMonitor = (address: string, signal: AbortSignal) =>
  fetchJson<MonitorState>(`/api/hosts/monitor?address=${encodeURIComponent(address)}`, signal);
