import { fetchJson, fetchOk } from './lib/http';
import type { Range } from './lib/range';

export type Parameter = { name: string; unit: string | null; min: number; max: number };
export type Channel = { name: string; unit: string };
/** Sampling inputs and the grid every field comes on, `dimensions` and `spacing` (m) in x, y, z. */
export type Config = {
  parameters: Parameter[];
  channels: Channel[];
  dimensions: [number, number, number];
  spacing: [number, number, number];
};
export type Design = { id: number; params: Record<string, number> };
export type Sample = { designs: Design[]; matched: number };
/** Narrowed parameter ranges only, omitted parameters span their full bounds. */
export type SampleQuery = { count: number; seed: number; ranges: Record<string, Range> };
/** x-fastest voxel values, `mask` 1 = fluid with a finite value, `min`/`max` over fluid voxels. */
export type Field = { values: Float32Array; mask: Uint8Array; min: number; max: number };
/** Props of every field renderer. */
export type FieldViewProps = { field: Field; config: Config; range: Range; slice: number };
/** Numeric dataset column, `counts` = histogram of all rows over [min, max]. */
export type DatasetColumn = { name: string; unit: string | null; min: number; max: number; counts: number[] };
/** Static overview, `header` is the column order of `fetchRows` rows. */
export type Dataset = { total: number; header: string[]; columns: DatasetColumn[]; runs: Record<string, number> };
/** Narrowed ranges only, `runs` undefined = every run. */
export type DatasetFilter = { ranges: Record<string, Range>; runs?: string[] };
export type RowsQuery = DatasetFilter & { sort: string; descending: boolean; offset: number; limit: number };
export type Row = (number | string)[];
/** Filtered row count, histograms on the overview bins, run counts ignoring the run filter. */
export type DatasetSummary = { matched: number; histograms: Record<string, number[]>; runs: Record<string, number> };

export const fetchConfig = (signal: AbortSignal) => fetchJson<Config>('/api/config', signal);
export const fetchSample = (query: SampleQuery, signal: AbortSignal) => fetchJson<Sample>('/api/sample', signal, query);
export const fetchDataset = (signal: AbortSignal) => fetchJson<Dataset>('/api/dataset', signal);
export const fetchSummary = (filter: DatasetFilter, signal: AbortSignal) => fetchJson<DatasetSummary>('/api/dataset/summary', signal, filter);
export const fetchRows = (query: RowsQuery, signal: AbortSignal) => fetchJson<Row[]>('/api/dataset/rows', signal, query);

/** One channel of one design: float32 values then uint8 mask, `size` voxels each. */
export async function fetchField(designId: number, channel: number, size: number, signal: AbortSignal): Promise<Field> {
  const buffer = await (await fetchOk(`/api/fields/${designId}?channel=${channel}`, signal)).arrayBuffer();
  if (buffer.byteLength !== size * 5) throw new Error('The field dimensions do not match the configured grid.');
  const values = new Float32Array(buffer, 0, size);
  const mask = new Uint8Array(buffer, size * 4, size);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < size; i++) {
    if (!mask[i]) continue;
    min = Math.min(min, values[i]);
    max = Math.max(max, values[i]);
  }
  return { values, mask, min, max };
}
