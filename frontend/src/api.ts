import type { ColorScale, ScaleKind } from './lib/colormap';
import { fetchJson, fetchOk } from './lib/http';
import type { Range } from './lib/range';

export type Parameter = { name: string; unit: string | null; min: number; max: number };
/** One field or surface quantity, `scale` = how it maps to color. */
export type Channel = { name: string; unit: string; scale: ScaleKind };
/** Sampling inputs and the grid every field comes on, `dimensions` and `spacing` (m) in x, y, z. */
export type Config = {
  parameters: Parameter[];
  channels: Channel[];
  surface_channels: Channel[];
  dimensions: [number, number, number];
  spacing: [number, number, number];
};
export type Design = { id: number; params: Record<string, number> };
export type Sample = { designs: Design[]; matched: number };
/** Narrowed parameter ranges only, omitted parameters span their full bounds. */
export type SampleQuery = { count: number; seed: number; ranges: Record<string, Range> };
/** x-fastest voxel values, `mask` 1 = fluid with a finite value, `min`/`max` over fluid voxels. */
export type Field = { values: Float32Array; mask: Uint8Array; min: number; max: number };

/** Fluid values of `field`, all voxels or one Z slice. */
export function fluidValues(field: Field, config: Config, slice?: number): Float32Array {
  const sliceSize = config.dimensions[0] * config.dimensions[1];
  const [start, end] = slice === undefined ? [0, field.values.length] : [slice * sliceSize, (slice + 1) * sliceSize];
  const values: number[] = [];
  for (let index = start; index < end; index++) if (field.mask[index]) values.push(field.values[index]);
  return Float32Array.from(values);
}

/** Boundary faces: x, y, z per point in m, legacy VTK `polygons` [n, i0, …], face `values` per surface channel, NaN where a boundary lacks it. */
export type Surface = { points: Float32Array; polygons: Uint32Array; values: Float32Array[] };
/** Props of every field renderer. */
export type FieldViewProps = { field: Field; config: Config; scale: ColorScale; slice: number };
/** Numeric dataset column, `counts` = histogram of all rows over [min, max]. */
export type DatasetColumn = { name: string; unit: string | null; min: number; max: number; counts: number[] };
/** Static overview, `header` is the column order of `fetchRows` rows. */
export type Dataset = {
  total: number; header: string[]; columns: DatasetColumn[]; runs: Record<string, number>;
};
/** Narrowed ranges only, `search` = design id prefix ("" = all), `runs` undefined = every run. */
export type DatasetFilter = { ranges: Record<string, Range>; search: string; runs?: string[] };
export type RowsQuery = DatasetFilter & { sort: string; descending: boolean; offset: number; limit: number };
export type Row = (number | string)[];
/** Filtered row count, histograms on the overview bins, run counts ignoring the run filter. */
export type DatasetSummary = { matched: number; histograms: Record<string, number[]>; runs: Record<string, number> };
/** Per channel in config order, `nrmse` in percent, `spectral_loss` on the per-design [-1, 1] scale. */
export type ReconstructionMetrics = { nrmse: number[]; spectral_loss: number[] };

export const fetchConfig = (signal: AbortSignal) => fetchJson<Config>('/api/config', signal);
export const fetchSample = (query: SampleQuery, signal: AbortSignal) => fetchJson<Sample>('/api/sample', signal, query);
export const fetchDataset = (signal: AbortSignal) => fetchJson<Dataset>('/api/dataset', signal);
export const fetchSummary = (filter: DatasetFilter, signal: AbortSignal) => fetchJson<DatasetSummary>('/api/dataset/summary', signal, filter);
export const fetchRows = (query: RowsQuery, signal: AbortSignal) => fetchJson<Row[]>('/api/dataset/rows', signal, query);
export const fetchDesigns = (ids: number[], signal: AbortSignal) => fetchJson<Design[]>(`/api/designs?${ids.map((id) => `id=${id}`).join('&')}`, signal);
export const fetchReconstructionMetrics = (designId: number, signal: AbortSignal) => fetchJson<ReconstructionMetrics>(`/api/reconstruction/${designId}/metrics`, signal);

/** One channel of one design, simulated or Tadpole-B's zero-shot reconstruction: float32 values then uint8 mask, `size` voxels each. */
export async function fetchField(designId: number, channel: number, size: number, signal: AbortSignal, source: 'fields' | 'reconstruction' = 'fields'): Promise<Field> {
  const buffer = await (await fetchOk(`/api/${source}/${designId}?channel=${channel}`, signal)).arrayBuffer();
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

/** Boundary surface of one design with every surface channel, see `Surface`. */
export async function fetchSurface(designId: number, channelCount: number, signal: AbortSignal): Promise<Surface> {
  const buffer = await (await fetchOk(`/api/surface/${designId}`, signal)).arrayBuffer();
  const [pointCount, polygonLength, faceCount] = new Uint32Array(buffer, 0, 3);
  const polygonsStart = 12 + pointCount * 12;
  const valuesStart = polygonsStart + polygonLength * 4;
  if (buffer.byteLength !== valuesStart + channelCount * faceCount * 4) throw new Error('The surface does not match the configured surface channels.');
  return {
    points: new Float32Array(buffer, 12, pointCount * 3),
    polygons: new Uint32Array(buffer, polygonsStart, polygonLength),
    values: Array.from({ length: channelCount }, (_, channel) => new Float32Array(buffer, valuesStart + channel * faceCount * 4, faceCount)),
  };
}
