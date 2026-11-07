import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { channelIndex, fetchField, fetchSurface, fluidValues } from './api';
import type { Config, Design, Field, Surface as SurfaceData } from './api';
import Heatmap from './Heatmap';
import { clipping, colorScaleOf } from './lib/colormap';
import type { ColorScale, ScaleKind } from './lib/colormap';
import { errorMessage, formatNumber } from './lib/format';
import { Colorbar, Panel, Segmented, Slider } from './lib/ui';

const Volume = lazy(() => import('./Volume'));
const Surface = lazy(() => import('./Volume').then((module) => ({ default: module.Surface })));

type Loaded = { field: Field } | { surface: SurfaceData };
type FieldState = Loaded | { error: string };
/** Shared color scale of a gallery and whether loaded values fall below or above it. */
type ColorLegend = { scale: ColorScale; clipped: { below: boolean; above: boolean } };
/** Per-design load state of one channel or of the surfaces, `color` shared across loaded designs. */
type FieldBatch = { fields: Record<number, FieldState>; color?: ColorLegend; pending: number; failed: number };
const NO_FIELDS: Record<number, FieldState> = {};
/** What every card of a gallery shows, `surfaceChannel` indexes `config.surface_channels`. */
export type FieldDisplay = { mode: '2d' | '3d' | 'surface'; channel: number; surfaceChannel: number; slice: number };

/** Temperature on the middle Z slice in 2D, unless overridden. */
export function defaultDisplay(config: Config, overrides: Partial<FieldDisplay> = {}): FieldDisplay {
  return { mode: '2d', channel: channelIndex(config, 'T'), surfaceChannel: 0, slice: Math.floor(config.dimensions[2] / 2), ...overrides };
}

/** Z slice slider with its height in mm. */
export function SliceControl({ config, slice, onChange }: { config: Config; slice: number; onChange: (slice: number) => void }) {
  return <>
    <div className="w-32"><Slider label="Z slice" min={0} max={config.dimensions[2] - 1} values={[slice]} onChange={([next]) => onChange(next)} /></div>
    <span className="min-w-16 whitespace-nowrap rounded-lg bg-surface-elevated px-2 py-0.5 text-right text-xs tabular-nums shadow-[var(--shadow-sm)]">{formatNumber(slice * config.spacing[2] * 1000)} mm</span>
  </>;
}

/** Render mode, field and Z slice controls, then the shared colorbar once `color` is known. */
export function FieldControls({ config, display, onChange, color }: {
  config: Config; display: FieldDisplay; onChange: (display: FieldDisplay) => void; color: ColorLegend | undefined;
}) {
  const update = (patch: Partial<FieldDisplay>) => onChange({ ...display, ...patch });
  const isSurface = display.mode === 'surface';
  const channels = isSurface ? config.surface_channels : config.channels;
  const shown = isSurface ? display.surfaceChannel : display.channel;
  const channel = channels[shown];
  return <>
    <Segmented label="Rendering mode" value={display.mode} onChange={(mode) => update({ mode })} options={[{ value: '2d', label: '2D' }, { value: '3d', label: '3D' }, { value: 'surface', label: 'Surface' }]} />
    <Segmented label="Field" value={shown} onChange={(index) => update(isSurface ? { surfaceChannel: index } : { channel: index })} options={channels.map(({ name }, index) => ({ value: index, label: name }))} />
    {!isSurface && <div className="flex items-center gap-3 rounded-xl bg-hover-strong py-1.5 pr-1.5 pl-3">
      <span className="text-xs text-ink-muted">Z</span>
      <SliceControl config={config} slice={display.slice} onChange={(slice) => update({ slice })} />
    </div>}
    {color && <div className="flex items-center gap-3 rounded-xl bg-hover-strong px-3 py-1.5 text-xs">
      <Colorbar scale={color.scale} clipped={color.clipped} unit={channel.unit} label={<span className="font-medium text-ink">{channel.name}</span>} />
      <span className="flex items-center gap-1.5 text-ink-muted"><i className="size-2.5 rounded-sm bg-solid" />{isSurface ? 'Solid cut · not stored' : 'Solid'}</span>
    </div>}
  </>;
}

/** Scale of the displayed quantity over all loaded designs, surface faces trimmed to the 1st–99th percentile since a few extreme faces would wash out the rest. */
function colorLegendOf(states: Loaded[], config: Config, kind: ScaleKind, surfaceChannel: number): ColorLegend | undefined {
  const values = concatenate(states.map((state) => 'field' in state
    ? fluidValues(state.field, config)
    : state.surface.values[surfaceChannel].filter(Number.isFinite)));
  const scale = colorScaleOf(kind, values, states.some((state) => 'surface' in state) ? 0.01 : 0);
  return scale && { scale, clipped: clipping(values, scale) };
}

function concatenate(arrays: Float32Array[]): Float32Array {
  const joined = new Float32Array(arrays.reduce((total, array) => total + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    joined.set(array, offset);
    offset += array.length;
  }
  return joined;
}

/** Load the displayed channel, or all surfaces in surface mode, of every design progressively, aborting and hiding the previous batch on change. */
export function useFieldBatch(designs: Design[] | undefined, display: FieldDisplay, config: Config): FieldBatch {
  const source = display.mode === 'surface' ? 'surface' : display.channel;
  const [batch, setBatch] = useState<{ designs: Design[]; source: number | 'surface'; fields: Record<number, FieldState> }>();
  useEffect(() => {
    if (!designs) return;
    const controller = new AbortController();
    const settle = (designId: number, state: FieldState) => {
      if (!controller.signal.aborted) setBatch((previous) => previous && { ...previous, fields: { ...previous.fields, [designId]: state } });
    };
    setBatch({ designs, source, fields: {} });
    for (const { id } of designs) {
      const loading: Promise<FieldState> = source === 'surface'
        ? fetchSurface(config, id, controller.signal).then((surface) => ({ surface }))
        : fetchField(config, id, source, controller.signal).then((field) => ({ field }));
      loading.then((state) => settle(id, state), (error) => settle(id, { error: errorMessage(error) }));
    }
    return () => controller.abort();
  }, [designs, source, config]);

  // Until the effect above resets it, `batch` still holds the previous designs or source
  const fields = batch?.designs === designs && batch?.source === source ? batch.fields : NO_FIELDS;
  const kind = (display.mode === 'surface' ? config.surface_channels[display.surfaceChannel] : config.channels[display.channel]).scale;
  return useMemo(() => {
    const loaded = Object.values(fields).filter((state): state is Loaded => !('error' in state));
    const failed = Object.keys(fields).length - loaded.length;
    return { fields, color: colorLegendOf(loaded, config, kind, display.surfaceChannel), failed, pending: (designs?.length ?? 0) - loaded.length - failed };
  }, [fields, designs, config, kind, display.surfaceChannel]);
}

function DesignCard({ design, state, config, scale, display, highlight }: {
  design: Design; state: FieldState | undefined; config: Config; scale: ColorScale | undefined; display: FieldDisplay; highlight: string[];
}) {
  const { mode, slice } = display;
  const units = Object.fromEntries(config.parameters.map((parameter) => [parameter.name, parameter.unit]));
  const placeholder = <div className="mx-4 h-72 animate-pulse rounded-xl bg-paper" />;
  return <Panel className="overflow-hidden pb-3">
    <div className="px-4 pt-4 pb-2">
      <h3 className="text-sm font-medium">Design {design.id}</h3>
      {highlight.length > 0 && <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-ink-muted tabular-nums">
        {highlight.map((name) => <span key={name}>{name} <span className="text-ink">{formatNumber(design.params[name])}</span>{units[name] && ` ${units[name]}`}</span>)}
      </p>}
    </div>
    {state && 'error' in state
      ? <p className="mx-4 rounded-xl bg-danger-soft p-4 text-xs break-words text-danger" role="alert">{state.error}</p>
      : !state || !scale
        ? <div className="mx-4 grid h-40 animate-pulse place-items-center rounded-xl bg-paper text-xs text-ink-muted" role="status">Loading field · uncached designs can take minutes</div>
        : 'surface' in state
          ? <Suspense fallback={placeholder}><Surface surface={state.surface} channel={display.surfaceChannel} config={config} scale={scale} /></Suspense>
          : mode === '2d'
            ? <Heatmap field={state.field} config={config} scale={scale} slice={slice} />
            : <Suspense fallback={placeholder}><Volume field={state.field} config={config} scale={scale} slice={slice} /></Suspense>}
  </Panel>;
}

/** Two cards per row, `highlight` parameters show under each card title. */
export default function Gallery({ designs, batch, config, display, label = 'Sampled simulations', highlight = [] }: {
  designs: Design[]; batch: FieldBatch; config: Config; display: FieldDisplay; label?: string; highlight?: string[];
}) {
  return <section aria-label={label} aria-busy={batch.pending > 0} className={`grid gap-5 ${designs.length > 1 ? 'grid-cols-2 max-lg:grid-cols-1' : ''}`}>
    {designs.map((design) => <DesignCard key={design.id} design={design} state={batch.fields[design.id]} config={config} scale={batch.color?.scale} display={display} highlight={highlight} />)}
  </section>;
}
