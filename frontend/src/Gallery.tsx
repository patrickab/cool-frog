import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { fetchField, fetchSurface } from './api';
import type { Config, Design, Field, Surface as SurfaceData } from './api';
import Heatmap from './Heatmap';
import { gradient } from './lib/colormap';
import { errorMessage, formatNumber } from './lib/format';
import type { Range } from './lib/range';
import { Panel, Segmented, Slider } from './lib/ui';

const Volume = lazy(() => import('./Volume'));
const Surface = lazy(() => import('./Volume').then((module) => ({ default: module.Surface })));

export type RenderMode = '2d' | '3d' | 'surface';
type FieldState = { field: Field } | { surface: SurfaceData } | { error: string };
/** Per-design load state of one channel or of the surfaces, `colorRange` shared across loaded designs. */
export type FieldBatch = { fields: Record<number, FieldState>; colorRange?: Range; pending: number; failed: number };
const NO_FIELDS: Record<number, FieldState> = {};
const TEMPERATURE_CHANNEL = 4;
/** What every card of a gallery shows, `surfaceChannel` indexes `config.surface_channels`. */
export type FieldDisplay = { mode: RenderMode; channel: number; surfaceChannel: number; slice: number };

/** Temperature on the middle Z slice in 2D, unless overridden. */
export function defaultDisplay(config: Config, overrides: Partial<FieldDisplay> = {}): FieldDisplay {
  return { mode: '2d', channel: Math.min(TEMPERATURE_CHANNEL, config.channels.length - 1), surfaceChannel: 0, slice: Math.floor(config.dimensions[2] / 2), ...overrides };
}

/** Render mode, field and Z slice controls, then the shared color legend once `colorRange` is known. */
export function FieldControls({ config, display, onChange, colorRange }: {
  config: Config; display: FieldDisplay; onChange: (display: FieldDisplay) => void; colorRange: Range | undefined;
}) {
  const update = (patch: Partial<FieldDisplay>) => onChange({ ...display, ...patch });
  const isSurface = display.mode === 'surface';
  const channels = isSurface ? config.surface_channels : config.channels;
  const channelIndex = isSurface ? display.surfaceChannel : display.channel;
  const channel = channels[channelIndex];
  return <>
    <Segmented label="Rendering mode" value={display.mode} onChange={(mode) => update({ mode })} options={[{ value: '2d', label: '2D' }, { value: '3d', label: '3D' }, { value: 'surface', label: 'Surface' }]} />
    <Segmented label="Field" value={channelIndex} onChange={(index) => update(isSurface ? { surfaceChannel: index } : { channel: index })} options={channels.map(({ name }, index) => ({ value: index, label: name }))} />
    {!isSurface && <div className="flex items-center gap-3 rounded-xl bg-hover-strong py-1.5 pr-1.5 pl-3">
      <span className="text-xs text-ink-muted">Z</span>
      <div className="w-32"><Slider label="Z slice" min={0} max={config.dimensions[2] - 1} values={[display.slice]} onChange={([slice]) => update({ slice })} /></div>
      <span className="min-w-16 whitespace-nowrap rounded-lg bg-surface-elevated px-2 py-0.5 text-right text-xs tabular-nums shadow-[var(--shadow-sm)]">{formatNumber(display.slice * config.spacing[2] * 1000)} mm</span>
    </div>}
    {colorRange && <div className="flex items-center gap-3 rounded-xl bg-hover-strong px-3 py-1 text-xs">
      <span><span className="font-medium">{channel.name}</span> <span className="text-ink-muted">{channel.unit}</span></span>
      <div className="w-40">
        <div className="h-2 rounded-full" style={{ background: gradient }} />
        <div className="mt-0.5 flex justify-between text-[10px] leading-none text-ink-muted tabular-nums"><span>{formatNumber(colorRange[0])}</span><span>{formatNumber(colorRange[1])}</span></div>
      </div>
      <span className="flex items-center gap-1.5 text-ink-muted"><i className="size-2.5 rounded-sm bg-solid" />{isSurface ? 'Solid cut · not stored' : 'Solid'}</span>
    </div>}
  </>;
}

/** Min to max over volume fields, 1st to 99th percentile over surface faces, whose few extreme faces would wash out the rest. */
function colorRangeOf(states: FieldState[], surfaceChannel: number): Range | undefined {
  const fields = states.flatMap((state) => 'field' in state ? [state.field] : []);
  if (fields.length) return [Math.min(...fields.map((field) => field.min)), Math.max(...fields.map((field) => field.max))];
  const values = states.flatMap((state) => 'surface' in state ? Array.from(state.surface.values[surfaceChannel]).filter(Number.isFinite) : []).sort((a, b) => a - b);
  return values.length ? [values[Math.floor(values.length * 0.01)], values[Math.ceil(values.length * 0.99) - 1]] : undefined;
}

/** Load the displayed channel, or all surfaces in surface mode, of every design progressively, aborting and hiding the previous batch on change. */
export function useFieldBatch(designs: Design[] | undefined, display: FieldDisplay, config: Config): FieldBatch {
  const source = display.mode === 'surface' ? 'surface' : display.channel;
  const [batch, setBatch] = useState<{ designs: Design[]; source: number | 'surface'; fields: Record<number, FieldState> }>();
  useEffect(() => {
    if (!designs) return;
    const controller = new AbortController();
    const size = config.dimensions[0] * config.dimensions[1] * config.dimensions[2];
    const settle = (designId: number, state: FieldState) => {
      if (!controller.signal.aborted) setBatch((previous) => previous && { ...previous, fields: { ...previous.fields, [designId]: state } });
    };
    setBatch({ designs, source, fields: {} });
    for (const { id } of designs) {
      const loading: Promise<FieldState> = source === 'surface'
        ? fetchSurface(id, config.surface_channels.length, controller.signal).then((surface) => ({ surface }))
        : fetchField(id, source, size, controller.signal).then((field) => ({ field }));
      loading.then((state) => settle(id, state), (error) => settle(id, { error: errorMessage(error) }));
    }
    return () => controller.abort();
  }, [designs, source, config]);

  // Until the effect above resets it, `batch` still holds the previous designs or source
  const fields = batch?.designs === designs && batch?.source === source ? batch.fields : NO_FIELDS;
  return useMemo(() => {
    const loaded = Object.values(fields).filter((state) => !('error' in state));
    const failed = Object.keys(fields).length - loaded.length;
    return { fields, colorRange: colorRangeOf(loaded, display.surfaceChannel), failed, pending: (designs?.length ?? 0) - loaded.length - failed };
  }, [fields, designs, display.surfaceChannel]);
}

function DesignCard({ design, state, config, colorRange, display, highlight }: {
  design: Design; state: FieldState | undefined; config: Config; colorRange: Range | undefined; display: FieldDisplay; highlight: string[];
}) {
  const { mode, slice } = display;
  const unitOf = (name: string) => config.parameters.find((parameter) => parameter.name === name)?.unit;
  const placeholder = <div className="mx-4 h-72 animate-pulse rounded-xl bg-paper" />;
  return <Panel className="overflow-hidden">
    <div className="px-4 pt-4 pb-2">
      <h3 className="text-sm font-medium">Design {design.id}</h3>
      {highlight.length > 0 && <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-ink-muted tabular-nums">
        {highlight.map((name) => <span key={name}>{name} <span className="text-ink">{formatNumber(design.params[name])}</span>{unitOf(name) && ` ${unitOf(name)}`}</span>)}
      </p>}
    </div>
    {state && colorRange && 'field' in state
      ? mode === '2d'
        ? <Heatmap field={state.field} config={config} range={colorRange} slice={slice} />
        : <Suspense fallback={placeholder}><Volume field={state.field} config={config} range={colorRange} slice={slice} /></Suspense>
      : state && colorRange && 'surface' in state
        ? <Suspense fallback={placeholder}><Surface surface={state.surface} channel={display.surfaceChannel} config={config} range={colorRange} /></Suspense>
        : state && 'error' in state
          ? <p className="mx-4 rounded-xl bg-danger-soft p-4 text-xs break-words text-danger" role="alert">{state.error}</p>
          : <div className="mx-4 grid h-40 animate-pulse place-items-center rounded-xl bg-paper text-xs text-ink-muted" role="status">Loading field · uncached designs can take minutes</div>}
  </Panel>;
}

/** Two cards per row, `highlight` parameters show under each card title. */
export default function Gallery({ designs, batch, config, display, label = 'Sampled simulations', highlight = [] }: {
  designs: Design[]; batch: FieldBatch; config: Config; display: FieldDisplay; label?: string; highlight?: string[];
}) {
  return <section aria-label={label} aria-busy={batch.pending > 0} className={`grid gap-5 ${designs.length > 1 ? 'grid-cols-2 max-lg:grid-cols-1' : ''}`}>
    {designs.map((design) => <DesignCard key={design.id} design={design} state={batch.fields[design.id]} config={config} colorRange={batch.colorRange} display={display} highlight={highlight} />)}
  </section>;
}
