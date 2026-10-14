import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { fetchField } from './api';
import type { Config, Design, Field } from './api';
import Heatmap from './Heatmap';
import { errorMessage, formatNumber } from './lib/format';
import type { Range } from './lib/range';
import { Disclosure, Panel } from './lib/ui';

const Volume = lazy(() => import('./Volume'));

export type RenderMode = '2d' | '3d';
type FieldState = { field: Field } | { error: string };
/** Per-design load state of one channel, `colorRange` shared across loaded fields. */
export type FieldBatch = { fields: Record<number, FieldState>; colorRange?: Range; pending: number; failed: number };
const NO_FIELDS: Record<number, FieldState> = {};

/** Load `channel` of every design progressively, aborting and hiding the previous batch on change. */
export function useFieldBatch(designs: Design[] | undefined, channel: number, dimensions: Config['dimensions']): FieldBatch {
  const [batch, setBatch] = useState<{ designs: Design[]; channel: number; fields: Record<number, FieldState> }>();
  useEffect(() => {
    if (!designs) return;
    const controller = new AbortController();
    const size = dimensions[0] * dimensions[1] * dimensions[2];
    const settle = (designId: number, state: FieldState) => {
      if (!controller.signal.aborted) setBatch((previous) => previous && { ...previous, fields: { ...previous.fields, [designId]: state } });
    };
    setBatch({ designs, channel, fields: {} });
    for (const { id } of designs) {
      fetchField(id, channel, size, controller.signal).then((field) => settle(id, { field }), (error) => settle(id, { error: errorMessage(error) }));
    }
    return () => controller.abort();
  }, [designs, channel, dimensions]);

  // Until the effect above resets it, `batch` still holds the previous designs or channel
  const fields = batch?.designs === designs && batch?.channel === channel ? batch.fields : NO_FIELDS;
  return useMemo(() => {
    const loaded = Object.values(fields).flatMap((state) => 'field' in state ? [state.field] : []);
    const failed = Object.keys(fields).length - loaded.length;
    const colorRange: Range | undefined = loaded.length ? [Math.min(...loaded.map((field) => field.min)), Math.max(...loaded.map((field) => field.max))] : undefined;
    return { fields, colorRange, failed, pending: (designs?.length ?? 0) - loaded.length - failed };
  }, [fields, designs]);
}

function DesignCard({ design, state, config, colorRange, mode, slice }: {
  design: Design; state: FieldState | undefined; config: Config; colorRange: Range | undefined; mode: RenderMode; slice: number;
}) {
  const field = state && 'field' in state ? state.field : undefined;
  return <Panel className="overflow-hidden">
    <h3 className="px-4 pt-4 pb-2 text-sm font-medium">Design {design.id}</h3>
    {field && colorRange
      ? mode === '2d'
        ? <Heatmap field={field} config={config} range={colorRange} slice={slice} />
        : <Suspense fallback={<div className="mx-4 h-72 animate-pulse rounded-xl bg-paper" />}><Volume field={field} config={config} range={colorRange} slice={slice} /></Suspense>
      : state && 'error' in state
        ? <p className="mx-4 rounded-xl bg-danger-soft p-4 text-xs break-words text-danger" role="alert">{state.error}</p>
        : <div className="mx-4 grid h-40 animate-pulse place-items-center rounded-xl bg-paper text-xs text-ink-muted" role="status">Loading field · uncached designs can take minutes</div>}
    <Disclosure label="Parameters" className="px-4 pt-2 pb-4">
      <dl className="mt-3 grid grid-cols-3 gap-x-5 gap-y-1.5">{Object.entries(design.params).map(([name, value]) => <div key={name} className="flex justify-between"><dt className="text-ink-muted">{name}</dt><dd className="tabular-nums">{formatNumber(value)}</dd></div>)}</dl>
    </Disclosure>
  </Panel>;
}

export default function Gallery({ designs, batch, config, mode, slice }: { designs: Design[]; batch: FieldBatch; config: Config; mode: RenderMode; slice: number }) {
  return <section aria-label="Sampled simulations" aria-busy={batch.pending > 0} className="grid grid-cols-2 gap-5 max-lg:grid-cols-1">
    {designs.map((design) => <DesignCard key={design.id} design={design} state={batch.fields[design.id]} config={config} colorRange={batch.colorRange} mode={mode} slice={slice} />)}
  </section>;
}
