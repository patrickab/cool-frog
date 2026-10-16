import { Fragment, useMemo, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { fetchField, fetchReconstructionMetrics, fluidValues, voxelCount } from './api';
import type { Config, Field } from './api';
import Heatmap from './Heatmap';
import { clipping, colorScaleOf } from './lib/colormap';
import type { ColorScale } from './lib/colormap';
import { formatNumber } from './lib/format';
import { Colorbar, HEADING, Layout, Panel, Pending, Segmented, Slider } from './lib/ui';
import { useAsync } from './lib/useAsync';

/** Signed error (reconstruction − simulation), its magnitude, and |error| max and 99th percentile over fluid voxels. */
function compareFields(simulation: Field, reconstruction: Field): { signed: Field; absolute: Field; maximum: number; percentile99: number } {
  const signed = new Float32Array(simulation.values.length);
  const absolute = new Float32Array(simulation.values.length);
  const fluidErrors: number[] = [];
  let [minimum, maximum] = [0, 0];
  for (let index = 0; index < signed.length; index++) {
    if (!simulation.mask[index]) continue;
    signed[index] = reconstruction.values[index] - simulation.values[index];
    absolute[index] = Math.abs(signed[index]);
    [minimum, maximum] = [Math.min(minimum, signed[index]), Math.max(maximum, signed[index])];
    fluidErrors.push(absolute[index]);
  }
  const sorted = Float32Array.from(fluidErrors).sort();
  const largest = sorted[sorted.length - 1];
  return {
    signed: { values: signed, mask: simulation.mask, min: minimum, max: maximum },
    absolute: { values: absolute, mask: simulation.mask, min: 0, max: largest },
    maximum: largest,
    percentile99: sorted[Math.floor(0.99 * (sorted.length - 1))],
  };
}

/** Range of the simulation and reconstruction colors: all 16 layers, or only the shown slice. */
const FIELD_SCALES = [{ value: 'volume', label: 'Volume' }, { value: 'slice', label: 'Slice' }] as const;
type FieldScale = (typeof FIELD_SCALES)[number]['value'];
/** Signed error on a diverging map shows over- vs under-prediction, absolute error on a single-hue map shows magnitude only. */
const ERROR_KINDS = [{ value: 'signed', label: 'Signed' }, { value: 'absolute', label: 'Absolute' }] as const;
type ErrorKind = (typeof ERROR_KINDS)[number]['value'];
/** End of each channel's error color scale: max |error| over the whole 3D tensor, its 99th percentile, or max |error| in the shown slice. */
const ERROR_SCALES = [
  { value: 'volume', label: 'Volume max' },
  { value: 'percentile', label: 'Volume p99' },
  { value: 'slice', label: 'Slice max' },
] as const;
type ErrorScale = (typeof ERROR_SCALES)[number]['value'];

function ControlGroup({ label, children }: { label: string; children: ReactNode }) {
  return <div className="flex items-center gap-2 text-xs"><span className="text-ink-muted">{label}</span>{children}</div>;
}

/**
 * Simulation, zero-shot Tadpole-B reconstruction and error of every channel at a chosen height, one row per channel.
 *
 * - Simulation and reconstruction share one colorbar per row, the error has its own
 * - Colorbars mark with ≤/≥ when values in the shown slice are clipped to the end colors
 */
export function Reconstruction({ config, designId }: { config: Config; designId: number }) {
  const [slice, setSlice] = useState(Math.floor(config.dimensions[2] / 2));
  const [fieldScale, setFieldScale] = useState<FieldScale>('volume');
  const [errorKind, setErrorKind] = useState<ErrorKind>('signed');
  const [errorScale, setErrorScale] = useState<ErrorScale>('percentile');
  const loaded = useAsync((signal) => Promise.all([
    Promise.all(config.channels.map((_, channel) => Promise.all([
      fetchField(designId, channel, voxelCount(config), signal),
      fetchField(designId, channel, voxelCount(config), signal, 'reconstruction'),
    ]))),
    fetchReconstructionMetrics(designId, signal),
  ]), [designId, config]);
  const comparisons = useMemo(() => loaded.status === 'done' ? loaded.data[0].map((pair) => compareFields(...pair)) : undefined, [loaded]);
  const volumeScales = useMemo(() => loaded.status === 'done'
    ? loaded.data[0].map(([simulation], channel) => colorScaleOf(config.channels[channel].scale, fluidValues(simulation, config))!)
    : undefined, [loaded, config]);
  const rows = useMemo(() => {
    if (loaded.status !== 'done' || !comparisons || !volumeScales) return undefined;
    return loaded.data[0].map(([simulation, reconstruction], channel) => {
      const shown = [fluidValues(simulation, config, slice), fluidValues(reconstruction, config, slice)];
      const scale = fieldScale === 'volume' ? volumeScales[channel] : colorScaleOf(config.channels[channel].scale, shown[0])!;
      const comparison = comparisons[channel];
      const error = errorKind === 'signed' ? comparison.signed : comparison.absolute;
      const shownError = fluidValues(error, config, slice);
      const bound = {
        volume: comparison.maximum,
        percentile: comparison.percentile99,
        slice: fluidValues(comparison.absolute, config, slice).reduce((maximum, value) => Math.max(maximum, value), 0),
      }[errorScale] || 1;
      const errorScaleOf: ColorScale = errorKind === 'signed'
        ? { colormap: 'diverging', range: [-bound, bound], log: false }
        : { colormap: 'magnitude', range: [0, bound], log: false };
      return {
        simulation, reconstruction, scale, error, errorScale: errorScaleOf,
        simulationClipped: clipping(shown[0], scale),
        reconstructionClipped: clipping(shown[1], scale),
        errorClipped: clipping(shownError, errorScaleOf),
      };
    });
  }, [loaded, comparisons, volumeScales, config, slice, fieldScale, errorKind, errorScale]);
  const height = formatNumber(slice * config.spacing[2] * 1000);
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
      <ControlGroup label="Height">
        <div className="w-32"><Slider label="Z slice" min={0} max={config.dimensions[2] - 1} values={[slice]} onChange={([next]) => setSlice(next)} /></div>
        <span className="min-w-16 whitespace-nowrap rounded-lg bg-surface-elevated px-2 py-0.5 text-right tabular-nums shadow-[var(--shadow-sm)]">{height} mm</span>
      </ControlGroup>
      <ControlGroup label="Field scale"><Segmented label="Field color range" value={fieldScale} onChange={setFieldScale} options={[...FIELD_SCALES]} /></ControlGroup>
      <ControlGroup label="Error"><Segmented label="Error kind" value={errorKind} onChange={setErrorKind} options={[...ERROR_KINDS]} /></ControlGroup>
      <ControlGroup label="Error scale"><Segmented label="Error color scale" value={errorScale} onChange={setErrorScale} options={[...ERROR_SCALES]} /></ControlGroup>
    </div>
    {loaded.status !== 'done' || !rows
      ? <Pending state={loaded} message={`Running Tadpole-B on design ${designId} · uncached designs take about a minute on the CPU`} />
      : <Panel className="grid gap-y-2 py-4 lg:grid-cols-[11rem_repeat(3,minmax(0,1fr))]">
        <p className="px-4 text-xs text-ink-muted tabular-nums max-lg:hidden">z = {height} mm</p>
        {[`Simulation, design ${designId}`, 'Tadpole-B reconstruction', errorKind === 'signed' ? 'Error, reconstruction − simulation' : 'Absolute error'].map((title) =>
          <p key={title} className="px-4 text-xs font-medium max-lg:hidden">{title}</p>)}
        {rows.map((row, channel) => {
          const { name, unit } = config.channels[channel];
          const metrics = loaded.data[1];
          const last = channel === rows.length - 1;
          return <Fragment key={name}>
            <dl className="row-span-2 grid h-fit grid-cols-[auto_auto] gap-x-2 px-4 pt-4 text-xs whitespace-nowrap tabular-nums lg:pt-6">
              <dt className="text-ink-muted">Metric:</dt><dd><span className="font-medium">{name}</span> <span className="text-ink-muted">{unit}</span></dd>
              <dt className="mt-2 text-ink-muted">NRMSE:</dt><dd className="mt-2 font-medium">{metrics.nrmse[channel].toFixed(1)} %</dd>
              <dt className="text-ink-muted">Spectral loss:</dt><dd className="font-medium">{metrics.spectral_loss[channel].toFixed(3)}</dd>
            </dl>
            <div className="flex justify-start px-4 pt-4"><Colorbar scale={row.scale} clipped={row.simulationClipped} unit={unit} /></div>
            <div className="flex justify-start px-4 pt-4"><Colorbar scale={row.scale} clipped={row.reconstructionClipped} unit={unit} /></div>
            <div className="flex justify-start px-4 pt-4"><Colorbar scale={row.errorScale} clipped={row.errorClipped} unit={unit} /></div>
            <Heatmap field={row.simulation} config={config} scale={row.scale} slice={slice} axis={last} />
            <Heatmap field={row.reconstruction} config={config} scale={row.scale} slice={slice} axis={last} />
            <Heatmap field={row.error} config={config} scale={row.errorScale} slice={slice} axis={last} />
          </Fragment>;
        })}
      </Panel>}
  </div>;
}

/** Cool Autoencoder: any design's zero-shot reconstruction and error, picked by id in the sidebar. */
export default function AutoencoderView({ config, nav, initialDesign }: { config: Config; nav: ReactNode; initialDesign: number }) {
  const [designId, setDesignId] = useState(initialDesign);
  const [draft, setDraft] = useState(String(initialDesign));
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const parsed = Number(draft);
    if (Number.isInteger(parsed) && parsed >= 0) setDesignId(parsed);
  };
  const sidebar = <Panel className="space-y-3 p-4">
    <form onSubmit={submit} className="flex items-center justify-between gap-3">
      <label htmlFor="autoencoder-design" className={HEADING}>Design</label>
      <input id="autoencoder-design" inputMode="numeric" value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={submit}
        className="w-24 rounded-lg bg-surface-elevated px-2.5 py-1 text-right text-xs tabular-nums shadow-[var(--shadow-sm)] outline-none focus-visible:ring-2 focus-visible:ring-divider" />
    </form>
    <p className="text-xs text-ink-muted">Encoded and decoded by Tadpole-B, never trained on CoolData.</p>
  </Panel>;
  return <Layout nav={nav} sidebar={sidebar}>
    <Reconstruction config={config} designId={designId} />
  </Layout>;
}
