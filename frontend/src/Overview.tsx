import { Fragment, lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { fetchDesigns, fetchField, fetchReconstructionMetrics, fluidValues } from './api';
import type { Config, Dataset, Design, Field } from './api';
import type { View } from './App';
import Gallery, { defaultDisplay, FieldControls, useFieldBatch } from './Gallery';
import type { FieldDisplay } from './Gallery';
import Heatmap from './Heatmap';
import { AXIS_COLORS, clipping, colorScaleOf, oklchCss } from './lib/colormap';
import type { ColorScale } from './lib/colormap';
import { formatNumber, formatTick } from './lib/format';
import { Colorbar, Disclosure, HEADING, Layout, Panel, Segmented, Slider } from './lib/ui';
import { useAsync } from './lib/useAsync';
import type { AsyncState } from './lib/useAsync';

// three.js loads only once a 3D scene renders
const TensorExplorer = lazy(() => import('./TensorScenes').then((module) => ({ default: module.TensorExplorer })));
const ConvolutionScene = lazy(() => import('./TensorScenes').then((module) => ({ default: module.ConvolutionScene })));
const AttentionScene = lazy(() => import('./TensorScenes').then((module) => ({ default: module.AttentionScene })));
const SizeTable = lazy(() => import('./TensorScenes').then((module) => ({ default: module.SizeTable })));
const CoolDataLatent = lazy(() => import('./TensorScenes').then((module) => ({ default: module.CoolDataLatent })));

/** Showcase designs, picked from the metadata as pairs that differ mostly in one parameter. */
const SHOWCASES = {
  hero: [228603],
  velocity: [233558, 237503],
  temperature: [38623, 228576],
  wall: [190038],
  cylinders: [182262, 21432],
  heights: [125002],
  turbulence: [39106, 182004],
};
const SHOWCASE_IDS = [...new Set(Object.values(SHOWCASES).flat())];
/** Design every Tadpole step shows, its zero-shot errors in the prose were measured offline. */
const TADPOLE_DESIGN = SHOWCASES.hero[0];

/** Topics, one expanded at a time, step ids are unique across topics, numbering restarts per topic. */
const TOPICS = {
  cooldata: {
    label: 'CoolData',
    steps: [
      { id: 'setup', title: 'One sample: air cooling a circuit board' },
      { id: 'fields', title: 'What a sample stores' },
      { id: 'parameters', title: 'The 35 parameters' },
      { id: 'velocity', title: 'Inlet velocity V' },
      { id: 'temperature', title: 'Body temperatures T1–T6' },
      { id: 'footprint', title: 'Footprint: blocking the channel' },
      { id: 'cylinders', title: 'Cylinders and their radius' },
      { id: 'turbulence', title: 'Turbulence k and ε' },
      { id: 'height', title: 'Height zs: a caveat' },
      { id: 'dataset', title: 'The dataset as a whole' },
      { id: 'next', title: 'Keep exploring' },
    ],
  },
  architecture: {
    label: 'Architecture',
    steps: [
      { id: 'arch-crops', title: 'Fields become a batch of crops' },
      { id: 'arch-convolution', title: 'Convolutions: embed and halve' },
      { id: 'arch-attention', title: 'Window attention' },
      { id: 'arch-network', title: 'Encoder, KL head, decoder' },
      { id: 'arch-layout', title: 'Latent layout for downstream models' },
      { id: 'arch-rationale', title: 'Why it is built this way' },
      { id: 'arch-whole', title: 'After training: whole domains' },
    ],
  },
  overview: {
    label: 'Overview',
    steps: [
      { id: 'tadpole', title: 'A foundation model for 3D PDEs' },
      { id: 'pretraining', title: 'Pre-training on synthetic PDEs' },
      { id: 'scalar-fields', title: 'One scalar field at a time' },
      { id: 'architecture', title: 'Inside the autoencoder' },
      { id: 'objective', title: 'What training optimizes' },
      { id: 'downstream', title: 'Downstream tasks' },
    ],
  },
  zeroshot: {
    label: 'Zero-shot autoencoding',
    steps: [
      { id: 'prepare', title: 'Preparing CoolData for Tadpole' },
      { id: 'latent', title: 'The latent of a CoolData design' },
      { id: 'zero-shot', title: 'Zero-shot on CoolData' },
    ],
  },
} as const;
type Topic = keyof typeof TOPICS;
/** Sidebar entries, a group with several topics nests them under its label. */
const SIDEBAR_GROUPS: { label: string; topics: Topic[] }[] = [
  { label: 'CoolData', topics: ['cooldata'] },
  { label: 'Tadpole', topics: ['overview', 'architecture', 'zeroshot'] },
];
type StepId = (typeof TOPICS)[Topic]['steps'][number]['id'];
const ALL_STEPS = Object.values(TOPICS).flatMap((topic): readonly { id: StepId; title: string }[] => topic.steps);

/** Parameter groups of the metadata, `columns` resolve to live bounds from `Config`. */
const PARAMETER_GROUPS = [
  { columns: ['V'], meaning: 'Inlet air velocity, uniform over the inlet plane' },
  { columns: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'], meaning: 'Surface temperature of each body in °C. Inlet air is 20 °C (293 K)' },
  { columns: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'], meaning: 'Body center along the flow in m' },
  // Live bounds include the absent marker 1.0, active bodies span 0–0.099 m
  { columns: ['y1', 'y2', 'y3', 'y4', 'y5', 'y6'], meaning: 'Body center across the channel in m. y = 1.0 marks an absent body (bodies 3–6 only)', range: '0 – 0.099' },
  { columns: ['xs1', 'xs2', 'xs3', 'xs4'], meaning: 'Cuboid length along the flow in m (bodies 1–4)' },
  { columns: ['ys1', 'ys2', 'ys3', 'ys4'], meaning: 'Cuboid width across the channel in m (bodies 1–4)' },
  { columns: ['r5', 'r6'], meaning: 'Cylinder radius in m (bodies 5–6)' },
  { columns: ['zs1', 'zs2', 'zs3', 'zs4', 'zs5', 'zs6'], meaning: 'Nominal body height in m, see step 9' },
];

/** Tadpole model sizes, paper Table 12, Section 5.1 and Appendix C.4. */
const TADPOLE_SIZES = [
  { size: 'S', parameters: '8.8 M', compression: '16×', error: '5.48', cost: '372 h L40S' },
  { size: 'B', parameters: '38.1 M', compression: '8×', error: '2.83', cost: '620 h A100' },
  { size: 'L', parameters: '152.1 M', compression: '4×', error: '2.06', cost: '2,300 h A100' },
];

/** Tadpole-B encoder stages, `divisor` shrinks every grid axis, shapes traced on the 400 × 80 × 16 grid. */
const ENCODER_STAGES = [
  { stage: 'Input', features: 1, divisor: 1 },
  { stage: 'Convolution', features: 64, divisor: 1 },
  { stage: 'Convolution, ↓2', features: 128, divisor: 2 },
  { stage: 'Convolution, ↓2', features: 128, divisor: 4 },
  { stage: 'Attention, convolution ↓2', features: 256, divisor: 8 },
  { stage: 'Attention, convolution ↓2, attention', features: 512, divisor: 16 },
  { stage: 'Mean and log-variance', features: 1024, divisor: 16 },
  { stage: 'Sampled latent z', features: 512, divisor: 16 },
];
const LATENT_DIVISOR = 16;

/** Pre-training data path, paper Section 3.2 and Appendix B.2. */
const PRETRAINING_PIPELINE = [
  { title: 'GPU solver', detail: '7 PDEs at 64³ to 384³' },
  { title: 'Pre-crop', detail: '96³, cuts transfer' },
  { title: 'FIFO queue', detail: 'Asynchronous to trainers' },
  { title: 'MFU cache', detail: 'Batches drawn here' },
  { title: 'Random crop', detail: '64³, one channel' },
  { title: 'Autoencoder', detail: 'Plus discriminator' },
];

function channelIndex(config: Config, name: string): number {
  return Math.max(0, config.channels.findIndex((channel) => channel.name === name));
}

function voxelCount(config: Config): number {
  return config.dimensions[0] * config.dimensions[1] * config.dimensions[2];
}

/** Id of the step currently crossing the upper third of the viewport, hidden topics never intersect. */
function useActiveStep(): [StepId, (step: StepId) => void] {
  const [active, setActive] = useState<StepId>(ALL_STEPS[0].id);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.find((entry) => entry.isIntersecting);
      if (visible) setActive(visible.target.id as StepId);
    }, { rootMargin: '-30% 0px -65% 0px' });
    for (const { id } of ALL_STEPS) {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    }
    return () => observer.disconnect();
  }, []);
  return [active, setActive];
}

function Step({ id, children }: { id: StepId; children: ReactNode }) {
  const steps: readonly { id: StepId; title: string }[] = Object.values(TOPICS).find((topic) => topic.steps.some((step) => step.id === id))!.steps;
  const index = steps.findIndex((step) => step.id === id);
  return <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-6 space-y-4">
    <header>
      <p className={HEADING}>Step {index + 1}</p>
      <h2 id={`${id}-title`} className="mt-1 text-xl font-semibold tracking-tight">{steps[index].title}</h2>
    </header>
    {children}
  </section>;
}

/** Explanatory text, readable line length, `strong` for terms. */
function Prose({ children }: { children: ReactNode }) {
  return <div className="max-w-3xl space-y-3 text-sm leading-relaxed text-ink-muted [&_strong]:font-medium [&_strong]:text-ink">{children}</div>;
}

/** Designs with their own field controls, highlighted parameters under each card title. */
function Showcase({ designs, config, initial, highlight, onOpen }: {
  designs: Design[]; config: Config; initial: Partial<FieldDisplay>; highlight: string[]; onOpen: (designs: Design[]) => void;
}) {
  const [display, setDisplay] = useState(() => defaultDisplay(config, initial));
  const batch = useFieldBatch(designs, display, config);
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-3">
      <FieldControls config={config} display={display} onChange={setDisplay} color={batch.color} />
      <button onClick={() => onOpen(designs)} className="ml-auto text-xs text-ink-muted transition hover:text-ink">Open in Cool Flowfields →</button>
    </div>
    <Gallery designs={designs} batch={batch} config={config} display={display} label={`Designs ${designs.map(({ id }) => id).join(', ')}`} highlight={highlight} />
  </div>;
}

function ParameterTable({ config }: { config: Config }) {
  const bounds = Object.fromEntries(config.parameters.map((parameter) => [parameter.name, parameter]));
  return <Panel className="overflow-x-auto">
    <table className="w-full text-left text-xs">
      <thead className="text-ink-muted shadow-[inset_0_-1px_0_var(--divider-soft)]">
        <tr><th scope="col" className="px-4 py-2 font-medium">Columns</th><th scope="col" className="px-2 py-2 font-medium">Meaning</th><th scope="col" className="px-4 py-2 text-right font-medium">Range in data</th></tr>
      </thead>
      <tbody>
        {PARAMETER_GROUPS.map(({ columns, meaning, range }) => {
          const minimum = Math.min(...columns.map((name) => bounds[name].min));
          const maximum = Math.max(...columns.map((name) => bounds[name].max));
          const unit = bounds[columns[0]].unit;
          return <tr key={columns[0]} className="align-top">
            <td className="px-4 py-2 font-medium whitespace-nowrap">{columns.length > 1 ? `${columns[0]}–${columns.at(-1)}` : columns[0]}</td>
            <td className="px-2 py-2 text-ink-muted">{meaning}</td>
            <td className="px-4 py-2 text-right whitespace-nowrap tabular-nums">{range ?? `${formatNumber(+minimum.toFixed(3))} – ${formatNumber(+maximum.toFixed(3))}`}{unit && ` ${unit}`}</td>
          </tr>;
        })}
      </tbody>
    </table>
  </Panel>;
}

/** Labeled horizontal bars, widths relative to the largest count. */
function CountBars({ label, counts, total }: { label: (key: string) => string; counts: Record<string, number>; total: number }) {
  const peak = Math.max(...Object.values(counts));
  return <Panel className="max-w-3xl space-y-2 p-4 text-xs tabular-nums">
    {Object.entries(counts).map(([key, count]) => <div key={key} className="grid grid-cols-[6rem_1fr_7rem] items-center gap-3">
      <span>{label(key)}</span>
      <div className="h-2 rounded-full bg-hover-strong"><div className="h-full rounded-full bg-ink-subtle" style={{ width: `${(count / peak) * 100}%` }} /></div>
      <span className="text-right text-ink-muted">{formatNumber(count)} · {((count / total) * 100).toFixed(1)}%</span>
    </div>)}
  </Panel>;
}

/** Stages left to right with arrows, wrapping on narrow screens. */
function Pipeline({ label, stages }: { label: string; stages: readonly { title: string; detail: string }[] }) {
  return <Panel className="w-fit max-w-full p-4">
    <ol aria-label={label} className="flex flex-wrap items-center gap-2 text-xs">
      {stages.map(({ title, detail }, index) => <li key={title} className="flex items-center gap-2">
        {index > 0 && <span aria-hidden className="text-ink-subtle">→</span>}
        <div className="rounded-xl bg-paper px-3 py-2">
          <p className="font-medium">{title}</p>
          <p className="text-ink-muted">{detail}</p>
        </div>
      </li>)}
    </ol>
  </Panel>;
}

/** Symbolic field geometry and feature slots, not measured values or model activations. */
function ReconstructionDiagram() {
  return <Panel className="max-w-3xl space-y-4 p-4">
    <figure aria-label="A scalar field passes through an encoder, a learned feature bottleneck and a decoder to reconstruct the same field at the same time. There is no encoder-to-decoder bypass.">
      <div className="grid items-center gap-3 text-center text-xs sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {[0, 1, 2].map((stage) => <Fragment key={stage}>
          {stage > 0 && <div className="text-ink-muted"><span className="block text-xl sm:-rotate-90" aria-hidden>↓</span><span>{stage === 1 ? 'Encoder' : 'Decoder'}</span></div>}
          <div className="space-y-2">
            <p className="font-medium">{['Input field · time t', 'Learned bottleneck', 'Reconstruction · time t'][stage]}</p>
            <svg viewBox="0 0 160 112" aria-hidden="true" className="mx-auto h-28 w-40 max-w-full" fill="none">
              {stage === 1 ? <g stroke={oklchCss(AXIS_COLORS.feature)} strokeWidth="1.5">
                {[0, 1, 2].map((layer) => <g key={layer} transform={`translate(${48 + layer * 9} ${36 - layer * 9})`}>
                  <rect width="48" height="48" fill="var(--surface)" />
                  <path d="M16 0v48M32 0v48M0 16h48M0 32h48" />
                </g>)}
              </g> : <>
                <g stroke={oklchCss(AXIS_COLORS.spatial)} strokeWidth="1">
                  <path d="M20 30h96v66H20zM20 30l24-18h96v66l-24 18M116 30l24-18M44 12v18M116 78h24" />
                  <path d="M44 30v66M68 30v66M92 30v66M20 52h96M20 74h96" strokeOpacity="0.3" />
                </g>
                <g stroke={oklchCss(AXIS_COLORS.channel)} strokeWidth="2">
                  <path d={stage === 0 ? 'M24 79C45 79 48 41 72 41S95 73 112 57' : 'M24 78C46 78 49 43 72 43S95 71 112 58'} />
                  <path d="M24 88C47 88 54 60 73 60S97 87 112 76" />
                </g>
              </>}
            </svg>
            <p className="text-ink-muted">{['One physical variable on a 3D grid', 'Fewer positions, many feature channels', 'Original grid, approximate values'][stage]}</p>
          </div>
        </Fragment>)}
      </div>
      <figcaption className="mt-4 text-xs text-ink-muted"><strong className="font-medium text-ink">No skip connections:</strong> everything the decoder needs passes through the latent.</figcaption>
    </figure>
  </Panel>;
}

function TadpoleSizeTable() {
  return <Panel className="max-w-3xl overflow-x-auto">
    <table className="w-full text-left text-xs tabular-nums">
      <thead className="text-ink-muted shadow-[inset_0_-1px_0_var(--divider-soft)]">
        <tr>
          <th scope="col" className="px-4 py-2 font-medium">Size</th>
          <th scope="col" className="px-2 py-2 text-right font-medium">Parameters</th>
          <th scope="col" className="px-2 py-2 text-right font-medium">Compression</th>
          <th scope="col" className="px-2 py-2 text-right font-medium">Pre-training RMSE ×10⁻³</th>
          <th scope="col" className="px-4 py-2 text-right font-medium">Pre-training GPU hours</th>
        </tr>
      </thead>
      <tbody>
        {TADPOLE_SIZES.map(({ size, parameters, compression, error, cost }) => <tr key={size}>
          <td className="px-4 py-2 font-medium">{size}</td>
          <td className="px-2 py-2 text-right">{parameters}</td>
          <td className="px-2 py-2 text-right">{compression}</td>
          <td className="px-2 py-2 text-right">{error}</td>
          <td className="px-4 py-2 text-right">{cost}</td>
        </tr>)}
      </tbody>
    </table>
  </Panel>;
}

/** One 64³ crop through the Tadpole-B encoder, each footprint drawn to scale against the input. */
function EncoderStages() {
  const footprint = 160;
  return <Panel className="max-w-3xl overflow-x-auto">
    <table className="w-full text-left text-xs tabular-nums">
      <caption className="sr-only">Tensor shape of one 64³ crop through the Tadpole-B encoder</caption>
      <thead className="text-ink-muted shadow-[inset_0_-1px_0_var(--divider-soft)]">
        <tr>
          <th scope="col" className="px-4 py-2 font-medium">Stage</th>
          <th scope="col" className="px-2 py-2 font-medium">Grid to scale</th>
          <th scope="col" className="px-2 py-2 text-right font-medium">Features × grid</th>
          <th scope="col" className="px-4 py-2 text-right font-medium">Values</th>
        </tr>
      </thead>
      <tbody>
        {ENCODER_STAGES.map(({ stage, features, divisor }, index) => {
          const edge = 64 / divisor;
          return <tr key={index}>
            <td className="px-4 py-2 font-medium whitespace-nowrap">{stage}</td>
            <td className="px-2 py-2" style={{ width: footprint + 16 }}>
              <div aria-hidden className="rounded-[2px] bg-ink-subtle" style={{ width: footprint / divisor, height: footprint / divisor }} />
            </td>
            <td className="px-2 py-2 text-right whitespace-nowrap">{features} × {edge}³</td>
            <td className="px-4 py-2 text-right text-ink-muted">{formatNumber(features * edge ** 3)}</td>
          </tr>;
        })}
      </tbody>
    </table>
  </Panel>;
}

/** Every channel of one design on its own linear color scale, as Tadpole receives them after per-channel min-max scaling. */
function ScalarFields({ config, fields }: { config: Config; fields: Field[] }) {
  const slice = Math.floor(config.dimensions[2] / 2);
  return <Panel className="grid gap-x-0 gap-y-3 py-4 lg:grid-cols-2">
    {fields.map((field, index) => <div key={index}>
      <p className="px-4 pb-1 text-xs">
        <span className="font-medium">{config.channels[index].name}</span> <span className="text-ink-muted">{config.channels[index].unit}</span>
        <span className="ml-2 text-ink-muted tabular-nums">{formatTick(field.min)} → −1, {formatTick(field.max)} → 1</span>
      </p>
      <Heatmap field={field} config={config} scale={{ colormap: 'sequential', range: [field.min, field.max], log: false }} slice={slice} axis={false} />
    </div>)}
  </Panel>;
}

/** Lazily loaded 3D scene with a placeholder in its place while three.js loads. */
function Lazy3D({ children }: { children: ReactNode }) {
  return <Suspense fallback={<Panel className="grid h-40 animate-pulse place-items-center text-xs text-ink-muted"><span role="status">Loading 3D view…</span></Panel>}>{children}</Suspense>;
}

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
function ZeroShot({ config, initialDesign }: { config: Config; initialDesign: number }) {
  const [designId, setDesignId] = useState(initialDesign);
  const [draft, setDraft] = useState(String(initialDesign));
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
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const parsed = Number(draft);
    if (Number.isInteger(parsed) && parsed >= 0) setDesignId(parsed);
  };
  const height = formatNumber(slice * config.spacing[2] * 1000);
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
      <form onSubmit={submit} className="flex items-center gap-2 text-xs">
        <label htmlFor="zero-shot-design" className="text-ink-muted">Design</label>
        <input id="zero-shot-design" inputMode="numeric" value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={submit}
          className="w-24 rounded-lg bg-surface-elevated px-2.5 py-1 text-right tabular-nums shadow-[var(--shadow-sm)] outline-none focus-visible:ring-2 focus-visible:ring-divider" />
      </form>
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

/** Inline error, or a pulsing placeholder with one factual line while `state` loads. */
function Pending({ state, message }: { state: AsyncState<unknown>; message: string }) {
  return state.status === 'error'
    ? <Panel className="p-4 text-sm text-danger"><p role="alert">{state.error}</p></Panel>
    : <Panel className="grid h-40 animate-pulse place-items-center text-xs text-ink-muted"><span role="status">{message}</span></Panel>;
}

/** Guided tour in two topics: the Tadpole model and the CoolData dataset, one shown at a time. */
export default function Overview({ config, dataset, nav, onView, onOpen }: {
  config: Config; dataset: Dataset; nav: ReactNode; onView: (view: View) => void; onOpen: (designs: Design[]) => void;
}) {
  const [topic, setTopic] = useState<Topic>('cooldata');
  const [active, setActive] = useActiveStep();
  const loaded = useAsync((signal) => fetchDesigns(SHOWCASE_IDS, signal), []);
  const showcases = useMemo(() => {
    if (loaded.status !== 'done') return undefined;
    const byId = Object.fromEntries(loaded.data.map((design) => [design.id, design]));
    return Object.fromEntries(Object.entries(SHOWCASES).map(([name, ids]) => [name, ids.map((id) => byId[id])])) as Record<keyof typeof SHOWCASES, Design[]>;
  }, [loaded]);
  const tadpoleFields = useAsync((signal) => Promise.all(config.channels.map((_, channel) => fetchField(TADPOLE_DESIGN, channel, voxelCount(config), signal))), [config]);
  const [velocity, pressure, temperature, turbulentEnergy] = ['Vx', 'p', 'T', 'k'].map((name) => channelIndex(config, name));
  const showcase = (name: keyof typeof SHOWCASES, initial: Partial<FieldDisplay>, highlight: string[]) => showcases
    ? <Showcase designs={showcases[name]} config={config} initial={initial} highlight={highlight} onOpen={onOpen} />
    : <Pending state={loaded} message="Loading designs…" />;
  const [nx, ny, nz] = config.dimensions;
  const latentCells = [nx, ny, nz].map((voxels) => voxels / LATENT_DIVISOR);
  const latentCellSize = config.spacing.slice(0, 2).map((spacing) => Math.round(spacing * LATENT_DIVISOR * 1000));
  const openTopic = (next: Topic) => {
    setTopic(next);
    setActive(TOPICS[next].steps[0].id);
    window.scrollTo({ top: 0 });
  };

  const stepList = (key: Topic) => <ol aria-label={`${TOPICS[key].label} steps`} className="space-y-0.5 text-xs">
    {TOPICS[key].steps.map(({ id, title }, index) => <li key={id}>
      <button onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })} aria-current={active === id ? 'step' : undefined}
        className={`flex w-full gap-2 rounded-lg px-2.5 py-1.5 text-left transition hover:bg-hover hover:text-ink ${active === id ? 'bg-hover-strong text-ink' : 'text-ink-muted'}`}>
        <span className="w-4 shrink-0 tabular-nums">{index + 1}</span>{title}
      </button>
    </li>)}
  </ol>;
  const topicButton = (label: string, count: number, open: boolean, onClick: () => void, nested = false) => <button onClick={onClick} aria-expanded={open}
    className={`flex w-full items-center justify-between rounded-lg px-2.5 text-left font-medium transition hover:text-ink ${nested ? 'py-1.5 text-xs' : 'py-2 text-sm'} ${open ? 'text-ink' : 'text-ink-muted hover:bg-hover'}`}>
    <span><span aria-hidden className={`mr-1.5 inline-block transition ${open ? 'rotate-90' : ''}`}>›</span>{label}</span>
    <span className="text-xs font-normal text-ink-muted tabular-nums">{count}</span>
  </button>;
  const sidebar = <Panel className="p-2">
    <nav aria-label="Overview topics" className="space-y-1">
      {SIDEBAR_GROUPS.map(({ label, topics }) => {
        const open = topics.includes(topic);
        const count = topics.reduce((total, key) => total + TOPICS[key].steps.length, 0);
        return <section key={label} className={open ? 'rounded-xl bg-paper p-1' : ''}>
          {topicButton(label, count, open, () => openTopic(topics[0]))}
          {open && (topics.length === 1 ? stepList(topics[0]) : <div className="space-y-0.5 pl-3">
            {topics.map((key) => <section key={key}>
              {topicButton(TOPICS[key].label, TOPICS[key].steps.length, key === topic, () => openTopic(key), true)}
              {key === topic && stepList(key)}
            </section>)}
          </div>)}
        </section>;
      })}
    </nav>
  </Panel>;

  return <Layout nav={nav} sidebar={sidebar}>
    <div hidden={topic !== 'cooldata'} className="space-y-16 pb-24">
      <Step id="setup">
        <Prose>
          <p><strong>CoolData</strong> holds {formatNumber(dataset.total)} CFD simulations of air cooling heated electronic components. Each simulation is one <strong>design</strong>:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Channel</strong>: 0.5 m long (x, flow direction), 0.1 m wide (y), 20 mm high (z)</li>
            <li><strong>Inflow</strong>: air at 20 °C enters at x = 0 with velocity V and leaves at the far end</li>
            <li><strong>Bodies</strong>: 2–6 heated cuboids (chips) and cylinders (capacitors) on the floor, within the first 0.3 m. The last 0.2 m lets the flow settle</li>
            <li><strong>Walls</strong>: floor and ceiling are walls, the sides are symmetry planes</li>
          </ul>
          <p>A horizontal slice through one design, bodies in grey:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>T</strong>: a plume of warmed air trails every body</li>
            <li><strong>Vx</strong>: air speeds up beside the bodies to 10.5 m/s (2.7 × V) and flows backwards (negative Vx) in their wakes</li>
          </ul>
        </Prose>
        {showcase('hero', {}, ['V'])}
      </Step>

      <Step id="fields">
        <Prose>
          <p>Each design is a <strong>steady-state</strong> solution (Simcenter STAR-CCM+, RANS with a k-ε turbulence model). There is no time axis: one design is one averaged flow. On its mesh of roughly 160,000 cells it stores:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Vx, Vy, Vz</strong>: velocity in m/s along the flow, across it and upwards</li>
            <li><strong>p</strong>: static pressure in Pa, fixed to 0 at the outlet, so the inlet pressure is the pressure drop</li>
            <li><strong>T</strong>: air temperature in K, from 293 K at the inlet up to the body temperatures</li>
            <li><strong>k</strong>: turbulent kinetic energy in m²/s², how strongly the air fluctuates around the averaged flow</li>
            <li><strong>ε</strong>: dissipation rate of k in m²/s³, how fast that turbulence decays into heat</li>
          </ul>
          <p>On the boundaries (walls, bodies, inlet, outlet, symmetry planes), a separate surface mesh stores per face:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>h</strong>: heat transfer coefficient in W/(m²·K), how well the air carries heat away from a wall</li>
            <li><strong>τx, τy, τz</strong>: wall shear stress in Pa, the drag the flow exerts along a wall</li>
            <li><strong>p</strong>, <strong>T</strong>: pressure and temperature at the wall</li>
          </ul>
          <p>Coolfrog resamples every volume mesh onto a {config.dimensions.join(' × ')} voxel grid of about {config.spacing.map((spacing) => formatNumber(spacing * 1000)).join(' × ')} mm. Voxels inside bodies have no flow values and render as solid. The <strong>Surface</strong> mode shows the boundary data on the original faces. Inlet, outlet and symmetry planes store no wall shear stress.</p>
        </Prose>
      </Step>

      <Step id="parameters">
        <Prose>
          <p>A design is fully described by <strong>35 numbers</strong>: the inlet velocity plus position, size and temperature of up to six bodies. Bodies 1–4 are cuboids, 5–6 cylinders. Positions are body <strong>centers</strong>, so a body near the inlet or a side wall extends past it and gets cut off. Values were drawn by Latin hypercube sampling, so every range is covered roughly evenly.</p>
        </Prose>
        <ParameterTable config={config} />
      </Step>

      <Step id="velocity">
        <Prose>
          <p>These two designs have nearly the same geometry and body temperatures. Only the inlet velocity differs, <strong>1.2 vs 7.0 m/s</strong>, and the fastest air in the channel reaches 1.6 vs 8.2 m/s. Switch to <strong>T</strong>: the faster air warms less and leaves at 22.1 °C instead of 24.0 °C. The cost is pressure: the drop across the channel grows from 0.8 to 3.5 Pa.</p>
        </Prose>
        {showcase('velocity', { channel: velocity }, ['V', 'T1', 'T2'])}
      </Step>

      <Step id="temperature">
        <Prose>
          <p>Same idea, now holding geometry and velocity (V ≈ 1.5 m/s) and changing the body temperatures from <strong>below 30 °C to above 70 °C</strong>. The air warms from a mean of 20.5 °C to 31.0 °C, while the velocity fields stay similar (Vx differs by 0.25 m/s on average). Switch to Vx to compare.</p>
        </Prose>
        {showcase('temperature', { channel: temperature }, ['T1', 'T2', 'V'])}
      </Step>

      <Step id="footprint">
        <Prose>
          <p>Footprint (xs, ys, r) and position decide how much of the channel cross-section a body blocks. Body 1 here spans <strong>nearly the full width</strong> (ys1 ≈ 0.098 m), leaving a 4.3 mm gap at the side. At the body center, 86% of the air squeezes through the 5 mm gap above it and 14% through the side gap. The pressure drop reaches <strong>70 Pa</strong>, compared with 0.8–3.5 Pa for the open designs above. Switch to <strong>Vx</strong>: at only V = 2.5 m/s the jet over the body reaches 12.2 m/s, and negative Vx behind it marks recirculation.</p>
        </Prose>
        {showcase('wall', { channel: pressure }, ['ys1', 'xs1', 'V'])}
      </Step>

      <Step id="cylinders">
        <Prose>
          <p>Cylinder radii go up to 0.1 m, so a cylinder can be <strong>wider than the channel</strong> and then acts like a wall with rounded ends. At the same V ≈ 4 m/s, an 8.6 cm cylinder causes a 211 Pa pressure drop and a 16.6 m/s jet, a 6 mm cylinder 24 Pa and 9.6 m/s.</p>
        </Prose>
        {showcase('cylinders', { channel: velocity }, ['r5', 'x5', 'V'])}
      </Step>

      <Step id="turbulence">
        <Prose>
          <p>RANS solves for the averaged flow only. The fluctuations around it are summarized by two fields:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>k</strong>: the kinetic energy of the fluctuations. √(2k/3) is roughly how far the velocity swings around its average</li>
            <li><strong>ε</strong>: how fast that energy is destroyed and turned into heat, highest where fluctuations are squeezed against walls</li>
          </ul>
          <p>These are the two extremes found among 10 candidates, both with two cuboids and no other bodies:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Calm</strong>: V = 1.0 m/s past two thin bodies. Mean k is 0.009 m²/s², so the velocity swings by about 0.08 m/s, 8% of V</li>
            <li><strong>Violent</strong>: V = 6.7 m/s, but both bodies span nearly the full width. The air squeezes through the 5 mm gap above them at up to 39 m/s. Mean k is 6.9 m²/s² (760× higher) and the velocity swings by about 2.1 m/s, 32% of V</li>
          </ul>
          <p>Both cards share one color scale, so the calm design stays dark. Mean ε differs even more, 0.21 vs 6,653 m²/s³. Switch to <strong>ε</strong> and move Z to the top: in the violent design, 97% of the most intense 1% of ε sits in the voxel layer under the ceiling, where the jet scrapes along it above the bodies. k itself peaks lower, 12–17 mm up, where the jet shears against the slower air around it.</p>
        </Prose>
        {showcase('turbulence', { channel: turbulentEnergy, slice: 10 }, ['V', 'ys1', 'ys2'])}
      </Step>

      <Step id="height">
        <Prose>
          <p>The metadata lists a height <strong>zs</strong> per body, spread evenly between 1 and 18 mm. The simulations <strong>ignore it</strong>: each body slot has a fixed height. Cuboids 1, 2 and 4 are 15 mm, cuboid 3 is 10 mm, cylinder 5 is 17 mm and cylinder 6 is 10 mm. The design below lists 16.7, 1.1 and 12.9 mm, yet its three bodies all stand 15 mm tall.</p>
          <p>These are exactly the zs1–zs6 values stored in the published simulation template, so the sampled heights were apparently never applied. Heights read from the surface files of 786 bodies in 297 designs, across all five runs, show no exception. zs says nothing about the geometry, so a model should not get it as input.</p>
        </Prose>
        {showcase('heights', { mode: '3d' }, ['zs1', 'zs2', 'zs4'])}
      </Step>

      <Step id="dataset">
        <Prose>
          <p>The designs come from five simulation <strong>runs</strong>. Only converged simulations were published, and ids 39,915–125,001 have parameters but no simulation. Run 7 mostly repeats run 4 parameter sets, so Coolfrog keeps one simulation per parameter set.</p>
        </Prose>
        <CountBars counts={dataset.runs} total={dataset.total} label={(run) => run} />
      </Step>

      <Step id="next">
        <Prose>
          <p>Continue with the model Coolfrog builds on, or explore the data in two views:</p>
        </Prose>
        <div className="flex flex-wrap gap-3">
          <button onClick={() => openTopic('overview')} className="rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-paper transition hover:opacity-90">Continue with Tadpole →</button>
          <button onClick={() => onView('dataset')} className="rounded-xl border border-divider-soft px-4 py-2.5 text-sm text-ink-muted transition hover:border-divider hover:text-ink">Cool Dataframe: filter and sort all designs</button>
          <button onClick={() => onView('flow')} className="rounded-xl border border-divider-soft px-4 py-2.5 text-sm text-ink-muted transition hover:border-divider hover:text-ink">Cool Flowfields: sample designs by parameter range</button>
        </div>
      </Step>
    </div>

    <div hidden={topic !== 'architecture'} className="space-y-16 pb-24">
      <Step id="arch-crops">
        <Prose>
          <p><strong>The encoder only ever sees single-channel crops.</strong> Before the network runs, a batch of PDE states is rearranged: variables fold into the batch, and a large domain can be tiled into crops that join the batch as well.</p>
          <p>Pure bookkeeping: values move, none change. Each batch entry is then encoded on its own.</p>
          <Disclosure label="The same in the paper's notation · §3.3">
            <div className="mt-3 space-y-3">
              <p>A batch has shape (B, C, X, Y, Z): B samples, C variables, three spatial axes. Tiled into U · V · W crops of H<sub>X</sub> × H<sub>Y</sub> × H<sub>Z</sub>:</p>
              <p className="overflow-x-auto text-ink">(B, C, U·H<sub>X</sub>, V·H<sub>Y</sub>, W·H<sub>Z</sub>) → (B·C·U·V·W, 1, H<sub>X</sub>, H<sub>Y</sub>, H<sub>Z</sub>)</p>
              <p>Pre-training draws random crops instead of tiling. After training, tiling or whole-domain encoding (step 7) covers larger domains.</p>
            </div>
          </Disclosure>
        </Prose>
        <Lazy3D><TensorExplorer range="rearrange" label="Toy batch of fields cut into crops and folded into one batch of single-channel crops" /></Lazy3D>
      </Step>

      <Step id="arch-convolution">
        <Prose>
          <p>The encoder opens with <strong>convolutions</strong>: one lifts the field to learned features, two stride-2 ones halve the grid twice, 64³ → 16³. Build one up from 2D:</p>
        </Prose>
        <Lazy3D><ConvolutionScene /></Lazy3D>
        <Prose>
          <Disclosure label="What can one latent value see?">
            <p className="mt-3">Each layer reads neighbours of neighbours, and every stride-2 step doubles how far one kernel step reaches. So a latent value depends on far more than its 16³ block.</p>
          </Disclosure>
          <Disclosure label="Why a network trained on 64³ crops runs on other grids · §3.4">
            <p className="mt-3">The same weights apply at every position, so the input size doesn’t matter. Tadpole, trained on 64³, also runs on 32³, 48³ or whole domains. Stride and padding make this only approximately shift-equivariant.</p>
          </Disclosure>
        </Prose>
      </Step>

      <Step id="arch-attention">
        <Prose>
          <p>From 16³ on, each grid position is a <strong>token</strong> and transformer blocks take over. Unlike a kernel, attention weights depend on content. Tadpole restricts them to fixed <strong>4 × 4 × 4 windows</strong>:</p>
        </Prose>
        <Lazy3D><AttentionScene /></Lazy3D>
        <Prose>
          <Disclosure label="Why windows, and how do windows talk to each other?">
            <p className="mt-3">Full attention over 16³ tokens: 16.8 M comparisons per head. In 4³ windows: 262,144. Windows never shift, so they connect only through the stride-2 convolutions between stages, and each coarser stage lets a window cover more of the crop.</p>
          </Disclosure>
          <Disclosure label="Attention details · Appendix C.1, P3D code">
            <p className="mt-3">Two blocks per stage, attention plus MLP on residual paths. Cosine similarity with a learned scale and a relative-position bias, as in Swin Transformer V2.</p>
          </Disclosure>
        </Prose>
      </Step>

      <Step id="arch-network">
        <Prose>
          <p><strong>Four halvings make the bottleneck</strong>: 64³ → 4³. The KL head then predicts a mean and a log-variance per latent value, and training samples z = μ + σ · ε.</p>
          <Disclosure label="Why the decoder is not a mirrored encoder · code trace">
            <div className="mt-3 space-y-3">
              <p>Each decoder upsampling is a convolution plus a <strong>pixel shuffle</strong>, which moves 8 channels into a 2 × 2 × 2 block. No attention at 4³, attention at 8³ and 16³.</p>
              <p>Compression per crop: 64³ / (C<sub>l</sub> · 4³) = 4,096 / C<sub>l</sub>.</p>
            </div>
          </Disclosure>
        </Prose>
        <Lazy3D><TensorExplorer range="network" diagram label="Toy crop through encoder, KL head and decoder" /></Lazy3D>
        <Disclosure label="Model sizes" className="max-w-3xl"><div className="mt-3"><Lazy3D><SizeTable /></Lazy3D></div></Disclosure>
      </Step>

      <Step id="arch-layout">
        <Prose>
          <p><strong>Put the variables back together.</strong> For dynamics, Tadpole-DFT stacks the latents of all variables, so its sub-network S can learn how they interact:</p>
          <Disclosure label="The same in the paper's notation · §4, TadpoleDFT.forward">
            <div className="mt-3 space-y-3">
              <p className="overflow-x-auto text-ink">(B·C·U·V·W, C<sub>l</sub>, X<sub>l</sub>, Y<sub>l</sub>, Z<sub>l</sub>) → (B, C·C<sub>l</sub>, U·X<sub>l</sub>, V·Y<sub>l</sub>, W·Z<sub>l</sub>)</p>
              <p>Tadpole-DFT encodes without crops by default, so U = V = W = 1 and the layout only stacks the variables.</p>
            </div>
          </Disclosure>
        </Prose>
        <Lazy3D><TensorExplorer range="layout" label="Sampled crop latents reassembled into one coarse volume per sample" /></Lazy3D>
      </Step>

      <Step id="arch-rationale">
        <Prose>
          <p>One goal: <strong>a reusable representation of single fields first, task models later</strong>.</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Scalar fields</strong>: any variable count, but no cross-variable learning</li>
            <li><strong>Random crops</strong>: any simulation resolution, no learned solver periodicity</li>
            <li><strong>Local operations</strong>: run on other grid sizes, e.g. 48³ and 32³ crops zero-shot</li>
            <li><strong>No skip connections</strong>: everything passes the latent</li>
            <li><strong>No PDE-parameter embeddings</strong>: reconstruction needs only the field</li>
          </ul>
        </Prose>
      </Step>

      <Step id="arch-whole">
        <Prose>
          <p><strong>64³ is the training crop size, not a limit.</strong> After training, a whole domain can be encoded in one pass if each axis is divisible by 16. No crop borders means no seams.</p>
          <p>On turbulent channel flow (96² × 192), whole-domain encoding changed RMSE by −2.8 % for Tadpole-S and +0.8 % for Tadpole-B, and the seams disappeared.</p>
        </Prose>
        <div className="flex flex-wrap gap-3">
          <button onClick={() => openTopic('zeroshot')} className="rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-paper transition hover:opacity-90">Continue with Zero-shot autoencoding →</button>
        </div>
      </Step>
    </div>

    <div hidden={topic !== 'overview'} className="space-y-16 pb-24">
      <Step id="tadpole">
        <Prose>
          <p><strong>Tadpole</strong> (Liu, Koehler, Holzschuh and Thuerey, TUM) is a foundation model for 3D PDEs: pre-trained once, reused on systems it has never seen. Most PDE foundation models learn dynamics, u<sub>t</sub> → u<sub>t+Δt</sub>, from stored simulations. In 3D that hits three walls:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Data</strong>: 3D simulations are expensive to generate, store and read</li>
            <li><strong>Transfer</strong>: most models need full fine-tuning on every new system</li>
            <li><strong>Function</strong>: a dynamics model serves dynamics only</li>
          </ul>
          <p>Tadpole’s three answers:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Reconstruct, don’t predict</strong>: a VAE learns the spatial structure of PDE solutions. Its latent then serves compression, dynamics and generation</li>
            <li><strong>Generate data while training</strong>: a GPU solver produces fresh solutions of seven PDEs on the fly. Nothing is stored, about 202 TB streamed</li>
            <li><strong>One scalar crop at a time</strong>: every variable becomes a single-channel 64³ crop, so one network handles any system</li>
          </ul>
        </Prose>
        <ReconstructionDiagram />
        <Prose>
          <p>Zero-shot, Tadpole-B reconstructs four unseen 3D flows with 1.3–7.9 % NRMSE, beating models trained from scratch on three. Fine-tuned for dynamics, it beats MORPH and DPOT, and on a turbulent boundary layer the 200× larger Walrus.</p>
          <Disclosure label="Why reconstruction is the better pre-training target · §3.1, Appendix A.1">
            <ul className="mt-3 list-disc space-y-1 pl-5">
              <li><strong>One right answer</strong>: a state can evolve into very different futures depending on PDE and parameters. Reconstruction always has one target, the input</li>
              <li><strong>Easier</strong>: PDE solutions lie on a smooth, low-dimensional manifold. Reconstruction learns the manifold, dynamics must also learn the flow along it</li>
              <li><strong>Broader</strong>: a latent of the solution space serves compression, prediction and generation</li>
            </ul>
          </Disclosure>
          <Disclosure label="Three model sizes · §5.1, Table 12, Appendix C.4">
            <div className="mt-3"><TadpoleSizeTable /></div>
          </Disclosure>
        </Prose>
      </Step>

      <Step id="pretraining">
        <Prose>
          <p><strong>No stored dataset.</strong> A GPU solver simulates while the model trains, cycling through seven PDE families with fresh coefficients and initial conditions:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Diffusion</strong> and <strong>hyper-diffusion</strong>: smoothing</li>
            <li><strong>Burgers</strong>: shocks in a three-component velocity field</li>
            <li><strong>Korteweg-de Vries</strong>: dispersive waves</li>
            <li><strong>Kuramoto-Sivashinsky</strong>: chaos</li>
            <li><strong>Fisher-KPP</strong>: reaction-diffusion fronts</li>
            <li><strong>Swift-Hohenberg</strong>: pattern formation</li>
          </ul>
          <p>Simulations run at 64³ to 384³. Training sees random 64³ crops, so borders aren’t periodic and the same crop size shows different scales:</p>
        </Prose>
        <Pipeline label="Online pre-training data path" stages={PRETRAINING_PIPELINE} />
        <Prose>
          <Disclosure label="Does the variety of PDEs matter? · §5.2, Fig. 2b">
            <p className="mt-3">Yes. Pre-trained on three PDEs, on one, or on initial conditions only, Tadpole-B reconstructed unseen flows worse in every case than with all seven.</p>
          </Disclosure>
          <Disclosure label="Why generate online? · §3.2, Appendix B.2">
            <p className="mt-3">Storing this much data is impractical and reading it would stall training. Buffers hide the solver’s latency, and online training was 1.8× faster than reading pre-generated data from fast SSDs.</p>
          </Disclosure>
        </Prose>
      </Step>

      <Step id="scalar-fields">
        <Prose>
          <p>Systems differ in their number of variables, from one for diffusion to ten for magnetohydrodynamics. Tadpole only ever looks at one: (B, C, X, Y, Z) → (B·C, 1, X, Y, Z), then random 64³ crops:</p>
        </Prose>
        <Pipeline label="From a multi-variable sample to single-channel crops" stages={[
          { title: 'Sample', detail: 'C variables on X × Y × Z' },
          { title: 'Split variables', detail: 'C scalar fields' },
          { title: 'Random crops', detail: '64³ from 96³ pre-crops' },
          { title: 'Batch', detail: 'Independent 1 × 64³ crops' },
        ]} />
        <Prose>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Gain</strong>: every system yields the same input shape</li>
            <li><strong>Price</strong>: the autoencoder never sees how variables relate. Tadpole-DFT recombines them downstream (step 6)</li>
          </ul>
          <Disclosure label="Why this works for reconstruction but not for dynamics · §3.3">
            <p className="mt-3">Predicting velocity needs pressure, and errors at crop borders accumulate over a rollout. A crop contains everything needed to rebuild itself.</p>
          </Disclosure>
          <p><button onClick={() => openTopic('architecture')} className="font-medium text-ink underline decoration-divider underline-offset-2 transition hover:decoration-ink">Architecture</button> step 1 animates the rearrangement.</p>
        </Prose>
      </Step>

      <Step id="architecture">
        <Prose>
          <p>The backbone is <strong>P3D</strong>, a hybrid of convolutions and a transformer, without its PDE-parameter embeddings and skip connections:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Convolutions</strong> at high resolution: 64³ → 16³</li>
            <li><strong>Window attention</strong> at low resolution, stride-2 convolutions in between: 16³ → 4³</li>
            <li><strong>Latent</strong>: a mean and a log-variance per value</li>
            <li><strong>Decoder</strong>: grows the grid back with pixel shuffle</li>
          </ul>
          <p>One 64³ crop through the Tadpole-B encoder:</p>
        </Prose>
        <EncoderStages />
        <Prose>
          <Disclosure label="Why convolutions first and attention later?">
            <p className="mt-3">Attention cost grows with the number of tokens: 262,144 at 64³, 4,096 at 16³. Convolutions handle fine detail cheaply, attention runs where it is affordable.</p>
          </Disclosure>
          <Disclosure label="Why remove the skip connections? · §3.4, §4">
            <p className="mt-3">Skips would let fine detail bypass the latent, so the latent would not need to describe the field. Tadpole-DFT restores them for dynamics, scaled from zero.</p>
          </Disclosure>
          <p>Every operation in 3D: <button onClick={() => openTopic('architecture')} className="font-medium text-ink underline decoration-divider underline-offset-2 transition hover:decoration-ink">Architecture</button>.</p>
        </Prose>
      </Step>

      <Step id="objective">
        <Prose>
          <p>Three loss terms:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Reconstruction</strong>: the decoded field should match the input</li>
            <li><strong>KL divergence</strong>, weight 10⁻⁶: pulls the latent toward a standard normal</li>
            <li><strong>Adversarial</strong>, weight up to 10⁻⁴: a patch discriminator pushes toward sharp output</li>
          </ul>
          <Disclosure label="Why such small weights? · Appendix C.2">
            <p className="mt-3">Reconstruction dominates, so the latent spends its capacity on detail. The adversarial term counters blur and starts only once reconstruction error is below 0.001, then ramps up.</p>
          </Disclosure>
        </Prose>
      </Step>

      <Step id="downstream">
        <Prose>
          <p><strong>One representation, three tasks.</strong> Encoder and decoder stay frozen, are fully retrained, or get <strong>LoRA</strong> adapters (W = W₀ + AB).</p>
        </Prose>
        <div className="grid max-w-5xl gap-5 lg:grid-cols-3">
          {[
            { task: 'Autoencoding', text: 'Compress and restore. LoRA-32 fine-tuning cuts the error on magnetohydrodynamics by over 60 % against training from scratch.' },
            { task: 'Dynamics (Tadpole-DFT)', text: 'A small transformer between encoder and decoder steps the latents of all variables forward. Skips and LoRA start at zero, so training starts from the autoencoder.' },
            { task: 'Generative modeling', text: 'Flow matching generates latents, the decoder turns them into fields. On channel flow it beat a pixel-space U-Net that samples about 180× slower.' },
          ].map(({ task, text }) => <Panel key={task} className="space-y-1.5 p-4 text-xs">
            <h3 className="text-sm font-medium">{task}</h3>
            <p className="leading-relaxed text-ink-muted">{text}</p>
          </Panel>)}
        </div>
        <div className="flex flex-wrap gap-3">
          <button onClick={() => openTopic('architecture')} className="rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-paper transition hover:opacity-90">Continue with Architecture →</button>
        </div>
      </Step>
    </div>

    <div hidden={topic !== 'zeroshot'} className="space-y-16 pb-24">
      <Step id="prepare">
        <Prose>
          <p>CoolData stores seven variables on a {config.dimensions.join(' × ')} grid. It cannot be tiled into 64³ crops: 400 and 80 are no multiples of 64, and the {nz}-voxel height is shorter than one crop. Every axis is divisible by 16, so Coolfrog encodes each variable as a whole domain with the published Tadpole-B weights:</p>
        </Prose>
        <Pipeline label="CoolData shapes through Tadpole-B, whole domain" stages={[
          { title: 'Fields', detail: `(1, ${config.channels.length}, ${config.dimensions.join(', ')})` },
          { title: 'Fold variables', detail: `(${config.channels.length}, 1, ${config.dimensions.join(', ')})` },
          { title: 'Encode', detail: `(${config.channels.length}, 512, ${latentCells.join(', ')})` },
          { title: 'Decode', detail: `(${config.channels.length}, 1, ${config.dimensions.join(', ')})` },
        ]} />
        <Prose>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Scale to [−1, 1]</strong> per variable by its fluid minimum and maximum, the value range of the pre-training data</li>
            <li><strong>Fill solids</strong> with the average of neighbouring fluid, repeated until every body is filled. Tadpole has no mask input, and a constant inside bodies would create jumps at every wall</li>
            <li><strong>Mask again</strong> after decoding, errors count fluid voxels only</li>
          </ul>
          <p>The {latentCells.join(' × ')} latent is padded to {latentCells.map((cells) => Math.ceil(cells / 4) * 4).join(' × ')} for the 4³ attention windows, {latentCells.map((cells) => Math.ceil(cells / 4)).reduce((product, count) => product * count, 1)} windows in total. Design {TADPOLE_DESIGN} as Tadpole receives it, seven independent fields at mid height, each scaled on its own:</p>
        </Prose>
        {tadpoleFields.status === 'done' ? <ScalarFields config={config} fields={tadpoleFields.data} /> : <Pending state={tadpoleFields} message="Loading fields…" />}
      </Step>

      <Step id="latent">
        <Prose>
          <p>On the {config.dimensions.join(' × ')} grid, the latent of one channel is <strong>512 × {latentCells.join(' × ')}</strong>: {latentCells[0] * latentCells[1] * latentCells[2]} positions with 512 numbers each, {formatNumber(512 * latentCells[0] * latentCells[1] * latentCells[2])} values for {formatNumber(nx * ny * nz)} voxels. Each position summarizes 16 × 16 × 16 voxels, about {latentCellSize.join(' × ')} mm of floor over the <strong>full channel height</strong>. The temperature of design {TADPOLE_DESIGN} under these cells, hover to link a cell and its voxels, scrub to collapse the height and merge every block into its pillar:</p>
        </Prose>
        {tadpoleFields.status === 'done' ? <Lazy3D><CoolDataLatent config={config} field={tadpoleFields.data[temperature]} /></Lazy3D> : <Pending state={tadpoleFields} message="Loading fields…" />}
        <Prose>
          <p>The paper's datasets span 96 to 1,024 voxels per axis. CoolData is only {nz} voxels tall, so its height collapses into a <strong>single latent cell</strong>. Every vertical structure, from the boundary layer on the floor to the jet over a body, has to fit into the 512 features of its column. The 16³ block is the cell's stride footprint: 3³ kernels and attention windows let each cell also see its neighbours.</p>
        </Prose>
      </Step>

      <Step id="zero-shot">
        <Prose>
          <p>Tadpole-B as published, without any training on CoolData: each channel is scaled by the design's own min and max, encoded whole and decoded again. Over 23 designs, the median <strong>NRMSE</strong> (RMSE divided by the field's standard deviation, 100 % is no better than predicting the mean everywhere) is:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>p</strong>: 4.9 %. Pressure falls smoothly along the channel</li>
            <li><strong>Vx, Vy, Vz, T, k</strong>: 28–39 %</li>
            <li><strong>ε</strong>: 110 %, worse than the mean</li>
          </ul>
          <p>The paper reports 1.3–7.9 % on its four datasets. Two effects measured on design {TADPOLE_DESIGN} hint at why CoolData is harder:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Walls</strong>: the Vx error is 65 % in the floor voxel layer and 72 % under the ceiling, against about 10 % at mid height. The boundary layers there share one latent column with everything above them (step 1)</li>
            <li><strong>Skewed values</strong>: 87 % of fluid voxels hold ε below 1 % of its range. After scaling to [−1, 1], nearly the whole field sits at −1, while a few voxels near the walls span the rest</li>
          </ul>
          <p>Every channel at one height, with two error measures from <code>coolfrog.lib.metrics</code> over the whole 3D field:</p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>NRMSE</strong>: as above, fluid voxels only</li>
            <li><strong>Spectral loss</strong>: mean absolute difference of log(1 + |FFT amplitude|) between reconstruction and simulation, solids set to 0. Computed on the [−1, 1] scale Tadpole sees, so channels compare. 0 is a perfect spectrum, larger values mean missing or spurious scales</li>
          </ul>
          <p>Simulation and reconstruction show the same range, taken from the simulation over all layers or over the shown slice. Every field uses viridis from the minimum to the 99th percentile, so a few extreme voxels don't flatten the rest, with a 0 tick where values change sign. k and ε use a log scale. The error is reconstruction − simulation by default, red where Tadpole overshoots, blue where it undershoots, or its magnitude on a single-hue map. Its scale ends at the channel's largest |error| in the 3D volume, at the 99th percentile, or at the largest |error| in the slice. ≤ and ≥ on a colorbar mark values clipped to the end colors. Any design id with a simulation works:</p>
        </Prose>
        <ZeroShot config={config} initialDesign={TADPOLE_DESIGN} />
      </Step>
    </div>
  </Layout>;
}
