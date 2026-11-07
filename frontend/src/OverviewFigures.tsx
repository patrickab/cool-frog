import { Fragment, useState } from 'react';
import type { ReactNode } from 'react';
import type { Config, Field } from './api';
import Heatmap from './Heatmap';
import { oklchCss } from './lib/colormap';
import { formatCount, formatNumber, formatTick } from './lib/format';
import { DataTable, Disclosure, HEADING, Panel, Segmented, useTheme } from './lib/ui';

/** Explanatory text, readable line length, `strong` for terms. */
export function Prose({ children }: { children: ReactNode }) {
  return <div className="max-w-3xl space-y-3 text-sm leading-relaxed text-ink-muted [&_strong]:font-medium [&_strong]:text-ink">{children}</div>;
}

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

/** Tadpole-B encoder stages on one 64³ crop, `divisor` shrinks every grid axis. */
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

/** Tadpole-B autoencoding NRMSE ×10⁻² on the paper's four datasets, Table 2. */
const FINETUNE_RESULTS = [
  { method: 'Zero-shot', errors: ['3.23', '7.87', '1.27', '3.42'], trainable: '0' },
  { method: 'From scratch', errors: ['7.17', '4.40', '1.67', '8.22'], trainable: '38.1 M' },
  { method: 'Full fine-tuning (FPFT)', errors: ['2.73', '2.94', '0.50', '2.50'], trainable: '38.1 M' },
  { method: 'LoRA-32', errors: ['3.01', '3.82', '0.78', '3.18'], trainable: '2.8 M' },
];
const PAPER_DATASETS = ['Iso', 'TCF', 'TBL', 'MHD'];

/** Trainable Tadpole-B parameters per fine-tuning state, counted on `TadpoleAutoencoder`. */
const TRAINABLE_PARAMETERS = [
  { state: 'LoRA-4', parameters: 370_697 },
  { state: 'LoRA-8', parameters: 719_185 },
  { state: 'LoRA-16', parameters: 1_416_161 },
  { state: 'LoRA-32', parameters: 2_810_113 },
  { state: 'LoRA-64', parameters: 5_598_017 },
  { state: 'FPFT', parameters: 38_068_329 },
];

export function ParameterTable({ config }: { config: Config }) {
  const bounds = Object.fromEntries(config.parameters.map((parameter) => [parameter.name, parameter]));
  return <DataTable head={<>
    <th scope="col" className="px-4 py-2 font-medium">Columns</th>
    <th scope="col" className="px-2 py-2 font-medium">Meaning</th>
    <th scope="col" className="px-4 py-2 text-right font-medium">Range in data</th>
  </>}>
    {PARAMETER_GROUPS.map(({ columns, meaning, range }) => {
      const minimum = Math.min(...columns.map((name) => bounds[name].min));
      const maximum = Math.max(...columns.map((name) => bounds[name].max));
      const unit = bounds[columns[0]].unit;
      return <tr key={columns[0]} className="align-top">
        <td className="px-4 py-2 font-medium whitespace-nowrap">{columns.length > 1 ? `${columns[0]}–${columns.at(-1)}` : columns[0]}</td>
        <td className="px-2 py-2 text-ink-muted">{meaning}</td>
        <td className="px-4 py-2 text-right whitespace-nowrap">{range ?? `${formatNumber(+minimum.toFixed(3))} – ${formatNumber(+maximum.toFixed(3))}`}{unit && ` ${unit}`}</td>
      </tr>;
    })}
  </DataTable>;
}

/** Labeled horizontal bars, widths relative to the largest count. */
export function CountBars({ label, counts, total }: { label: (key: string) => string; counts: Record<string, number>; total: number }) {
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
export function Pipeline({ label, stages }: { label: string; stages: readonly { title: string; detail: string }[] }) {
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
export function ReconstructionDiagram() {
  const { axes } = useTheme().colors
  return <Panel className="max-w-3xl space-y-4 p-4">
    <figure aria-label="A scalar field passes through an encoder, a learned feature bottleneck and a decoder to reconstruct the same field at the same time. There is no encoder-to-decoder bypass.">
      <div className="grid items-center gap-3 text-center text-xs sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {[0, 1, 2].map((stage) => <Fragment key={stage}>
          {stage > 0 && <div className="text-ink-muted"><span className="block text-xl sm:-rotate-90" aria-hidden>↓</span><span>{stage === 1 ? 'Encoder' : 'Decoder'}</span></div>}
          <div className="space-y-2">
            <p className="font-medium">{['Input field · time t', 'Learned bottleneck', 'Reconstruction · time t'][stage]}</p>
            <svg viewBox="0 0 160 112" aria-hidden="true" className="mx-auto h-28 w-40 max-w-full" fill="none">
              {stage === 1 ? <g stroke={oklchCss(axes.feature)} strokeWidth="1.5">
                {[0, 1, 2].map((layer) => <g key={layer} transform={`translate(${48 + layer * 9} ${36 - layer * 9})`}>
                  <rect width="48" height="48" fill="var(--surface)" />
                  <path d="M16 0v48M32 0v48M0 16h48M0 32h48" />
                </g>)}
              </g> : <>
                <g stroke={oklchCss(axes.spatial)} strokeWidth="1">
                  <path d="M20 30h96v66H20zM20 30l24-18h96v66l-24 18M116 30l24-18M44 12v18M116 78h24" />
                  <path d="M44 30v66M68 30v66M92 30v66M20 52h96M20 74h96" strokeOpacity="0.3" />
                </g>
                <g stroke={oklchCss(axes.channel)} strokeWidth="2">
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

export function TadpoleSizeTable() {
  return <DataTable className="max-w-3xl" head={<>
    <th scope="col" className="px-4 py-2 font-medium">Size</th>
    <th scope="col" className="px-2 py-2 text-right font-medium">Parameters</th>
    <th scope="col" className="px-2 py-2 text-right font-medium">Compression</th>
    <th scope="col" className="px-2 py-2 text-right font-medium">Pre-training RMSE ×10⁻³</th>
    <th scope="col" className="px-4 py-2 text-right font-medium">Pre-training GPU hours</th>
  </>}>
    {TADPOLE_SIZES.map(({ size, parameters, compression, error, cost }) => <tr key={size}>
      <td className="px-4 py-2 font-medium">{size}</td>
      <td className="px-2 py-2 text-right">{parameters}</td>
      <td className="px-2 py-2 text-right">{compression}</td>
      <td className="px-2 py-2 text-right">{error}</td>
      <td className="px-4 py-2 text-right">{cost}</td>
    </tr>)}
  </DataTable>;
}

export function FinetuneResultsTable() {
  return <figure className="max-w-3xl space-y-2">
    <DataTable caption="Tadpole-B autoencoding NRMSE ×10⁻², paper Table 2" head={<>
      <th scope="col" className="px-4 py-2 font-medium">Tadpole-B</th>
      {PAPER_DATASETS.map((dataset) => <th key={dataset} scope="col" className="px-2 py-2 text-right font-medium">{dataset}</th>)}
      <th scope="col" className="px-4 py-2 text-right font-medium">Trainable</th>
    </>}>
      {FINETUNE_RESULTS.map(({ method, errors, trainable }) => <tr key={method}>
        <th scope="row" className="px-4 py-2 font-medium">{method}</th>
        {errors.map((error, index) => <td key={PAPER_DATASETS[index]} className="px-2 py-2 text-right">{error}</td>)}
        <td className="px-4 py-2 text-right">{trainable}</td>
      </tr>)}
    </DataTable>
    <figcaption className="text-xs text-ink-muted">NRMSE ×10⁻², Table 2 · Iso: isotropic turbulence · TCF: channel flow · TBL: boundary layer · MHD: magnetohydrodynamics</figcaption>
  </figure>;
}

/** Explain a dense LoRA adapter with column-vector notation and a shape-only rank example. */
export function LoraExplainer() {
  const [rank, setRank] = useState(2)
  const parameters = rank * (8 + 8)
  return <div className="max-w-3xl space-y-4">
    <Panel className="p-4">
      <h3 className={HEADING}>One layer, two parallel paths</h3>
      <div className="grid items-center gap-5 lg:grid-cols-2">
        <figure>
          <svg viewBox="0 0 320 310" role="img" aria-labelledby="lora-forward-title lora-forward-description" className="mx-auto w-full max-w-80" fill="none">
            <title id="lora-forward-title">LoRA forward pass</title>
            <desc id="lora-forward-description">The same input x enters frozen W zero and trainable A then B in parallel. Scale the adapter output by s and add it to W zero x to obtain y.</desc>
            <g stroke="var(--ink-muted)" strokeWidth="1.5">
              <path d="M160 39v13H70v46M160 52h90v23M70 161v93h72M250 119v28M250 191v63h-72M160 272v19" />
              <path d="M66 92l4 6 4-6M246 69l4 6 4-6M246 141l4 6 4-6M136 250l6 4-6 4M184 250l-6 4 6 4M156 285l4 6 4-6" />
              <rect x="20" y="98" width="100" height="63" rx="8" strokeDasharray="4 3" />
              <rect x="200" y="75" width="100" height="44" rx="8" />
              <rect x="200" y="147" width="100" height="44" rx="8" />
              <circle cx="160" cy="254" r="18" />
            </g>
            <g fill="var(--ink)" textAnchor="middle" fontSize="14">
              <text x="160" y="27">x ∈ ℝᵏ</text>
              <text x="70" y="124">W₀ · frozen</text>
              <text x="70" y="147">d × k</text>
              <text x="250" y="102">A · r × k</text>
              <text x="250" y="174">B · d × r</text>
              <text x="160" y="259" fontSize="20">+</text>
              <text x="160" y="307">y ∈ ℝᵈ</text>
            </g>
            <g fill="var(--ink-muted)" fontSize="12">
              <text x="173" y="137">Ax ∈ ℝʳ</text>
              <text x="258" y="226">× s</text>
              <text x="23" y="225">W₀x</text>
            </g>
          </svg>
          <figcaption className="mt-3 text-center text-xs text-ink-muted">One dense layer · biases and optional dropout omitted</figcaption>
        </figure>
        <div className="space-y-3 text-sm leading-relaxed text-ink-muted">
          <p className="text-lg text-ink">y = W₀x + sB(Ax)</p>
          <p><strong className="font-medium text-ink">A reads, B writes.</strong> A turns k input features into r coefficients. B combines r learned output directions into a d-dimensional correction. Both matrices learn from the task loss.</p>
          <p>There is <strong className="font-medium text-ink">no activation between A and B</strong>. Their composition is a linear update ΔW = sBA, not another nonlinear network.</p>
          <p>In original LoRA, s = α/r. Rank r controls capacity, α controls the update scale, and the optimizer learning rate controls parameter steps. These are different knobs.</p>
        </div>
      </div>
    </Panel>
    <Panel className="space-y-4 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className={HEADING}>What does low rank constrain?</h3>
        <Segmented label="Illustrative adapter rank" options={[1, 2, 4, 8].map((value) => ({ value, label: `r = ${value}` }))} value={rank} onChange={setRank} />
      </div>
      <figure aria-label={`Shape example: B is 8 by ${rank}, A is ${rank} by 8, and their product is 8 by 8 with rank at most ${rank}.`}>
        <div className="grid grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-1 text-center text-sm tabular-nums">
          {[{ name: 'B', rows: 8, columns: rank }, { name: 'A', rows: rank, columns: 8 }, { name: 'BA', rows: 8, columns: 8 }].map(({ name, rows, columns }, index) => <Fragment key={name}>
            {index > 0 && <span aria-hidden className="text-ink-muted">{index === 1 ? '×' : '='}</span>}
            <div>
              <svg viewBox="0 0 128 128" aria-hidden="true" className="mx-auto w-full max-w-40" fill="none">
                <g transform={`translate(${(128 - columns * 14) / 2} ${(128 - rows * 14) / 2})`}>
                  <rect width={columns * 14} height={rows * 14} fill="var(--hover-strong)" stroke="var(--ink-muted)" />
                  {Array.from({ length: columns - 1 }, (_, column) => <path key={`c${column}`} d={`M${(column + 1) * 14} 0v${rows * 14}`} stroke="var(--divider)" />)}
                  {Array.from({ length: rows - 1 }, (_, row) => <path key={`r${row}`} d={`M0 ${(row + 1) * 14}h${columns * 14}`} stroke="var(--divider)" />)}
                </g>
              </svg>
              <p className="font-medium">{name}</p>
              <p className="text-xs text-ink-muted">{rows} × {columns}</p>
            </div>
          </Fragment>)}
        </div>
        <figcaption className="mt-4 text-xs leading-relaxed text-ink-muted">Illustrative 8 × 8 layer, not Tadpole weights. Cells show parameter slots, not values. The product can be dense even at rank 1. During training, compute B(Ax) without materializing BA.</figcaption>
      </figure>
      <p aria-live="polite" className="text-sm tabular-nums"><strong className="font-medium">{parameters} trainable parameters</strong> instead of 64 for a full update · {parameters / 64 * 100} % of the dense count.{rank === 4 ? ' No parameter saving at this rank.' : rank === 8 ? ' Twice the dense count, with no rank restriction.' : ''}</p>
      <div className="space-y-2 text-sm leading-relaxed text-ink-muted">
        <p>In general: <strong className="font-medium text-ink">r(k + d) parameters instead of dk</strong>. Each column of B is one output direction, weighted by a coefficient from Ax. Thus rank(ΔW) ≤ r, even though all dk entries may change.</p>
        <p>The constraint is on <strong className="font-medium text-ink">the correction, not W₀</strong>. LoRA does not compress the base model or take its SVD. The output directions are learned, not fixed beforehand. A larger r permits richer corrections, but does not guarantee better validation error.</p>
      </div>
    </Panel>
    <Prose>
      <h3 className={HEADING}>From initialization to inference</h3>
      <ol className="list-decimal space-y-2 pl-5">
        <li><strong>Start unchanged.</strong> Standard dense LoRA initializes A randomly and B to zero, so ΔW = 0. The first forward pass matches the base layer.</li>
        <li><strong>Learn the correction.</strong> Backpropagate the task loss and update A and B, keeping W₀ fixed. Initially the loss gradient for A is zero because B = 0, while B can receive a gradient. Once B changes, both can learn.</li>
        <li><strong>Store or merge.</strong> Save the small adapter alongside a reference to the base checkpoint. For inference, a compatible dense adapter can be merged as Wmerged = W₀ + sBA, removing the extra branch. This is an algebraic property, not a claim that Coolfrog currently exports merged weights.</li>
      </ol>
      <Disclosure label="How can gradients flow through a frozen layer?">
        <div className="mt-3 space-y-2">
          <p>Frozen means no optimizer update to W₀, not no differentiation through the layer. With g = ∂L/∂y, the input gradient is ∂L/∂x = W₀ᵀg + sAᵀBᵀg, so earlier trainable layers still receive a learning signal.</p>
          <p>The adapter gradients are ∂L/∂B = sg(Ax)ᵀ and ∂L/∂A = sBᵀgxᵀ. Setting both A and B to zero would make both loss gradients zero. Random A breaks that dead start.</p>
        </div>
      </Disclosure>
      <p className="text-xs">Sources: <a href="https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html" className="underline underline-offset-2">Raschka’s visual introduction</a> for the intuition, <a href="https://arxiv.org/html/2106.09685v2#S4.SS1" className="underline underline-offset-2">Hu et al., §4.1</a> for the formulation, and <a href="https://github.com/microsoft/LoRA/blob/main/loralib/layers.py" className="underline underline-offset-2">Microsoft’s dense-layer implementation</a> for initialization, scaling and merging. Diagrams here are original schematics.</p>
    </Prose>
  </div>
}

export function TrainableParametersTable() {
  const total = TRAINABLE_PARAMETERS[TRAINABLE_PARAMETERS.length - 1].parameters;
  return <DataTable className="max-w-md" caption="Trainable Tadpole-B parameters per fine-tuning state" head={<>
    <th scope="col" className="px-4 py-2 font-medium">State</th>
    <th scope="col" className="px-2 py-2 text-right font-medium">Trainable parameters</th>
    <th scope="col" className="px-4 py-2 text-right font-medium">Share</th>
  </>}>
    {TRAINABLE_PARAMETERS.map(({ state, parameters }) => <tr key={state}>
      <th scope="row" className="px-4 py-2 font-medium">{state}</th>
      <td className="px-2 py-2 text-right">{formatCount(parameters)}</td>
      <td className="px-4 py-2 text-right">{(100 * parameters / total).toFixed(1)} %</td>
    </tr>)}
  </DataTable>;
}

/** One 64³ crop through the Tadpole-B encoder, each footprint drawn to scale against the input. */
export function EncoderStages() {
  const footprint = 160;
  return <DataTable className="max-w-3xl" caption="Tensor shape of one 64³ crop through the Tadpole-B encoder" head={<>
    <th scope="col" className="px-4 py-2 font-medium">Stage</th>
    <th scope="col" className="px-2 py-2 font-medium">Grid to scale</th>
    <th scope="col" className="px-2 py-2 text-right font-medium">Features × grid</th>
    <th scope="col" className="px-4 py-2 text-right font-medium">Values</th>
  </>}>
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
  </DataTable>;
}

/** Every channel of one design on its own linear color scale, as Tadpole receives them after per-channel min-max scaling. */
export function ScalarFields({ config, fields }: { config: Config; fields: Field[] }) {
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
