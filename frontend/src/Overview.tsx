import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { fetchDesigns } from './api';
import type { Config, Dataset, Design } from './api';
import type { View } from './App';
import Gallery, { defaultDisplay, FieldControls, useFieldBatch } from './Gallery';
import type { FieldDisplay } from './Gallery';
import { formatNumber } from './lib/format';
import { HEADING, Layout, Panel } from './lib/ui';
import { useAsync } from './lib/useAsync';

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

const STEPS = [
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
] as const;
type StepId = (typeof STEPS)[number]['id'];

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
  { columns: ['zs1', 'zs2', 'zs3', 'zs4', 'zs5', 'zs6'], meaning: 'Nominal body height in m, see step 8' },
];

function channelIndex(config: Config, name: string): number {
  return Math.max(0, config.channels.findIndex((channel) => channel.name === name));
}

/** Id of the step currently crossing the upper third of the viewport. */
function useActiveStep(): StepId {
  const [active, setActive] = useState<StepId>(STEPS[0].id);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.find((entry) => entry.isIntersecting);
      if (visible) setActive(visible.target.id as StepId);
    }, { rootMargin: '-30% 0px -65% 0px' });
    for (const { id } of STEPS) {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    }
    return () => observer.disconnect();
  }, []);
  return active;
}

function Step({ id, children }: { id: StepId; children: ReactNode }) {
  const index = STEPS.findIndex((step) => step.id === id);
  return <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-6 space-y-4">
    <header>
      <p className={HEADING}>Step {index + 1}</p>
      <h2 id={`${id}-title`} className="mt-1 text-xl font-semibold tracking-tight">{STEPS[index].title}</h2>
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
      <FieldControls config={config} display={display} onChange={setDisplay} colorRange={batch.colorRange} />
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

/** Guided tour: setup, fields, parameters with controlled showcase pairs, data caveats, dataset composition. */
export default function Overview({ config, dataset, nav, onView, onOpen }: {
  config: Config; dataset: Dataset; nav: ReactNode; onView: (view: View) => void; onOpen: (designs: Design[]) => void;
}) {
  const loaded = useAsync((signal) => fetchDesigns(SHOWCASE_IDS, signal), []);
  const showcases = useMemo(() => {
    if (loaded.status !== 'done') return undefined;
    const byId = Object.fromEntries(loaded.data.map((design) => [design.id, design]));
    return Object.fromEntries(Object.entries(SHOWCASES).map(([name, ids]) => [name, ids.map((id) => byId[id])])) as Record<keyof typeof SHOWCASES, Design[]>;
  }, [loaded]);
  const active = useActiveStep();
  const [velocity, pressure, temperature, turbulentEnergy] = ['Vx', 'p', 'T', 'k'].map((name) => channelIndex(config, name));
  const showcase = (name: keyof typeof SHOWCASES, initial: Partial<FieldDisplay>, highlight: string[]) => showcases
    ? <Showcase designs={showcases[name]} config={config} initial={initial} highlight={highlight} onOpen={onOpen} />
    : loaded.status === 'error'
      ? <Panel className="p-4 text-sm text-danger"><p role="alert">{loaded.error}</p></Panel>
      : <Panel className="grid h-40 place-items-center text-xs text-ink-muted"><span role="status">Loading designs…</span></Panel>;

  const sidebar = <Panel className="p-2">
    <nav aria-label="Steps">
      <ol className="space-y-0.5 text-xs">
        {STEPS.map(({ id, title }, index) => <li key={id}>
          <button onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })} aria-current={active === id ? 'step' : undefined}
            className={`flex w-full gap-2 rounded-lg px-2.5 py-1.5 text-left transition hover:bg-hover hover:text-ink ${active === id ? 'bg-hover-strong text-ink' : 'text-ink-muted'}`}>
            <span className="w-4 shrink-0 tabular-nums">{index + 1}</span>{title}
          </button>
        </li>)}
      </ol>
    </nav>
  </Panel>;

  return <Layout nav={nav} sidebar={sidebar}>
    <div className="space-y-16 pb-24">
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
          <p>Two views let you go further:</p>
        </Prose>
        <div className="flex flex-wrap gap-3">
          <button onClick={() => onView('dataset')} className="rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-paper transition hover:opacity-90">Cool Dataframe: filter and sort all designs</button>
          <button onClick={() => onView('flow')} className="rounded-xl border border-divider-soft px-4 py-2.5 text-sm text-ink-muted transition hover:border-divider hover:text-ink">Cool Flowfields: sample designs by parameter range</button>
        </div>
      </Step>
    </div>
  </Layout>;
}
