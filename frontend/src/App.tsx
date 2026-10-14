import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { fetchConfig, fetchDataset, fetchSample } from './api';
import type { Config, Design, Sample, SampleQuery } from './api';
import DatasetExplorer from './Dataset';
import Gallery, { useFieldBatch } from './Gallery';
import type { RenderMode } from './Gallery';
import Sidebar from './Sidebar';
import { gradient } from './lib/colormap';
import { formatNumber } from './lib/format';
import { Layout, Panel, Segmented, Select, Slider } from './lib/ui';
import { useAsync } from './lib/useAsync';

const TEMPERATURE_CHANNEL = 4;
const VIEWS = { flow: 'Cool Flowfields', dataset: 'Cool Dataframe' } as const;
type View = keyof typeof VIEWS;
/** A seeded sample from the sidebar or designs picked in the dataset explorer. */
type FlowRequest = SampleQuery | Design[];

/** Subtitle-styled select switching between views. */
function ViewSelect({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return <div className="mt-1">
    <Select variant="inline" label="View" value={view} onChange={onChange} options={Object.entries(VIEWS).map(([value, label]) => ({ value: value as View, label }))} />
  </div>;
}

function Explorer({ config, nav, request, onRequest }: { config: Config; nav: ReactNode; request: FlowRequest | undefined; onRequest: (request: FlowRequest) => void }) {
  const sample = useAsync<Sample & { seed?: number }>(request && ((signal) => Array.isArray(request)
    ? Promise.resolve({ designs: request, matched: request.length })
    : fetchSample(request, signal).then((result) => ({ ...result, seed: request.seed }))), [request]);
  const [channel, setChannel] = useState(Math.min(TEMPERATURE_CHANNEL, config.channels.length - 1));
  const [mode, setMode] = useState<RenderMode>('2d');
  const [slice, setSlice] = useState(Math.floor(config.dimensions[2] / 2));
  const designs = sample.status === 'done' ? sample.data.designs : undefined;
  const batch = useFieldBatch(designs, channel, config.dimensions);
  const { colorRange } = batch;

  const status = sample.status === 'loading' ? 'Selecting designs…'
    : sample.status === 'done' ? [
      sample.data.seed === undefined
        ? `${sample.data.designs.length} designs picked in the dataset explorer`
        : `${sample.data.designs.length} of ${formatNumber(sample.data.matched)} matching designs · seed ${sample.data.seed}`,
      batch.pending > 0 && `${batch.pending} loading`,
      batch.failed > 0 && `${batch.failed} failed`,
    ].filter(Boolean).join(' · ')
    : '';

  return <Layout nav={nav} sidebar={<Sidebar config={config} busy={sample.status === 'loading'} onSample={onRequest} />}>
    <div className="flex flex-wrap items-center gap-3">
      <Segmented label="Rendering mode" value={mode} onChange={setMode} options={[{ value: '2d', label: '2D' }, { value: '3d', label: '3D' }]} />
      <Segmented label="Field" value={channel} onChange={setChannel} options={config.channels.map(({ name }, index) => ({ value: index, label: name }))} />
      <div className="flex items-center gap-3 rounded-xl bg-hover-strong py-1.5 pr-1.5 pl-3">
        <span className="text-xs text-ink-muted">Z</span>
        <div className="w-32"><Slider label="Z slice" min={0} max={config.dimensions[2] - 1} values={[slice]} onChange={([value]) => setSlice(value)} /></div>
        <span className="min-w-16 whitespace-nowrap rounded-lg bg-surface-elevated px-2 py-0.5 text-right text-xs tabular-nums shadow-[var(--shadow-sm)]">{formatNumber(slice * config.spacing[2] * 1000)} mm</span>
      </div>
      {colorRange && <div className="flex items-center gap-3 rounded-xl bg-hover-strong px-3 py-1 text-xs">
        <span><span className="font-medium">{config.channels[channel].name}</span> <span className="text-ink-muted">{config.channels[channel].unit}</span></span>
        <div className="w-40">
          <div className="h-2 rounded-full" style={{ background: gradient }} />
          <div className="mt-0.5 flex justify-between text-[10px] leading-none text-ink-muted tabular-nums"><span>{formatNumber(colorRange[0])}</span><span>{formatNumber(colorRange[1])}</span></div>
        </div>
        <span className="flex items-center gap-1.5 text-ink-muted"><i className="size-2.5 rounded-sm bg-solid" />Solid</span>
      </div>}
      <span className="ml-auto text-xs text-ink-muted" role="status">{status}</span>
    </div>

    {sample.status === 'error' && <Panel className="p-4 text-sm text-danger"><p role="alert">{sample.error}</p></Panel>}
    {designs
      ? <Gallery designs={designs} batch={batch} config={config} mode={mode} slice={slice} />
      : <Panel className="grid place-items-center py-32 text-sm text-ink-muted">{sample.status === 'loading' ? 'Selecting designs…' : 'Set parameter ranges and sample, or pick designs in the dataset explorer.'}</Panel>}
  </Layout>;
}

export default function App() {
  const [attempt, setAttempt] = useState(0);
  const [view, setView] = useState<View>(() => location.hash === '#dataset' ? 'dataset' : 'flow');
  const [flowRequest, setFlowRequest] = useState<FlowRequest>();
  const loaded = useAsync((signal) => Promise.all([fetchConfig(signal), fetchDataset(signal)]), [attempt]);
  useEffect(() => {
    history.replaceState(null, '', `#${view}`);
    document.title = `Coolfrog · ${VIEWS[view]}`;
  }, [view]);
  const nav = <ViewSelect view={view} onChange={setView} />;

  if (loaded.status === 'done') {
    const [config, dataset] = loaded.data;
    // Both views stay mounted so switching keeps loaded fields and filters
    return <>
      <div hidden={view !== 'flow'}><Explorer config={config} nav={nav} request={flowRequest} onRequest={setFlowRequest} /></div>
      <div hidden={view !== 'dataset'}>
        <DatasetExplorer dataset={dataset} config={config} nav={nav} onOpen={(designs) => { setFlowRequest(designs.sort((a, b) => a.id - b.id)); setView('flow'); }} />
      </div>
    </>;
  }
  return <Layout nav={nav}>
    <Panel className="p-6 text-sm text-ink-muted">
      {loaded.status === 'error'
        ? <div role="alert" className="flex items-center justify-between gap-4"><span className="text-danger">{loaded.error}</span><button className="rounded-lg border border-divider-soft px-3 py-1 text-xs hover:text-ink" onClick={() => setAttempt((value) => value + 1)}>Retry</button></div>
        : 'Loading metadata…'}
    </Panel>
  </Layout>;
}
