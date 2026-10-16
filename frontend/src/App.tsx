import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { fetchConfig, fetchDataset, fetchSample } from './api';
import type { Config, Design, Sample, SampleQuery } from './api';
import AutoencoderView from './Autoencoder';
import DatasetExplorer from './Dataset';
import Gallery, { defaultDisplay, FieldControls, useFieldBatch } from './Gallery';
import Overview, { TADPOLE_DESIGN } from './Overview';
import Sidebar from './Sidebar';
import { formatNumber } from './lib/format';
import { Layout, Panel, Select } from './lib/ui';
import { useAsync } from './lib/useAsync';

const VIEWS = { overview: 'Cool Overview', flow: 'Cool Flowfields', dataset: 'Cool Dataframe', autoencoder: 'Cool Autoencoder' } as const;
export type View = keyof typeof VIEWS;
/** A seeded sample from the sidebar or designs picked in the overview or dataframe. */
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
  const [display, setDisplay] = useState(() => defaultDisplay(config));
  const designs = sample.status === 'done' ? sample.data.designs : undefined;
  const batch = useFieldBatch(designs, display, config);

  const status = sample.status === 'loading' ? 'Selecting designs…'
    : sample.status === 'done' ? [
      sample.data.seed === undefined
        ? `${sample.data.designs.length} picked designs`
        : `${sample.data.designs.length} of ${formatNumber(sample.data.matched)} matching designs · seed ${sample.data.seed}`,
      batch.pending > 0 && `${batch.pending} loading`,
      batch.failed > 0 && `${batch.failed} failed`,
    ].filter(Boolean).join(' · ')
    : '';

  return <Layout nav={nav} sidebar={<Sidebar config={config} busy={sample.status === 'loading'} onSample={onRequest} />}>
    <div className="flex flex-wrap items-center gap-3">
      <FieldControls config={config} display={display} onChange={setDisplay} color={batch.color} />
      <span className="ml-auto text-xs text-ink-muted" role="status">{status}</span>
    </div>

    {sample.status === 'error' && <Panel className="p-4 text-sm text-danger"><p role="alert">{sample.error}</p></Panel>}
    {designs
      ? <Gallery designs={designs} batch={batch} config={config} display={display} />
      : <Panel className="grid place-items-center py-32 text-sm text-ink-muted">{sample.status === 'loading' ? 'Selecting designs…' : 'Set parameter ranges and sample, or pick designs in Cool Dataframe.'}</Panel>}
  </Layout>;
}

export default function App() {
  const [attempt, setAttempt] = useState(0);
  const [view, setView] = useState<View>(() => (location.hash.slice(1) in VIEWS ? location.hash.slice(1) as View : 'overview'));
  const [flowRequest, setFlowRequest] = useState<FlowRequest>();
  const loaded = useAsync((signal) => Promise.all([fetchConfig(signal), fetchDataset(signal)]), [attempt]);
  useEffect(() => {
    history.replaceState(null, '', `#${view}`);
    document.title = `Coolfrog · ${VIEWS[view]}`;
  }, [view]);
  const nav = <ViewSelect view={view} onChange={setView} />;

  if (loaded.status === 'done') {
    const [config, dataset] = loaded.data;
    const openInFlow = (designs: Design[]) => { setFlowRequest([...designs].sort((a, b) => a.id - b.id)); setView('flow'); };
    // All views stay mounted so switching keeps loaded fields and filters
    return <>
      <div hidden={view !== 'overview'}><Overview config={config} dataset={dataset} nav={nav} onView={setView} onOpen={openInFlow} /></div>
      <div hidden={view !== 'flow'}><Explorer config={config} nav={nav} request={flowRequest} onRequest={setFlowRequest} /></div>
      <div hidden={view !== 'dataset'}><DatasetExplorer dataset={dataset} config={config} nav={nav} onOpen={openInFlow} /></div>
      <div hidden={view !== 'autoencoder'}><AutoencoderView config={config} nav={nav} initialDesign={TADPOLE_DESIGN} /></div>
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
