import { useState } from 'react';
import type { Config, SampleQuery } from './api';
import { withRange } from './lib/range';
import type { Range } from './lib/range';
import { Badge, Disclosure, HEADING, Panel, RangeControl, Slider } from './lib/ui';

const MAX_SEED = 2147483646;

/** Sampling inputs: count, narrowed parameter ranges and seed. */
export default function Sidebar({ config, busy, onSample }: { config: Config; busy: boolean; onSample: (query: SampleQuery) => void }) {
  const [count, setCount] = useState(4);
  const [seed, setSeed] = useState(42);
  const [narrowed, setNarrowed] = useState<Record<string, Range>>({});
  const narrowedCount = Object.keys(narrowed).length;
  const sample = (nextSeed: number) => {
    setSeed(nextSeed);
    onSample({ count, seed: nextSeed, ranges: narrowed });
  };

  return <Panel className="divide-y divide-divider-soft">
    <section className="space-y-3 p-4">
      <div className="flex items-center justify-between"><h2 className={HEADING}>Samples</h2><Badge>{count}</Badge></div>
      <Slider label="Number of samples" min={1} max={12} values={[count]} onChange={([value]) => setCount(value)} />
    </section>

    <section className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h2 className={HEADING}>Parameter ranges</h2>
        <span className="flex items-center gap-2">
          {narrowedCount > 0 && <button className="text-[11px] text-ink-muted transition hover:text-ink" onClick={() => setNarrowed({})}>Reset</button>}
          <Badge>{narrowedCount}</Badge>
        </span>
      </div>
      <Disclosure label="All parameters" aside={<span className="tabular-nums">{config.parameters.length}</span>} className="rounded-xl bg-paper" summaryClassName="px-3 py-2">
        <div className="max-h-64 space-y-4 overflow-y-auto border-t border-divider-soft px-3 py-3 [scrollbar-width:thin]">
          {config.parameters.map((parameter) => <RangeControl key={parameter.name} bounded={parameter} range={narrowed[parameter.name]} onChange={(range) => setNarrowed((previous) => withRange(previous, parameter, range))} />)}
        </div>
      </Disclosure>
    </section>

    <section className="space-y-4 p-4">
      <label className="flex items-center justify-between text-sm">
        <span className={HEADING}>Seed</span>
        <input type="number" min="0" max={MAX_SEED} step="1" value={seed} className="w-24 rounded-lg bg-paper px-2.5 py-1 text-right text-xs tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-divider" onChange={(event) => setSeed(Math.max(0, Math.min(MAX_SEED, Math.floor(Number(event.target.value) || 0))))} />
      </label>
      <div className="flex gap-2">
        <button disabled={busy} onClick={() => sample(seed)} className="flex-1 rounded-xl bg-ink py-2.5 text-sm font-medium text-paper transition hover:opacity-90 disabled:opacity-40">{busy ? 'Selecting…' : 'Sample'}</button>
        <button disabled={busy} onClick={() => sample(crypto.getRandomValues(new Uint32Array(1))[0] % (MAX_SEED + 1))} title="Sample with a new seed" className="rounded-xl border border-divider-soft px-3.5 text-sm text-ink-muted transition hover:border-divider hover:text-ink disabled:opacity-40">Shuffle</button>
      </div>
    </section>
  </Panel>;
}
