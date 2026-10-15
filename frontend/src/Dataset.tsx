import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { fetchRows, fetchSummary } from './api';
import type { Config, Dataset, DatasetColumn, DatasetFilter, DatasetSummary, Design, Row, RowsQuery } from './api';
import { errorMessage, formatNumber } from './lib/format';
import { withRange } from './lib/range';
import type { Range } from './lib/range';
import { Badge, HEADING, Layout, Panel, RangeControl } from './lib/ui';
import { useAsync } from './lib/useAsync';

/** Most designs sent to the flow explorer at once, its sample count limit. */
const MAX_PICKED = 12;
/** Fixed row height in px (`h-6`), lets the table render only the visible rows. */
const ROW_HEIGHT = 24;
/** Rows per `/api/dataset/rows` request. */
const BLOCK_SIZE = 100;
/** Rows rendered beyond each viewport edge. */
const OVERSCAN = 20;
/** Filtered and sorted rows, everything a row block depends on except its position. */
type RowSource = Omit<RowsQuery, 'offset' | 'limit'>;
type RowBlocks = { source: RowSource; blocks: Record<number, Row[]>; error?: string };

/** All-row distribution behind the filtered one, each scaled to its own peak to compare shapes. */
function Histogram({ all, filtered }: { all: number[]; filtered: number[] | undefined }) {
  const bars = (counts: number[], className: string) => {
    const peak = Math.max(...counts);
    return counts.map((count, index) => count > 0 && <rect key={index} x={index + 0.1} width={0.8} y={1 - count / peak} height={count / peak} className={className} />);
  };
  return <svg viewBox={`0 0 ${all.length} 1`} preserveAspectRatio="none" className="block h-8 w-full px-2" aria-hidden>
    {bars(all, 'fill-hover-strong')}
    {filtered && bars(filtered, 'fill-ink-subtle')}
  </svg>;
}

/** Run checkboxes styled like `Segmented`, `selected` undefined = every run, `counts` may omit empty runs. */
function RunFilter({ runs, counts, selected, onChange }: {
  runs: string[]; counts: Record<string, number>; selected: string[] | undefined; onChange: (runs: string[] | undefined) => void;
}) {
  const toggle = (run: string) => {
    const current = selected ?? runs;
    const next = current.includes(run) ? current.filter((other) => other !== run) : [...current, run];
    onChange(next.length === runs.length ? undefined : next);
  };
  return <fieldset className="flex flex-wrap gap-1 rounded-xl bg-hover-strong p-1">
    <legend className="sr-only">Runs</legend>
    {runs.map((run) => <label key={run} className="cursor-pointer">
      <input type="checkbox" className="peer sr-only" checked={!selected || selected.includes(run)} onChange={() => toggle(run)} />
      <span className="block rounded-lg px-2.5 py-1 text-xs text-ink-muted transition peer-checked:bg-surface-elevated peer-checked:text-ink peer-checked:shadow-[var(--shadow-sm)] peer-focus-visible:ring-2 peer-focus-visible:ring-divider hover:text-ink">
        {run} <span className="tabular-nums text-ink-muted">{formatNumber(counts[run] ?? 0)}</span>
      </span>
    </label>)}
  </fieldset>;
}

function formatCell(value: number | string): string {
  return typeof value === 'number' && !Number.isInteger(value) ? formatNumber(value) : String(value);
}

/** Rows of `source` in `BLOCK_SIZE` blocks: fetch each block covering [first, last) once, start over when `source` changes. */
function useRowBlocks(source: RowSource, first: number, last: number): Omit<RowBlocks, 'source'> {
  const [state, setState] = useState<RowBlocks>({ source, blocks: {} });
  const requested = useRef(new Set<number>());
  const controller = useRef(new AbortController());
  useEffect(() => {
    controller.current = new AbortController();
    requested.current = new Set();
    return () => controller.current.abort();
  }, [source]);
  useEffect(() => {
    const { signal } = controller.current;
    // Successful blocks clear `error`, failed ones set it and are retried on the next scroll
    const settle = (update: (blocks: Record<number, Row[]>) => Omit<RowBlocks, 'source'>) =>
      setState((previous) => ({ source, ...update(previous.source === source ? previous.blocks : {}) }));
    for (let block = Math.floor(first / BLOCK_SIZE); block * BLOCK_SIZE < last; block++) {
      if (requested.current.has(block)) continue;
      requested.current.add(block);
      fetchRows({ ...source, offset: block * BLOCK_SIZE, limit: BLOCK_SIZE }, signal).then(
        (rows) => settle((blocks) => ({ blocks: { ...blocks, [block]: rows } })),
        (error) => {
          if (signal.aborted) return;
          requested.current.delete(block);
          settle((blocks) => ({ blocks, error: errorMessage(error) }));
        },
      );
    }
  }, [source, first, last]);
  return state.source === source ? state : { blocks: {} };
}

/** Opaque cell that tints with its row, so sticky cells hide what scrolls beneath them. */
const STICKY_CELL = 'sticky z-[1] bg-surface group-hover:shadow-[inset_0_0_0_100vmax_var(--hover)]';
const STICKY_PICKED = 'shadow-[inset_0_0_0_100vmax_var(--hover-strong)] group-hover:shadow-[inset_0_0_0_100vmax_var(--hover-strong)]';

/** Sortable `design_id` header whose lens swaps the title for a digits-only prefix search, Escape or a second click clears it. */
function DesignIdHeader({ column, source, onSort, search, onSearch }: {
  column: string; source: RowSource; onSort: (column: string) => void; search: string; onSearch: (search: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const close = () => { setOpen(false); onSearch(''); };
  const sorted = source.sort === column;
  return <div className="flex h-5 items-center justify-end gap-1.5">
    {open
      ? <input
        autoFocus type="search" inputMode="numeric" value={search} placeholder="Search id" aria-label="Search design id"
        onChange={(event) => onSearch(event.target.value.replace(/\D/g, ''))}
        onKeyDown={(event) => event.key === 'Escape' && close()}
        onBlur={() => !search && setOpen(false)}
        className="min-w-0 flex-1 rounded-md bg-paper px-1.5 py-0.5 text-xs font-normal text-ink outline-none placeholder:text-ink-subtle focus-visible:ring-2 focus-visible:ring-divider [&::-webkit-search-cancel-button]:hidden"
      />
      : <button className={`transition hover:text-ink ${sorted ? 'text-ink' : ''}`} onClick={() => onSort(column)}>{column}{sorted && (source.descending ? ' ↓' : ' ↑')}</button>}
    <button aria-label="Search design id" aria-pressed={open} onClick={() => (open ? close() : setOpen(true))} className={`shrink-0 transition hover:text-ink ${open ? 'text-ink' : ''}`}>
      <svg viewBox="0 0 16 16" className="size-3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden><circle cx="7" cy="7" r="4.5" /><path d="m10.5 10.5 3.5 3.5" /></svg>
    </button>
  </div>;
}

/** Endless table over all `rowCount` rows: renders the visible window, numeric headers sort, rows toggle picking. */
function Table({ header, columns, rowCount, source, onSort, search, onSearch, isPicked, canPick, onPick }: {
  header: string[]; columns: Record<string, DatasetColumn>; rowCount: number; source: RowSource; onSort: (column: string) => void;
  search: string; onSearch: (search: string) => void; isPicked: (row: Row) => boolean; canPick: boolean; onPick: (row: Row) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ top: 0, height: 0 });
  useEffect(() => {
    const element = scroller.current!;
    const observer = new ResizeObserver(() => setViewport((previous) => ({ ...previous, height: element.clientHeight })));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { scroller.current?.scrollTo({ top: 0 }); }, [source]);

  const first = Math.max(0, Math.floor(viewport.top / ROW_HEIGHT) - OVERSCAN);
  const last = Math.min(rowCount, Math.ceil((viewport.top + viewport.height) / ROW_HEIGHT) + OVERSCAN);
  const { blocks, error } = useRowBlocks(source, first, last);

  return <div ref={scroller} onScroll={(event) => { const top = event.currentTarget.scrollTop; setViewport((previous) => ({ ...previous, top })); }} className="min-h-0 flex-1 overflow-auto [scrollbar-width:thin]">
    <table className="min-w-full table-fixed text-xs tabular-nums" style={{ width: `${3.5 + header.length * 5 + header.filter((column) => !(column in columns)).length + 1}rem` }}>
      <thead className="sticky top-0 z-10 bg-surface text-left text-ink-muted shadow-[inset_0_-1px_0_var(--divider-soft)]">
        <tr>
          <th className={`${STICKY_CELL} left-0 w-10 px-4`}><span className="sr-only">Picked</span></th>
          {header.map((column, index) => <th key={column} scope="col" className={`${index === 0 ? `${STICKY_CELL} left-10 w-28` : column in columns ? 'w-20' : 'w-24'} truncate px-3 py-2 font-medium ${column in columns ? 'text-right' : ''}`} aria-sort={source.sort === column ? (source.descending ? 'descending' : 'ascending') : undefined}>
            {index === 0
              ? <DesignIdHeader column={column} source={source} onSort={onSort} search={search} onSearch={onSearch} />
              : column in columns
                ? <button className={`transition hover:text-ink ${source.sort === column ? 'text-ink' : ''}`} onClick={() => onSort(column)}>{column}{source.sort === column && (source.descending ? ' ↓' : ' ↑')}</button>
                : column}
          </th>)}
        </tr>
      </thead>
      <tbody>
        {first > 0 && <tr aria-hidden style={{ height: first * ROW_HEIGHT }} />}
        {Array.from({ length: last - first }, (_, offset) => {
          const index = first + offset;
          const row = blocks[Math.floor(index / BLOCK_SIZE)]?.[index % BLOCK_SIZE];
          if (!row) return <tr key={index} className="h-6"><td colSpan={header.length + 1} className="px-4"><div className="h-2 w-1/3 animate-pulse rounded-full bg-hover" /></td></tr>;
          const picked = isPicked(row);
          const stickyCell = `${STICKY_CELL} ${picked ? STICKY_PICKED : ''}`;
          return <tr key={index} onClick={() => (picked || canPick) && onPick(row)} className={`group h-6 cursor-pointer transition hover:bg-hover ${picked ? 'bg-hover-strong' : ''}`}>
            <td className={`${stickyCell} left-0 px-4`}><input type="checkbox" readOnly checked={picked} disabled={!picked && !canPick} aria-label={`Pick design ${row[0]}`} className="block accent-ink" /></td>
            {row.map((value, column) => <td key={header[column]} className={`truncate px-3 ${header[column] in columns ? 'text-right' : 'text-ink-muted'} ${column === 0 ? `${stickyCell} left-10` : ''}`}>{formatCell(value)}</td>)}
          </tr>;
        })}
        {last < rowCount && <tr aria-hidden style={{ height: (rowCount - last) * ROW_HEIGHT }} />}
      </tbody>
    </table>
    {error && <p role="alert" className="sticky left-0 p-4 text-xs text-danger">{error}</p>}
  </div>;
}

/** Filter, sort and scroll the metadata, then open picked designs in the flow explorer. */
export default function DatasetExplorer({ dataset, config, nav, onOpen }: {
  dataset: Dataset; config: Config; nav: ReactNode; onOpen: (designs: Design[]) => void;
}) {
  const [ranges, setRanges] = useState<Record<string, Range>>({});
  const [search, setSearch] = useState('');
  const [runs, setRuns] = useState<string[]>();
  const [sort, setSort] = useState({ sort: 'design_id', descending: false });
  const [picked, setPicked] = useState<Record<number, Design>>({});
  const filter = useMemo<DatasetFilter>(() => ({ ranges, search, runs }), [ranges, search, runs]);
  const source = useMemo<RowSource>(() => ({ ...filter, ...sort }), [filter, sort]);
  const result = useAsync((signal) => fetchSummary(filter, signal), [filter]);
  // Keep showing the last summary while the next one loads, so dragging a slider doesn't flicker
  const latest = useRef<DatasetSummary>(undefined);
  if (result.status === 'done') latest.current = result.data;
  const summary = latest.current;

  const columns = useMemo(() => Object.fromEntries(dataset.columns.map((column) => [column.name, column])), [dataset]);
  const narrowedCount = Object.keys(ranges).length;
  const pickedCount = Object.keys(picked).length;

  const designFromRow = (row: Row): Design => ({
    id: row[dataset.header.indexOf('design_id')] as number,
    params: Object.fromEntries(config.parameters.map(({ name }) => [name, row[dataset.header.indexOf(name)] as number])),
  });
  const togglePick = (row: Row) => {
    const design = designFromRow(row);
    setPicked(({ [design.id]: existing, ...rest }) => existing ? rest : { ...rest, [design.id]: design });
  };

  const sidebar = <Panel className="flex max-h-[calc(100vh-8rem)] flex-col divide-y divide-divider-soft">
    <section className="space-y-3 p-4">
      <h2 className={HEADING}>Runs</h2>
      <RunFilter runs={Object.keys(dataset.runs)} counts={summary?.runs ?? dataset.runs} selected={runs} onChange={setRuns} />
    </section>
    <section className="flex min-h-0 flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <h2 className={HEADING}>Column ranges</h2>
        <span className="flex items-center gap-2">
          {narrowedCount > 0 && <button className="text-[11px] text-ink-muted transition hover:text-ink" onClick={() => setRanges({})}>Reset</button>}
          <Badge>{narrowedCount}</Badge>
        </span>
      </div>
      <div className="-mx-4 min-h-0 space-y-4 overflow-y-auto px-4 [scrollbar-width:thin]">
        {dataset.columns.map((column) => <RangeControl key={column.name} bounded={column} range={ranges[column.name]} onChange={(range) => setRanges((previous) => withRange(previous, column, range))}>
          <Histogram all={column.counts} filtered={summary?.histograms[column.name]} />
        </RangeControl>)}
      </div>
    </section>
  </Panel>;

  return <Layout nav={nav} sidebar={sidebar}>
    <div className="flex h-[calc(100dvh-3rem)] flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs text-ink-muted" role="status">
          {summary ? `${formatNumber(summary.matched)} of ${formatNumber(dataset.total)} rows` : 'Filtering…'}
          {result.status === 'loading' && summary && ' · updating'}
        </span>
        <span className="ml-auto flex items-center gap-3 text-xs text-ink-muted">
          {pickedCount > 0 && <button className="transition hover:text-ink" onClick={() => setPicked({})}>Clear</button>}
          <span className="tabular-nums">{pickedCount} / {MAX_PICKED} picked</span>
          <button disabled={!pickedCount} onClick={() => onOpen(Object.values(picked))} className="rounded-xl bg-ink px-3.5 py-1.5 font-medium text-paper transition hover:opacity-90 disabled:opacity-40">Open in flow explorer</button>
        </span>
      </div>

      {result.status === 'error' && <Panel className="p-4 text-sm text-danger"><p role="alert">{result.error}</p></Panel>}

      <Panel className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <Table
          header={dataset.header} columns={columns} rowCount={summary?.matched ?? 0} source={source}
          onSort={(column) => setSort((previous) => ({ sort: column, descending: previous.sort === column && !previous.descending }))}
          search={search} onSearch={setSearch}
          isPicked={(row) => designFromRow(row).id in picked} canPick={pickedCount < MAX_PICKED} onPick={togglePick}
        />
      </Panel>
    </div>
  </Layout>;
}
