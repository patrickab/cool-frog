import { useEffect, useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { gradientCss, scaleValue } from './colormap';
import type { ColorScale } from './colormap';
import { formatNumber, formatPower, formatTick } from './format';
import type { Bounded, Range } from './range';
import type { AsyncState } from './useAsync';

const COLORBAR_WIDTH = 192;
/** Approximate width of an 11 px tabular label, enough to keep labels apart. */
const labelWidth = (text: string) => text.length * 5.8 + 6;

/** Middle ticks: 0 when the range crosses it (diverging, or linear with negative values), else the midpoint, powers of ten when logarithmic. */
function innerTicks(scale: ColorScale): [number, string][] {
  const [min, max] = scale.range;
  if (!scale.log) return min < 0 && max > 0 ? [[-min / (max - min), '0']] : [[0.5, formatTick(scaleValue(0.5, scale))]];
  const [low, high] = scale.range.map(Math.log10);
  const exponents = Array.from({ length: Math.max(0, Math.floor(high) - Math.ceil(low) + 1) }, (_, index) => Math.ceil(low) + index);
  return exponents.map((exponent) => [(exponent - low) / (high - low), formatPower(exponent)]);
}

/**
 * End ticks plus inner ticks that fit without overlapping.
 *
 * - Inner ticks colliding with an end label are dropped
 * - The rest is thinned to every 2nd, 3rd, … tick until neighbours no longer overlap, so spacing stays even
 */
function fittingTicks(start: string, inner: [number, string][], end: string): [number, string][] {
  const span = ([position, text]: [number, string]) => [position * COLORBAR_WIDTH - labelWidth(text) / 2, position * COLORBAR_WIDTH + labelWidth(text) / 2];
  const free = inner.filter((tick) => span(tick)[0] >= labelWidth(start) && span(tick)[1] <= COLORBAR_WIDTH - labelWidth(end));
  const isSpread = (ticks: [number, string][]) => ticks.every((tick, index) => index === 0 || span(tick)[0] >= span(ticks[index - 1])[1]);
  const stride = Array.from({ length: free.length }, (_, index) => index + 1)
    .find((step) => isSpread(free.filter((_, index) => index % step === 0)));
  return [[0, start], ...(stride ? free.filter((_, index) => index % stride === 0) : []), [1, end]];
}

/**
 * Horizontal colorbar with ticks at both ends and inside, see `innerTicks`.
 *
 * - `clipped.below`/`above` prefix the end labels with ≤/≥: values beyond the range show in the end colors
 * - Log scales carry a "log scale" tag next to the unit
 * - `label` (e.g. the quantity) sits left of the bar, `unit` after the ticks
 */
export function Colorbar({ scale, unit, label, clipped = { below: false, above: false } }: {
  scale: ColorScale; unit?: string | null; label?: ReactNode; clipped?: { below: boolean; above: boolean };
}) {
  const ticks = fittingTicks(
    `${clipped.below ? '≤ ' : ''}${formatTick(scale.range[0])}`,
    innerTicks(scale),
    `${clipped.above ? '≥ ' : ''}${formatTick(scale.range[1])}`,
  );
  return <div className="flex items-center gap-2 text-[11px] text-ink-muted tabular-nums">
    {label}
    <div style={{ width: COLORBAR_WIDTH }}>
      <div className="h-2 rounded-sm" style={{ background: gradientCss(scale.colormap) }} />
      <div className="relative mt-0.5 h-3.5">
        {ticks.map(([position, text]) => <span key={position} className="absolute top-0 flex flex-col whitespace-nowrap leading-none"
          style={{ left: `${position * 100}%`, transform: `translateX(${position === 0 ? '0' : position === 1 ? '-100%' : '-50%'})`, alignItems: position === 0 ? 'flex-start' : position === 1 ? 'flex-end' : 'center' }}>
          <i aria-hidden className="h-1 w-px bg-ink-subtle" />{text}
        </span>)}
      </div>
    </div>
    {unit && <span className="whitespace-nowrap">{unit}</span>}
    {scale.log && <span className="rounded-md bg-paper px-1.5 py-0.5 whitespace-nowrap text-ink">log scale</span>}
  </div>;
}

/** Page shell: sticky sidebar with the title and `nav` (view select) left, content right. */
export function Layout({ nav, sidebar, children }: { nav: ReactNode; sidebar?: ReactNode; children: ReactNode }) {
  return <div className="min-h-screen bg-paper font-sans text-ink antialiased">
    <div className="flex gap-6 p-6 max-md:flex-col">
      <aside className="w-72 shrink-0 max-md:w-full">
        <div className="sticky top-6 space-y-5">
          <header className="px-1">
            <h1 className="text-2xl font-semibold tracking-tight">Coolfrog</h1>
            {nav}
          </header>
          {sidebar}
        </div>
      </aside>
      <main className="min-w-0 flex-1 space-y-5">{children}</main>
    </div>
  </div>;
}

/** Section heading class: small uppercase muted label. */
export const HEADING = 'text-[11px] font-medium uppercase tracking-[0.12em] text-ink-muted';

/** Elevated container on the page: surface layer, warm shadow, top inner highlight. */
export function Panel({ className = '', children }: { className?: string; children: ReactNode }) {
  return <div className={`rounded-2xl bg-surface shadow-[var(--shadow-sm),var(--inner-highlight)] ${className}`}>{children}</div>;
}

/** Inline error, or a pulsing placeholder with one factual line while `state` loads. */
export function Pending({ state, message }: { state: AsyncState<unknown>; message: string }) {
  return state.status === 'error'
    ? <Panel className="p-4 text-sm text-danger"><p role="alert">{state.error}</p></Panel>
    : <Panel className="grid h-40 animate-pulse place-items-center text-xs text-ink-muted"><span role="status">{message}</span></Panel>;
}

export function Badge({ children }: { children: ReactNode }) {
  return <span className="rounded-md bg-paper px-1.5 py-0.5 text-[11px] text-ink-muted tabular-nums">{children}</span>;
}

/** Collapsed `details` with a rotating chevron, `aside` sits right in the summary. */
export function Disclosure({ label, aside, className = '', summaryClassName = '', children }: {
  label: string; aside?: ReactNode; className?: string; summaryClassName?: string; children: ReactNode;
}) {
  return <details className={`group text-xs ${className}`}>
    <summary className={`flex cursor-pointer list-none items-center justify-between text-ink-muted transition hover:text-ink ${summaryClassName}`}>
      <span><span className="mr-1.5 inline-block transition group-open:rotate-90">›</span>{label}</span>
      {aside}
    </summary>
    {children}
  </details>;
}

/** Radio group styled as a segmented control, arrow keys work natively, `name` unique per instance. */
export function Segmented<T extends string | number>({ label, options, value, onChange }: {
  label: string; options: { value: T; label: string }[]; value: T; onChange: (value: T) => void;
}) {
  const name = useId();
  return <fieldset className="inline-flex rounded-xl bg-hover-strong p-1">
    <legend className="sr-only">{label}</legend>
    {options.map((option) => <label key={option.value} className="cursor-pointer">
      <input type="radio" name={name} className="peer sr-only" checked={option.value === value} onChange={() => onChange(option.value)} />
      <span className="block rounded-lg px-3 py-1 text-xs text-ink-muted transition peer-checked:bg-surface-elevated peer-checked:text-ink peer-checked:shadow-[var(--shadow-sm)] peer-focus-visible:ring-2 peer-focus-visible:ring-divider hover:text-ink">{option.label}</span>
    </label>)}
  </fieldset>;
}

/** One thumb per value: `[value]` slider or `[min, max]` range. */
export function Slider<Values extends [number] | [number, number]>({ label, min, max, step, values, onChange, disabled }: {
  label: string; min: number; max: number; step?: number; values: Values; onChange: (values: Values) => void; disabled?: boolean;
}) {
  const span = max - min || 1;
  const isRange = values.length > 1;
  const fillStart = isRange ? ((values[0] - min) / span) * 100 : 0;
  const fillEnd = ((values[values.length - 1] - min) / span) * 100;
  const update = (index: number, value: number) => {
    const next = [...values] as Values;
    next[index] = isRange ? (index ? Math.max(value, values[0]) : Math.min(value, values[1]!)) : value;
    onChange(next);
  };
  return <div className="relative h-4">
    <div className="absolute inset-x-2 top-1/2 h-1 -translate-y-1/2 rounded-full bg-hover-strong">
      <div className="absolute h-full rounded-full bg-ink" style={{ left: `${fillStart}%`, right: `${100 - fillEnd}%` }} />
    </div>
    {values.map((value, index) => <input
      key={index} type="range" className="slider" min={min} max={max} step={step} value={value} disabled={disabled}
      aria-label={isRange ? `${label} ${index ? 'maximum' : 'minimum'}` : label}
      style={{ zIndex: index === 0 && value > min + span / 2 ? 2 : 1 }}
      onChange={(event) => update(index, +event.target.value)}
    />)}
  </div>;
}

/** Labeled range slider over the full bounds, `children` (e.g. a histogram) sit above the slider. */
export function RangeControl({ bounded, range, onChange, children }: {
  bounded: Bounded; range: Range | undefined; onChange: (range: Range) => void; children?: ReactNode;
}) {
  const { name, unit, min, max } = bounded;
  const value: Range = range ?? [min, max];
  return <div className="space-y-2">
    <div className="flex items-baseline justify-between gap-2 text-sm">
      <span>{name}{unit && <span className="ml-1 text-xs text-ink-muted">{unit}</span>}</span>
      <span className={`text-xs tabular-nums ${range ? 'text-ink' : 'text-ink-muted'}`}>{formatNumber(value[0])} – {formatNumber(value[1])}</span>
    </div>
    {children}
    <Slider label={name} min={min} max={max} step={(max - min) / 1000 || 1} values={value} disabled={min === max} onChange={onChange} />
  </div>;
}

/** Listbox select ported from gigachad-bot `StyledSelect`: bordered "box" field or "inline" text trigger. */
export function Select<T extends string>({ label, options, value, onChange, variant = 'box' }: {
  label: string; options: { value: T; label: string }[]; value: T; onChange: (value: T) => void; variant?: 'box' | 'inline';
}) {
  const inline = variant === 'inline';
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', closeOutside);
    return () => document.removeEventListener('pointerdown', closeOutside);
  }, [open]);
  const select = (next: T) => { onChange(next); setOpen(false); };
  const move = (step: 1 | -1) => {
    const index = Math.max(0, options.findIndex((option) => option.value === value));
    select(options[(index + step + options.length) % options.length].value);
  };
  const keys: Record<string, () => void> = {
    ArrowDown: () => (open ? move(1) : setOpen(true)),
    ArrowUp: () => (open ? move(-1) : setOpen(true)),
    Home: () => select(options[0].value),
    End: () => select(options[options.length - 1].value),
    Enter: () => setOpen(!open),
    ' ': () => setOpen(!open),
    Escape: () => setOpen(false),
  };

  return <div ref={root} className={inline ? 'relative inline-block' : 'relative'}>
    <button type="button" aria-label={label} aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)}
      onKeyDown={(event) => { if (event.key in keys) { event.preventDefault(); keys[event.key](); } }}
      className={inline
        ? 'flex cursor-pointer items-center gap-1 text-xs text-ink-muted transition hover:text-ink focus:outline-none focus-visible:text-ink'
        : 'flex w-full items-center gap-2 rounded-lg border border-divider bg-paper px-2.5 py-1.5 text-left text-xs text-ink shadow-[var(--inner-highlight)] transition-colors hover:bg-surface-elevated focus:outline-none focus-visible:border-ink-subtle'}>
      <span className="min-w-0 flex-1 truncate">{options.find((option) => option.value === value)?.label ?? value}</span>
      <svg aria-hidden viewBox="0 0 16 16" className={`size-3.5 shrink-0 fill-none stroke-current stroke-[1.5] text-ink-subtle transition-transform ${open ? 'rotate-180' : ''}`}><path d="m4 6 4 4 4-4" /></svg>
    </button>
    {open && <div role="listbox" aria-label={label} className={`absolute z-50 mt-1 max-h-52 overflow-y-auto rounded-lg border border-divider bg-surface p-1 shadow-[var(--shadow-md)] [scrollbar-width:thin] ${inline ? 'left-0 w-max' : 'w-full'}`}>
      {options.map((option) => <button key={option.value} type="button" role="option" aria-selected={option.value === value} onClick={() => select(option.value)}
        className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors ${option.value === value ? 'bg-surface-elevated text-ink' : 'text-ink-muted hover:bg-surface-elevated hover:text-ink'}`}>
        <span className="min-w-0 flex-1 truncate">{option.label}</span>
        {option.value === value && <svg aria-hidden viewBox="0 0 16 16" className="size-3.5 shrink-0 fill-none stroke-current stroke-2"><path d="m3.5 8.5 3 3 6-7" /></svg>}
      </button>)}
    </div>}
  </div>;
}
