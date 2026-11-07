import { useEffect, useRef, useState } from 'react';
import type { Config, Data, Layout } from 'plotly.js';
import type Plotly from 'plotly.js-basic-dist-min';
import { oklchToRgb } from '../lib/colormap';
import { HEADING, Panel, useTheme } from '../lib/ui';

const HEIGHT = 300;
const MAX_MARKED_POINTS = 40; // sparse series such as validations also mark each logged point
const LOG_SCALE_RATIO = 10; // positive values spanning more than this factor start on a log axis, as in the terminal dashboard
// gigachad-bot's muted categorical hues plus one, distinct for the 7 channels, darker on light paper.
// As rgb, Plotly parses no oklch
const HUES = [250, 25, 150, 320, 200, 60, 110];
const rgb = (lightness: number, chroma: number) => HUES.map((hue) => `rgb(${oklchToRgb([lightness, chroma, hue]).join(',')})`);
const COLORWAY = { light: rgb(0.52, 0.13), dark: rgb(0.72, 0.11) };
const CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d'],
  toImageButtonOptions: { format: 'svg' },
};

// Loaded once, on the first chart, keeps Plotly out of the page's first load
let plotly: Promise<{ default: typeof Plotly }> | undefined;
const loadPlotly = () => (plotly ??= import('plotly.js-basic-dist-min'));

const cssVariable = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export type ChartSeries = { label: string; values: (number | null)[] };

/** Whether all finite values are positive and span more than `LOG_SCALE_RATIO`. */
function spansDecades(series: ChartSeries[]): boolean {
  // Loops, not spreads, so tables with hundreds of thousands of rows stay within the call stack
  let smallest = Infinity;
  let largest = -Infinity;
  for (const { values } of series) {
    for (const value of values) {
      if (value === null || !Number.isFinite(value)) continue;
      if (value < smallest) smallest = value;
      if (value > largest) largest = value;
    }
  }
  return smallest > 0 && smallest !== Infinity && largest / smallest > LOG_SCALE_RATIO;
}

/**
 * One chart group of a tracker table as a Plotly figure: raw logged values, no smoothing, one y axis.
 *
 * - x spans 0 to `xMax` in every chart, `xMax` being the largest step of that x column over all tables, so sparse validations line up with the training curves
 * - Log axis when the values span decades, the axis button switches it, power-of-ten labels and minor ticks there
 * - Hover reads every series at the hovered x to 4 significant digits, drag zooms, the zoomed x range survives polls until a double-click or autoscale resets it
 * - NaN and inf arrive as null and show as gaps in the line
 * - Camera icon exports SVG
 */
export function LineChart({ title, xLabel, yLabel, x, series, xMax }: {
  title: string; xLabel: string; yLabel: string; x: number[]; series: ChartSeries[]; xMax: number;
}) {
  const element = useRef<HTMLDivElement>(null);
  const zoomedRange = useRef<[number, number]>(undefined);
  const listening = useRef<HTMLDivElement>(undefined);
  const { theme } = useTheme();
  const [logOverride, setLogOverride] = useState<boolean>();
  const logY = logOverride ?? spansDecades(series);
  const hasData = x.length > 0;

  useEffect(() => {
    const target = element.current;
    if (!target || !hasData) return;
    const [ink, inkMuted, grid, surface, font] = ['--ink', '--ink-muted', '--chart-grid', '--surface-elevated', '--font-sans'].map(cssVariable);
    const colors = COLORWAY[theme];
    const data: Data[] = series.map((line, index) => {
      const points = line.values.filter((value) => value !== null).length;
      return {
        type: 'scatter',
        name: line.label,
        x,
        y: line.values.map((value) => (value !== null && Number.isFinite(value) && (!logY || value > 0) ? value : null)),
        connectgaps: false, // null (missing, NaN or inf) breaks the line, so a non-finite value shows as a gap
        mode: points <= MAX_MARKED_POINTS ? 'lines+markers' : 'lines',
        line: { color: colors[index % colors.length], width: 1.5 },
        marker: { size: 5 },
        hovertemplate: '%{y:.4~g}',
      };
    });
    const axis = {
      gridcolor: grid, linecolor: inkMuted, tickcolor: inkMuted, showline: true, mirror: true, ticks: 'outside' as const, zeroline: false,
      title: { font: { size: 11, color: inkMuted } },
    };
    const layout: Partial<Layout> = {
      margin: { t: 8, r: 16, b: 44, l: 64 },
      modebar: { bgcolor: 'transparent', color: inkMuted, activecolor: ink },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: font, size: 11, color: ink },
      uirevision: `${title}/${logY}`,
      hovermode: 'x unified',
      hoverlabel: { bgcolor: surface, bordercolor: grid, font: { color: ink, family: font } },
      showlegend: series.length > 1,
      // Below the x axis title, Plotly grows the bottom margin to fit it
      legend: { orientation: 'h', x: 0, y: -0.22, yanchor: 'top', font: { size: 11 } },
      // 2 % padding keeps markers at x = 0 off the axis line
      xaxis: { ...axis, title: { ...axis.title, text: xLabel }, range: zoomedRange.current ?? [-0.02 * Math.max(xMax, 1), 1.02 * Math.max(xMax, 1)], separatethousands: true },
      yaxis: {
        ...axis,
        title: { ...axis.title, text: yLabel },
        type: logY ? 'log' : 'linear',
        exponentformat: 'power',
        minor: logY ? { ticks: 'outside', tickcolor: inkMuted, showgrid: true, gridcolor: grid, griddash: 'dot' } : undefined,
      },
    };
    let cancelled = false;
    void loadPlotly().then(async ({ default: Plotly }) => {
      if (cancelled) return;
      const plot = await Plotly.react(target, data, layout, CONFIG);
      if (listening.current === target) return;
      listening.current = target;
      // Keep the user's x zoom across polls, autoscale and double-click go back to the shared range
      plot.on('plotly_relayout', (event) => {
        const update = event as Record<string, unknown>;
        if (update['xaxis.autorange']) zoomedRange.current = undefined;
        else if (typeof update['xaxis.range[0]'] === 'number' && typeof update['xaxis.range[1]'] === 'number') {
          zoomedRange.current = [update['xaxis.range[0]'], update['xaxis.range[1]']];
        } else if (Array.isArray(update['xaxis.range'])) zoomedRange.current = update['xaxis.range'] as [number, number];
      });
    });
    return () => { cancelled = true; };
  }, [title, xLabel, yLabel, x, series, xMax, logY, theme, hasData]);

  useEffect(() => {
    const target = element.current;
    return () => { if (target) void loadPlotly().then(({ default: Plotly }) => Plotly.purge(target)); };
  }, [hasData]);

  return <Panel className="space-y-2 p-4">
    <div className="flex items-baseline justify-between gap-3">
      <h2 className={HEADING}>{title}</h2>
      <button onClick={() => setLogOverride(!logY)} aria-pressed={logY} title="Switch the y axis between log and linear"
        className="rounded-md px-1.5 py-0.5 text-[11px] text-ink-muted transition hover:bg-hover hover:text-ink">{logY ? 'log y' : 'linear y'}</button>
    </div>
    {hasData
      ? <div ref={element} role="figure" aria-label={title} style={{ height: HEIGHT }} />
      : <p className="grid place-items-center text-xs text-ink-muted" style={{ height: HEIGHT }}>Nothing logged yet</p>}
  </Panel>;
}
