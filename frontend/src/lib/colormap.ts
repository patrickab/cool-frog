import { nonDegenerate } from './range';
import type { Range } from './range';

type Rgb = readonly [number, number, number];
/** How a field maps to color: linear, or logarithmic for positive values spanning decades. */
export type ScaleKind = 'linear' | 'log';
/**
 * Perceptually ordered colormaps, 9 stops sampled from matplotlib.
 *
 * - `sequential`: viridis, monotonic lightness, colorblind-safe, no grey midtones that would read as solids
 * - `diverging`: RdBu reversed, light at 0, blue negative, red positive, for signed errors
 * - `magnitude`: Reds, light at 0, for nonnegative errors
 */
export type Colormap = 'sequential' | 'diverging' | 'magnitude';
const COLORMAPS: Record<Colormap, readonly Rgb[]> = {
  sequential: [[68, 1, 84], [71, 45, 123], [59, 82, 139], [44, 114, 142], [33, 145, 140], [40, 174, 128], [94, 201, 98], [173, 220, 48], [253, 231, 37]],
  diverging: [[5, 48, 97], [42, 113, 178], [107, 172, 209], [194, 221, 236], [247, 246, 246], [251, 204, 180], [228, 128, 102], [186, 40, 50], [103, 0, 31]],
  magnitude: [[255, 245, 240], [254, 224, 210], [252, 187, 161], [252, 146, 114], [251, 105, 74], [238, 58, 44], [202, 24, 29], [163, 15, 21], [103, 0, 13]],
};
/** Colormap over `range`, `log` maps log10 of the value, values outside `range` clip to the end colors. */
export type ColorScale = { colormap: Colormap; range: Range; log: boolean };

export const gradientCss = (colormap: Colormap) => `linear-gradient(90deg, ${COLORMAPS[colormap].map((color) => `rgb(${color.join(',')})`).join(', ')})`;

/** Position of `value` on `scale` in [0, 1], clipped. */
export function scalePosition(value: number, { range, log }: ColorScale): number {
  const transform = log ? (input: number) => Math.log10(Math.max(input, Number.MIN_VALUE)) : (input: number) => input;
  const [min, max] = nonDegenerate([transform(range[0]), transform(range[1])]);
  return Math.max(0, Math.min(1, (transform(value) - min) / (max - min)));
}

/** Value at `position` in [0, 1] of `scale`, the inverse of `scalePosition`. */
export function scaleValue(position: number, { range, log }: ColorScale): number {
  return log ? range[0] * (range[1] / range[0]) ** position : range[0] + (range[1] - range[0]) * position;
}

export function colorAt(value: number, scale: ColorScale): Rgb {
  const stops = COLORMAPS[scale.colormap];
  const position = scalePosition(value, scale) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const fraction = position - index;
  return stops[index].map((component, axis) => Math.round(component + (stops[index + 1][axis] - component) * fraction)) as unknown as Rgb;
}

/** Evenly spaced stops of `colormap` from position 0 to 1, for renderers that interpolate themselves. */
export const colormapStops = (colormap: Colormap): readonly Rgb[] => COLORMAPS[colormap];

/** Linear scales end at this quantile so a few extreme voxels (e.g. hot walls) don't compress the rest into one color. */
const LINEAR_UPPER_QUANTILE = 0.99;

/**
 * Scale of `kind` over finite `values`.
 *
 * - `linear`: from the minimum to the 99th percentile, higher values clip to the top color
 * - `log`: from the 1st percentile of positive values, floors such as k = 1e-10 clip to the bottom color
 * - `trim`: fraction dropped at each end before taking the range, e.g. 0.01 for surface faces with a few extremes
 */
export function colorScaleOf(kind: ScaleKind, values: Float32Array, trim = 0): ColorScale | undefined {
  if (!values.length) return undefined;
  const sorted = values.slice().sort();
  const quantile = (fraction: number) => sorted[Math.round(fraction * (sorted.length - 1))];
  const [min, max] = [quantile(trim), quantile(1 - trim)];
  if (kind === 'log' && max > 0) {
    const firstPositive = sorted.findIndex((value) => value > 0);
    const floor = sorted[firstPositive + Math.floor(0.01 * (sorted.length - 1 - firstPositive))];
    if (floor < max) return { colormap: 'sequential', range: [floor, max], log: true };
  }
  const upper = quantile(Math.min(1 - trim, LINEAR_UPPER_QUANTILE));
  return { colormap: 'sequential', range: [min, upper > min ? upper : max], log: false };
}

/** Tensor axis kinds of the Tadpole shape notation, one fixed color each in formulas and 3D scenes. */
export type Axis = 'batch' | 'channel' | 'crop' | 'spatial' | 'feature';
type Oklch = readonly [number, number, number]
type ThemeColors = { surface: Oklch; ink: Oklch; solid: Oklch; outline: Oklch; axes: Record<Axis, Oklch> }
/** Renderer colors mirror CSS roles, axis hues retain their meaning in both themes. */
const THEME_COLORS: Record<'light' | 'dark', ThemeColors> = {
  light: {
    surface: [0.975, 0.012, 85], ink: [0.3, 0.012, 85], solid: [0.84, 0.012, 85], outline: [0.72, 0.008, 85],
    axes: { batch: [0.5, 0.13, 255], channel: [0.5, 0.12, 150], crop: [0.52, 0.12, 55], spatial: [0.42, 0.015, 85], feature: [0.5, 0.15, 320] },
  },
  dark: {
    surface: [0.235, 0.008, 85], ink: [0.92, 0.012, 85], solid: [0.44, 0.008, 85], outline: [0.55, 0.008, 85],
    axes: { batch: [0.76, 0.10, 255], channel: [0.76, 0.10, 150], crop: [0.78, 0.10, 55], spatial: [0.78, 0.015, 85], feature: [0.76, 0.12, 320] },
  },
}

export const themeColors = (theme: 'light' | 'dark') => THEME_COLORS[theme]

export const oklchCss = ([lightness, chroma, hue]: readonly number[]) => `oklch(${lightness} ${chroma} ${hue})`;

/** OKLCH to linear sRGB in [0, 1], the working color space of three.js. */
export function oklchToLinear([lightness, chroma, hue]: readonly number[]): [number, number, number] {
  const a = chroma * Math.cos((hue * Math.PI) / 180);
  const b = chroma * Math.sin((hue * Math.PI) / 180);
  const long = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const medium = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const short = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
    -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
    -0.0041960863 * long - 0.7034186147 * medium + 1.707614701 * short,
  ].map((component) => Math.min(1, Math.max(0, component))) as [number, number, number];
}

/** Convert theme colors to 8-bit sRGB for canvas pixels and vtk.js. */
export function oklchToRgb(color: readonly number[]): [number, number, number] {
  return oklchToLinear(color).map((component) => Math.round(255 * (component <= 0.0031308
    ? 12.92 * component
    : 1.055 * component ** (1 / 2.4) - 0.055))) as [number, number, number]
}

/** Whether any of `values` falls below or above the scale and shows in an end color. */
export function clipping(values: Float32Array, { range }: ColorScale): { below: boolean; above: boolean } {
  let [below, above] = [false, false];
  for (const value of values) { below ||= value < range[0]; above ||= value > range[1]; }
  return { below, above };
}
