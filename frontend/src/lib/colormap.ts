import { nonDegenerate } from './range';
import type { Range } from './range';

export const colors = [[25, 55, 105], [30, 117, 151], [49, 185, 166], [184, 220, 137], [250, 221, 106], [244, 134, 74]] as const;
// `--solid` from style.css
export const solid = [206, 202, 194] as const;
export const gradient = `linear-gradient(90deg, ${colors.map((color) => `rgb(${color.join(',')})`).join(', ')})`;

export function colorAt(value: number, range: Range): readonly number[] {
  const [min, max] = nonDegenerate(range);
  const position = Math.max(0, Math.min(1, (value - min) / (max - min))) * (colors.length - 1);
  const index = Math.min(Math.floor(position), colors.length - 2);
  const fraction = position - index;
  return colors[index].map((component, axis) => Math.round(component + (colors[index + 1][axis] - component) * fraction));
}
