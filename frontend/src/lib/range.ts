/** Inclusive numeric interval. */
export type Range = [min: number, max: number];
/** Named value with its full bounds, e.g. a parameter or dataset column. */
export type Bounded = { name: string; unit: string | null; min: number; max: number };

/** Widen a zero-width range so it can be divided by. */
export function nonDegenerate([min, max]: Range): Range {
  return min === max ? [min, min + Math.max(Math.abs(min) * 0.01, 1)] : [min, max];
}

/** Store `range` for `bounded` only while it excludes part of its bounds. */
export function withRange(narrowed: Record<string, Range>, bounded: Bounded, range: Range): Record<string, Range> {
  const { [bounded.name]: _previous, ...rest } = narrowed;
  return range[0] > bounded.min || range[1] < bounded.max ? { ...rest, [bounded.name]: range } : rest;
}
