const numberFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });
const scientificFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 3, notation: 'scientific' });
const tickFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 3 });
const SUPERSCRIPTS: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

/** 4 significant digits, scientific below 1e-4 so near-zero values stay short. */
export function formatNumber(value: number): string {
  return value !== 0 && Math.abs(value) < 1e-4 ? scientificFormat.format(value) : numberFormat.format(value);
}

/** `10⁻³` style power of ten. */
export function formatPower(exponent: number): string {
  return `10${[...String(exponent)].map((character) => SUPERSCRIPTS[character]).join('')}`;
}

/** Axis and colorbar label: 3 significant digits, `1.23×10⁻⁴` outside [0.01, 10⁴). */
export function formatTick(value: number): string {
  if (value === 0 || (Math.abs(value) >= 0.01 && Math.abs(value) < 1e4)) return tickFormat.format(value);
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = tickFormat.format(value / 10 ** exponent);
  const power = formatPower(exponent);
  return mantissa === '1' ? power : mantissa === '-1' ? `−${power}` : `${mantissa}×${power}`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
