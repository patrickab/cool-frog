const numberFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });
const scientificFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 3, notation: 'scientific' });

/** 4 significant digits, scientific below 1e-4 so near-zero values stay short. */
export function formatNumber(value: number): string {
  return value !== 0 && Math.abs(value) < 1e-4 ? scientificFormat.format(value) : numberFormat.format(value);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
