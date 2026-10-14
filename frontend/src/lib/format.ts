const numberFormat = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });

export function formatNumber(value: number): string {
  return numberFormat.format(value);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
