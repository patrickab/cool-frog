/** Fetch `url`, POSTing `body` as JSON when given, throw the server's `detail` or text unless OK. */
export async function fetchOk(url: string, signal: AbortSignal, body?: unknown): Promise<Response> {
  const response = await fetch(url, {
    signal,
    ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  });
  if (response.ok) return response;
  const text = await response.text();
  let message = text;
  try {
    const error = JSON.parse(text);
    message = typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail ?? error);
  } catch { /* A proxy or upstream server may return plain text. */ }
  throw new Error(message || `Request failed (${response.status})`);
}

export async function fetchJson<T>(url: string, signal: AbortSignal, body?: unknown): Promise<T> {
  return (await fetchOk(url, signal, body)).json() as Promise<T>;
}
