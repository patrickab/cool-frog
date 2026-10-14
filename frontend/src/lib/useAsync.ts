import { useEffect, useState } from 'react';
import type { DependencyList } from 'react';
import { errorMessage } from './format';

export type AsyncState<T> = { status: 'idle' } | { status: 'loading' } | { status: 'done'; data: T } | { status: 'error'; error: string };

/** Run `load` whenever `deps` change, aborting the previous run. No `load` = idle. */
export function useAsync<T>(load: ((signal: AbortSignal) => Promise<T>) | undefined, deps: DependencyList): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: load ? 'loading' : 'idle' });
  useEffect(() => {
    if (!load) {
      setState({ status: 'idle' });
      return;
    }
    const controller = new AbortController();
    const settle = (next: AsyncState<T>) => { if (!controller.signal.aborted) setState(next); };
    setState({ status: 'loading' });
    load(controller.signal).then((data) => settle({ status: 'done', data }), (error) => settle({ status: 'error', error: errorMessage(error) }));
    return () => controller.abort();
  }, deps);
  return state;
}
