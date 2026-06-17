import { createContext, useEffect, useState, type ReactNode } from 'react';
import * as Comlink from 'comlink';
import type { WorkerApi } from './bridge';

type BootStatus = 'booting' | 'ready' | 'error';

interface PyodideContextValue {
  api: Comlink.Remote<WorkerApi> | null;
  status: BootStatus;
  error: string | null;
}

export const PyodideContext = createContext<PyodideContextValue>({
  api: null,
  status: 'booting',
  error: null,
});

// Module-level singleton: survives React StrictMode's mount/unmount/mount in dev
// so we only ever spawn one worker and call loadPyodide once.
let singleton: { worker: Worker; api: Comlink.Remote<WorkerApi> } | null = null;
function getSingleton() {
  if (!singleton) {
    const worker = new Worker(new URL('./pyodide.worker.ts', import.meta.url), {
      type: 'module',
    });
    singleton = { worker, api: Comlink.wrap<WorkerApi>(worker) };
  }
  return singleton;
}

export function PyodideProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<BootStatus>('booting');
  const [error, setError] = useState<string | null>(null);
  const { api } = getSingleton();

  useEffect(() => {
    let cancelled = false;
    api.ready().then(
      () => !cancelled && setStatus('ready'),
      (e: unknown) => {
        if (!cancelled) {
          setError(String(e));
          setStatus('error');
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api]);

  return (
    <PyodideContext.Provider value={{ api, status, error }}>{children}</PyodideContext.Provider>
  );
}
