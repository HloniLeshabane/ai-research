import { useCallback, useRef, useState } from 'react';
import { reviveNdArrays } from './bridge';
import { usePyodide } from './usePyodide';
import type { LabModule, RunStatus } from '../modules/types';

// Owns execution for one module: loads its packages once, runs code, and
// guards against stale results so fast slider drags can't paint an old frame.
export function useRunner<TState>(module: LabModule<TState>) {
  const { api } = usePyodide();
  const [state, setState] = useState<TState | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [stdout, setStdout] = useState('');
  const runId = useRef(0);
  const packagesLoaded = useRef(false);

  const run = useCallback(
    async (code: string, params: Record<string, number>) => {
      if (!api) return;
      const myId = ++runId.current;
      try {
        if (!packagesLoaded.current && module.packages.length) {
          setRunStatus('loading-packages');
          await api.ensurePackages(module.packages);
          packagesLoaded.current = true;
        }
        setRunStatus('running');
        const res = await api.runAndExtract(code, module.outputs, params);
        if (myId !== runId.current) return; // a newer run superseded this one
        setStdout(res.stdout + res.stderr);
        if (!res.ok) {
          setError(res.error);
          setRunStatus('error');
          return;
        }
        setError(null);
        setState(module.extract(reviveNdArrays(res.values)));
        setRunStatus('ready');
      } catch (e) {
        if (myId === runId.current) {
          setError(String(e));
          setRunStatus('error');
        }
      }
    },
    [api, module],
  );

  return { state, runStatus, error, stdout, run };
}
