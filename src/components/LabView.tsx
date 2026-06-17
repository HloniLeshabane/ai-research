import { useEffect, useRef, useState } from 'react';
import { usePyodide } from '../pyodide/usePyodide';
import { useRunner } from '../pyodide/useRunner';
import { useDebouncer } from '../lib/debounce';
import { SplitPane } from './SplitPane';
import { CodeEditor } from './CodeEditor';
import { RunBar } from './RunBar';
import { SliderRow } from './controls/SliderRow';
import type { LabModule, RunStatus } from '../modules/types';

export function LabView<TState>({ module }: { module: LabModule<TState> }) {
  const { status: pyStatus } = usePyodide();
  const { state, runStatus, error, stdout, run } = useRunner(module);
  const [code, setCode] = useState(module.defaultCode);
  const [params, setParams] = useState<Record<string, number>>(() =>
    Object.fromEntries((module.controls ?? []).map((c) => [c.name, c.default])),
  );
  const debounce = useDebouncer();
  const didInit = useRef(false);

  // Initial run once Pyodide is ready.
  useEffect(() => {
    if (pyStatus === 'ready' && !didInit.current) {
      didInit.current = true;
      void run(code, params);
    }
  }, [pyStatus, run, code, params]);

  const onCode = (v: string) => {
    setCode(v);
    debounce(() => void run(v, params), 400); // editing re-runs after you pause
  };
  const onParam = (name: string, value: number) => {
    const next = { ...params, [name]: value };
    setParams(next);
    debounce(() => void run(code, next), 60); // sliders feel instant
  };

  const displayStatus: RunStatus = pyStatus === 'ready' ? runStatus : 'booting';
  const Visual = module.Visual;

  return (
    <SplitPane
      left={
        <div className="flex h-full flex-col">
          <div className="min-h-0 flex-1 overflow-auto">
            <CodeEditor value={code} onChange={onCode} />
          </div>
          <RunBar status={displayStatus} error={error} stdout={stdout} />
        </div>
      }
      right={
        <div className="flex h-full flex-col bg-neutral-950">
          <div className="min-h-0 flex-1">
            <Visual state={state} status={runStatus} />
          </div>
          {module.controls && module.controls.length > 0 && (
            <div className="border-t border-neutral-800 bg-neutral-900 px-4 py-3">
              {module.controls.map((c) => (
                <SliderRow
                  key={c.name}
                  spec={c}
                  value={params[c.name]}
                  onChange={(v) => onParam(c.name, v)}
                />
              ))}
            </div>
          )}
        </div>
      }
    />
  );
}
