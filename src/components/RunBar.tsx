import type { RunStatus } from '../modules/types';

const LABELS: Record<RunStatus, string> = {
  idle: 'idle',
  booting: 'booting Pyodide…',
  'loading-packages': 'loading numpy…',
  running: 'running…',
  ready: 'ready',
  error: 'error',
};

export function RunBar({
  status,
  error,
  stdout,
}: {
  status: RunStatus;
  error: string | null;
  stdout: string;
}) {
  const color =
    status === 'error'
      ? 'text-rose-400'
      : status === 'ready'
        ? 'text-emerald-400'
        : 'text-amber-400';

  return (
    <div className="border-t border-neutral-800 bg-neutral-900 px-3 py-2 font-mono text-xs">
      <span className="text-neutral-500">status: </span>
      <span className={color}>{LABELS[status]}</span>
      {error && <pre className="mt-1 whitespace-pre-wrap text-rose-400">{error}</pre>}
      {!error && stdout.trim() && (
        <pre className="mt-1 whitespace-pre-wrap text-neutral-400">{stdout.trim()}</pre>
      )}
    </div>
  );
}
