// Shared, framework-free protocol between the Pyodide worker and the main thread.
// The worker imports ONLY from this file (no React), so its bundle stays clean.

export type Dtype = 'f32';

// Reconstructed, ready-to-render array on the main thread.
export interface NdArray {
  data: Float32Array;
  shape: number[];
  dtype: Dtype;
}

// Wire format: the transferred ArrayBuffer plus its shape. `__nd__` tags it so
// reviveNdArrays can tell arrays apart from plain scalars.
export interface NdArrayMsg {
  __nd__: true;
  buffer: ArrayBuffer;
  shape: number[];
  dtype: Dtype;
}

export type RunValue = number | boolean | string | null | NdArrayMsg | unknown;

export interface RunResult {
  ok: boolean;
  error: string | null;
  stdout: string;
  stderr: string;
  values: Record<string, RunValue>;
}

// The surface Comlink exposes from the worker.
export interface WorkerApi {
  ready(): Promise<void>;
  ensurePackages(names: string[]): Promise<void>;
  runAndExtract(
    code: string,
    outputs: string[],
    params?: Record<string, number>,
  ): Promise<RunResult>;
}

export function isNdArrayMsg(v: unknown): v is NdArrayMsg {
  return typeof v === 'object' && v !== null && (v as { __nd__?: unknown }).__nd__ === true;
}

// Turn the wire `values` into render-ready values: NdArrayMsg -> NdArray, scalars untouched.
export function reviveNdArrays(
  values: Record<string, RunValue>,
): Record<string, number | boolean | string | null | NdArray | unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(values)) {
    out[key] = isNdArrayMsg(v)
      ? ({ data: new Float32Array(v.buffer), shape: v.shape, dtype: v.dtype } satisfies NdArray)
      : v;
  }
  return out;
}
