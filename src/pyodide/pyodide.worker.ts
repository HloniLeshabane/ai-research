import * as Comlink from 'comlink';
import type { NdArrayMsg, RunResult, RunValue, WorkerApi } from './bridge';

// --- Pyodide loaded from the pinned CDN at runtime ---------------------------
// We use a runtime dynamic import (with @vite-ignore) of the pinned CDN module
// so Vite leaves it untouched in both dev and build, and the worker fetches it
// natively. This honours "import Pyodide from the pinned CDN (v0.29.4)" without
// pulling pyodide into the npm/Vite dependency graph.
const PYODIDE_VERSION = 'v0.29.4';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`;

// Minimal local typings for the slice of the Pyodide API we touch.
interface PyBufferView {
  data: Float32Array;
  shape: number[];
  c_contiguous: boolean;
  release(): void;
}
interface PyProxy {
  getBuffer(format?: string): PyBufferView;
  toJs(opts?: { create_proxies?: boolean }): unknown;
  destroy(): void;
}
interface PyodideInterface {
  runPythonAsync(code: string): Promise<unknown>;
  loadPackage(names: string | string[]): Promise<void>;
  globals: { get(name: string): unknown; set(name: string, value: unknown): void };
  setStdout(opts: { batched: (s: string) => void }): void;
  setStderr(opts: { batched: (s: string) => void }): void;
}

let py: PyodideInterface;

// Single-init: every call awaits the same boot promise.
const boot = (async () => {
  const mod = (await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`)) as {
    loadPyodide: (config: { indexURL: string }) => Promise<PyodideInterface>;
  };
  py = await mod.loadPyodide({ indexURL: INDEX_URL });
})();

function hasGetBuffer(o: unknown): o is PyProxy {
  return typeof o === 'object' && o !== null && typeof (o as PyProxy).getBuffer === 'function';
}
function isProxy(o: unknown): o is PyProxy {
  return typeof o === 'object' && o !== null && typeof (o as PyProxy).destroy === 'function';
}

const api: WorkerApi = {
  async ready() {
    await boot;
  },

  async ensurePackages(names: string[]) {
    await boot;
    if (names.length) await py.loadPackage(names);
  },

  async runAndExtract(code, outputs, params = {}) {
    await boot;

    // Slider values are injected as Python globals before the code runs.
    for (const [k, v] of Object.entries(params)) py.globals.set(k, v);

    let stdout = '';
    let stderr = '';
    py.setStdout({ batched: (s) => { stdout += s + '\n'; } });
    py.setStderr({ batched: (s) => { stderr += s + '\n'; } });

    try {
      await py.runPythonAsync(code);
    } catch (e) {
      return { ok: false, error: String(e), stdout, stderr, values: {} };
    }

    const values: Record<string, RunValue> = {};
    const transfers: ArrayBuffer[] = [];

    for (const name of outputs) {
      const obj = py.globals.get(name);

      if (obj == null) {
        values[name] = null;
        continue;
      }

      // Python float/int/str/bool auto-convert to JS primitives — no proxy, no leak.
      if (typeof obj !== 'object') {
        values[name] = obj as RunValue;
        continue;
      }

      if (hasGetBuffer(obj)) {
        // NumPy array: get a zero-copy VIEW into the WASM heap, copy it out ONCE,
        // then release the buffer and destroy the proxy in finally so a fast
        // slider drag can never leak WASM memory or read a stale view.
        let buf: PyBufferView | undefined;
        try {
          buf = obj.getBuffer('f32');
          if (!buf.c_contiguous) {
            throw new Error(`'${name}' is not C-contiguous; wrap it in np.ascontiguousarray(...)`);
          }
          const copy = new Float32Array(buf.data); // the single copy out of the heap
          const msg: NdArrayMsg = {
            __nd__: true,
            buffer: copy.buffer,
            shape: [...buf.shape],
            dtype: 'f32',
          };
          values[name] = msg;
          transfers.push(copy.buffer);
        } finally {
          buf?.release();
          obj.destroy();
        }
      } else if (isProxy(obj)) {
        // Some other Python object (e.g. a list) — convert and destroy.
        try {
          values[name] = obj.toJs({ create_proxies: false }) as RunValue;
        } finally {
          obj.destroy();
        }
      } else {
        values[name] = obj as RunValue;
      }
    }

    const result: RunResult = { ok: true, error: null, stdout, stderr, values };
    // Hand the ArrayBuffers to the main thread by ownership transfer (no 2nd copy).
    return Comlink.transfer(result, transfers);
  },
};

Comlink.expose(api);
