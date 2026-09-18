import type { NdArray } from '../../pyodide/bridge';

export interface GDState {
  traj: [number, number][];
  pole: number;
  finalLoss: number;
  kappa: number;
  lr: number;
  diverged: boolean;
}

function toNum(v: unknown): number {
  return typeof v === 'number' ? v : Number(v);
}

export function extract(values: Record<string, unknown>): GDState {
  const nd = values.traj as NdArray | null;
  const flat = nd?.data ? Array.from(nd.data) : [];
  const n = nd?.shape?.[0] ?? 0;
  const traj: [number, number][] = [];
  for (let i = 0; i < n; i++) traj.push([flat[i * 2], flat[i * 2 + 1]]);
  const finalLoss = toNum(values.final_loss);
  return {
    traj,
    pole: toNum(values.pole),
    finalLoss,
    kappa: toNum(values.kappa),
    lr: toNum(values.lr),
    diverged: !isFinite(finalLoss) || finalLoss > 1e6,
  };
}
