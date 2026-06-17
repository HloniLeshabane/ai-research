import type { NdArray } from '../../pyodide/bridge';

export interface PerceptronState {
  x: number[];
  w: number[];
  contributions: number[];
  activation: number;
  bias: number;
}

function toArr(v: unknown): number[] {
  const nd = v as NdArray | null;
  return nd?.data ? Array.from(nd.data) : [];
}

function toNum(v: unknown): number {
  return typeof v === 'number' ? v : Number(v);
}

export function extract(values: Record<string, unknown>): PerceptronState {
  return {
    x: toArr(values.X),
    w: toArr(values.W),
    contributions: toArr(values.contributions),
    activation: toNum(values.activation),
    bias: toNum(values.bias),
  };
}
