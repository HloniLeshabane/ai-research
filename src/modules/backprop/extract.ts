import type { NdArray } from '../../pyodide/bridge';

export interface BackpropState {
  X: [number, number][];
  y: number[];
  frames: Float32Array; // flat, length K*R*R — class-1 probability per grid cell
  K: number;
  R: number;
  frameEpochs: number[];
  loss: number[];
  extent: [number, number, number, number]; // [xmin, xmax, ymin, ymax]
  finalAcc: number;
  finalLoss: number;
  datasetName: string;
  actName: string;
  hidden: number;
  lr: number;
  diverged: boolean;
  gradcheckOk: boolean;
  gradcheckMaxErr: number;
  gradcheckWorst: string;
}

function toNum(v: unknown): number {
  return typeof v === 'number' ? v : Number(v);
}

function pairs(nd: NdArray | null): [number, number][] {
  if (!nd?.data) return [];
  const n = nd.shape[0] ?? 0;
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) out.push([nd.data[i * 2], nd.data[i * 2 + 1]]);
  return out;
}

function nums(nd: NdArray | null): number[] {
  return nd?.data ? Array.from(nd.data) : [];
}

export function extract(values: Record<string, unknown>): BackpropState {
  const framesNd = values.frames as NdArray | null;
  const shape = framesNd?.shape ?? [0, 0, 0];
  const K = shape[0] ?? 0;
  const R = shape[1] ?? 0;
  const ext = nums(values.extent as NdArray | null);
  const finalLoss = toNum(values.final_loss);

  return {
    X: pairs(values.Xdata as NdArray | null),
    y: nums(values.ydata as NdArray | null),
    frames: framesNd?.data ?? new Float32Array(0),
    K,
    R,
    frameEpochs: nums(values.frame_epochs as NdArray | null),
    loss: nums(values.loss_hist as NdArray | null),
    extent: [ext[0] ?? -1, ext[1] ?? 1, ext[2] ?? -1, ext[3] ?? 1],
    finalAcc: toNum(values.final_acc),
    finalLoss,
    datasetName: String(values.dataset_name ?? ''),
    actName: String(values.act_name ?? ''),
    hidden: toNum(values.H),
    lr: toNum(values.lr),
    diverged: !isFinite(finalLoss) || finalLoss > 1e6,
    gradcheckOk: values.gradcheck_ok === true || values.gradcheck_ok === 1,
    gradcheckMaxErr: toNum(values.gradcheck_max_err),
    gradcheckWorst: String(values.gradcheck_worst ?? ''),
  };
}
