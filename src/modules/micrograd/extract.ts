// The Python side emits JSON strings (no numpy in this module — the graph is
// ~40 scalar nodes and the trace ~270 events, so JSON is simpler and boots
// without a package download).

export type NodeKind = 'input' | 'const' | 'param' | 'op';

export interface GraphNode {
  id: string;
  kind: NodeKind;
  op: string; // '' for leaves; 'add' | 'mul' | 'tanh' | 'sub' | 'pow2' for ops
  label: string;
  d0: number; // node .data at the start of the traced epoch
}

export interface BwdContrib {
  id: string; // the child receiving the gradient
  d: number; // the += amount this op contributed
  g: number; // the child's accumulated grad after the +=
}

// One replayable debugger event. Which fields are set depends on `t`:
//   sample: si, x, y · fwd: id, v · seed: id · bwd: id, g, cs ·
//   upd: id, old, g, new · end: k, loss · zero: (none)
export interface TraceEvent {
  t: 'sample' | 'fwd' | 'zero' | 'seed' | 'bwd' | 'upd' | 'end';
  si?: number;
  x?: [number, number];
  y?: number;
  id?: string;
  v?: number;
  g?: number;
  cs?: BwdContrib[];
  old?: number;
  new?: number;
  k?: number;
  loss?: number;
}

export interface MicrogradState {
  nodes: GraphNode[];
  edges: [string, string][]; // deduped — d·d contributes a single d→L edge
  lossIds: string[]; // op nodes downstream of the target y; rendered as the LOSS phase
  events: TraceEvent[];
  surfaces: number[][]; // 1 + 4 frames, each R*R, row 0 = top, values in [-1, 1]
  R: number;
  dom: number; // surface half-extent in input space
  data: [number, number, number][]; // [x1, x2, y] per training sample
  loss: number[]; // mean loss per epoch, full run
  tracedEpoch: number;
  lr: number;
  finalLoss: number;
  finalAcc: number;
  gradcheckOk: boolean;
  gradcheckErr: number;
  gradcheckWorst: string; // label of the param with the worst analytic-vs-numeric error
}

function toNum(v: unknown): number {
  return typeof v === 'number' ? v : Number(v);
}

function parse<T>(v: unknown, fallback: T): T {
  if (typeof v !== 'string') return fallback;
  try {
    return JSON.parse(v) as T;
  } catch {
    return fallback;
  }
}

export function extract(values: Record<string, unknown>): MicrogradState {
  const structure = parse<{ nodes: GraphNode[]; edges: [string, string][]; loss?: string[] } | null>(
    values.structure_json,
    null,
  );
  return {
    nodes: structure?.nodes ?? [],
    edges: structure?.edges ?? [],
    lossIds: structure?.loss ?? [],
    events: parse<TraceEvent[]>(values.trace_json, []),
    surfaces: parse<number[][]>(values.surfaces_json, []),
    R: toNum(values.surf_r),
    dom: toNum(values.surf_dom),
    data: parse<[number, number, number][]>(values.data_json, []),
    loss: parse<number[]>(values.loss_json, []),
    tracedEpoch: toNum(values.traced_epoch),
    lr: toNum(values.lr_used),
    finalLoss: toNum(values.final_loss),
    finalAcc: toNum(values.final_acc),
    gradcheckOk: values.gradcheck_ok === true || values.gradcheck_ok === 1,
    gradcheckErr: toNum(values.gradcheck_err),
    gradcheckWorst: String(values.gradcheck_worst ?? ''),
  };
}
