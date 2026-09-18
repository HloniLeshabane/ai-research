import { useEffect, useMemo, useRef, useState } from 'react';
import type { VisualProps } from '../types';
import type { GraphNode, MicrogradState, TraceEvent } from './extract';

// data / grad / param / loss — the micrograd color convention throughout
const FWD = '#60a5fa'; // blue-400
const GRAD = '#fbbf24'; // amber-400
const PARAM = '#a78bfa'; // violet-400
const LOSS = '#fb7185'; // rose-400
const INK = '#e5e7eb';
const MUTED = '#6b7280';
const EDGE = '#3a4150';

const NW = 112;
const NH = 46;
const COLW = 142;
const ROWH = 60;
const PAD = 14;

const OPSYM: Record<string, string> = { mul: '×', add: '+', tanh: 'tanh', sub: '−', pow2: '( )²' };

type Phase = 'forward' | 'loss' | 'backward' | 'update';
const PHASES: Phase[] = ['forward', 'loss', 'backward', 'update'];
const PHASE_COLOR: Record<Phase, string> = { forward: FWD, loss: LOSS, backward: GRAD, update: PARAM };

const fmt = (v: number | null | undefined) => (v == null ? '—' : (Object.is(v, -0) ? 0 : v).toFixed(3));

// ---------------------------------------------------------------------------
// Layered DAG layout: ops at 1 + max(input col); leaves hug their consumers.
// Generic over whatever graph the Python side emitted.
// ---------------------------------------------------------------------------
interface Laid extends GraphNode {
  inputs: string[];
  col: number;
  row: number;
  x: number;
  y: number;
}

function layoutGraph(nodes: GraphNode[], edges: [string, string][]) {
  const byId = new Map<string, Laid>();
  nodes.forEach((n) => byId.set(n.id, { ...n, inputs: [], col: 0, row: 0, x: 0, y: 0 }));
  const consumers = new Map<string, string[]>();
  for (const [s, d] of edges) {
    byId.get(d)?.inputs.push(s);
    if (!consumers.has(s)) consumers.set(s, []);
    consumers.get(s)!.push(d);
  }
  // nodes arrive in topo order, so inputs always have their col before we need it
  for (const n of nodes) {
    const ln = byId.get(n.id)!;
    if (ln.kind === 'op') ln.col = 1 + Math.max(0, ...ln.inputs.map((i) => byId.get(i)?.col ?? 0));
  }
  for (const n of nodes) {
    const ln = byId.get(n.id)!;
    if (ln.kind !== 'op') {
      const cons = (consumers.get(n.id) ?? []).map((c) => byId.get(c)!.col);
      ln.col = cons.length ? Math.min(...cons) - 1 : 0;
    }
  }
  const cols = new Map<number, Laid[]>();
  for (const n of nodes) {
    const ln = byId.get(n.id)!;
    if (!cols.has(ln.col)) cols.set(ln.col, []);
    cols.get(ln.col)!.push(ln);
  }
  const maxCol = Math.max(0, ...[...cols.keys()]);
  let maxRows = 0;
  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
  const key = new Map<string, number>();
  for (let c = 0; c <= maxCol; c++) {
    const list = cols.get(c) ?? [];
    const ops = list.filter((n) => n.kind === 'op');
    ops.sort((a, b) => mean(a.inputs.map((i) => byId.get(i)?.row ?? 0)) - mean(b.inputs.map((i) => byId.get(i)?.row ?? 0)));
    ops.forEach((n, i) => key.set(n.id, i));
    const leaves = list.filter((n) => n.kind !== 'op');
    leaves.forEach((n, i) => {
      if (c === 0) {
        key.set(n.id, i);
      } else {
        // sit just below the op this leaf feeds (its consumer's other input)
        const con = byId.get((consumers.get(n.id) ?? [])[0] ?? '');
        const sib = con?.inputs.map((i2) => byId.get(i2)!).find((s) => s.id !== n.id && s.col === c && s.kind === 'op');
        key.set(n.id, (sib ? key.get(sib.id)! : ops.length) + 0.5);
      }
    });
    const all = [...ops, ...leaves].sort((a, b) => key.get(a.id)! - key.get(b.id)!);
    all.forEach((n, i) => (n.row = i));
    maxRows = Math.max(maxRows, all.length);
  }
  for (const n of byId.values()) {
    const count = (cols.get(n.col) ?? []).length || 1;
    n.x = PAD + n.col * COLW;
    n.y = PAD + ((maxRows - count) / 2) * ROWH + n.row * ROWH;
  }
  return { byId, consumers, width: PAD * 2 + maxCol * COLW + NW, height: PAD * 2 + maxRows * ROWH };
}

// ---------------------------------------------------------------------------
// Replay: fold events[0..cursor) into a display state + explainer for the
// last-applied event. ~270 events, so recomputing per step is free.
// ---------------------------------------------------------------------------
interface ExplainLine {
  rule?: string;
  nums: string;
}
interface Replay {
  data: Record<string, number | undefined>;
  grad: Record<string, number | undefined>;
  stale: Record<string, boolean>;
  active: string[];
  hot: Set<string>; // `${src}>${dst}` edges lit by the current event
  hotBack: boolean;
  phase: Phase | null;
  title: string;
  lines: ExplainLine[];
  note: string;
  sampleIdx: number | null;
  itersDone: number;
  lastLoss: number | null;
}

const FWD_NOTES: Record<string, string> = {
  mul: 'Multiply — and (crucially for later) each input’s local derivative is just the other input.',
  add: 'Add. During backprop this routes the gradient through untouched.',
  tanh: 'Squash into (−1, 1). The nonlinearity that lets the boundary bend.',
  sub: 'The signed error: how far the prediction missed, and in which direction.',
  pow2: 'Squared error — always positive, punishes big misses hardest. This one number is what training shrinks.',
};

function evPhase(ev: TraceEvent, lossSet: Set<string>): Phase {
  switch (ev.t) {
    case 'sample':
      return 'forward';
    case 'fwd':
      return lossSet.has(ev.id!) ? 'loss' : 'forward';
    case 'zero':
    case 'seed':
    case 'bwd':
      return 'backward';
    default:
      return 'update';
  }
}

function buildReplay(state: MicrogradState, byId: Map<string, Laid>, lossSet: Set<string>, cursor: number): Replay {
  const r: Replay = {
    data: {},
    grad: {},
    stale: {},
    active: [],
    hot: new Set(),
    hotBack: false,
    phase: null,
    title: 'press step to begin',
    lines: [{ nums: `replaying epoch ${state.tracedEpoch} — 4 SGD steps, one event at a time` }],
    note: 'Forward fills the graph left to right, the loss measures the miss, backward runs the chain rule right to left, update nudges every violet parameter downhill. Click any node to inspect how its numbers are made.',
    sampleIdx: null,
    itersDone: 0,
    lastLoss: null,
  };
  for (const n of state.nodes) if (n.kind !== 'op') r.data[n.id] = n.d0;

  const lbl = (id: string | undefined) => (id && byId.get(id)?.label) || id || '?';

  for (let i = 0; i < cursor && i < state.events.length; i++) {
    const ev = state.events[i];
    const last = i === cursor - 1;
    if (last) {
      r.phase = evPhase(ev, lossSet);
      r.active = ev.id ? [ev.id] : [];
      r.hot = new Set();
      r.hotBack = ev.t === 'bwd';
      r.lines = [];
      r.note = '';
    }
    const n = ev.id ? byId.get(ev.id) : undefined;
    switch (ev.t) {
      case 'sample': {
        const inputs = state.nodes.filter((nd) => nd.kind === 'input');
        const target = state.nodes.find((nd) => nd.kind === 'const');
        inputs.forEach((nd, k) => (r.data[nd.id] = ev.x![k]));
        if (target) r.data[target.id] = ev.y;
        state.nodes.forEach((nd) => {
          if (nd.kind === 'op') r.stale[nd.id] = true;
        });
        r.sampleIdx = ev.si!;
        if (last) {
          r.active = [...inputs.map((nd) => nd.id), ...(target ? [target.id] : [])];
          r.title = `load sample ${ev.si! + 1}/4`;
          r.lines = [{ nums: `x = (${ev.x![0]}, ${ev.x![1]})   target y = ${ev.y}` }];
          r.note = 'Fresh inputs in the leaves; everything downstream is stale (dimmed) until the forward pass recomputes it.';
        }
        break;
      }
      case 'fwd': {
        r.data[ev.id!] = ev.v;
        r.stale[ev.id!] = false;
        if (last && n) {
          // a mul with one (deduped) input is a self-product like L = d·d
          const selfProduct = n.op === 'mul' && n.inputs.length === 1;
          const [a, b0] = n.inputs;
          const b = selfProduct ? a : b0;
          n.inputs.forEach((s) => r.hot.add(`${s}>${n.id}`));
          const da = fmt(r.data[a]);
          const db = b ? fmt(r.data[b]) : '';
          const v = fmt(ev.v);
          if (n.op === 'mul') { r.title = `${n.label} = ${lbl(a)} × ${lbl(b)}`; r.lines = [{ nums: `= ${da} × ${db} = ${v}` }]; }
          else if (n.op === 'add') { r.title = `${n.label} = ${lbl(a)} + ${lbl(b)}`; r.lines = [{ nums: `= ${da} + ${db} = ${v}` }]; }
          else if (n.op === 'sub') { r.title = `${n.label} = ${lbl(a)} − ${lbl(b)}`; r.lines = [{ nums: `= ${da} − ${db} = ${v}` }]; }
          else if (n.op === 'tanh') { r.title = `${n.label} = tanh(${lbl(a)})`; r.lines = [{ nums: `= tanh(${da}) = ${v}` }]; }
          else if (n.op === 'pow2') { r.title = `${n.label} = ${lbl(a)}²`; r.lines = [{ nums: `= (${da})² = ${v}` }]; }
          else { r.title = `${n.label} = ${n.op}(…)`; r.lines = [{ nums: `= ${v}` }]; }
          r.note = selfProduct
            ? 'Both factors are the same node — squared error as d·d. Watch backprop here: d will receive its gradient TWICE, and the += is what makes that come out as 2d.'
            : FWD_NOTES[n.op] ?? '';
        }
        break;
      }
      case 'zero': {
        state.nodes.forEach((nd) => (r.grad[nd.id] = 0));
        if (last) {
          r.title = 'zero_grad()';
          r.lines = [{ nums: 'every node.grad ← 0' }];
          r.note = 'Gradients accumulate with +=, so the slate is wiped before each backward pass. Forgetting this is the classic bug.';
        }
        break;
      }
      case 'seed': {
        r.grad[ev.id!] = 1;
        r.stale[ev.id!] = false;
        if (last) {
          r.title = '∂L/∂L = 1';
          r.lines = [{ nums: 'L.grad ← 1.000' }];
          r.note = 'The seed of backprop: the loss moves 1:1 with itself. Every other grad answers “nudge me — how much does L move?”';
        }
        break;
      }
      case 'bwd': {
        for (const c of ev.cs ?? []) {
          r.grad[c.id] = c.g;
          r.stale[c.id] = false;
        }
        if (last && n) {
          n.inputs.forEach((s) => r.hot.add(`${s}>${n.id}`));
          r.title = `backprop ${n.label}  (${OPSYM[n.op] ?? n.op})`;
          const g = fmt(ev.g);
          const nv = r.data[n.id];
          r.lines = (ev.cs ?? []).map((c, k) => {
            const child = byId.get(c.id);
            const cl = child?.label ?? c.id;
            const other = n.inputs.find((s) => s !== c.id);
            const arrow = `→ ${cl}.grad = ${fmt(c.g)}`;
            if (n.op === 'mul' && !other) {
              // self-product (d·d): the child sits in both slots, so the
              // += fires twice — its one deduped entry carries 2·d
              const dv = r.data[c.id];
              return { rule: `∂L/∂${cl} += 2·${cl} · ∂L/∂${n.label}   (both slots)`, nums: `= ${fmt(dv == null ? null : 2 * dv)} × ${g} ${arrow}` };
            }
            if (n.op === 'mul') return { rule: `∂L/∂${cl} += ${lbl(other)} · ∂L/∂${n.label}`, nums: `= ${fmt(r.data[other ?? ''])} × ${g} ${arrow}` };
            if (n.op === 'add') return { rule: `∂L/∂${cl} += ∂L/∂${n.label}`, nums: `= ${g} ${arrow}` };
            if (n.op === 'tanh') return { rule: `∂L/∂${cl} += (1 − ${n.label}²) · ∂L/∂${n.label}`, nums: `= ${fmt(nv == null ? null : 1 - nv * nv)} × ${g} ${arrow}` };
            if (n.op === 'sub') return k === 0
              ? { rule: `∂L/∂${cl} += ∂L/∂${n.label}`, nums: `= ${g} ${arrow}` }
              : { rule: `∂L/∂${cl} −= ∂L/∂${n.label}`, nums: `= −(${g}) ${arrow} (y is data — computed, never used)` };
            if (n.op === 'pow2') return { rule: `∂L/∂${cl} += 2·${cl} · ∂L/∂${n.label}`, nums: `= ${fmt(r.data[c.id] == null ? null : 2 * r.data[c.id]!)} × ${g} ${arrow}` };
            return { rule: `∂L/∂${cl} += Δ`, nums: `= ${fmt(c.d)} ${arrow}` };
          });
          r.note =
            n.op === 'mul' && n.inputs.length === 1
              ? 'The a·a moment from the video: d is self AND other, so `self.grad +=` and `other.grad +=` both land on d — 2d total. Change += to = in __mul__ and this halves to d, silently slowing all of training by 2×.'
            : n.op === 'mul' ? 'Chain rule through ×: each input’s grad is the other input times the output’s grad.'
            : n.op === 'add' ? 'Adds route the gradient to both inputs unchanged. The += is where separate paths merge.'
            : n.op === 'tanh' ? 'Local gain 1 − tanh². Near saturation (±1) it → 0 and the gradient dies here — vanishing gradients in miniature.'
            : n.op === 'pow2' ? 'd(d²)/dd = 2d. Sign points uphill — the update walks the other way.'
            : '';
        }
        break;
      }
      case 'upd': {
        r.data[ev.id!] = ev.new;
        r.stale[ev.id!] = false;
        if (last && n) {
          r.title = `${n.label} ← ${n.label} − η · ∂L/∂${n.label}`;
          r.lines = [{ nums: `= ${fmt(ev.old)} − ${state.lr.toFixed(2)} × (${fmt(ev.g)}) = ${fmt(ev.new)}` }];
          r.note = 'Gradient descent: step against the gradient. η is the loop gain.';
        }
        break;
      }
      case 'end': {
        r.itersDone += 1;
        r.lastLoss = ev.loss ?? null;
        if (last) {
          r.title = `step ${ev.k! + 1}/4 done — loss ${fmt(ev.loss)}`;
          r.lines = [];
          r.note =
            ev.k === 3
              ? 'Epoch traced. The decision surface shows the four nudges; drag “trace ep” to x-ray a different epoch.'
              : 'Same four phases, next sample.';
        }
        break;
      }
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Click-to-inspect: how THIS node's numbers are made. data = its forward
// expression over its inputs; grad = one line per consumer showing the
// chain-rule contribution that flows back into it (the += story).
// ---------------------------------------------------------------------------
interface Inspector {
  title: string;
  sections: { head: string; color: string; lines: ExplainLine[] }[];
  note: string;
}

function buildInspector(
  n: Laid,
  byId: Map<string, Laid>,
  consumers: Map<string, string[]>,
  r: Replay,
  lr: number,
): Inspector {
  const lbl = (id?: string) => (id && byId.get(id)?.label) || id || '?';
  const dv = (id?: string) => (id == null ? undefined : r.data[id]);

  // --- data: where this node's value comes from -----------------------------
  const dSec: ExplainLine[] = [];
  if (n.kind === 'op') {
    const selfProduct = n.op === 'mul' && n.inputs.length === 1;
    const [a, b0] = n.inputs;
    const b = selfProduct ? a : b0;
    if (n.op === 'mul') dSec.push({ rule: `${n.label} = ${lbl(a)} × ${lbl(b)}`, nums: `= ${fmt(dv(a))} × ${fmt(dv(b))} = ${fmt(dv(n.id))}` });
    else if (n.op === 'add') dSec.push({ rule: `${n.label} = ${lbl(a)} + ${lbl(b)}`, nums: `= ${fmt(dv(a))} + ${fmt(dv(b))} = ${fmt(dv(n.id))}` });
    else if (n.op === 'sub') dSec.push({ rule: `${n.label} = ${lbl(a)} − ${lbl(b)}`, nums: `= ${fmt(dv(a))} − ${fmt(dv(b))} = ${fmt(dv(n.id))}` });
    else if (n.op === 'tanh') dSec.push({ rule: `${n.label} = tanh(${lbl(a)})`, nums: `= tanh(${fmt(dv(a))}) = ${fmt(dv(n.id))}` });
    else dSec.push({ rule: `${n.label} = ${n.op}(${n.inputs.map((i) => lbl(i)).join(', ')})`, nums: `= ${fmt(dv(n.id))}` });
    if (r.stale[n.id]) dSec.push({ nums: '(stale — recomputed when the forward pass reaches it)' });
  } else if (n.kind === 'param') {
    dSec.push({ rule: `${n.label} = ${fmt(dv(n.id))}`, nums: `learned — changes ONLY in update: ${n.label} ← ${n.label} − η·∂L/∂${n.label}  (η = ${lr.toFixed(2)})` });
  } else {
    dSec.push({ rule: `${n.label} = ${fmt(dv(n.id))}`, nums: 'set when the sample loads — training data, not computed' });
  }

  // --- grad: one contribution per consumer, then the accumulated total ------
  const gSec: ExplainLine[] = [];
  const cons = consumers.get(n.id) ?? [];
  if (!cons.length) {
    gSec.push({ rule: `∂L/∂${n.label} = 1`, nums: 'the seed — nothing consumes the loss, backprop starts here' });
  } else {
    for (const cid of cons) {
      const c = byId.get(cid)!;
      const cg = r.grad[cid];
      if (c.op === 'mul' && c.inputs.length === 1) {
        const d2 = dv(n.id);
        gSec.push({ rule: `+= 2·${n.label} · ∂L/∂${c.label}`, nums: `= ${fmt(d2 == null ? null : 2 * d2)} × ${fmt(cg)}   (${n.label} fills both slots of ${c.label})` });
      } else if (c.op === 'mul') {
        const other = c.inputs.find((s) => s !== n.id);
        gSec.push({ rule: `+= ${lbl(other)} · ∂L/∂${c.label}`, nums: `= ${fmt(dv(other))} × ${fmt(cg)}   (via ${c.label}, ×)` });
      } else if (c.op === 'add') {
        gSec.push({ rule: `+= ∂L/∂${c.label}`, nums: `= ${fmt(cg)}   (via ${c.label}, + routes it through)` });
      } else if (c.op === 'sub') {
        const isNegSide = c.inputs.indexOf(n.id) === 1;
        gSec.push(
          isNegSide
            ? { rule: `−= ∂L/∂${c.label}`, nums: `= −(${fmt(cg)})   (via ${c.label}; computed, never used)` }
            : { rule: `+= ∂L/∂${c.label}`, nums: `= ${fmt(cg)}   (via ${c.label}, −)` },
        );
      } else if (c.op === 'tanh') {
        const t = dv(cid);
        gSec.push({ rule: `+= (1 − ${c.label}²) · ∂L/∂${c.label}`, nums: `= ${fmt(t == null ? null : 1 - t * t)} × ${fmt(cg)}   (via ${c.label})` });
      } else {
        gSec.push({ rule: `+= ∂${c.label}/∂${n.label} · ∂L/∂${c.label}`, nums: `(via ${c.label})` });
      }
    }
    if (cons.length > 1) gSec.push({ nums: `${cons.length} consumers → contributions ACCUMULATE (the +=)` });
    gSec.push({ nums: `current: ∂L/∂${n.label} = ${fmt(r.grad[n.id])}` });
  }

  const kindWord =
    n.kind === 'param' ? 'parameter'
    : n.kind === 'input' ? 'input leaf'
    : n.kind === 'const' ? (n.label === 'y' ? 'target (data)' : 'literal (data)')
    : `scalar op ${OPSYM[n.op] ?? n.op}`;

  const note =
    n.kind === 'param' ? 'A learned parameter (violet). It has no forward calculation — the update phase walks it against its gradient.'
    : n.kind === 'input' ? 'Training data. Its grad fans in from every neuron\'s first × — shown for the chain-rule story, but never used in an update.'
    : n.kind === 'const' ? 'Data, not a parameter — its gradient is computed by the machinery and thrown away.'
    : n.op === 'tanh' && n.label.startsWith('h')
      ? `${n.label} IS what people call "neuron ${[...n.label.slice(1)].map((ch) => { const i = '₀₁₂₃₄₅₆₇₈₉'.indexOf(ch); return i >= 0 ? String(i) : ch; }).join('')}'s activation" — one neuron = the whole chain of five scalar ops (two ×, two +, tanh) ending here.`
      : 'One node = one scalar op, not a neuron. A neuron is a whole chain of these — e.g. h₁ is the end of neuron 1\'s five-op chain.';

  const dataSection = { head: 'data — forward', color: FWD, lines: dSec };
  const gradSection = { head: 'grad — backward', color: GRAD, lines: gSec };
  const backwardish = r.phase === 'backward' || r.phase === 'update';

  return {
    title: `${n.label} · ${kindWord}`,
    sections: backwardish ? [gradSection, dataSection] : [dataSection, gradSection],
    note,
  };
}

// ---------------------------------------------------------------------------
export function MicrogradVisual({ state, status }: VisualProps<MicrogradState>) {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(6);
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const selectedRef = useRef<string | null>(null);
  selectedRef.current = selected;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const layout = useMemo(() => (state ? layoutGraph(state.nodes, state.edges) : null), [state]);
  const lossSet = useMemo(() => new Set(state?.lossIds ?? []), [state]);
  const replay = useMemo(
    () => (state && layout ? buildReplay(state, layout.byId, lossSet, cursor) : null),
    [state, layout, lossSet, cursor],
  );
  const total = state?.events.length ?? 0;

  // fresh run (slider drag / edit) → rewind the tape; drop a selection whose
  // node no longer exists (edited architecture), keep it otherwise — watching
  // one node across epochs is the point
  useEffect(() => {
    setCursor(0);
    setPlaying(false);
    if (selectedRef.current && layout && !layout.byId.has(selectedRef.current)) setSelected(null);
  }, [state, layout]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setCursor((c) => {
        if (c + 1 >= total) setPlaying(false);
        return Math.min(c + 1, total);
      });
    }, 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed, total]);

  const step = () => setCursor((c) => Math.min(c + 1, total));
  const stepPhase = () => {
    if (!state || !layout) return;
    setCursor((c) => {
      if (c >= total) return c;
      const ph = evPhase(state.events[c], lossSet);
      let i = c;
      while (i < total && evPhase(state.events[i], lossSet) === ph) i++;
      return i;
    });
  };
  const stepIter = () => {
    if (!state) return;
    setCursor((c) => {
      let i = c;
      while (i < total && state.events[i].t !== 'end') i++;
      return Math.min(i + 1, total);
    });
  };
  // backward = decrement cursor: the replay is derived from events[0..cursor),
  // so every panel (graph, surface, inspector) rewinds for free
  const stepBack = () => setCursor((c) => Math.max(c - 1, 0));
  const stepPhaseBack = () => {
    if (!state) return;
    setCursor((c) => {
      if (c <= 0) return 0;
      const ph = evPhase(state.events[c - 1], lossSet);
      let i = c;
      while (i > 0 && evPhase(state.events[i - 1], lossSet) === ph) i--;
      return i;
    });
  };
  const stepIterBack = () => {
    if (!state) return;
    setCursor((c) => {
      let s = -1;
      for (let i = Math.min(c, total) - 1; i >= 0; i--) {
        if (state.events[i].t === 'sample') { s = i; break; }
      }
      if (s === -1) return 0;
      if (c - 1 === s) {
        // already sitting on this iteration's start — jump to the previous one
        for (let i = s - 1; i >= 0; i--) if (state.events[i].t === 'sample') return i + 1;
        return 0;
      }
      return s + 1;
    });
  };

  // keyboard transport (ignored while typing in the editor pane)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        // first Esc closes the inspector, second exits full screen
        if (selectedRef.current) setSelected(null);
        else setExpanded(false);
        return;
      }
      const t = e.target as HTMLElement | null;
      // stay out of the way while typing/adjusting — but a focused BUTTON
      // (e.g. right after clicking step or ⛶) must not swallow the transport keys
      if (t && typeof t.closest === 'function' && (t.closest('.cm-editor') || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key === 'ArrowRight' && e.shiftKey) { e.preventDefault(); setPlaying(false); stepPhase(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); setPlaying(false); step(); }
      else if (e.key === 'ArrowLeft' && e.shiftKey) { e.preventDefault(); setPlaying(false); stepPhaseBack(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setPlaying(false); stepBack(); }
      else if (e.key === 'p') setPlaying((p) => !p);
      else if (e.key === 'f') setExpanded((x) => !x);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, layout, total]);

  // follow the active node
  useEffect(() => {
    const el = scrollRef.current;
    const id = replay?.active[0];
    if (!el || !id || !layout) return;
    const n = layout.byId.get(id);
    if (!n) return;
    el.scrollTo({
      left: n.x - el.clientWidth / 2 + NW / 2,
      top: Math.max(0, n.y - el.clientHeight / 2 + NH / 2),
      behavior: speed > 10 ? 'auto' : 'smooth',
    });
  }, [replay?.active[0], layout, speed]); // eslint-disable-line react-hooks/exhaustive-deps

  // decision surface, updated after each traced SGD step
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state || !replay || !state.surfaces.length) return;
    const { R, dom } = state;
    const frame = state.surfaces[Math.min(replay.itersDone, state.surfaces.length - 1)];
    const off = document.createElement('canvas');
    off.width = R;
    off.height = R;
    const octx = off.getContext('2d');
    const ctx = canvas.getContext('2d');
    if (!octx || !ctx) return;
    const POS = [96, 165, 250];
    const NEG = [251, 191, 36];
    const BASE = [17, 18, 22];
    const img = octx.createImageData(R, R);
    for (let i = 0; i < R * R; i++) {
      const p = frame[i];
      const c = p >= 0 ? POS : NEG;
      const t = Math.min(1, Math.abs(p)) * 0.7;
      const o = i * 4;
      img.data[o] = BASE[0] + (c[0] - BASE[0]) * t;
      img.data[o + 1] = BASE[1] + (c[1] - BASE[1]) * t;
      img.data[o + 2] = BASE[2] + (c[2] - BASE[2]) * t;
      img.data[o + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const S = canvas.width;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(off, 0, 0, S, S);
    for (const [x1, x2, y] of state.data) {
      const px = ((x1 + dom) / (2 * dom)) * S;
      const py = ((dom - x2) / (2 * dom)) * S;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = y > 0 ? FWD : GRAD;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.stroke();
    }
  }, [state, replay?.itersDone]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!state || !layout || !replay) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-sm text-neutral-500">
        {status === 'error' ? 'see error in the editor panel' : 'booting…'}
      </div>
    );
  }

  // loss curve (log-y) with the traced epoch marked
  const LW = 340;
  const LH = 34;
  const logLoss = state.loss.map((v) => Math.log10(Math.max(v, 1e-6)));
  const lo = Math.min(...logLoss, -3);
  const hi = Math.max(...logLoss, 0.1);
  const span = hi - lo || 1;
  const lossPath = logLoss
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${((i / Math.max(state.loss.length - 1, 1)) * LW).toFixed(1)},${(LH - ((v - lo) / span) * LH).toFixed(1)}`)
    .join(' ');
  const markerX = (state.tracedEpoch / Math.max(state.loss.length - 1, 1)) * LW;

  const curPhase = replay.phase;
  const selNode = selected ? layout.byId.get(selected) : undefined;
  const insp = selNode ? buildInspector(selNode, layout.byId, layout.consumers, replay, state.lr) : null;
  const btn = 'rounded border border-neutral-700 bg-neutral-800 px-2 py-0.5 font-mono text-[11px] text-neutral-200 hover:border-neutral-500 disabled:opacity-40';

  return (
    // overflow-y-auto: on short panes (big lesson + sliders) the debugger
    // scrolls internally instead of crushing the graph to zero height.
    // expanded: the same tree as a fixed full-screen overlay (f / Esc).
    <div
      className={
        expanded
          ? 'fixed inset-0 z-50 flex flex-col gap-1.5 overflow-y-auto bg-neutral-950 p-3'
          : 'flex h-full flex-col gap-1.5 overflow-y-auto p-2'
      }
    >
      {/* transport + phase pills */}
      <div className="flex shrink-0 flex-wrap items-center gap-1.5">
        <button className={btn} title="rewind the trace" onClick={() => { setPlaying(false); setCursor(0); }}>⟲</button>
        <button className={btn} title="back one iteration" onClick={() => { setPlaying(false); stepIterBack(); }} disabled={cursor <= 0}>⏮⏮</button>
        <button className={btn} title="back one phase (Shift+←)" onClick={() => { setPlaying(false); stepPhaseBack(); }} disabled={cursor <= 0}>⏮</button>
        <button className={btn} title="back one event (←)" onClick={() => { setPlaying(false); stepBack(); }} disabled={cursor <= 0}>⏴</button>
        <button className={`${btn} !border-blue-500/60 !bg-blue-500/15 !text-blue-300`} title="one event (→)" onClick={() => { setPlaying(false); step(); }} disabled={cursor >= total}>step ⏵</button>
        <button className={btn} title="finish this phase (Shift+→)" onClick={() => { setPlaying(false); stepPhase(); }} disabled={cursor >= total}>phase ⏭</button>
        <button className={btn} title="finish this SGD step" onClick={() => { setPlaying(false); stepIter(); }} disabled={cursor >= total}>iter ⏭⏭</button>
        <button className={btn} title="auto-step (p)" onClick={() => setPlaying((p) => !p)} disabled={cursor >= total}>{playing ? '❚❚' : '▶'}</button>
        <input type="range" min={1} max={20} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-16 accent-blue-400" title="steps / second" />
        <span className="font-mono text-[10px] tabular-nums text-neutral-500">ev {cursor}/{total}</span>
        <button className={btn} title={expanded ? 'exit full screen (Esc)' : 'full screen (f)'} onClick={() => setExpanded((x) => !x)}>
          {expanded ? '🗙 close' : '⛶'}
        </button>
        <span className="ml-auto flex items-center gap-1">
          {PHASES.map((p) => (
            <span
              key={p}
              className="rounded px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider"
              style={curPhase === p ? { color: PHASE_COLOR[p], background: `${PHASE_COLOR[p]}22`, fontWeight: 700 } : { color: MUTED }}
            >
              {p}
            </span>
          ))}
        </span>
      </div>

      {/* the computation graph */}
      <div ref={scrollRef} className="relative min-h-[240px] flex-1 overflow-auto rounded border border-neutral-800 bg-neutral-900/50">
        <svg
          width={layout.width}
          height={layout.height}
          role="img"
          aria-label="computation graph — click a node to inspect it"
          onClick={() => setSelected(null)}
        >
          {state.edges.map(([s, d]) => {
            const a = layout.byId.get(s)!;
            const b = layout.byId.get(d)!;
            const x1 = a.x + NW;
            const y1 = a.y + NH / 2;
            const x2 = b.x;
            const y2 = b.y + NH / 2;
            const mx = (x1 + x2) / 2;
            const hot = replay.hot.has(`${s}>${d}`);
            return (
              <path
                key={`${s}>${d}`}
                d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={hot ? (replay.hotBack ? GRAD : FWD) : EDGE}
                strokeWidth={hot ? 2.2 : 1.1}
                opacity={hot ? 1 : 0.8}
              />
            );
          })}
          {state.nodes.map((n) => {
            const ln = layout.byId.get(n.id)!;
            const active = replay.active.includes(n.id);
            const stale = replay.stale[n.id];
            const nameFill = n.kind === 'param' ? PARAM : n.label === 'L' ? LOSS : INK;
            const ring = active && curPhase ? PHASE_COLOR[curPhase] : n.kind === 'param' ? '#4c3a78' : '#333a47';
            const isSel = selected === n.id;
            return (
              <g
                key={n.id}
                transform={`translate(${ln.x},${ln.y})`}
                opacity={1}
                className="cursor-pointer"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelected((s) => (s === n.id ? null : n.id));
                }}
              >
                <title>{`inspect ${n.label}`}</title>
                {isSel && (
                  <rect x={-3} y={-3} width={NW + 6} height={NH + 6} rx={7} fill="none" stroke="#34d399" strokeWidth={1.5} strokeDasharray="5 3" />
                )}
                <rect
                  width={NW}
                  height={NH}
                  rx={5}
                  fill={n.kind === 'param' ? '#a78bfa14' : '#1c212b'}
                  stroke={ring}
                  strokeWidth={active ? 2 : 1}
                  strokeDasharray={n.kind === 'input' || n.kind === 'const' ? '3 3' : undefined}
                  style={active && curPhase ? { filter: `drop-shadow(0 0 4px ${PHASE_COLOR[curPhase]})` } : undefined}
                />
                <text x={7} y={13} fontSize={10} fontWeight={700} fontFamily="monospace" fill={nameFill}>{n.label}</text>
                {n.op && (
                  <text x={NW - 7} y={13} fontSize={9} textAnchor="end" fontFamily="monospace" fill={MUTED}>{OPSYM[n.op] ?? n.op}</text>
                )}
                <text x={7} y={28} fontSize={8} fontFamily="monospace" fill={MUTED}>d</text>
                <text x={18} y={28} fontSize={10} fontFamily="monospace" fill={FWD} opacity={stale ? 0.35 : 1}>{fmt(replay.data[n.id])}</text>
                <text x={7} y={41} fontSize={8} fontFamily="monospace" fill={MUTED}>g</text>
                <text x={18} y={41} fontSize={10} fontFamily="monospace" fill={GRAD} opacity={stale ? 0.35 : 1}>{fmt(replay.grad[n.id])}</text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* explainer — event readout, or node inspector when a node is clicked */}
      {insp && selNode ? (
        <div className="h-[128px] shrink-0 grow-0 basis-[128px] overflow-y-auto rounded border border-emerald-800/60 bg-neutral-900 px-2.5 py-1.5">
          <div className="flex items-baseline gap-2">
            <span className="rounded px-1.5 font-mono text-[9px] font-bold uppercase tracking-wider text-emerald-300" style={{ background: '#34d39922' }}>
              inspect
            </span>
            <span className="font-mono text-[12.5px] font-bold text-neutral-100">{insp.title}</span>
            <button
              className="ml-auto rounded px-1.5 font-mono text-[10px] text-neutral-400 hover:text-neutral-100"
              title="back to the event stream (Esc)"
              onClick={() => setSelected(null)}
            >
              ✕
            </button>
          </div>
          {insp.sections.map((sec) => (
            <div key={sec.head} className="mt-1">
              <div className="font-mono text-[9px] font-bold uppercase tracking-wider" style={{ color: sec.color }}>{sec.head}</div>
              <div className="flex flex-col font-mono text-[11px] leading-4 tabular-nums">
                {sec.lines.map((l, i) => (
                  <div key={i}>
                    {l.rule && <div className="text-neutral-200">{l.rule}</div>}
                    <div className="text-amber-300/90">{' '}{l.nums}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <p className="mt-1 text-[11px] leading-4 text-neutral-400">{insp.note}</p>
        </div>
      ) : (
      <div className="h-[104px] shrink-0 grow-0 basis-[104px] overflow-y-auto rounded border border-neutral-800 bg-neutral-900 px-2.5 py-1.5" aria-live="polite">
        <div className="flex items-baseline gap-2">
          {curPhase && (
            <span className="rounded px-1.5 font-mono text-[9px] font-bold uppercase tracking-wider" style={{ color: PHASE_COLOR[curPhase], background: `${PHASE_COLOR[curPhase]}22` }}>
              {curPhase}
            </span>
          )}
          <span className="font-mono text-[12.5px] font-bold text-neutral-100">{replay.title}</span>
        </div>
        <div className="mt-0.5 flex flex-col font-mono text-[11px] leading-4 tabular-nums">
          {replay.lines.map((l, i) => (
            <div key={i}>
              {l.rule && <div className="text-neutral-200">{l.rule}</div>}
              <div className="text-amber-300/90">{' '}{l.nums}</div>
            </div>
          ))}
        </div>
        {replay.note && <p className="mt-0.5 text-[11px] leading-4 text-neutral-400">{replay.note}</p>}
      </div>
      )}

      {/* surface + loss + verdicts */}
      <div className="flex shrink-0 items-stretch gap-2">
        <div className="shrink-0">
          <canvas ref={canvasRef} width={126} height={126} className="rounded border border-neutral-800" style={{ width: 126, height: 126 }} />
          <div className="mt-0.5 text-center font-mono text-[9px] text-neutral-500">surface · after step {Math.min(replay.itersDone, 4)}/4</div>
        </div>
        <div className="flex min-w-0 flex-1 flex-col justify-between">
          <svg viewBox={`0 0 ${LW} ${LH}`} className="w-full" style={{ height: LH }} preserveAspectRatio="none">
            <rect x={0} y={0} width={LW} height={LH} fill="rgba(255,255,255,0.03)" />
            <path d={lossPath} fill="none" stroke={LOSS} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
            <line x1={markerX} y1={0} x2={markerX} y2={LH} stroke={GRAD} strokeWidth={1} vectorEffect="non-scaling-stroke" />
          </svg>
          <div className="font-mono text-[10.5px] leading-4 text-neutral-400">
            loss (log) · <span style={{ color: GRAD }}>│</span> ep {state.tracedEpoch} · step loss{' '}
            <span className="text-neutral-200">{fmt(replay.lastLoss)}</span> · final{' '}
            <span className={state.finalAcc === 1 ? 'text-emerald-400' : 'text-neutral-200'}>
              {state.finalLoss.toFixed(4)} / acc {(state.finalAcc * 100).toFixed(0)}%
            </span>
          </div>
          <div>
            <span className={`inline-block rounded px-1.5 py-0.5 font-mono text-[10px] font-medium text-white ${state.gradcheckOk ? 'bg-emerald-600/85' : 'bg-rose-600/90'}`}>
              {state.gradcheckOk
                ? `✓ chain rule = finite diff (all params) · err ${state.gradcheckErr.toExponential(1)}`
                : `✗ a _backward is WRONG · worst ${state.gradcheckWorst || '?'} · err ${state.gradcheckErr.toExponential(1)}`}
            </span>
            <span className="ml-2 font-mono text-[10px] text-neutral-600">
              <span style={{ color: FWD }}>d</span> data · <span style={{ color: GRAD }}>g</span> grad ·{' '}
              <span style={{ color: PARAM }}>▢</span> param · ⌁ dashed = training data · click a node to inspect
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
