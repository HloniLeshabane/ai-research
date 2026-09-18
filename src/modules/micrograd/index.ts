import type { LabModule } from '../types';
import defaultCode from './default.py?raw';
import { extract, type MicrogradState } from './extract';
import { MicrogradVisual } from './MicrogradVisual';

export const micrograd: LabModule<MicrogradState> = {
  id: 'micrograd',
  title: 'Autograd from scratch (micrograd)',
  section: 'The neuron',
  blurb: 'Single-step one SGD iteration: every multiply, every chain-rule hop, every weight nudge.',
  lesson: {
    concept:
      'Backprop is not a formula, it is a graph walk. Every scalar op remembers its inputs; the forward pass fills in values left-to-right; then the chain rule runs the graph in reverse, each node handing its children "how much does the loss move if you nudge?" — local derivative × incoming gradient, accumulated with +=. This module runs YOUR Value class on a 2→3→1 tanh net learning XOR, one sample per step, and the debugger replays the traced epoch event by event. Step through the whole cycle: forward → loss → zero_grad → backward → update.',
    bridge:
      'The computation graph is a signal-flow graph, and backprop is the adjoint network: same topology, edges reversed, each op replaced by its small-signal gain (mul → the other input, tanh → 1−t², add → unity). Sensitivity analysis in SPICE does exactly this walk. Saturated tanh is a stage biased into the rails: gain ≈ 0, and the gradient signal dies there — watch g go to ~0 on h-nodes whose data is near ±1.',
    tryThis: [
      'Step through backprop on Σ nodes: adds route the gradient through unchanged; the += on shared nodes is the chain rule merging two paths.',
      'Drag "trace ep" from 0 to a late epoch: same graph, but the grads have shrunk by orders of magnitude — the loss curve marker shows why.',
      'In the editor, break tanh: drop the (1 - t*t) in its _backward. Predict what the gradcheck badge and the h-node grads will show; then look.',
      'Change += to = in __mul__\'s _backward: d sits in BOTH slots of L = d·d, so ∂L/∂d halves to d — training still converges, just at half speed, and only the gradcheck badge sees the lie. (In __add__ the same edit is a no-op here: no add-child has fan-out — check the graph.)',
      'Crank lr to 0.5, trace an early epoch, and watch an update step overshoot: a weight flips sign and the next iteration\'s loss jumps.',
    ],
  },
  packages: [], // pure-Python module — no numpy download, fastest boot in the lab
  defaultCode,
  outputs: [
    'structure_json',
    'trace_json',
    'surfaces_json',
    'loss_json',
    'data_json',
    'traced_epoch',
    'surf_r',
    'surf_dom',
    'lr_used',
    'final_loss',
    'final_acc',
    'gradcheck_ok',
    'gradcheck_err',
    'gradcheck_worst',
  ],
  controls: [
    { name: 'trace_epoch', label: 'trace ep', min: 0, max: 199, step: 1, default: 0 },
    { name: 'seed', label: 'seed', min: 0, max: 20, step: 1, default: 8 },
    { name: 'lr', label: 'lr', min: 0.05, max: 0.5, step: 0.05, default: 0.2 },
    { name: 'epochs', label: 'epochs', min: 20, max: 200, step: 20, default: 100 },
  ],
  extract,
  Visual: MicrogradVisual,
};
