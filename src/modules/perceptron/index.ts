import type { LabModule } from '../types';
import defaultCode from './default.py?raw';
import { extract, type PerceptronState } from './extract';
import { PerceptronVisual } from './PerceptronVisual';

export const perceptron: LabModule<PerceptronState> = {
  id: 'perceptron',
  title: 'The perceptron',
  section: 'The neuron',
  blurb: 'A linear neuron: activation = X·W + b.',
  lesson: {
    concept:
      'A neuron multiplies each input by a weight, sums them with a bias, and outputs the result. This weighted sum is the atom every neural network is built from.',
    bridge:
      "It's a summing junction — like a multi-input op-amp adder. Weights are gains, the bias is a DC offset, and each wire's colour shows its gain's sign.",
    tryThis: [
      'Drag a weight negative — its wire turns red (inhibitory) and the output drops.',
      'Edit X in the code; the input-node wire widths redraw when it re-runs.',
      'Zero every weight — the output collapses to just the bias.',
    ],
  },
  packages: ['numpy'],
  defaultCode,
  outputs: ['X', 'W', 'contributions', 'activation', 'bias'],
  controls: [
    { name: 'w0', label: 'w₀', min: -2, max: 2, step: 0.05, default: 0.6 },
    { name: 'w1', label: 'w₁', min: -2, max: 2, step: 0.05, default: -0.4 },
    { name: 'w2', label: 'w₂', min: -2, max: 2, step: 0.05, default: 0.9 },
    { name: 'bias', label: 'b', min: -2, max: 2, step: 0.05, default: 0.1 },
  ],
  extract,
  Visual: PerceptronVisual,
};
