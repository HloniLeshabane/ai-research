import type { LabModule } from '../types';
import defaultCode from './default.py?raw';
import { extract, type PerceptronState } from './extract';
import { PerceptronVisual } from './PerceptronVisual';

export const perceptron: LabModule<PerceptronState> = {
  id: 'perceptron',
  title: 'The Interactive Perceptron',
  blurb: 'A linear neuron: activation = X·W + b. Wire width ∝ |input|, opacity ∝ |weight|, colour = sign.',
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
