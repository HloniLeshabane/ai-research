import type { LabModule } from '../types';
import defaultCode from './default.py?raw';
import { extract, type GDState } from './extract';
import { GradientDescentVisual } from './GradientDescentVisual';

export const gradientDescent: LabModule<GDState> = {
  id: 'gradient-descent',
  title: 'Gradient descent as a control loop',
  section: 'Foundations',
  blurb: 'Step downhill against the gradient; the learning rate is loop gain.',
  lesson: {
    concept:
      'Gradient descent walks downhill by repeatedly stepping against the gradient. The learning rate sets the step size — too small and it crawls, too large and it overshoots and diverges.',
    bridge:
      "It's a discrete feedback loop: the learning rate is loop gain, and the per-step factor (1 − lr·curvature) is the closed-loop pole. |pole| < 1 — inside the unit circle — is stable; cross it and the loss explodes.",
    tryThis: [
      'Push lr past 2.0 — the path spirals out as the pole leaves the unit circle.',
      'Set lr ≈ 1.0 for near one-step convergence on the stiff axis.',
      'Raise κ (condition number) and watch the path zig-zag and crawl down the valley.',
    ],
  },
  packages: ['numpy'],
  defaultCode,
  outputs: ['traj', 'pole', 'final_loss', 'kappa', 'lr'],
  controls: [
    { name: 'lr', label: 'lr', min: 0, max: 2.4, step: 0.01, default: 0.9 },
    { name: 'kappa', label: 'κ', min: 1, max: 40, step: 1, default: 8 },
  ],
  extract,
  Visual: GradientDescentVisual,
};
