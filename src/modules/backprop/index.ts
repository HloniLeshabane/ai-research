import type { LabModule } from '../types';
import defaultCode from './default.py?raw';
import { extract, type BackpropState } from './extract';
import { BackpropVisual } from './BackpropVisual';

export const backprop: LabModule<BackpropState> = {
  id: 'backprop',
  title: 'Backpropagation',
  section: 'The neuron',
  blurb: 'Watch your from-scratch net bend a flat plane into the data as it learns.',
  lesson: {
    concept:
      'Backprop trains a network by running the chain rule in reverse: each layer takes the loss-sensitivity of its output and hands back the sensitivity of its input and its weights. Do that top-to-bottom, step every weight downhill, repeat — and a flat decision boundary curves to fit data a straight line never could. This runs your actual numpy nn.py; drag the epoch scrubber to watch the boundary form.',
    bridge:
      'The training loop is a feedback controller: loss is the error signal, the gradient is the plant sensitivity, and the learning rate is loop gain. Push lr too high and it overshoots and diverges (the loss curve turns red); too low and it crawls. The nonlinearity is what lets stacked layers be more than one big linear block — remove it and XOR becomes unsolvable.',
    tryThis: [
      'Start on XOR: watch the boundary go from a flat wash to the four-quadrant checkerboard.',
      'Drop hidden units to 2 — barely enough capacity; watch it struggle or find a lopsided fit.',
      'Crank lr toward 1.5 and watch the loss curve oscillate or blow up (loop gain past stability).',
      'Switch to spirals and raise hidden units — see how much capacity a tight nonlinearity needs.',
      'In the editor, break tanh_backward (e.g. drop the 1 − A²): the green ✓ gradient badge flips to red ✗ and the boundary stops forming — a live gradient check catching the lie.',
    ],
  },
  packages: ['numpy'],
  defaultCode,
  outputs: [
    'Xdata',
    'ydata',
    'frames',
    'frame_epochs',
    'loss_hist',
    'extent',
    'final_acc',
    'final_loss',
    'dataset_name',
    'act_name',
    'H',
    'lr',
    'gradcheck_ok',
    'gradcheck_max_err',
    'gradcheck_worst',
  ],
  // Terse labels (the label column is narrow); the Visual's readout shows what
  // the dataset/activation indices resolve to — e.g. "XOR · tanh · hidden 16".
  controls: [
    { name: 'dataset', label: 'data', min: 0, max: 2, step: 1, default: 0 }, // 0=XOR 1=moons 2=spirals
    { name: 'activation', label: 'act', min: 0, max: 1, step: 1, default: 0 }, // 0=tanh 1=relu
    { name: 'hidden', label: 'hidden', min: 2, max: 32, step: 1, default: 16 },
    { name: 'lr', label: 'lr', min: 0.01, max: 1.5, step: 0.01, default: 0.3 },
    { name: 'epochs', label: 'epochs', min: 100, max: 2000, step: 100, default: 1200 },
  ],
  extract,
  Visual: BackpropVisual,
};
