import type { LabModule } from './types';
import { perceptron } from './perceptron';
import { gradientDescent } from './gradient_descent';
import { micrograd } from './micrograd';
import { backprop } from './backprop';

// One topic in the course. `ready` entries carry a built module; `soon` entries
// are roadmap placeholders shown (disabled) in the nav so the whole arc is visible.
export interface CurriculumEntry {
  id: string;
  title: string;
  status: 'ready' | 'soon';
  module?: LabModule<unknown>;
}

export interface CurriculumSection {
  title: string;
  entries: CurriculumEntry[];
}

export const CURRICULUM: CurriculumSection[] = [
  {
    title: 'Foundations',
    entries: [
      { id: 'gradient-descent', title: 'Gradient descent as a control loop', status: 'ready', module: gradientDescent as LabModule<unknown> },
      { id: 'vectors', title: 'Vectors & dot products', status: 'soon' },
    ],
  },
  {
    title: 'The neuron',
    entries: [
      { id: 'perceptron', title: 'The perceptron', status: 'ready', module: perceptron as LabModule<unknown> },
      { id: 'activation', title: 'Activation functions', status: 'soon' },
      { id: 'training', title: 'Training a neuron', status: 'soon' },
      { id: 'mlp', title: 'Multi-layer perceptron', status: 'soon' },
      { id: 'micrograd', title: 'Autograd from scratch (micrograd)', status: 'ready', module: micrograd as LabModule<unknown> },
      { id: 'backprop', title: 'Backpropagation', status: 'ready', module: backprop as LabModule<unknown> },
    ],
  },
  {
    title: 'Vision',
    entries: [
      { id: 'convolution', title: 'Convolution', status: 'soon' },
      { id: 'feature-maps', title: 'Filters & feature maps', status: 'soon' },
    ],
  },
  {
    title: 'Sequences & representations',
    entries: [
      { id: 'embeddings', title: 'Embeddings', status: 'soon' },
      { id: 'attention', title: 'Attention', status: 'soon' },
      { id: 'tiny-gpt', title: 'A tiny GPT', status: 'soon' },
    ],
  },
  {
    title: 'Frontier',
    entries: [{ id: 'mech-interp', title: 'Mechanistic interpretability', status: 'soon' }],
  },
];

export const READY_ENTRIES = CURRICULUM.flatMap((s) => s.entries).filter(
  (e): e is CurriculumEntry & { module: LabModule<unknown> } => e.status === 'ready' && !!e.module,
);
