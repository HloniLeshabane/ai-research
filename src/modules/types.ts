import type { ComponentType } from 'react';
import type { NdArray } from '../pyodide/bridge';

export type { NdArray };

// A slider that patches a Python global of the same `name`, then re-runs.
export interface ControlSpec {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export type RunStatus =
  | 'idle'
  | 'booting'
  | 'loading-packages'
  | 'running'
  | 'ready'
  | 'error';

export interface VisualProps<TState> {
  state: TState | null;
  status: RunStatus;
}

// The teaching content shown above the editor for each module.
export interface Lesson {
  concept: string; // what it is and why it matters
  bridge?: string; // the electrical-engineering analogy
  tryThis?: string[]; // suggested experiments
}

// The contract every lab module implements. Adding a concept = implement this
// interface and add a CurriculumEntry pointing at it.
export interface LabModule<TState> {
  id: string;
  title: string;
  section: string;
  blurb: string;
  lesson: Lesson;
  packages: string[]; // pyodide packages to ensure are loaded, e.g. ['numpy']
  defaultCode: string; // prebuilt Python shown in the editor
  outputs: string[]; // Python global names to extract after each run
  controls?: ControlSpec[]; // optional sliders injected as Python globals
  extract(values: Record<string, unknown>): TState; // raw extraction -> typed state
  Visual: ComponentType<VisualProps<TState>>; // right-panel renderer
}
