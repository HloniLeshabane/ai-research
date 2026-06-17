import type { LabModule } from './types';
import { perceptron } from './perceptron';

// Every module the lab knows about. Add a concept = implement LabModule + push here.
export const MODULES: LabModule<unknown>[] = [perceptron as LabModule<unknown>];
