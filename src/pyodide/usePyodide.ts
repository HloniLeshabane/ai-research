import { useContext } from 'react';
import { PyodideContext } from './PyodideProvider';

export function usePyodide() {
  return useContext(PyodideContext);
}
