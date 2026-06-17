import { useCallback, useEffect, useRef } from 'react';

// Returns a stable `schedule(fn, delay)` that coalesces rapid calls — code
// edits run on a long delay (~400ms), slider ticks on a short one.
export function useDebouncer() {
  const timer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  return useCallback((fn: () => void, delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(fn, delay);
  }, []);
}
