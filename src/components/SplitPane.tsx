import { useRef, useState, type ReactNode } from 'react';

// Simple draggable 50/50 split. The divider drags between 25% and 75%.
export function SplitPane({ left, right }: { left: ReactNode; right: ReactNode }) {
  const [pct, setPct] = useState(48);
  const dragging = useRef(false);

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const p = ((e.clientX - rect.left) / rect.width) * 100;
    setPct(Math.min(75, Math.max(25, p)));
  };
  const stop = () => {
    dragging.current = false;
    document.body.style.cursor = '';
  };

  return (
    <div
      className="flex h-full w-full select-none"
      onMouseMove={onMove}
      onMouseUp={stop}
      onMouseLeave={stop}
    >
      <div style={{ width: `${pct}%` }} className="h-full min-w-0">
        {left}
      </div>
      <div
        onMouseDown={() => {
          dragging.current = true;
          document.body.style.cursor = 'col-resize';
        }}
        className="w-1.5 shrink-0 cursor-col-resize bg-neutral-800 transition-colors hover:bg-emerald-600/60"
      />
      <div style={{ width: `${100 - pct}%` }} className="h-full min-w-0">
        {right}
      </div>
    </div>
  );
}
