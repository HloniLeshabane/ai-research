import { useCallback, useEffect, useRef, useState } from 'react';
import type { VisualProps } from '../types';
import type { BackpropState } from './extract';

const C0 = [96, 165, 250]; // class 0 — blue
const C1 = [251, 113, 133]; // class 1 — red
const FPS = 14;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export function BackpropVisual({ state, status }: VisualProps<BackpropState>) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const roRef = useRef<ResizeObserver | null>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [side, setSide] = useState(300); // display px — a square that fits the pane

  const K = state?.K ?? 0;

  // Callback ref: attach the observer the moment the wrapper mounts (which is
  // *after* the null-state "booting" view, so a plain effect would miss it) and
  // keep the boundary square sized to whatever room the pane gives it.
  const setWrap = useCallback((node: HTMLDivElement | null) => {
    roRef.current?.disconnect();
    roRef.current = null;
    if (!node) return;
    const ro = new ResizeObserver(() => {
      const s = Math.max(80, Math.floor(Math.min(node.clientWidth, node.clientHeight)));
      setSide(s);
    });
    ro.observe(node);
    roRef.current = ro;
  }, []);

  // New run → restart the animation from the untrained boundary.
  useEffect(() => {
    setFrame(0);
    setPlaying(true);
  }, [state]);

  // Play loop (throttled), pausing at the last frame.
  useEffect(() => {
    if (!playing || K === 0) return;
    let raf = 0;
    let last = 0;
    const tick = (t: number) => {
      if (t - last >= 1000 / FPS) {
        last = t;
        setFrame((f) => {
          if (f + 1 >= K) {
            setPlaying(false);
            return f;
          }
          return f + 1;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, K]);

  // Draw the current frame's boundary + data points.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state || state.K === 0) return;
    const { frames, R, extent, X, y } = state;
    const idx = Math.min(frame, state.K - 1);

    if (!offRef.current) offRef.current = document.createElement('canvas');
    const off = offRef.current;
    off.width = R;
    off.height = R;
    const octx = off.getContext('2d');
    const ctx = canvas.getContext('2d');
    if (!octx || !ctx) return;

    // Build the R×R probability image (flip rows: canvas top = world ymax).
    const img = octx.createImageData(R, R);
    const base = idx * R * R;
    for (let j = 0; j < R; j++) {
      const srcRow = (R - 1 - j) * R;
      for (let i = 0; i < R; i++) {
        const p = frames[base + srcRow + i]; // class-1 probability
        const shade = 0.5 + 0.5 * Math.abs(2 * p - 1); // dim near the boundary
        const o = (j * R + i) * 4;
        img.data[o] = lerp(C0[0], C1[0], p) * shade;
        img.data[o + 1] = lerp(C0[1], C1[1], p) * shade;
        img.data[o + 2] = lerp(C0[2], C1[2], p) * shade;
        img.data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    // Scale up smoothly onto the display canvas.
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, side, side);
    ctx.drawImage(off, 0, 0, side, side);

    // Overlay data points (same world→pixel mapping, y flipped).
    const [xmin, xmax, ymin, ymax] = extent;
    const toPx = (px: number, py: number): [number, number] => [
      ((px - xmin) / (xmax - xmin)) * side,
      (1 - (py - ymin) / (ymax - ymin)) * side,
    ];
    const r = Math.max(2, side / 130);
    for (let k = 0; k < X.length; k++) {
      const [sx, sy] = toPx(X[k][0], X[k][1]);
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      const c = y[k] >= 0.5 ? C1 : C0;
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.stroke();
    }
  }, [state, frame, side]);

  if (!state) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-sm text-neutral-500">
        {status === 'error' ? 'see error in the editor panel' : 'booting…'}
      </div>
    );
  }

  const { loss, frameEpochs, K: nFrames, finalAcc, finalLoss, datasetName, actName, hidden, diverged } = state;
  const { gradcheckOk, gradcheckMaxErr, gradcheckWorst } = state;
  const gcErr = Number.isFinite(gradcheckMaxErr) ? gradcheckMaxErr.toExponential(1) : '—';
  const epoch = frameEpochs[Math.min(frame, nFrames - 1)] ?? 0;

  // Loss curve (log-y so the early plunge is visible), with an epoch marker.
  const LW = 340;
  const LH = 40;
  const logLoss = loss.map((v) => Math.log10(Math.max(v, 1e-6)));
  const lo = Math.min(...logLoss, 0);
  const hi = Math.max(...logLoss, -1);
  const span = hi - lo || 1;
  const lossPath = loss.length
    ? logLoss
        .map((v, i) => {
          const x = (i / Math.max(loss.length - 1, 1)) * LW;
          const yy = LH - ((v - lo) / span) * LH;
          return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${yy.toFixed(1)}`;
        })
        .join(' ')
    : '';
  const markerX = (epoch / Math.max(loss.length, 1)) * LW;

  return (
    <div className="flex h-full flex-col gap-1.5 p-2">
      {/* Boundary fills the pane; the transport bar overlays its bottom edge. */}
      <div ref={setWrap} className="relative flex min-h-0 w-full flex-1 items-center justify-center">
        <canvas
          ref={canvasRef}
          width={side}
          height={side}
          className="rounded border border-neutral-800"
          style={{ width: side, height: side }}
        />
        {/* Live gradient-check verdict — flips to red if backward() disagrees
            with finite differences (e.g. someone breaks tanh_backward). */}
        <div
          className={`absolute left-2 top-2 rounded px-2 py-1 font-mono text-[11px] font-medium text-white shadow ${
            gradcheckOk ? 'bg-emerald-600/85' : 'bg-rose-600/90'
          }`}
        >
          {gradcheckOk
            ? `✓ backprop verified · err ${gcErr}`
            : `✗ backprop WRONG · ${gradcheckWorst || 'grad'} err ${gcErr}`}
        </div>
        <div
          className="absolute bottom-2 flex items-center gap-2 rounded-md bg-black/55 px-2 py-1 backdrop-blur-sm"
          style={{ width: Math.min(side - 16, 320) }}
        >
          <button
            onClick={() => {
              if (frame >= nFrames - 1) setFrame(0);
              setPlaying((p) => !p);
            }}
            className="rounded bg-neutral-100/90 px-2 py-0.5 font-mono text-xs text-neutral-900 hover:bg-white"
          >
            {playing ? '❚❚' : '▶'}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(nFrames - 1, 0)}
            value={Math.min(frame, nFrames - 1)}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
            className="flex-1 accent-amber-400"
          />
          <span className="w-14 shrink-0 text-right font-mono text-[11px] text-neutral-200">ep {Math.round(epoch)}</span>
        </div>
      </div>

      {/* Compact loss sparkline (log-y) with an epoch marker. */}
      <svg viewBox={`0 0 ${LW} ${LH}`} className="w-full shrink-0" style={{ height: LH }} preserveAspectRatio="none">
        <rect x={0} y={0} width={LW} height={LH} fill="rgba(255,255,255,0.03)" />
        <path d={lossPath} fill="none" stroke={diverged ? '#fb7185' : '#34d399'} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
        <line x1={markerX} y1={0} x2={markerX} y2={LH} stroke="#f5b301" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      </svg>

      <div className="shrink-0 font-mono text-[11px] leading-4 text-neutral-400">
        <span className="text-neutral-200">{datasetName}</span> · {actName} · hidden {hidden} · loss (log){' '}
        acc <span className={finalAcc > 0.95 ? 'text-emerald-400' : 'text-neutral-200'}>{(finalAcc * 100).toFixed(0)}%</span> ·{' '}
        {diverged ? <span className="text-rose-400">diverged</span> : `loss ${finalLoss.toFixed(3)}`}
      </div>
    </div>
  );
}
