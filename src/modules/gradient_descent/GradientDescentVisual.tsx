import type { VisualProps } from '../types';
import type { GDState } from './extract';

const OK = '#34d399';
const BAD = '#fb7185';

export function GradientDescentVisual({ state, status }: VisualProps<GDState>) {
  if (!state) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-sm text-neutral-500">
        {status === 'error' ? 'see error in the editor panel' : 'booting…'}
      </div>
    );
  }

  const { traj, pole, finalLoss, kappa, lr, diverged } = state;
  const W = 480;
  const H = 380;
  const cx = W / 2;
  const cy = H / 2;
  const scale = H / 9;
  const w2s = (x: number, y: number): [number, number] => [cx + x * scale, cy - y * scale];
  const inView = (sx: number, sy: number) => sx >= -12 && sx <= W + 12 && sy >= -12 && sy <= H + 12;
  const clamp = (v: number, hi: number) => Math.max(-12, Math.min(hi + 12, v));

  const levels = Array.from({ length: 6 }, (_, k) => {
    const yk = (4.2 * (k + 1)) / 6;
    return { ry: yk * scale, rx: (yk / Math.sqrt(kappa)) * scale };
  });

  let d = '';
  traj.forEach((p, i) => {
    const [sx, sy] = w2s(p[0], p[1]);
    d += (i === 0 ? 'M' : 'L') + clamp(sx, W).toFixed(1) + ',' + clamp(sy, H).toFixed(1) + ' ';
  });

  const col = diverged ? BAD : OK;
  const ap = Math.abs(pole);
  const regime = diverged
    ? 'unstable — diverging'
    : ap < 0.05
      ? 'well-tuned — near one-step'
      : pole < 0
        ? 'underdamped — zig-zag'
        : 'overdamped — smooth';
  const [ox, oy] = w2s(0, 0);
  const [s0x, s0y] = traj.length ? w2s(traj[0][0], traj[0][1]) : [cx, cy];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      {levels.map((l, i) => (
        <ellipse key={i} cx={cx} cy={cy} rx={Math.max(l.rx, 0.5)} ry={Math.max(l.ry, 0.5)} fill="none" stroke="rgba(255,255,255,0.10)" />
      ))}
      <line x1={0} y1={cy} x2={W} y2={cy} stroke="rgba(255,255,255,0.14)" />
      <line x1={cx} y1={0} x2={cx} y2={H} stroke="rgba(255,255,255,0.14)" />
      <path d={d} fill="none" stroke={col} strokeWidth={1.8} />
      {traj.map((p, i) => {
        const [sx, sy] = w2s(p[0], p[1]);
        return inView(sx, sy) ? <circle key={i} cx={sx} cy={sy} r={2.3} fill={col} /> : null;
      })}
      <circle cx={s0x} cy={s0y} r={4} fill="#888780" />
      <g stroke="#f5b301" strokeWidth={2}>
        <line x1={ox - 6} y1={oy - 6} x2={ox + 6} y2={oy + 6} />
        <line x1={ox + 6} y1={oy - 6} x2={ox - 6} y2={oy + 6} />
      </g>
      <text x={14} y={24} fill="#e5e7eb" fontSize="12" fontFamily="monospace">
        lr = {lr.toFixed(2)} · pole 1−lr = {pole.toFixed(2)}
      </text>
      <text x={14} y={42} fill={diverged ? BAD : OK} fontSize="12" fontFamily="monospace">
        {regime}
        {!diverged && ap < 1 ? ' · |pole| < 1' : ''}
      </text>
      <text x={14} y={H - 14} fill="#9ca3af" fontSize="11" fontFamily="monospace">
        loss after {Math.max(traj.length - 1, 0)} steps: {diverged ? 'diverged' : finalLoss.toExponential(1)}
      </text>
    </svg>
  );
}
