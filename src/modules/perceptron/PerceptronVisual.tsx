import type { VisualProps } from '../types';
import type { PerceptronState } from './extract';

const POS = '#34d399'; // emerald — positive weight
const NEG = '#fb7185'; // rose — negative weight

const wireWidth = (input: number) => 1.2 + Math.min(Math.abs(input), 1.5) * 3.6;
const wireOpacity = (weight: number) => 0.18 + Math.min(Math.abs(weight) / 2, 1) * 0.82;
const fmt = (n: number) => (n >= 0 ? '+' : '') + n.toFixed(2);

export function PerceptronVisual({ state, status }: VisualProps<PerceptronState>) {
  if (!state) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-sm text-neutral-500">
        {status === 'error' ? 'see error in the editor panel' : 'booting…'}
      </div>
    );
  }

  const { x, w, activation, bias } = state;
  // Leads: each input x_i with weight w_i, plus the bias as a constant-1 input.
  const leads = [
    ...x.map((xi, i) => ({ label: `x${i}`, input: xi, weight: w[i] ?? 0 })),
    { label: '1', input: 1, weight: bias },
  ];

  const W = 520;
  const H = 360;
  const inX = 84;
  const sumX = 300;
  const outX = 446;
  const midY = H / 2;
  const n = leads.length;
  const span = H - 84;
  const nodeY = (i: number) => 42 + (n > 1 ? (i * span) / (n - 1) : span / 2);

  const t = Math.tanh(activation);
  const outColor = t >= 0 ? POS : NEG;
  const outFill = 0.12 + Math.abs(t) * 0.8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      {/* input -> sum wires (width = |input|, opacity = |weight|, colour = sign) */}
      {leads.map((l, i) => {
        const y = nodeY(i);
        const color = l.weight >= 0 ? POS : NEG;
        return (
          <g key={l.label}>
            <line
              x1={inX + 18}
              y1={y}
              x2={sumX - 30}
              y2={midY}
              stroke={color}
              strokeWidth={wireWidth(l.input)}
              strokeOpacity={wireOpacity(l.weight)}
              strokeLinecap="round"
            />
            <text
              x={(inX + sumX) / 2}
              y={(y + midY) / 2 - 6}
              fill="#9ca3af"
              fontSize="11"
              fontFamily="monospace"
              textAnchor="middle"
            >
              {fmt(l.weight)}
            </text>
          </g>
        );
      })}

      {/* sum -> output */}
      <line
        x1={sumX + 30}
        y1={midY}
        x2={outX - 26}
        y2={midY}
        stroke={outColor}
        strokeWidth={2.5}
        strokeOpacity={0.9}
        strokeLinecap="round"
      />

      {/* input nodes */}
      {leads.map((l, i) => {
        const y = nodeY(i);
        return (
          <g key={`node-${l.label}`}>
            <circle cx={inX} cy={y} r={18} fill="#1f2937" stroke="#4b5563" strokeWidth={1.5} />
            <text x={inX} y={y + 4} fill="#e5e7eb" fontSize="12" fontFamily="monospace" textAnchor="middle">
              {l.label}
            </text>
            <text x={inX - 28} y={y + 4} fill="#9ca3af" fontSize="11" fontFamily="monospace" textAnchor="end">
              {l.input.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* summing node */}
      <circle cx={sumX} cy={midY} r={30} fill="#111827" stroke="#6b7280" strokeWidth={1.5} />
      <text x={sumX} y={midY + 8} fill="#e5e7eb" fontSize="24" fontFamily="monospace" textAnchor="middle">
        Σ
      </text>

      {/* output node (fill intensity = |tanh(activation)|) */}
      <circle cx={outX} cy={midY} r={26} fill={outColor} fillOpacity={outFill} stroke={outColor} strokeWidth={2} />
      <text x={outX} y={midY + 5} fill="#f9fafb" fontSize="13" fontFamily="monospace" textAnchor="middle">
        {activation.toFixed(2)}
      </text>
      <text x={outX} y={midY + 50} fill="#9ca3af" fontSize="11" fontFamily="monospace" textAnchor="middle">
        activation
      </text>
    </svg>
  );
}
