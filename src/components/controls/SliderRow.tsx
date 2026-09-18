import type { ControlSpec } from '../../modules/types';

export function SliderRow({
  spec,
  value,
  onChange,
}: {
  spec: ControlSpec;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex items-center gap-3 py-1">
      <label className="w-16 shrink-0 font-mono text-sm text-neutral-300">{spec.label}</label>
      <input
        type="range"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-emerald-500"
      />
      <span className="w-12 text-right font-mono text-sm text-neutral-100">{value.toFixed(2)}</span>
    </div>
  );
}
