import React from "react";

export const Sparkline = ({ points }: { points: { label: string; value: number }[] }) => {
  if (!points.length) return <div className="text-sm text-[var(--muted)]">Нет данных по скорости</div>;
  const width = 260;
  const height = 80;
  const max = Math.max(...points.map((p) => p.value), 1);
  const step = width / Math.max(points.length - 1, 1);
  const coords = points.map((p, idx) => [idx * step, height - (p.value / max) * height]);
  const path = coords
    .map(([x, y], idx) => `${idx === 0 ? "M" : "L"}${x},${y}`)
    .join(" ");

  return (
    <div className="flex flex-col gap-2">
      <svg width={width} height={height} className="text-xs text-[var(--muted)]">
        <defs>
          <linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#2f7dff" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#2f7dff" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${path}`} stroke="#2f7dff" strokeWidth={2} fill="none" />
        <path
          d={`${path} L${width},${height} L0,${height} Z`}
          fill="url(#spark)"
          opacity={0.6}
        />
        {coords.map(([x, y], idx) => (
          <circle key={idx} cx={x} cy={y} r={3} fill="#2f7dff" />
        ))}
      </svg>
      <div className="flex flex-wrap gap-2 text-[11px] text-[var(--muted)]">
        {points.map((p) => (
          <span key={p.label} className="rounded-full bg-slate-100 px-2 py-1 dark:bg-white/10">
            {p.label}
          </span>
        ))}
      </div>
    </div>
  );
};
