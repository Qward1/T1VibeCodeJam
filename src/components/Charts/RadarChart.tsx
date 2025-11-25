import { SkillStat } from "@/types";

// Простой радар-чарт без внешних библиотек
export const RadarChart = ({ data }: { data: SkillStat[] }) => {
  const size = 240;
  const center = size / 2;
  const radius = size / 2 - 20;
  const angleStep = (Math.PI * 2) / data.length;

  const toPoint = (value: number, index: number) => {
    const angle = index * angleStep - Math.PI / 2;
    const r = radius * (value / 100);
    return [center + r * Math.cos(angle), center + r * Math.sin(angle)];
  };

  const path = data
    .map((item, idx) => {
      const [x, y] = toPoint(item.value, idx);
      const prefix = idx === 0 ? "M" : "L";
      return `${prefix}${x},${y}`;
    })
    .join(" ") + " Z";

  return (
    <div className="relative">
      <svg width={size} height={size} className="text-sm text-[var(--muted)]">
        {[1, 0.75, 0.5, 0.25].map((ratio) => (
          <circle
            key={ratio}
            cx={center}
            cy={center}
            r={radius * ratio}
            fill="none"
            stroke="var(--border)"
            strokeDasharray="4 4"
          />
        ))}
        {data.map((item, idx) => {
          const angle = idx * angleStep - Math.PI / 2;
          const x = center + (radius + 12) * Math.cos(angle);
          const y = center + (radius + 12) * Math.sin(angle);
          return (
            <g key={item.label}>
              <line
                x1={center}
                y1={center}
                x2={center + radius * Math.cos(angle)}
                y2={center + radius * Math.sin(angle)}
                stroke="var(--border)"
              />
              <text x={x} y={y} textAnchor="middle" dominantBaseline="middle">
                {item.label}
              </text>
            </g>
          );
        })}
        <path
          d={path}
          fill="#2f7dff33"
          stroke="#2f7dff"
          strokeWidth={2}
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};
