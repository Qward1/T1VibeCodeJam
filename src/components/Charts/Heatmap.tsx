import { ErrorHeat } from "@/types";

// Минималистичный хитмэп: чем больше count, тем насыщеннее цвет
export const Heatmap = ({ data }: { data: ErrorHeat[] }) => {
  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-2">
      {data.map((item) => {
        const intensity = item.count / max;
        return (
          <div
            key={item.bucket}
            className="rounded-xl border border-[var(--border)] p-3"
            style={{
              background: `linear-gradient(90deg, rgba(47,125,255,${0.08 + intensity * 0.3}), rgba(47,125,255,0))`,
            }}
          >
            <div className="text-sm font-semibold">{item.bucket}</div>
            <div className="text-xs text-[var(--muted)]">{item.count} ошибок</div>
          </div>
        );
      })}
    </div>
  );
};
