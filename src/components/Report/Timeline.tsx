import { formatDate } from "@/utils";

export const Timeline = ({ events }: { events: { label: string; at: string }[] }) => (
  <div className="space-y-3">
    {events.map((e) => (
      <div key={e.at} className="flex items-center gap-3">
        <span className="h-10 w-10 rounded-full bg-vibe-50 text-center text-xs font-semibold leading-10 text-vibe-700 dark:bg-white/10 dark:text-white">
          {e.label}
        </span>
        <div>
          <div className="text-sm font-semibold">{e.label}</div>
          <div className="text-xs text-[var(--muted)]">{formatDate(e.at)}</div>
        </div>
      </div>
    ))}
  </div>
);
