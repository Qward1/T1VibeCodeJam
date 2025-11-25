import { formatDuration } from "@/utils";

type Props = {
  description: string;
  title?: string;
  timer: number;
  currentIndex: number;
  total: number;
};

export const TaskCard = ({ description, title, timer, currentIndex, total }: Props) => (
  <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-4 shadow-sm">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
      {title && <h3 className="text-lg font-semibold">Задание: {title}</h3>}
      <div className="flex items-center gap-2 rounded-full bg-vibe-50 px-4 py-2 text-sm font-semibold text-vibe-800 dark:bg-white/10 dark:text-white">
        <span>Задача {Math.min(currentIndex + 1, total)}/{total}</span>
        <span className="text-[var(--muted)]">•</span>
        <span>⏳ {formatDuration(timer)}</span>
      </div>
    </div>
    <p className="mt-1 text-base leading-relaxed text-vibe-900 dark:text-white/90">{description}</p>
  </div>
);
