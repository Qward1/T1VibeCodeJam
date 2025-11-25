"use client";
import { InterviewSession } from "@/types";
import { formatDuration } from "@/utils";

type Props = { session: InterviewSession; remaining?: number };

export const SessionHeader = ({ session, remaining }: Props) => {
  const timeLeft = remaining ?? session.timer;
  return (
    <div className="flex items-center justify-end gap-2 rounded-2xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 text-sm">
      <span className="rounded-full bg-vibe-50 px-3 py-1 text-base font-semibold text-vibe-700 dark:bg-white/10 dark:text-white">
        ⏳ {formatDuration(timeLeft)}
      </span>
    </div>
  );
};
