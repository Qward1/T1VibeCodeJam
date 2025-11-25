import { cn } from "@/utils";

type Props = { label: string; tone?: "info" | "success" | "warn" | "neutral"; className?: string };

export const Badge = ({ label, tone = "info", className }: Props) => {
  const toneClass: Record<typeof tone, string> = {
    info: "bg-vibe-100 text-vibe-700 dark:bg-white/10 dark:text-white",
    success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-100",
    warn: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-100",
    neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-100",
  };
  return (
    <span className={cn("rounded-full px-3 py-1 text-xs font-semibold", toneClass[tone], className)}>
      {label}
    </span>
  );
};
