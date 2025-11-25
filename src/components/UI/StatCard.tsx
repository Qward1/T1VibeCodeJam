import { ReactNode } from "react";
import { cn } from "@/utils";

type Props = {
  label: string;
  value: string | number;
  hint?: string;
  icon?: ReactNode;
  className?: string;
};

export const StatCard = ({ label, value, hint, icon, className }: Props) => (
  <div className={cn("glass rounded-2xl border border-[var(--border)] p-4", className)}>
    <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
      {icon && <span className="text-lg text-vibe-500">{icon}</span>}
      <span>{label}</span>
    </div>
    <div className="mt-2 text-2xl font-semibold">{value}</div>
    {hint && <div className="text-xs text-[var(--muted)]">{hint}</div>}
  </div>
);
