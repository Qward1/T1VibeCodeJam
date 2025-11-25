"use client";
import { cn } from "@/utils";

type BaseSwitchProps = {
  checked: boolean;
  onChange: (next: boolean) => void;
};

type Props = BaseSwitchProps & { label?: string };

export const Switch = ({ checked, onChange, label }: Props) => (
  <button
    type="button"
    onClick={() => onChange(!checked)}
    className="flex items-center gap-3 text-sm"
  >
    <span
      className={cn(
        "h-6 w-11 rounded-full border border-[var(--border)] px-[2px] py-[2px] transition",
        checked ? "bg-vibe-500" : "bg-[var(--card)]"
      )}
    >
      <span
        className={cn(
          "block h-full w-1/2 rounded-full bg-white shadow-sm transition-transform",
          checked ? "translate-x-5" : "translate-x-0"
        )}
      />
    </span>
    {label && <span className="text-[var(--muted)]">{label}</span>}
  </button>
);
