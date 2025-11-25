"use client";
import { InputHTMLAttributes } from "react";
import { cn } from "@/utils";

type Props = InputHTMLAttributes<HTMLInputElement> & { label?: string; hint?: string };

export const Input = ({ label, hint, className, ...props }: Props) => {
  return (
    <label className="flex w-full flex-col gap-2 text-sm">
      {label && <span className="text-sm font-semibold text-vibe-700 dark:text-vibe-100">{label}</span>}
      <input
        className={cn(
          "w-full rounded-xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 text-base shadow-sm transition focus:border-vibe-400 focus:outline-none focus:ring-2 focus:ring-vibe-200 dark:text-white",
          className
        )}
        {...props}
      />
      {hint && <span className="text-xs text-[var(--muted)]">{hint}</span>}
    </label>
  );
};
