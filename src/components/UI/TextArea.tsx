"use client";
import { TextareaHTMLAttributes } from "react";
import { cn } from "@/utils";

type Props = TextareaHTMLAttributes<HTMLTextAreaElement> & { label?: string };

export const TextArea = ({ label, className, ...props }: Props) => (
  <label className="flex w-full flex-col gap-2 text-sm">
    {label && <span className="text-sm font-semibold text-vibe-700 dark:text-vibe-100">{label}</span>}
    <textarea
      className={cn(
        "min-h-[120px] rounded-xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 text-base shadow-inner focus:border-vibe-400 focus:outline-none focus:ring-2 focus:ring-vibe-200 dark:text-white",
        className
      )}
      {...props}
    />
  </label>
);
