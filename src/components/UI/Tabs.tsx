"use client";
import { cn } from "@/utils";

type Tab = { value: string; label: string };

type Props = {
  value: string;
  onChange: (value: string) => void;
  tabs: Tab[];
};

export const Tabs = ({ value, onChange, tabs }: Props) => (
  <div className="flex items-center gap-2 rounded-full bg-[var(--card)] p-2 shadow-inner">
    {tabs.map((tab) => (
      <button
        key={tab.value}
        onClick={() => onChange(tab.value)}
        className={cn(
          "rounded-full px-4 py-2 text-sm font-semibold transition",
          tab.value === value
            ? "bg-vibe-500 text-white shadow"
            : "text-[var(--muted)] hover:bg-vibe-50 dark:hover:bg-white/10"
        )}
      >
        {tab.label}
      </button>
    ))}
  </div>
);
