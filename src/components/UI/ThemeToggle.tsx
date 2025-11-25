"use client";
import { useThemeStore } from "@/stores/theme";

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useThemeStore();
  return (
    <button
      onClick={toggleTheme}
      className="rounded-full border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--muted)] shadow-sm hover:border-vibe-400"
    >
      {theme === "light" ? "🌙 Тёмная" : "☀️ Светлая"}
    </button>
  );
};
