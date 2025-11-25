import { clsx, type ClassValue } from "clsx";

export const cn = (...inputs: ClassValue[]) => clsx(inputs);

// Форматирует дату в удобный вид
export const formatDate = (value: string | number | Date) =>
  new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));

// Форматирует длительность в м:с
export const formatDuration = (seconds: number) => {
  const m = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};

// Имитируем задержку сети, чтобы UI выглядел реалистично
export const withDelay = async <T>(data: T, latency = 450): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(data), latency));
