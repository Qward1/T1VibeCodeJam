import { TestResult } from "@/types";
import { Badge } from "@/components/UI/Badge";

export const TestResults = ({ result }: { result?: TestResult }) => {
  if (!result) return null;
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="font-semibold">Результат проверки</div>
        <Badge label={result.passed ? "Успех" : "Есть ошибки"} tone={result.passed ? "success" : "warn"} />
      </div>
      <p className="text-sm text-[var(--muted)]">{result.summary}</p>
      <div className="mt-3 space-y-2 text-sm">
        {result.cases.map((c) => (
          <div key={c.name} className="flex items-center justify-between rounded-xl bg-vibe-50 px-3 py-2 dark:bg-white/5">
            <span>{c.name}</span>
            <span className={c.passed ? "text-emerald-500" : "text-amber-500"}>{c.passed ? "OK" : "Fail"}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
