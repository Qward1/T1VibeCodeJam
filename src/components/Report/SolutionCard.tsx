import { TestResult } from "@/types";
import { TestResults } from "@/components/Interview/TestResults";

export const SolutionCard = ({
  title,
  code,
  errors,
  tests,
}: {
  title: string;
  code: string;
  errors?: string;
  tests: TestResult;
}) => (
  <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] p-4">
    <div className="mb-2 flex items-center justify-between">
      <div className="text-lg font-semibold">{title}</div>
      {errors && <span className="rounded-full bg-amber-100 px-3 py-1 text-xs text-amber-700 dark:bg-amber-900/40 dark:text-amber-100">{errors}</span>}
    </div>
    <pre className="overflow-x-auto rounded-xl bg-ink/90 p-4 text-sm text-emerald-50">
      <code>{code}</code>
    </pre>
    <div className="mt-3">
      <TestResults result={tests} />
    </div>
  </div>
);
