"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Card } from "@/components/UI/Card";
import { Badge } from "@/components/UI/Badge";

export default function ReportPage({ params }: { params: { id: string } }) {
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["report", params.id],
    queryFn: () => api.getReport(params.id),
  });

  if (isLoading || isFetching) {
    return (
      <main className="space-y-4">
        <div className="text-sm text-[var(--muted)]">Загрузка отчёта...</div>
        <h1 className="text-2xl font-semibold">Отчёт по сессии {params.id}</h1>
      </main>
    );
  }

  if (!data) return <div>Отчёт не найден</div>;

  const report: any = (data as any).report ?? data;
  const summary =
    report.candidate_answer_text ||
    report.candidate_report_text ||
    report.summary_candidate ||
    report.summary ||
    "Текст отчёта не найден";
  const score = report.score || report.metrics?.overall?.total_score || 0;
  const level = report.level || report.metrics?.overall?.level || "—";

  return (
    <main className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm text-[var(--muted)]">Итоговое досье</div>
          <h1 className="text-3xl font-semibold">Отчёт по сессии {report.id || params.id}</h1>
        </div>
        <div className="flex gap-2">
          <Badge label={`Score ${score}`} tone={score > 70 ? "success" : "warn"} />
          <Badge label={level} />
        </div>
      </div>

      <Card title="Резюме">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--muted)]">{summary}</p>
      </Card>
    </main>
  );
}
