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
  const speed = report.metrics?.speed || {};
  const overall = report.metrics?.overall || {};
  const errors = report.metrics?.errors || {};
  const formatMinutes = (sec?: number) => Math.round((sec || 0) / 60);
  const downloadTxt = () => {
    const lines = [
      `Отчёт по сессии ${report.id || params.id}`,
      `Score: ${score}`,
      `Level: ${level}`,
      "",
      "Резюме:",
      summary,
      "",
      "Скорость:",
      `- Длительность: ${formatMinutes(overall.total_time_sec)} мин`,
      `- Среднее на вопрос: ${Math.round(speed.avg_time_per_question_sec || 0)} сек`,
      `- Среднее по теории: ${Math.round(speed.avg_time_theory_sec || 0)} сек`,
      `- Среднее по коду: ${Math.round(speed.avg_time_coding_sec || 0)} сек`,
      `- Быстро решённых: ${speed.fast_questions_count ?? 0}`,
      "",
      "Ошибки:",
      `- Код — проваленных задач: ${errors.coding_failed ?? 0}`,
      `- Runtime-ошибок: ${errors.runtime_errors_total ?? 0}`,
      `- Среднее попыток по коду: ${errors.avg_attempts_per_coding ?? 0}`,
      `- Всего подсказок: ${errors.hints_used_total ?? 0}`,
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `report-${report.id || params.id}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

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
          <button
            onClick={downloadTxt}
            className="inline-flex items-center gap-2 rounded-full bg-vibe-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-vibe-700"
          >
            Скачать отчёт
          </button>
        </div>
      </div>

      <Card title="Резюме">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--muted)]">{summary}</p>
      </Card>

      <Card title="Скорость">
        {report.metrics ? (
          <div className="space-y-2 text-sm text-[var(--muted)]">
            <div>Длительность сессии: {formatMinutes(overall.total_time_sec)} мин</div>
            <div>Среднее время на вопрос: {Math.round(speed.avg_time_per_question_sec || 0)} сек</div>
            <div>Среднее по теории: {Math.round(speed.avg_time_theory_sec || 0)} сек</div>
            <div>Среднее по коду: {Math.round(speed.avg_time_coding_sec || 0)} сек</div>
            <div>Быстро решённых: {speed.fast_questions_count ?? 0}</div>
          </div>
        ) : (
          <div className="text-sm text-[var(--muted)]">Нет данных по скорости.</div>
        )}
      </Card>

      <Card title="Ошибки">
        {report.metrics ? (
          <div className="space-y-2 text-sm text-[var(--muted)]">
            <div>Код — проваленных задач: {errors.coding_failed ?? 0}</div>
            <div>Runtime-ошибок: {errors.runtime_errors_total ?? 0}</div>
            <div>Среднее попыток по коду: {errors.avg_attempts_per_coding ?? 0}</div>
            <div>Всего подсказок: {errors.hints_used_total ?? 0}</div>
          </div>
        ) : (
          <div className="text-sm text-[var(--muted)]">Нет данных по ошибкам.</div>
        )}
      </Card>
    </main>
  );
}
