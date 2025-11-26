"use client";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Card } from "@/components/UI/Card";
import { Badge } from "@/components/UI/Badge";
import { Timeline } from "@/components/Report/Timeline";
import { SolutionCard } from "@/components/Report/SolutionCard";
import { RadarChart } from "@/components/Charts/RadarChart";
import { Heatmap } from "@/components/Charts/Heatmap";
import { Sparkline } from "@/components/Charts/Sparkline";

export default function ReportPage({ params }: { params: { id: string } }) {
  const [progress, setProgress] = useState(12);
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["report", params.id],
    queryFn: () => api.getReport(params.id),
  });

  useEffect(() => {
    if (!isLoading && !isFetching) return;
    const timer = setInterval(() => {
      setProgress((p) => (p >= 90 ? 90 : p + 5));
    }, 300);
    return () => clearInterval(timer);
  }, [isLoading, isFetching]);

  if (isLoading || isFetching) {
    return (
      <main className="space-y-4">
        <div>
          <div className="text-sm text-[var(--muted)]">Формируем отчёт...</div>
          <h1 className="text-2xl font-semibold">Отчёт по сессии {params.id}</h1>
        </div>
        <div className="w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
          <div className="mb-2 text-sm text-[var(--muted)]">Ожидайте, идёт генерация</div>
          <div className="relative h-3 w-full overflow-hidden rounded-full bg-[var(--border)]">
            <div
              className="h-full rounded-full bg-gradient-to-r from-vibe-500 to-vibe-700 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </main>
    );
  }

  if (!data) return <div>Отчёт не найден</div>;

  return (
    <main className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-sm text-[var(--muted)]">Итоговое досье</div>
          <h1 className="text-3xl font-semibold">Отчёт по сессии {data.id}</h1>
        </div>
        <div className="flex gap-2">
          <Badge label={`Score ${data.score}`} tone={data.score > 70 ? "success" : "warn"} />
          <Badge label={data.level} />
        </div>
      </div>

      <Card title="Резюме">
        <p className="text-sm leading-relaxed text-[var(--muted)]">{data.summary}</p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Хронология">
          <Timeline events={data.timeline} />
        </Card>
        <Card title="Навыки">
          <RadarChart data={data.analytics.skillMap} />
        </Card>
        <Card title="Ошибки">
          <Heatmap data={data.analytics.errorHeatmap} />
        </Card>
      </div>

      <Card title="Скорость">
        <Sparkline points={data.analytics.speed} />
      </Card>

      <div className="space-y-4">
        {data.solutions.map((sol) => (
          <SolutionCard key={sol.title} title={sol.title} code={sol.code} errors={sol.errors} tests={sol.tests} />
        ))}
      </div>
    </main>
  );
}
