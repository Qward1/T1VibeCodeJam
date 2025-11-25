"use client";

import { useEffect, useMemo, useState } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { Button } from "@/components/UI/Button";
import { Card } from "@/components/UI/Card";
import { TextArea } from "@/components/UI/TextArea";
import { formatDuration } from "@/utils";

// Простой экран интервью без привязки к API: сохраняем разметку, цвета и таймер,
// чтобы дальше можно было строить новую логику поверх готового каркаса.

const DEFAULT_DURATION = 45 * 60; // 45 минут

export default function InterviewSkeletonPage() {
  const router = useRouter();
  const { sessionId } = router.query as { sessionId?: string };
  const [timeLeft, setTimeLeft] = useState<number>(DEFAULT_DURATION);
  const [code, setCode] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  const [finished, setFinished] = useState<boolean>(false);

  // Таймер с отсчётом до 0
  useEffect(() => {
    setTimeLeft(DEFAULT_DURATION);
    const startedAt = Date.now();
    const tick = () => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      setTimeLeft((prev) => Math.max(0, DEFAULT_DURATION - elapsed));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [sessionId]);

  const progress = useMemo(() => {
    return Math.round(((DEFAULT_DURATION - timeLeft) / DEFAULT_DURATION) * 100);
  }, [timeLeft]);

  const handleFinish = () => {
    setFinished(true);
  };

  return (
    <div className="min-h-screen bg-mesh bg-cover bg-fixed px-3 py-4 text-sm">
      <Head>
        <title>Интервью · Новый каркас</title>
      </Head>

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 shadow-lg">
        <div>
          <div className="text-xs text-[var(--muted)]">
            Сессия: {sessionId || "новая"} · Задача 1/3 · Уровень Middle
          </div>
          <div className="text-lg font-semibold">Каркас страницы интервью</div>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-vibe-50 px-3 py-2 text-xs font-semibold text-vibe-800 dark:bg-white/10 dark:text-white">
            ⏳ {formatDuration(timeLeft)}
          </div>
          <div className="rounded-full bg-emerald-100 px-3 py-2 text-xs font-semibold text-emerald-700">
            Прогресс: {progress}%
          </div>
          <Button variant="outline" onClick={handleFinish}>
            Завершить собеседование
          </Button>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[2fr_1.2fr]">
        <div className="space-y-3">
          <Card title="Редактор (заглушка)">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-3">
              <TextArea
                rows={10}
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Здесь будет редактор кода или Monaco. Сейчас это простая текстовая область."
              />
            </div>
            <div className="mt-3 flex gap-2">
              <Button className="bg-[rgba(109,65,128,0.25)] text-[rgb(109,65,128)] border border-[rgba(109,65,128,0.55)]">
                Запустить код
              </Button>
              <Button className="bg-gradient-to-r from-vibe-500 to-vibe-700 text-white">
                Отправить решение
              </Button>
            </div>
          </Card>

          <Card title="Видимые тесты">
            <div className="space-y-2 text-sm text-[var(--muted)]">
              <div className="rounded-xl border border-dashed border-[var(--border)] px-3 py-2">
                Тесты будут подставляться здесь после интеграции с новой логикой.
              </div>
            </div>
          </Card>

          <Card title="Вывод">
            <div className="text-xs text-[var(--muted)]">stdout / stderr</div>
            <div className="rounded-lg bg-[var(--card)] p-2 text-xs text-[var(--muted)]">Нет данных — ждём новую логику.</div>
          </Card>
        </div>

        <div className="space-y-3">
          <Card title="Чат (заглушка)">
            <div className="rounded-xl border border-dashed border-[var(--border)] bg-[var(--card)] p-3 text-sm text-[var(--muted)]">
              Здесь появится чат с интервьюером/поддержкой после обновления логики.
            </div>
          </Card>

          <Card title="Заметки">
            <TextArea
              rows={6}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Личные заметки кандидата..."
            />
          </Card>

          <Card title="Метрики">
            <div className="space-y-1 text-sm">
              <div>Решено скрытых: —</div>
              <div>Попыток: —</div>
              <div>Запусков кода: —</div>
              <div>Обращений в чат: —</div>
              <div>Чит-скор: —</div>
              <div>Прогресс: {progress}%</div>
            </div>
          </Card>
        </div>
      </div>

      {finished && (
        <Card title="Сессия завершена" className="mt-4">
          <div className="text-sm text-[var(--muted)]">
            Завершили интервью. Здесь можно будет показать отчёт, когда новая логика будет готова.
          </div>
        </Card>
      )}
    </div>
  );
}
