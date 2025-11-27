"use client";
import Link from "next/link";
import { Button } from "@/components/UI/Button";
import { Card } from "@/components/UI/Card";
import { useAuthStore } from "@/stores/auth";
import { useTranslation } from "@/utils/i18n";

const highlights = [
  {
    title: "Навыковые метрики",
    desc: "Отслеживаем скорость решений, ошибки, сложность задач и формируем точный отчет",
  },
  {
    title: "Умный античит",
    desc: "Невидимая защита: фиксирует попытки списывания, вкладки, вставки кода и подозрительные действия",
  },
  {
    title: "Уникальные и интересные задачи",
    desc: "Подбираем задания под уровень кандидата: редкие кейсы, реальные сценарии, живые задачи из индустрии и адаптивная генерация под стиль решения",
  },
];

export default function Home() {
  const user = useAuthStore((s) => s.user);
  const { t } = useTranslation();
  return (
    <main className="space-y-8">
      <section className="glass rounded-3xl border border-[var(--border)] bg-[var(--card)]/80 px-8 py-10 shadow-floating">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="space-y-4 md:max-w-xl">
            
            <h1 className="text-4xl font-bold leading-tight">Платформа технических собеседований с использованием AI</h1>
            <div className="flex flex-wrap gap-3">
              {!user && (
                <Link href="/auth/register">
                  <Button>{t("heroRegister")}</Button>
                </Link>
              )}
              <Link href="/interview/select">
                <Button
                  variant="outline"
                  size="lg"
                  className="border-vibe-300 bg-vibe-50 text-vibe-700 hover:border-vibe-400 hover:bg-vibe-100 dark:border-vibe-500/50 dark:bg-vibe-900/40 dark:text-vibe-100 dark:hover:bg-vibe-800/70"
                >
                  {t("heroStart")}
                </Button>
              </Link>
            </div>
          </div>
          <div className="relative w-full max-w-md overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--card)]/70 p-0 shadow-floating">
            <img src="/mock-preview.png" alt="Mock preview" className="h-full w-full object-cover" />
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {highlights.map((item) => (
          <Card key={item.title} title={item.title}>
            <p className="text-sm text-[var(--muted)]">{item.desc}</p>
          </Card>
        ))}
      </section>

    </main>
  );
}
