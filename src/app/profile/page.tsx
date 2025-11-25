"use client";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Card } from "@/components/UI/Card";
import { RadarChart } from "@/components/Charts/RadarChart";
import { Heatmap } from "@/components/Charts/Heatmap";
import { Sparkline } from "@/components/Charts/Sparkline";
import { Badge } from "@/components/UI/Badge";
import { StatCard } from "@/components/UI/StatCard";
import { formatDate } from "@/utils";
import { Button } from "@/components/UI/Button";
import { Switch } from "@/components/UI/Switch";
import { useThemeStore } from "@/stores/theme";
import { useEffect, useState } from "react";
import { useAuthStore } from "@/stores/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function ProfilePage() {
  const { theme, toggleTheme } = useThemeStore();
  const [language, setLanguage] = useState<"ru" | "en">("ru");
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: api.getProfile });
  const { data: history } = useQuery({ queryKey: ["history"], queryFn: api.getInterviewHistory });
  const logout = useAuthStore((s) => s.logout);
  const setUser = useAuthStore((s) => s.setUser);
  const router = useRouter();
  const changeLangMutation = useMutation({
    mutationFn: (lang: "ru" | "en") => api.changeLanguage(lang),
    onSuccess: (user) => {
      setLanguage(user.lang ?? "ru");
      setUser(user);
    },
  });

  useEffect(() => {
    if (profile?.user.lang) setLanguage(profile.user.lang);
  }, [profile?.user.lang]);

  return (
    <main className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm text-[var(--muted)]">Профиль</div>
          <h1 className="text-3xl font-semibold">{profile?.user.name ?? "Загрузка"}</h1>
          <div className="flex gap-2 text-sm text-[var(--muted)]">
            <span>{profile?.user.email}</span>
            <span>•</span>
            <span>{profile?.user.level}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Badge label="Стабильность" tone="info" />
          <Badge label="Заданий решено: 48" tone="neutral" />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Навыки">
          {profile && <RadarChart data={profile.stats.skillMap} />}
        </Card>
        <Card title="Ошибки">
          {profile && <Heatmap data={profile.stats.errorHeatmap} />}
        </Card>
        <Card title="Среднее время решения">
          {profile && (
            <div className="flex flex-col gap-4">
              <div className="text-4xl font-semibold">{profile.stats.avgSolveTime} мин</div>
              <Sparkline
                points={profile.stats.skillMap.map((s, idx) => ({ label: s.label, value: s.value - idx * 5 }))}
              />
            </div>
          )}
        </Card>
      </div>

      <Card title="История собеседований">
        <div className="space-y-3">
          {history?.map((item) => (
            <div
              key={item.id}
              className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-[var(--border)] bg-[var(--card)] px-4 py-3"
            >
              <div>
                <div className="font-semibold">{item.topic}</div>
                <div className="text-sm text-[var(--muted)]">{item.direction} • {item.level}</div>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <Badge label={`Score ${item.score}`} tone={item.score > 80 ? "success" : "info"} />
                <span className="text-[var(--muted)]">{formatDate(item.date)}</span>
                <a
                  href={`/report/${item.id}`}
                  className="inline-flex items-center rounded-full border border-vibe-300 px-4 py-2 text-sm font-semibold text-vibe-700 transition hover:bg-vibe-50 dark:border-white/20 dark:text-white dark:hover:bg-white/10"
                >
                  Отчёт
                </a>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Настройки">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <Switch checked={theme === "dark"} onChange={toggleTheme} label="Тёмная тема" />
          <div className="flex items-center gap-3 rounded-xl border border-[var(--border)] px-3 py-2">
            <span className="text-[var(--muted)]">Язык</span>
            <div className="flex gap-2">
              {(["ru", "en"] as const).map((lng) => (
                <button
                  key={lng}
                  onClick={() => changeLangMutation.mutate(lng)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    language === lng
                      ? "bg-vibe-600 text-white"
                      : "border border-[var(--border)] text-[var(--muted)]"
                  }`}
                >
                  {lng.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <Link href="/profile/change-password">
            <Button variant="outline">Сменить пароль</Button>
          </Link>
        </div>
      </Card>

      <div className="mt-4 flex justify-center">
        <Button
          variant="outline"
          className="scale-[1.2] border-rose-300 bg-rose-50 text-rose-700 shadow-sm hover:bg-rose-100 dark:border-rose-500/60 dark:bg-rose-900/20 dark:text-rose-200 dark:hover:bg-rose-900/30"
          onClick={() => {
            logout();
            router.push("/");
          }}
        >
          Выйти из аккаунта
        </Button>
      </div>
    </main>
  );
}
