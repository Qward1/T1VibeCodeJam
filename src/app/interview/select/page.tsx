"use client";
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { api } from "@/services/api";
import { Button } from "@/components/UI/Button";
import { Card } from "@/components/UI/Card";
import { useSessionStore } from "@/stores/session";
import { useAuthStore } from "@/stores/auth";
import { useEffect } from "react";
import { AssignedInterview } from "@/types";

const directions = ["Frontend", "Backend", "DS", "ML", "DevOps", "Fullstack"];
const difficulties = ["Junior", "Middle", "Senior"] as const;
const taskTypes = ["Coding", "Algorithms", "Debug", "Theory"];

type Level = (typeof difficulties)[number];

export default function SelectPage() {
  const router = useRouter();
  const setSession = useSessionStore((s) => s.setSession);
  const setInterviewId = useSessionStore((s) => s.setInterviewId);
  const reset = useSessionStore((s) => s.reset);
  const user = useAuthStore((s) => s.user);
  const [direction, setDirection] = useState(directions[0]);
  const [level, setLevel] = useState<Level>("Middle");
  const [format, setFormat] = useState<string>("Full interview");
  const [tasks, setTasks] = useState<string[]>(["Coding"]);
  const { data: assignedList } = useQuery<AssignedInterview[]>({
    queryKey: ["assigned-interviews"],
    queryFn: api.getAssignedInterviews,
    enabled: !!user,
    refetchInterval: 5000,
  });

  const toggleTask = (task: string) => {
    setTasks((prev) => (prev.includes(task) ? prev.filter((t) => t !== task) : [...prev, task]));
  };

  const mutation = useMutation({
    mutationFn: () => api.startInterview({ direction, level, format, tasks }),
    onSuccess: (session) => {
      reset();
      setSession(session);
      setInterviewId(session.id);
      router.push(`/interview/session/${session.id}`);
    },
  });

  const startAssigned = useMutation({
    mutationFn: (assignedId: string) => api.startAssignedInterview(assignedId),
    onSuccess: (session) => {
      reset();
      setSession(session);
      setInterviewId(session.id);
      router.push(`/interview/session/${session.id}`);
    },
  });

  useEffect(() => {
    const checkActive = async () => {
      if (!user) return;
      try {
        const active = await api.getActiveInterview();
        if (active?.id) {
          setSession(active);
          setInterviewId(active.id);
          router.replace(`/interview/session/${active.id}`);
        } else {
          reset();
        }
      } catch {
        reset();
      }
    };
    checkActive();
  }, [user, setSession, setInterviewId, router, reset]);

  const handleStart = async () => {
    try {
      const active = await api.getActiveInterview();
      if (active?.id) {
        setSession(active);
        setInterviewId(active.id);
        router.push(`/interview/session/${active.id}`);
        return;
      }
    } catch {
      // ignore, пойдём дальше
    }
    reset();
    mutation.mutate();
  };

  return (
    <main className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-[var(--muted)]">Шаг 1</div>
          <h1 className="text-3xl font-semibold">Выбор темы собеседования</h1>
        </div>
        <Button onClick={handleStart} disabled={mutation.isPending}>
          {mutation.isPending ? "Готовим..." : "Начать интервью"}
        </Button>
      </div>

      {assignedList && assignedList.length > 0 && (
        <Card title="Назначенные собеседования">
          <div className="space-y-3">
            {assignedList.map((assigned) => (
              <div key={assigned.id} className="rounded-xl border border-[var(--border)] p-3">
                <div className="grid gap-3 md:grid-cols-4">
                  <div>
                    <div className="text-sm text-[var(--muted)]">Статус</div>
                    <div className="text-lg font-semibold capitalize">{assigned.status}</div>
                  </div>
                  <div>
                    <div className="text-sm text-[var(--muted)]">Направление</div>
                    <div className="text-lg font-semibold">{assigned.direction}</div>
                  </div>
                  <div>
                    <div className="text-sm text-[var(--muted)]">Уровень</div>
                    <div className="text-lg font-semibold">{assigned.level}</div>
                  </div>
                  <div>
                    <div className="text-sm text-[var(--muted)]">Задачи</div>
                    <div className="text-sm text-[var(--muted)]">{assigned.tasks?.join(", ") || "—"}</div>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <div className="text-sm text-[var(--muted)]">
                    Назначено админом. Нажмите, чтобы продолжить или начать назначенное интервью.
                  </div>
                  <Button
                    onClick={() => assigned.id && startAssigned.mutate(assigned.id)}
                    disabled={startAssigned.isPending}
                    className="bg-gradient-to-r from-vibe-500 to-vibe-700 text-white"
                  >
                    {startAssigned.isPending ? "Запускаем..." : assigned.status === "active" ? "Продолжить" : "Приступить к интервью"}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Направление">
          <div className="flex flex-wrap gap-2">
            {directions.map((d) => (
              <button
                key={d}
                onClick={() => setDirection(d)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  d === direction
                    ? "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white"
                    : "border border-[var(--border)] text-[var(--muted)] hover:border-vibe-300"
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </Card>

        <Card title="Сложность">
          <div className="flex flex-wrap gap-3">
            {difficulties.map((lvl) => (
              <button
                key={lvl}
                onClick={() => setLevel(lvl)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  lvl === level
                    ? "bg-vibe-600 text-white"
                    : "border border-[var(--border)] text-[var(--muted)] hover:border-vibe-300"
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Типы задач">
        <div className="flex flex-wrap gap-2">
          {taskTypes.map((t) => {
            const active = tasks.includes(t);
            return (
              <button
                key={t}
                onClick={() => toggleTask(t)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  active
                    ? "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white"
                    : "border border-[var(--border)] text-[var(--muted)] hover:border-vibe-300"
                }`}
              >
                {t}
              </button>
            );
          })}
        </div>
      </Card>
    </main>
  );
}
