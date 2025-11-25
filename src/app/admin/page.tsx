"use client";
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Card } from "@/components/UI/Card";
import { Input } from "@/components/UI/Input";
import { Badge } from "@/components/UI/Badge";
import { formatDate } from "@/utils";
import { useAuthStore } from "@/stores/auth";
import { Button } from "@/components/UI/Button";
import { useEffect } from "react";
import { Select } from "@/components/UI/Select";

export default function AdminPage() {
  const user = useAuthStore((s) => s.user);
  const isAdmin = !!user?.admin || user?.role === "superadmin";
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ["admin"], queryFn: api.getAdminOverview, enabled: isAdmin });
  const { data: events } = useQuery({
    queryKey: ["admin-events"],
    queryFn: api.getAdminEvents,
    enabled: isAdmin,
    refetchInterval: 5000,
  });
  const { data: inbox } = useQuery({
    queryKey: ["support-inbox"],
    queryFn: api.getSupportInbox,
    enabled: isAdmin,
    refetchInterval: 5000,
  });
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [supportHistory, setSupportHistory] = useState<any[]>([]);
  const [supportInput, setSupportInput] = useState("");
  const supportMutation = useMutation({
    mutationFn: ({ userId, message }: { userId: string; message: string }) => api.sendSupportReply(userId, message),
    onSuccess: async (_, vars) => {
      const msgs = await api.getSupportHistory(vars.userId);
      setSupportHistory(msgs);
      setSupportInput("");
      queryClient.invalidateQueries({ queryKey: ["support-inbox"] });
    },
  });
  const closeMutation = useMutation({
    mutationFn: (userId: string) => api.closeSupportDialog(userId),
    onSuccess: () => {
      setSelectedUser(null);
      setSupportHistory([]);
      queryClient.invalidateQueries({ queryKey: ["support-inbox"] });
    },
  });
  // Подгружаем историю при выборе пользователя
  useEffect(() => {
    const load = async () => {
      if (!selectedUser) return;
      const msgs = await api.getSupportHistory(selectedUser);
      setSupportHistory(msgs);
    };
    load();
  }, [selectedUser]);
  const clearEvents = useMutation({
    mutationFn: () => api.clearAdminEvents(),
    onSuccess: () => {
      queryClient.setQueryData(["admin-events"], []);
    },
  });
  const [search, setSearch] = useState("");
  const isSuper = user?.role === "superadmin";
  const [tab, setTab] = useState<"overview" | "sessions" | "config">("overview");
  const [sessionFilters, setSessionFilters] = useState<{ level?: string; status?: string }>({});
  const { data: sessions } = useQuery({
    queryKey: ["admin-sessions", sessionFilters],
    queryFn: () => api.getAdminSessions(sessionFilters),
    enabled: isAdmin,
  });
  const [selectedSession, setSelectedSession] = useState<any | null>(null);
  const sessionDetailQuery = useQuery({
    queryKey: ["admin-session-detail", selectedSession?.id],
    queryFn: () => (selectedSession ? api.getAdminSessionDetail(selectedSession.id) : null),
    enabled: isAdmin && !!selectedSession?.id,
  });
  const { data: adminConfig } = useQuery({
    queryKey: ["admin-config"],
    queryFn: api.getAdminConfig,
    enabled: isAdmin,
  });
  const saveConfigMutation = useMutation({
    mutationFn: (cfg: any) => api.saveAdminConfig(cfg),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin-config"] }),
  });

  const grantMutation = useMutation({
    mutationFn: (targetUserId: string) => api.grantAdmin(targetUserId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin"] }),
  });
  const revokeMutation = useMutation({
    mutationFn: (targetUserId: string) => api.revokeAdmin(targetUserId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin"] }),
  });

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.candidates.filter(
      (c) => c.name.toLowerCase().includes(search.toLowerCase()) || c.email.toLowerCase().includes(search.toLowerCase())
    );
  }, [data, search]);

  if (!isAdmin) {
    return (
      <main className="space-y-4">
        <Card title="Нет доступа">
          <p className="text-sm text-[var(--muted)]">Требуются права администратора. Обратитесь к супер-админу для выдачи.</p>
        </Card>
      </main>
    );
  }

  return (
    <main className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-[var(--muted)]">Админ-панель</div>
          <h1 className="text-3xl font-semibold">Кандидаты и анти-чит</h1>
        </div>
      </div>
      <div className="flex gap-2">
        <Button variant={tab === "overview" ? "primary" : "outline"} onClick={() => setTab("overview")}>
          Обзор
        </Button>
        <Button variant={tab === "sessions" ? "primary" : "outline"} onClick={() => setTab("sessions")}>
          Сессии
        </Button>
        <Button variant={tab === "config" ? "primary" : "outline"} onClick={() => setTab("config")}>
          Настройки
        </Button>
      </div>

      {tab === "overview" && (
        <>
      <Card title="Список кандидатов">
        <div className="mb-3">
          <Input placeholder="Поиск по имени или email" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-[var(--muted)]">
              <tr>
                <th className="py-2">Имя</th>
                <th>Email</th>
                <th>Уровень</th>
                <th>Роль</th>
                <th>Тема</th>
                <th>Score</th>
                <th></th>
                {isSuper && <th></th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {filtered.map((c) => (
                <tr
                  key={c.id}
                  className={`hover:bg-vibe-50/60 dark:hover:bg-white/5 ${
                    c.hasFlags ? "bg-rose-50/70 dark:bg-rose-900/10" : ""
                  }`}
                >
                  <td className="py-2 font-semibold">{c.name}</td>
                  <td className="text-[var(--muted)]">{c.email}</td>
                  <td>{c.level}</td>
                  <td className="text-sm">{c.role ?? (c.admin ? "admin" : "user")}</td>
                  <td>{c.lastTopic}</td>
                  <td>
                    <Badge label={`${c.lastScore}`} tone={c.lastScore > 80 ? "success" : "info"} />
                  </td>
                  <td>
                    <a
                      href={`/report/${c.id}`}
                      className="text-sm text-vibe-600 underline decoration-dotted underline-offset-4"
                    >
                      Отчёт
                    </a>
                    {c.hasFlags && (
                      <div className="text-xs text-rose-500">Анти-чит: {c.flagsCount ?? 0}</div>
                    )}
                  </td>
                  {isSuper && (
                    <td className="space-x-2 whitespace-nowrap">
                      {c.id !== "super-1" && (
                        c.admin ? (
                          <Button
                            variant="outline"
                            onClick={() => revokeMutation.mutate(c.id)}
                            disabled={revokeMutation.isPending}
                          >
                            {revokeMutation.isPending ? "Снимаем..." : "Снять админа"}
                          </Button>
                        ) : (
                          <Button
                            variant="outline"
                            onClick={() => grantMutation.mutate(c.id)}
                            disabled={grantMutation.isPending}
                          >
                            {grantMutation.isPending ? "Выдаём..." : "Сделать админом"}
                          </Button>
                        )
                      )}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Вопросы поддержки">
          <div className="space-y-2">
            {(inbox ?? []).map((item) => (
              <div
                key={item.userId}
                className={`flex cursor-pointer items-center justify-between rounded-xl border border-[var(--border)] px-3 py-2 transition hover:border-vibe-300 ${
                  selectedUser === item.userId ? "border-vibe-400 bg-vibe-50 dark:bg-white/5" : ""
                }`}
                onClick={() => setSelectedUser(item.userId)}
              >
                <div>
                  <div className="font-semibold">Пользователь: {item.userName || item.userId}</div>
                  <div className="text-sm text-[var(--muted)] truncate">Последнее: {item.lastMessage}</div>
                </div>
                <Badge label={item.status === "new" ? "Новое" : "Открыто"} tone={item.status === "new" ? "warning" : "info"} />
              </div>
            ))}
            {(inbox ?? []).length === 0 && <div className="text-sm text-[var(--muted)]">Обращений пока нет</div>}
          </div>
        </Card>
        <Card title="Чат поддержки">
          {selectedUser ? (
            <div className="flex h-[420px] flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="text-sm text-[var(--muted)]">
                  Диалог с {inbox?.find((i: any) => i.userId === selectedUser)?.userName || selectedUser}
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => selectedUser && closeMutation.mutate(selectedUser)}
                  disabled={closeMutation.isPending}
                  className="text-rose-600 hover:border-rose-300 hover:bg-rose-50"
                >
                  {closeMutation.isPending ? "Закрываем..." : "Закрыть диалог"}
                </Button>
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto rounded-xl border border-[var(--border)] p-2">
                {supportHistory.length === 0 && (
                  <div className="text-center text-sm text-[var(--muted)]">Сообщений пока нет</div>
                )}
                {supportHistory.map((m) => (
                  <div key={m.id} className={`flex ${m.sender === "admin" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${
                        m.sender === "admin"
                          ? "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white"
                          : "bg-vibe-50 text-vibe-900 dark:bg-white/10 dark:text-white"
                      }`}
                    >
                      {m.message}
                      <div className="mt-1 text-[10px] text-white/70 dark:text-[var(--muted)]">
                        {formatDate(m.timestamp || m.createdAt || new Date().toISOString())}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <input
                  value={supportInput}
                  onChange={(e) => setSupportInput(e.target.value)}
                  placeholder="Ответить..."
                  className="flex-1 rounded-full border border-[var(--border)] bg-transparent px-3 py-2 text-sm"
                />
                <Button
                  size="sm"
                  onClick={() => {
                    if (!supportInput.trim() || !selectedUser) return;
                    supportMutation.mutate({ userId: selectedUser, message: supportInput.trim() });
                  }}
                  disabled={supportMutation.isPending}
                >
                  {supportMutation.isPending ? "Отправляем..." : "Отправить"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-sm text-[var(--muted)]">Выберите обращение слева</div>
          )}
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Анти-чит события" className="lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm text-[var(--muted)]">Последние триггеры</div>
            <button
              className="rounded-full border border-rose-300 px-3 py-1 text-xs font-semibold text-rose-600 transition hover:bg-rose-50"
              onClick={() => clearEvents.mutate()}
              disabled={clearEvents.isPending}
            >
              {clearEvents.isPending ? "Чистим..." : "Очистить историю"}
            </button>
          </div>
          <div className="space-y-3 text-sm">
            {(events ?? []).map((evt) => (
              <div
                key={evt.id}
                className="flex items-center justify-between rounded-2xl border border-[var(--border)] px-3 py-2"
              >
                <div>
                  <div className="font-semibold">{evt.eventType}</div>
                  <div className="text-[var(--muted)]">
                    {evt.payload || "—"} • риск: {evt.risk}
                  </div>
                  <div className="text-xs text-[var(--muted)]">Сессия: {evt.sessionId}</div>
                </div>
                <div className="text-right text-xs text-[var(--muted)]">
                  {formatDate(evt.createdAt)}
                  <div>
                    <a
                      href={`/interview/session/${evt.sessionId}`}
                      className="text-vibe-600 underline decoration-dotted"
                    >
                      Открыть сессию
                    </a>
                  </div>
                </div>
              </div>
            ))}
            {(events ?? []).length === 0 && <div className="text-[var(--muted)]">Событий пока нет</div>}
          </div>
        </Card>
        <Card title="Глобальная аналитика">
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span>Completion rate</span>
              <Badge label={`${Math.round((data?.analytics.completionRate ?? 0) * 100)}%`} />
            </div>
            <div className="flex items-center justify-between">
              <span>Avg score</span>
              <Badge label={`${Math.round((data?.analytics.avgScore ?? 0) * 100)}/100`} tone="success" />
            </div>
            <div>
              <div className="text-[var(--muted)]">Самые сложные темы</div>
              <div className="mt-2 space-y-2">
                {data?.analytics.hardestTopics.map((t) => (
                  <div key={t.name} className="flex items-center justify_between rounded-xl bg-vibe-50 px-3 py-2 text-vibe-800 dark:bg-white/5 dark:text-white">
                    <span>{t.name}</span>
                    <span>{t.score}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      </div>
      </>
      )}

      {tab === "sessions" && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Сессии">
            <div className="mb-2 flex gap-2">
              <select
                className="rounded-full border border-[var(--border)] px-3 py-2 text-sm"
                value={sessionFilters.level || ""}
                onChange={(e) => setSessionFilters((f) => ({ ...f, level: e.target.value || undefined }))}
              >
                <option value="">Все уровни</option>
                <option value="junior">Junior</option>
                <option value="middle">Middle</option>
                <option value="senior">Senior</option>
              </select>
              <select
                className="rounded-full border border-[var(--border)] px-3 py-2 text-sm"
                value={sessionFilters.status || ""}
                onChange={(e) => setSessionFilters((f) => ({ ...f, status: e.target.value || undefined }))}
              >
                <option value="">Все статусы</option>
                <option value="active">Active</option>
                <option value="finished">Finished</option>
              </select>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-[var(--muted)]">
                  <tr>
                    <th>Кандидат</th>
                    <th>Дата</th>
                    <th>Направление</th>
                    <th>Уровень</th>
                    <th>Чит</th>
                    <th>Статус</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {(sessions ?? []).map((s) => (
                    <tr
                      key={s.id}
                      className="cursor-pointer hover:bg-vibe-50/60 dark:hover:bg-white/5"
                      onClick={() => setSelectedSession(s)}
                    >
                      <td className="py-2 font-semibold">{s.candidate}</td>
                      <td className="text-[var(--muted)]">{formatDate(s.createdAt)}</td>
                      <td>{s.direction}</td>
                      <td>{s.level}</td>
                      <td>
                        <Badge label={`${s.cheat_score ?? 0}`} tone={(s.cheat_score ?? 0) > 5 ? "warning" : "info"} />
                      </td>
                      <td>{s.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(sessions ?? []).length === 0 && <div className="mt-2 text-sm text-[var(--muted)]">Сессий нет</div>}
            </div>
          </Card>
          <Card title="Детали сессии" className="lg:col-span-2">
            {sessionDetailQuery.data ? (
              <div className="space-y-2 text-sm">
                <div className="font-semibold">Сессия: {sessionDetailQuery.data.session.id}</div>
                <div>Направление: {sessionDetailQuery.data.session.direction}</div>
                <div>Уровень: {sessionDetailQuery.data.session.current_level || sessionDetailQuery.data.session.level}</div>
                <div>Статус: {sessionDetailQuery.data.session.status}</div>
                <div>Чит-скор: {sessionDetailQuery.data.session.cheat_score ?? 0}</div>
                <div className="mt-2 font-semibold">Вопросы</div>
                <ul className="space-y-1">
                  {sessionDetailQuery.data.questions.map((q: any) => (
                    <li key={q.id} className="rounded-lg border border-[var(--border)] px-2 py-1">
                      {q.title} ({q.status})</li>
                  ))}
                </ul>
                <div className="mt-2 font-semibold">Метрики</div>
                <pre className="rounded-lg bg-[var(--card)] p-2 text-xs whitespace-pre-wrap">
                  {JSON.stringify(sessionDetailQuery.data.metrics || {}, null, 2)}
                </pre>
                <div className="mt-2 font-semibold">Чат</div>
                <div className="max-h-48 overflow-y-auto space-y-1 text-xs">
                  {sessionDetailQuery.data.messages.map((m: any) => (
                    <div key={m.id} className={`flex ${m.role === "assistant" ? "justify-start" : "justify-end"}`}>
                      <div className="rounded-xl bg-[var(--card)] px-2 py-1">{m.content}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-[var(--muted)]">Выберите сессию</div>
            )}
          </Card>
        </div>
      )}

      {tab === "config" && (
        <Card title="Настройки">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="text-sm">
              Модель задач
              <Input
                value={(adminConfig?.task_model as string) || "qwen3-coder-30b-a3b-instruct-fp8"}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), task_model: e.target.value })}
              />
            </label>
            <label className="text-sm">
              Модель объяснений
              <Input
                value={(adminConfig?.chat_model as string) || "qwen3-32b-awq"}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), chat_model: e.target.value })}
              />
            </label>
            <label className="text-sm">
              Temperature
              <Input
                value={(adminConfig?.temperature as string) || "0.4"}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), temperature: e.target.value })}
              />
            </label>
            <label className="text-sm">
              Top_p
              <Input
                value={(adminConfig?.top_p as string) || "0.9"}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), top_p: e.target.value })}
              />
            </label>
            <label className="text-sm">
              Max tokens
              <Input
                value={(adminConfig?.max_tokens as string) || "900"}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), max_tokens: e.target.value })}
              />
            </label>
            <label className="text-sm md:col-span-2">
              Prompt генерации задач (system)
              <textarea
                className="mt-1 w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-2"
                rows={3}
                value={(adminConfig?.prompt_task as string) || ""}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), prompt_task: e.target.value })}
              />
            </label>
            <label className="text-sm md:col-span-2">
              Prompt оценки кода (system)
              <textarea
                className="mt-1 w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-2"
                rows={3}
                value={(adminConfig?.prompt_review as string) || ""}
                onChange={(e) => saveConfigMutation.mutate({ ...(adminConfig || {}), prompt_review: e.target.value })}
              />
            </label>
          </div>
        </Card>
      )}
    </main>
  );
}
