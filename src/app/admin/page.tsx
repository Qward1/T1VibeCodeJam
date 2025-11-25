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
    </main>
  );
}
