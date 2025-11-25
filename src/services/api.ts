import {
  AdminOverview,
  ErrorHeat,
  FlaggedEvent,
  InterviewHistoryItem,
  InterviewReport,
  InterviewSession,
  Level,
  LoginPayload,
  Message,
  RegisterPayload,
  SkillStat,
  StartInterviewPayload,
  TestResult,
  User,
} from "@/types";
import { withDelay } from "@/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL;
const useBackend = Boolean(API_BASE);
const uid = () => Math.random().toString(36).slice(2, 10);

// Текущее состояние пользователя — нужно для запросов, где требуется userId
let activeUser: User = {
  id: uid(),
  email: "user@vibe.dev",
  name: "Алексей Вибров",
  level: "Middle",
  lang: "ru",
};

// Локальное хранилище сессий/вопросов для оффлайн-режима
const localQuestions = [
  {
    id: "lq1",
    title: "Два числа",
    body: "Даны числа и target. Верните индексы двух чисел, сумма которых равна target. Используйте O(n).",
    useIDE: true,
  },
  {
    id: "lq2",
    title: "Минимальный путь",
    body: "Найдите кратчайший путь в ориентированном графе без отрицательных весов.",
    useIDE: false,
  },
  {
    id: "lq3",
    title: "Дедупликация логов",
    body: "Есть поток логов. Реализуйте структуру, которая выдаёт уникальные записи за последние 5 минут.",
    useIDE: false,
  },
];
const localSessions: Record<string, InterviewSession> = {};

// Позволяем подхватывать пользователя из стора после перезагрузки страницы
export const setActiveUser = (user?: User) => {
  if (user) {
    activeUser = user;
  }
};

const call = async <T>(path: string, init?: RequestInit): Promise<T> => {
  if (!API_BASE) throw new Error("API_BASE_NOT_SET");
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
      mode: "cors",
    });
  } catch (e) {
    throw new Error("FAILED_TO_FETCH_API");
  }
  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = (data as any)?.detail || (data as any)?.error || message;
    } catch {
      message = await res.text();
    }
    throw new Error(message || "REQUEST_FAILED");
  }
  return res.json() as Promise<T>;
};

const skillMap: SkillStat[] = [
  { label: "JS/TS", value: 82 },
  { label: "React", value: 88 },
  { label: "Architecture", value: 74 },
  { label: "Algorithms", value: 69 },
  { label: "Debug", value: 77 },
];

const heatmap: ErrorHeat[] = [
  { bucket: "Off-by-one", count: 3 },
  { bucket: "Types", count: 2 },
  { bucket: "Edge cases", count: 4 },
  { bucket: "Performance", count: 1 },
];

let history: InterviewHistoryItem[] = [
  {
    id: "sess-41",
    topic: "Реактивный поиск",
    direction: "Frontend",
    level: "Middle",
    score: 84,
    date: new Date().toISOString(),
  },
  {
    id: "sess-40",
    topic: "API оптимизация",
    direction: "Fullstack",
    level: "Senior",
    score: 78,
    date: new Date(Date.now() - 86400000 * 4).toISOString(),
  },
];

const baseStarter = `function twoSum(nums, target) {\n  // Используйте хеш-таблицу для O(n)\n  const map = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const complement = target - nums[i];\n    if (map.has(complement)) return [map.get(complement), i];\n    map.set(nums[i], i);\n  }\n  return [];\n}`;

const buildTimeline = () => [
  { label: "task_start", at: new Date(Date.now() - 600000).toISOString() },
  { label: "hint_used", at: new Date(Date.now() - 540000).toISOString() },
  { label: "tests_failed", at: new Date(Date.now() - 420000).toISOString() },
  { label: "completed", at: new Date().toISOString() },
];

const defaultTestResult: TestResult = {
  passed: false,
  summary: "Тест №3 падает на кейсе с пустым массивом",
  cases: [
    { name: "Возвращает индексы", passed: true },
    { name: "Работает с дубликатами", passed: true },
    { name: "Пустой массив", passed: false, details: "ожидалось []" },
  ],
};

const adminEvents: FlaggedEvent[] = [
  {
    id: uid(),
    candidateId: "cand-1",
    type: "paste_detected",
    at: new Date(Date.now() - 900000).toISOString(),
    details: "Вставлено 54 символа кода",
  },
  {
    id: uid(),
    candidateId: "cand-2",
    type: "tab_change",
    at: new Date(Date.now() - 720000).toISOString(),
    details: "Ушёл в другую вкладку на 35 секунд",
  },
];

export const api = {
  // Выдача прав админа (для супер-админа)
  async grantAdmin(targetUserId: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { user } = await call<{ user: User }>("/api/admin/grant", {
        method: "POST",
        body: JSON.stringify({ superId: activeUser.id, targetUserId }),
      });
      return user;
    }
    return withDelay({ ...activeUser, admin: true, role: "admin" } as User);
  },

  // Снятие прав админа (для супер-админа)
  async revokeAdmin(targetUserId: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { user } = await call<{ user: User }>("/api/admin/revoke", {
        method: "POST",
        body: JSON.stringify({ superId: activeUser.id, targetUserId }),
      });
      return user;
    }
    return withDelay({ ...activeUser, admin: false, role: "user" } as User);
  },

  // Смена пароля (требует авторизованного пользователя)
  async changePassword(oldPassword: string, newPassword: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>("/api/change-password", {
        method: "POST",
        body: JSON.stringify({ userId: activeUser.id, oldPassword, newPassword }),
      });
      return true;
    }
    // В мок-режиме просто откладываем успешный ответ
    return withDelay(true);
  },

  // Регистрация нового пользователя
  async register(payload: RegisterPayload): Promise<User> {
    if (useBackend && API_BASE) {
      const { user } = await call<{ user: User }>("/api/register", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      activeUser = user;
      return user;
    }
    activeUser = {
      id: uid(),
      email: payload.email,
      name: payload.name ?? "Новый кандидат",
      level: "Junior",
      lang: payload.lang ?? "ru",
    };
    return withDelay(activeUser);
  },

  // Логин возвращает профиль пользователя
  async login(payload: LoginPayload): Promise<User> {
    if (useBackend && API_BASE) {
      const { user } = await call<{ user: User }>("/api/login", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      activeUser = user;
      return user;
    }
    activeUser = { ...activeUser, email: payload.email };
    return withDelay(activeUser);
  },

  async getProfile() {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      return call(`/api/profile?userId=${activeUser.id}`);
    }
    return withDelay({
      user: activeUser,
      stats: {
        skillMap,
        avgSolveTime: 12,
        errorHeatmap: heatmap,
      },
    });
  },

  async getInterviewHistory(): Promise<InterviewHistoryItem[]> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { history } = await call<{ history: InterviewHistoryItem[] }>(
        `/api/history?userId=${activeUser.id}`
      );
      return history;
    }
    return withDelay(history);
  },

  async changeLanguage(lang: "ru" | "en") {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { user } = await call<{ user: User }>("/api/profile/language", {
        method: "POST",
        body: JSON.stringify({ userId: activeUser.id, lang }),
      });
      activeUser = user;
      return user;
    }
    activeUser = { ...activeUser, lang };
    return withDelay(activeUser);
  },

  async sendAntiCheat(event: { sessionId: string; eventType: string; payload?: any; risk?: string }) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      try {
        await call<{ status: string }>("/api/interview/event", {
          method: "POST",
          body: JSON.stringify({
            session_id: event.sessionId,
            ownerId: activeUser.id,
            event_type: event.eventType,
            payload: event.payload ?? {},
          }),
        });
        return true;
      } catch (e) {
        console.warn("sendAntiCheat backend failed", e);
      }
    }
    return withDelay(true);
  },

  async startInterview(payload: StartInterviewPayload): Promise<InterviewSession> {
    if (useBackend && API_BASE) {
      try {
        if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
        const { session } = await call<{ session: InterviewSession }>("/api/start-interview", {
          method: "POST",
          body: JSON.stringify({ ...payload, userId: activeUser.id }),
        });
        return session;
      } catch (e) {
        // если бекенд недоступен/404 — падаем в локальный сценарий
        console.warn("Backend start-interview failed, using local fallback", e);
      }
    }
    const session: InterviewSession = {
      id: uid(),
      direction: payload.direction,
      level: payload.level,
      format: payload.format,
      tasks: payload.tasks,
      questionId: localQuestions[0].id,
      questionTitle: localQuestions[0].title,
      description: localQuestions[0].body,
      starterCode: baseStarter,
      useIDE: !!localQuestions[0].useIDE,
      timer: 45 * 60,
      startedAt: new Date().toISOString(),
      solved: 0,
      total: localQuestions.length,
      usedQuestions: localQuestions.map((q) => ({ id: q.id, title: q.title })),
    };
    localSessions[session.id] = session;
    history = [
      {
        id: session.id,
        topic: session.description.slice(0, 25) + "...",
        direction: session.direction,
        level: session.level,
        score: 0,
        date: new Date().toISOString(),
      },
      ...history,
    ];
    return withDelay(session);
  },

  async getActiveInterview(): Promise<InterviewSession | null> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      try {
        const { session } = await call<{ session: InterviewSession }>(
          `/api/session/active?ownerId=${activeUser.id}`
        );
        return session;
      } catch (e: any) {
        return null;
      }
    }
    const local = Object.values(localSessions)[0];
    return local ?? null;
  },

  async getInterviewSession(id: string, questionId?: string): Promise<InterviewSession> {
    if (useBackend && API_BASE) {
      try {
        const url = questionId ? `/api/session/${id}?questionId=${questionId}` : `/api/session/${id}`;
        const { session } = await call<{ session: InterviewSession }>(url);
        return session;
      } catch (e) {
        console.warn("Backend getInterviewSession failed, using local fallback", e);
      }
    }
    // Ищем в истории, если нет — создаём мок
    const existing = localSessions[id];
    if (existing) {
      const target = questionId
        ? localQuestions.find((q) => q.id === questionId) ?? localQuestions[0]
        : localQuestions.find((q) => q.id === existing.questionId) ?? localQuestions[0];
      return {
        ...existing,
        questionId: target.id,
        questionTitle: target.title,
        description: target.body,
        useIDE: !!target.useIDE,
        startedAt: existing.startedAt,
        timer: existing.timer,
      };
    }
    const historyItem = history.find((h) => h.id === id);
    return withDelay({
      id,
      direction: historyItem?.direction ?? "Frontend",
      level: historyItem?.level ?? "Middle",
      format: "Full interview",
      tasks: ["Coding", "Algorithms"],
      description:
        historyItem?.topic ?? "Оптимизируйте структуру данных под частые чтения",
      starterCode: baseStarter,
      timer: 35 * 60,
      startedAt: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
      solved: 1,
      total: 3,
    });
  },

  async finishInterview(sessionId: string) {
    if (useBackend && API_BASE) {
      try {
        if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
        try {
          await call<{ status: string }>(`/api/session/finish`, {
            method: "POST",
            body: JSON.stringify({ sessionId, ownerId: activeUser.id }),
          });
          return true;
        } catch {
          // Совместимость со старым роутом, если вдруг нужен
          await call<{ status: string }>(`/api/interview/finish`, {
            method: "POST",
            body: JSON.stringify({ sessionId, ownerId: activeUser.id }),
          });
          return true;
        }
      } catch (e) {
        console.warn("finishInterview backend failed, falling back", e);
      }
    }
    // Чистим локальное активное интервью
    if (localSessions[sessionId]) {
      delete localSessions[sessionId];
    }
    return withDelay(true);
  },

  async saveAnswer(sessionId: string, questionId: string, content: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>(`/api/answer`, {
        method: "POST",
        body: JSON.stringify({ sessionId, questionId, content, ownerId: activeUser.id }),
      });
      return true;
    }
    return withDelay(true);
  },

  async getAnswer(sessionId: string, questionId: string): Promise<string> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { content } = await call<{ content: string }>(
        `/api/answer?sessionId=${sessionId}&questionId=${questionId}&ownerId=${activeUser.id}`
      );
      return content ?? "";
    }
    return withDelay("");
  },

  async sendMessage(sessionId: string, msg: string, questionId?: string): Promise<Message> {
    if (useBackend && API_BASE) {
      try {
        const { reply } = await call<{ reply: Message }>(`/api/session/${sessionId}/chat`, {
          method: "POST",
          body: JSON.stringify({ message: msg, questionId }),
        });
        return reply;
      } catch (e) {
        console.warn("sendMessage backend failed, using mock reply", e);
      }
    }
    const aiReply: Message = {
      id: uid(),
      role: "assistant",
      createdAt: new Date().toISOString(),
      content:
        "Хороший ход. Проверьте крайние случаи: пустой ввод и большие объёмы данных. Что по сложности?",
    };
    return withDelay(aiReply, 600);
  },

  async getChat(sessionId: string, questionId?: string) {
    if (useBackend && API_BASE) {
      try {
        const res = await call<{ chat: Message[] }>(
          `/api/session/${sessionId}/chat${questionId ? `?questionId=${questionId}` : ""}`
        );
        return res;
      } catch (e) {
        console.warn("getChat backend failed, returning empty log", e);
      }
    }
    return withDelay({ chat: [] });
  },

  async runCode(sessionId: string, code: string, language = "python") {
    if (useBackend && API_BASE) {
      try {
        const res = await call<{
          status: string;
          stdout: string;
          stderr: string;
          executionTimeMs: number;
          errorType?: string;
        }>("/api/run-code", {
          method: "POST",
          body: JSON.stringify({ sessionId, code, language }),
        });
        return res;
      } catch (e) {
        console.warn("runCode failed", e);
        throw e;
      }
    }
    return withDelay({
      status: "ok",
      stdout: "Hello from mock runner",
      stderr: "",
      executionTimeMs: 120,
    });
  },

  async sendSupport(message: string, userId: string) {
    if (useBackend && API_BASE) {
      await call<{ status: string }>(`/api/support/send`, {
        method: "POST",
        body: JSON.stringify({ userId, message }),
      });
      return true;
    }
    return withDelay(true);
  },

  async getSupportMessages(userId: string) {
    if (useBackend && API_BASE) {
      const { messages } = await call<{ messages: any[] }>(`/api/support/messages?userId=${userId}`);
      return messages;
    }
    return withDelay([]);
  },

  async getSupportInbox() {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { items } = await call<{ items: any[] }>(`/api/support/inbox?adminId=${activeUser.id}`);
      return items;
    }
    return withDelay([]);
  },

  async getSupportHistory(userId: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { messages } = await call<{ messages: any[] }>(
        `/api/support/history?userId=${userId}&adminId=${activeUser.id}`
      );
      return messages;
    }
    return withDelay([]);
  },

  async sendSupportReply(userId: string, message: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>(`/api/support/reply`, {
        method: "POST",
        body: JSON.stringify({ userId, message, adminId: activeUser.id }),
      });
      return true;
    }
    return withDelay(true);
  },

  async closeSupportDialog(userId: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>(`/api/support/close`, {
        method: "POST",
        body: JSON.stringify({ userId, adminId: activeUser.id }),
      });
      return true;
    }
    return withDelay(true);
  },

  async checkSolution(sessionId: string, code: string): Promise<TestResult> {
    if (useBackend && API_BASE) {
      const { result } = await call<{ result: TestResult }>(`/api/session/${sessionId}/check`, {
        method: "POST",
        body: JSON.stringify({ code }),
      });
      return result;
    }
    const isPassing = code.toLowerCase().includes("map") || Math.random() > 0.5;
    const result: TestResult = isPassing
      ? { ...defaultTestResult, passed: true, summary: "Все тесты зелёные" }
      : defaultTestResult;
    return withDelay(result, 700);
  },

  async getReport(id: string): Promise<InterviewReport> {
    if (useBackend && API_BASE) {
      const { report } = await call<{ report: InterviewReport }>(`/api/report/${id}`);
      return report;
    }
    return withDelay({
      id,
      score: 86,
      level: (history.find((h) => h.id === id)?.level as Level) ?? "Middle",
      summary:
        "Кандидат уверенно решает алгоритмические задачи, но стоит потренировать обработку ошибок и крайних кейсов.",
      timeline: buildTimeline(),
      solutions: [
        {
          title: "Оптимальный маршрут",
          code: baseStarter,
          errors: "Падал на пустом массиве",
          tests: defaultTestResult,
        },
      ],
      analytics: {
        skillMap,
        errorHeatmap: heatmap,
        speed: [
          { label: "Старт", value: 0 },
          { label: "Черновик", value: 6 },
          { label: "Отладка", value: 14 },
          { label: "Финал", value: 22 },
        ],
      },
    });
  },

  async getAdminOverview(): Promise<AdminOverview> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const data = await call<AdminOverview>(`/api/admin/overview?adminId=${activeUser.id}`);
      return data;
    }
    return withDelay({
      candidates: [
        {
          id: "cand-1",
          name: "Виктория Ким",
          email: "victoria@team.dev",
          level: "Senior",
          role: "admin",
          admin: true,
          lastScore: 91,
          lastTopic: "ML",
        },
        {
          id: "cand-2",
          name: "Сергей Романов",
          email: "romanov@team.dev",
          level: "Middle",
          lastScore: 73,
          lastTopic: "System Design",
        },
      ],
      flagged: adminEvents,
      analytics: {
        hardestTopics: [
          { name: "Concurrency", score: 62 },
          { name: "Graph", score: 58 },
          { name: "API design", score: 64 },
        ],
        completionRate: 0.78,
        avgScore: 0.81,
      },
    });
  },

  async getAdminEvents() {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      try {
        const { events } = await call<{ events: any[] }>(`/api/admin/events?adminId=${activeUser.id}`);
        return events;
      } catch (e) {
        console.warn("getAdminEvents backend failed", e);
      }
    }
    return withDelay([]);
  },

  async getAdminSessions(filters?: { level?: string; status?: string; date_from?: string; date_to?: string }) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const params = new URLSearchParams({ adminId: activeUser.id });
      if (filters?.level) params.append("level", filters.level);
      if (filters?.status) params.append("status", filters.status);
      if (filters?.date_from) params.append("date_from", filters.date_from);
      if (filters?.date_to) params.append("date_to", filters.date_to);
      const { sessions } = await call<{ sessions: any[] }>(`/api/admin/sessions?${params.toString()}`);
      return sessions;
    }
    return withDelay([]);
  },

  async getAdminSessionDetail(id: string) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const data = await call<{ session: any; questions: any[]; answers: any[]; events: any[]; metrics: any; messages: any[] }>(
        `/api/admin/session/${id}?adminId=${activeUser.id}`
      );
      return data;
    }
    return withDelay(null);
  },

  async getAdminConfig() {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { config } = await call<{ config: any }>(`/api/admin/config?adminId=${activeUser.id}`);
      return config;
    }
    return withDelay({});
  },

  async saveAdminConfig(cfg: any) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>(`/api/admin/config`, {
        method: "POST",
        body: JSON.stringify({ adminId: activeUser.id, config: cfg }),
      });
      return true;
    }
    return withDelay(true);
  },

  async clearAdminEvents() {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      await call<{ status: string }>(`/api/admin/events/clear`, {
        method: "POST",
        body: JSON.stringify({ adminId: activeUser.id }),
      });
      return true;
    }
    return withDelay(true);
  },

  async nextQuestion(sessionId: string) {
    if (useBackend && API_BASE) {
      try {
        if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
        const { session } = await call<{ session: InterviewSession }>(`/api/interview/next`, {
          method: "POST",
          body: JSON.stringify({ sessionId, ownerId: activeUser.id }),
        });
        return session;
      } catch (e) {
        console.warn("Backend nextQuestion failed", e);
        // Не уходим в мок, если нет локальной сессии — пробрасываем ошибку
        throw e instanceof Error ? e : new Error("NEXT_QUESTION_FAILED");
      }
    }
    const existing = localSessions[sessionId];
    if (!existing) throw new Error("SESSION_NOT_FOUND");
    const used = existing.usedQuestions ?? [];
    const pool = localQuestions.filter((q) => !used.find((u) => u.id === q.id));
    const next =
      pool.length > 0 ? pool[Math.floor(Math.random() * pool.length)] : localQuestions[used.length % localQuestions.length];
    const updated: InterviewSession = {
      ...existing,
      questionId: next.id,
      questionTitle: next.title,
      description: next.body,
      useIDE: !!next.useIDE,
      usedQuestions: used.find((u) => u.id === next.id) ? used : [...used, { id: next.id, title: next.title }],
      total: existing.total ?? localQuestions.length,
      startedAt: existing.startedAt,
      timer: existing.timer,
    };
    localSessions[sessionId] = updated;
    return withDelay(updated);
  },

  async streamChatSSE(params: {
    sessionId: string;
    questionId?: string;
    message: string;
    onDelta: (delta: string) => void;
    onError?: (err: any) => void;
    onEnd?: () => void;
  }) {
    if (!API_BASE) throw new Error("API_BASE_NOT_SET");
    try {
      const res = await fetch(`${API_BASE}/api/interview/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: params.sessionId, questionId: params.questionId, message: params.message }),
      });
      if (!res.body) throw new Error("NO_STREAM");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "");
          if (!line) continue;
          try {
            const obj = JSON.parse(line);
            if (obj.delta) params.onDelta(obj.delta);
          } catch {}
        }
      }
      params.onEnd?.();
    } catch (e) {
      params.onError?.(e);
    }
  },

  async streamNextSSE(params: {
    sessionId: string;
    ownerId: string;
    language?: string;
    onDelta: (delta: string) => void;
    onError?: (err: any) => void;
    onEnd?: () => void;
  }) {
    if (!API_BASE) throw new Error("API_BASE_NOT_SET");
    try {
      const res = await fetch(`${API_BASE}/api/interview/next/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: params.sessionId, ownerId: params.ownerId, language: params.language }),
      });
      if (!res.body) throw new Error("NO_STREAM");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "");
          if (!line) continue;
          try {
            const obj = JSON.parse(line);
            if (obj.delta) params.onDelta(obj.delta);
          } catch {}
        }
      }
      params.onEnd?.();
    } catch (e) {
      params.onError?.(e);
    }
  },
};
