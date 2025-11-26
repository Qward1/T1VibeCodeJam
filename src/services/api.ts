import {
  AdminOverview,
  ErrorHeat,
  FlaggedEvent,
  InterviewHistoryItem,
  InterviewReport,
  InterviewSession,
  AssignedInterview,
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
let localAssigned: AssignedInterview | null = null;
let localAssignedList: AssignedInterview[] = [];

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

  // Назначение интервью админом
  async assignInterview(candidateId: string, payload: { direction: string; level: Level; format: string; tasks: string[]; duration?: number }) {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { assigned } = await call<{ assigned: AssignedInterview }>("/api/admin/assign-interview", {
        method: "POST",
        body: JSON.stringify({
          adminId: activeUser.id,
          candidateId,
          direction: payload.direction,
          level: payload.level,
          format: payload.format,
          tasks: payload.tasks,
          duration: payload.duration,
        }),
      });
      return assigned;
    }
    localAssigned = {
      id: `ass-${uid()}`,
      candidateId,
      adminId: activeUser?.id || "admin",
      direction: payload.direction,
      level: payload.level,
      format: payload.format,
      tasks: payload.tasks,
      duration: payload.duration,
      status: "pending",
      createdAt: new Date().toISOString(),
    };
    localAssignedList = [localAssigned];
    return withDelay(localAssigned);
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

  async getAssignedInterview(): Promise<AssignedInterview | null> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { assigned } = await call<{ assigned: AssignedInterview | null }>(
        `/api/assigned-interview?userId=${activeUser.id}`
      );
      return assigned ?? null;
    }
    return withDelay(localAssigned);
  },

  async getAssignedInterviews(): Promise<AssignedInterview[]> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { assigned } = await call<{ assigned: AssignedInterview[] }>(
        `/api/assigned-interviews?userId=${activeUser.id}`
      );
      return assigned || [];
    }
    return withDelay(localAssignedList);
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
        await call<{ status: string }>("/api/anticheat/event", {
          method: "POST",
          body: JSON.stringify({
            sessionId: event.sessionId,
            ownerId: activeUser.id,
            eventType: event.eventType,
            payload: JSON.stringify(event.payload ?? {}),
            risk: event.risk ?? "medium",
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

  async startAssignedInterview(assignedId: string): Promise<InterviewSession> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const { session } = await call<{ session: InterviewSession }>(
        `/api/assigned-interview/start/${assignedId}`,
        {
          method: "POST",
          body: JSON.stringify({ userId: activeUser.id }),
        }
      );
      return session;
    }
    const session: InterviewSession = {
      id: uid(),
      ownerId: activeUser?.id,
      direction: localAssigned?.direction || "Frontend",
      level: (localAssigned?.level as Level) || "Middle",
      format: localAssigned?.format || "Full interview",
      tasks: localAssigned?.tasks || ["Coding"],
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
    if (localAssigned) {
      localAssigned = { ...localAssigned, status: "active", sessionId: session.id };
    }
    localAssignedList = localAssignedList.map((a) => (a.id === assignedId ? { ...a, status: "active", sessionId: session.id } : a));
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

  async finishInterview(sessionId: string): Promise<number | null> {
    if (useBackend && API_BASE) {
      try {
        if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
        try {
          const res = await call<{ status: string; score?: number }>(`/api/session/finish`, {
            method: "POST",
            body: JSON.stringify({ sessionId, ownerId: activeUser.id }),
          });
          return res?.score ?? null;
        } catch {
          // Совместимость со старым роутом, если вдруг нужен
          const res2 = await call<{ status: string; score?: number }>(`/api/interview/finish`, {
            method: "POST",
            body: JSON.stringify({ sessionId, ownerId: activeUser.id }),
          });
          return res2?.score ?? null;
        }
      } catch (e) {
        console.warn("finishInterview backend failed, falling back", e);
      }
    }
    // Чистим локальное активное интервью
    if (localSessions[sessionId]) {
      delete localSessions[sessionId];
    }
    return withDelay(null);
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

  async getAnswer(sessionId: string, questionId: string): Promise<{ content: string; decision?: string | null; score?: number | null; maxScore?: number | null }> {
    if (useBackend && API_BASE) {
      if (!activeUser?.id) throw new Error("NO_ACTIVE_USER");
      const res = await call<{ content: string; decision?: string | null; score?: number | null; maxScore?: number | null }>(
        `/api/answer?sessionId=${sessionId}&questionId=${questionId}&ownerId=${activeUser.id}`
      );
      return {
        content: res.content ?? "",
        decision: res.decision ?? null,
        score: res.score ?? null,
        maxScore: res.maxScore ?? null,
      };
    }
    return withDelay({ content: "", decision: null, score: null, maxScore: null });
  },

  async evalTheoryAnswer(params: {
    sessionId: string;
    questionId: string;
    ownerId: string;
    answer: string;
    baseQuestionJson?: any;
  }) {
    const payload = {
      sessionId: params.sessionId,
      questionId: params.questionId,
      ownerId: params.ownerId,
      answer: params.answer,
      isFollowup: false,
      baseQuestionJson: params.baseQuestionJson,
    };
    const res = await call<{
      decision: string;
      score: number;
      maxScore: number;
      coveredPoints: string[];
      missingPoints: string[];
      feedbackShort?: string;
      feedbackDetailed?: string;
      followUp?: { question?: string } | null;
    }>("/api/theory/eval", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return res;
  },

  async evalTheoryFollowup(params: {
    sessionId: string;
    questionId: string;
    ownerId: string;
    answer: string;
    followupQuestion: string;
    missingPoints: string[];
    baseQuestionJson?: any;
  }) {
    const payload = {
      sessionId: params.sessionId,
      questionId: params.questionId,
      ownerId: params.ownerId,
      answer: params.answer,
      isFollowup: true,
      followupQuestion: params.followupQuestion,
      missingPoints: params.missingPoints,
      baseQuestionJson: params.baseQuestionJson,
    };
    const res = await call<{
      decision: string;
      score: number;
      maxScore: number;
      coveredPoints: string[];
      missingPoints: string[];
      feedbackShort?: string;
    }>("/api/theory/eval", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return res;
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

  async runSamples(params: { sessionId: string; questionId: string; taskId: string; code: string; language: string; ownerId?: string }): Promise<{ tests: any[]; hasError: boolean }> {
    if (useBackend && API_BASE) {
      const ownerId = params.ownerId || activeUser?.id;
      if (!ownerId) throw new Error("NO_ACTIVE_USER");
      return call<{ tests: any[]; hasError: boolean }>(`/api/code/run-samples`, {
        method: "POST",
        body: JSON.stringify({ ...params, ownerId }),
      });
    }
    return withDelay({ tests: [], hasError: false });
  },

  async checkCode(params: { sessionId: string; questionId: string; taskId: string; code: string; language: string; ownerId?: string }): Promise<any> {
    if (useBackend && API_BASE) {
      const ownerId = params.ownerId || activeUser?.id;
      if (!ownerId) throw new Error("NO_ACTIVE_USER");
      return call<any>(`/api/code/check`, {
        method: "POST",
        body: JSON.stringify({ ...params, ownerId }),
      });
    }
    // mock: always passed public, hidden
    return withDelay({
      solved: true,
      attempt: 1,
      score: 10,
      maxScore: 10,
      publicTests: [
        { name: "пример_1", status: "passed", expected: 6, actual: 6 },
        { name: "пример_2", status: "passed", expected: -1, actual: -1 },
      ],
      hiddenPassed: true,
      hasError: false,
    });
  },

  async codeHint(params: { sessionId: string; questionId: string; taskId: string; language: string; ownerId?: string; userCode?: string }): Promise<{ hint: string | null; hintsUsed: number; effectiveMaxScore: number }> {
    if (useBackend && API_BASE) {
      const ownerId = params.ownerId || activeUser?.id;
      if (!ownerId) throw new Error("NO_ACTIVE_USER");
      return call(`/api/code/hint`, {
        method: "POST",
        body: JSON.stringify({ ...params, ownerId }),
      });
    }
    return withDelay({
      hint: "Подумайте о граничных случаях и проверьте работу с пустыми входами.",
      hintsUsed: 1,
      effectiveMaxScore: 8,
    });
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
};
