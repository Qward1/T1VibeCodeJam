"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/UI/Button";
import { Card } from "@/components/UI/Card";
import { TextArea } from "@/components/UI/TextArea";
import { formatDuration } from "@/utils";

// Динамический импорт Monaco
const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <div className="h-[420px] w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-3">
      Загрузка редактора...
    </div>
  ),
});

type TestCase = {
  name?: string;
  input?: string;
  output?: string;
  expected_output?: string;
  actual_output?: string;
  status?: string;
  stdout?: string;
};

type Question = {
  id: string;
  title: string;
  statement?: string;
  body?: string;
  language?: string;
  useIDE?: boolean;
  visible_tests?: { input: string; output: string }[];
  hidden_tests?: { input: string; output: string }[];
  starter_code?: string;
  session_question_id?: number;
  level?: string;
};

type InterviewState = {
  session: {
    id: string;
    ownerId?: string;
    current_level?: string;
    status?: string;
    total?: number;
    timer?: number;
    startedAt?: string;
    cheat_score?: number;
  };
  current_question: Question;
  questions: Question[];
  visible_tests: { input: string; output: string }[];
  metrics?: {
    progress_percent?: number;
    cheat_score?: number;
  };
  messages?: { role: "assistant" | "user"; content: string; id?: string }[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL;

const fetchJson = async <T,>(path: string, init?: RequestInit) => {
  if (!API_BASE) throw new Error("API_BASE_NOT_SET");
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = (data as any)?.detail || (data as any)?.error || msg;
    } catch {
      msg = await res.text();
    }
    throw new Error(msg || "REQUEST_FAILED");
  }
  return res.json() as Promise<T>;
};

// Компонент обёртка
export default function InterviewSessionPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const sessionId = params?.id;
  const [state, setState] = useState<InterviewState | null>(null);
  const [code, setCode] = useState<string>("");
  const [tests, setTests] = useState<TestCase[]>([]);
  const [stdout, setStdout] = useState("");
  const [stderr, setStderr] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState(45 * 60);
  const [chat, setChat] = useState<{ role: "assistant" | "user"; content: string; id?: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStreaming, setChatStreaming] = useState(false);
  const [cheatBanner, setCheatBanner] = useState(false);
  const [finishModal, setFinishModal] = useState(false);
  const [finished, setFinished] = useState(false);

  // загрузка состояния
  useEffect(() => {
    const load = async () => {
      if (!sessionId) return;
      try {
        setLoading(true);
        const data = await fetchJson<InterviewState>(`/api/interview/state/${sessionId}`);
        setState(data);
        setCode(data.current_question?.starter_code || "");
        setChat(data.messages || []);
        // таймер от startedAt
        const total = Number(data.session?.timer ?? 45 * 60);
        const startedRaw = data.session?.startedAt;
        const startedTs = startedRaw ? Date.parse(startedRaw) : Date.now();
        const tick = () => {
          const elapsed = Math.floor((Date.now() - startedTs) / 1000);
          setTimeLeft(Math.max(0, total - elapsed));
        };
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
      } catch (e: any) {
        setError(e.message || "Не удалось загрузить интервью");
      } finally {
        setLoading(false);
      }
    };
    const cleanup = load();
    return () => {
      if (typeof cleanup === "function") cleanup();
    };
  }, [sessionId]);

  // Анти-чит события
  useEffect(() => {
    if (!sessionId) return;
    const sendEvent = (event_type: string, payload?: any) =>
      fetchJson("/api/interview/event", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, event_type, payload }),
      }).catch(() => undefined);

    const onBlur = () => sendEvent("tab_switch", { state: "blur" });
    const onVis = () => sendEvent("visibilitychange", { state: document.visibilityState });
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key.toLowerCase() === "v") sendEvent("paste", { via: "ctrl+v" });
    };
    const onContext = (e: MouseEvent) => {
      e.preventDefault();
      sendEvent("context_blocked", { x: e.clientX, y: e.clientY });
    };
    const devtoolsCheck = () => {
      if (window.outerHeight - window.innerHeight > 150 || window.outerWidth - window.innerWidth > 150) {
        setCheatBanner(true);
        sendEvent("devtools", { delta: window.outerHeight - window.innerHeight });
      }
    };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("blur", onBlur);
    window.addEventListener("keydown", onKey);
    document.addEventListener("contextmenu", onContext);
    const devtoolsTimer = setInterval(devtoolsCheck, 3000);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("keydown", onKey);
      document.removeEventListener("contextmenu", onContext);
      clearInterval(devtoolsTimer);
    };
  }, [sessionId]);

  const current = state?.current_question;
  const progressLabel = useMemo(() => {
    const idx =
      state?.questions?.findIndex((q) => q.id === current?.id || q.session_question_id === current?.session_question_id) ?? 0;
    const total = state?.questions?.length || state?.session?.total || 1;
    return { current: idx + 1, total };
  }, [state?.questions, state?.session?.total, current?.id, current?.session_question_id]);

  const sendChat = async () => {
    if (!chatInput.trim() || !sessionId) return;
    const userMsg = { role: "user" as const, content: chatInput };
    setChat((c) => [...c, userMsg]);
    setChatInput("");
    try {
      setChatStreaming(true);
      const resp = await fetchJson<{ reply: string }>("/api/interview/chat", {
        method: "POST",
        body: JSON.stringify({ sessionId, questionId: current?.id, message: userMsg.content }),
      });
      setChat((c) => [...c, { role: "assistant", content: resp.reply }]);
    } catch (e: any) {
      setChat((c) => [...c, { role: "assistant", content: `Ошибка: ${e.message || "не удалось ответить"}` }]);
    } finally {
      setChatStreaming(false);
    }
  };

  const runCode = async () => {
    if (!sessionId || !current?.session_question_id) return;
    try {
      setLoading(true);
      const res = await fetchJson<{ tests: TestCase[]; stdout?: string; stderr?: string }>("/api/code/run", {
        method: "POST",
        body: JSON.stringify({
          session_question_id: current.session_question_id,
          sessionQuestionId: current.session_question_id,
          sessionId,
          code,
          language: current.language || "javascript",
        }),
      });
      setTests(res.tests || []);
      setStdout(res.stdout || "");
      setStderr(res.stderr || "");
    } catch (e: any) {
      setError(e.message || "Не удалось запустить код");
    } finally {
      setLoading(false);
    }
  };

  const submitCode = async () => {
    if (!sessionId || !current?.session_question_id) return;
    try {
      setLoading(true);
      const res = await fetchJson<{
        visible_tests?: TestCase[];
        hidden_tests?: TestCase[];
        stdout?: string;
        stderr?: string;
        passed_hidden?: boolean;
      }>("/api/code/submit", {
        method: "POST",
        body: JSON.stringify({
          session_question_id: current.session_question_id,
          sessionQuestionId: current.session_question_id,
          sessionId,
          code,
          language: current.language || "javascript",
        }),
      });
      setTests([...(res.visible_tests || []), ...(res.hidden_tests || [])]);
      setStdout(res.stdout || "");
      setStderr(res.stderr || "");
      // если задача пройдена — запрашиваем следующую
      if (res.passed_hidden) {
        await fetchJson("/api/interview/next", {
          method: "POST",
          body: JSON.stringify({ sessionId, ownerId: state?.session?.ownerId }),
        }).catch(() => undefined);
        const refreshed = await fetchJson<InterviewState>(`/api/interview/state/${sessionId}`);
        setState(refreshed);
        setCode(refreshed.current_question?.starter_code || "");
        setTests([]);
      }
    } catch (e: any) {
      setError(e.message || "Не удалось отправить решение");
    } finally {
      setLoading(false);
    }
  };

  const callFinish = async (keepalive = false) => {
    if (!sessionId) return;
    try {
      setLoading(true);
      await fetchJson("/api/interview/finish", {
        method: "POST",
        body: JSON.stringify({ sessionId, ownerId: state?.session?.ownerId }),
        keepalive,
      } as RequestInit);
      setFinished(true);
      setFinishModal(false);
      setChatStreaming(false);
      setLoading(false);
      // перенаправление на отчёт
      router.push(`/interview/${sessionId}/result`);
    } catch (e: any) {
      setError(e.message || "Не удалось завершить собеседование");
      setLoading(false);
    }
  };

  // Предупреждение при закрытии вкладки
  useEffect(() => {
    if (!sessionId) return;
    const handler = (e: BeforeUnloadEvent) => {
      if (finished) return;
      e.preventDefault();
      e.returnValue = "";
      callFinish(true);
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [sessionId, finished]);

  // автопереход на результат, если статус завершён или время вышло
  useEffect(() => {
    if (!state?.session?.status) return;
    if (state.session.status === "completed" || state.session.status === "finished" || timeLeft <= 0) {
      setFinished(true);
      router.push(`/interview/${sessionId}/result`);
    }
  }, [state?.session?.status, timeLeft, router, sessionId]);

  if (error) {
    return (
      <div className="p-6">
        <Card title="Ошибка">
          <div className="text-sm text-rose-500">{error}</div>
        </Card>
      </div>
    );
  }

  if (!state || !current) return <div className="p-6">Загрузка интервью...</div>;

  return (
    <div className="min-h-screen bg-mesh bg-cover bg-fixed px-3 py-4 text-sm">
      {cheatBanner && (
        <div className="fixed top-4 right-4 z-[2000] rounded-xl bg-amber-100 px-4 py-3 text-sm text-amber-900 shadow-lg border border-amber-300">
          Система фиксирует переключения вкладок/DevTools — это может повлиять на оценку
        </div>
      )}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 shadow-lg">
        <div>
          <div className="text-xs text-[var(--muted)]">
            Задача {progressLabel.current}/{progressLabel.total} · Уровень {current.level || state.session.current_level || "—"}
          </div>
          <div className="text-lg font-semibold">{current.title}</div>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-vibe-50 px-3 py-2 text-xs font-semibold text-vibe-800 dark:bg-white/10 dark:text-white">
            ⏳ {formatDuration(timeLeft)}
          </div>
          <div
            className={`rounded-full px-3 py-2 text-xs font-semibold ${
              (state.metrics?.cheat_score ?? state.session?.cheat_score ?? 0) > 5
                ? "bg-rose-100 text-rose-700"
                : "bg-emerald-100 text-emerald-700"
            }`}
          >
            Анти-чит: {state.metrics?.cheat_score ?? state.session?.cheat_score ?? 0}
          </div>
          <Button
            variant="outline"
            onClick={() => setFinishModal(true)}
            className="border-vibe-400 text-vibe-800 dark:text-white hover:bg-vibe-50"
            disabled={finished}
          >
            Завершить собеседование
          </Button>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[2fr_1.2fr]">
        <div className="space-y-3">
          <TaskDescription title={current.title} statement={current.statement || current.body || ""} tests={state.visible_tests} />
          <EditorPanel
            code={code}
            setCode={setCode}
            language={current.language || "javascript"}
            onRun={runCode}
            onSubmit={submitCode}
            loading={loading || finished}
          />
          <TestsPanel tests={tests} stdout={stdout} stderr={stderr} />
        </div>

        <div className="space-y-3">
          <ChatPanel
            chat={chat}
            input={chatInput}
            setInput={setChatInput}
            onSend={sendChat}
            streaming={chatStreaming}
            disabled={finished}
          />
          <Card title="Метрики">
            <div className="space-y-1 text-sm">
              <div>Прогресс: {state.metrics?.progress_percent ?? 0}%</div>
              <div>Чит-скор: {state.metrics?.cheat_score ?? state.session?.cheat_score ?? 0}</div>
            </div>
          </Card>
        </div>
      </div>

      {finished && (
        <div className="fixed top-4 left-1/2 z-[1500] -translate-x-1/2 rounded-xl bg-emerald-100 px-4 py-2 text-emerald-800 shadow">
          Собеседование завершено
        </div>
      )}

      {finishModal && (
        <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/50 px-4">
          <div className="w-full max-w-md rounded-2xl bg-[var(--card)] p-5 shadow-lg border border-[var(--border)]">
            <div className="text-lg font-semibold mb-2">Завершить собеседование?</div>
            <div className="text-sm text-[var(--muted)] mb-4">
              Вы уверены, что хотите завершить собеседование? После завершения вы не сможете вернуться к текущим задачам.
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setFinishModal(false)}>
                Отмена
              </Button>
              <Button className="bg-gradient-to-r from-vibe-500 to-vibe-700 text-white" onClick={() => callFinish()}>
                Завершить окончательно
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TaskDescription({
  title,
  statement,
  tests,
}: {
  title: string;
  statement: string;
  tests?: { input: string; output: string }[];
}) {
  return (
    <Card title="Задание">
      <div className="space-y-2">
        <div className="max-h-56 overflow-auto rounded-lg border border-[var(--border)] bg-[var(--card)] p-3 text-sm whitespace-pre-wrap">
          {statement || "Условие пока не загружено."}
        </div>
        {tests && tests.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-semibold text-[var(--muted)]">Примеры ввода/вывода</div>
            {tests.map((t, idx) => (
              <div key={idx} className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-2 text-xs">
                <div className="text-[var(--muted)]">Ввод:</div>
                <pre className="whitespace-pre-wrap">{t.input}</pre>
                <div className="text-[var(--muted)] mt-1">Ожидание:</div>
                <pre className="whitespace-pre-wrap">{t.output}</pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}

function EditorPanel({
  code,
  setCode,
  language,
  onRun,
  onSubmit,
  loading,
}: {
  code: string;
  setCode: (v: string) => void;
  language: string;
  onRun: () => void;
  onSubmit: () => void;
  loading: boolean;
}) {
  return (
    <Card title="Редактор">
      <MonacoEditor
        height="420px"
        defaultLanguage={language}
        theme="vs-dark"
        value={code}
        onChange={(v) => setCode(v ?? "")}
        options={{ minimap: { enabled: false }, renderValidationDecorations: "off" }}
      />
      <div className="mt-3 flex gap-2">
        <Button
          onClick={onRun}
          disabled={loading}
          className="bg-[rgba(109,65,128,0.15)] text-[rgb(109,65,128)] border border-[rgba(109,65,128,0.4)]"
        >
          Запустить код
        </Button>
        <Button onClick={onSubmit} disabled={loading} className="bg-gradient-to-r from-vibe-500 to-vibe-700 text-white">
          Отправить решение
        </Button>
      </div>
    </Card>
  );
}

function TestsPanel({ tests, stdout, stderr }: { tests: TestCase[]; stdout: string; stderr: string }) {
  return (
    <Card title="Результаты тестов">
      <div className="space-y-2">
        {tests.length === 0 && <div className="text-sm text-[var(--muted)]">Тесты ещё не запускались.</div>}
        {tests.map((t, idx) => (
          <div
            key={idx}
            className={`rounded-xl border px-3 py-2 text-sm ${
              t.status === "passed" ? "border-emerald-300 bg-emerald-50" : "border-rose-300 bg-rose-50"
            }`}
          >
            <div className="font-semibold">
              Тест {idx + 1}: {t.status || "pending"}
            </div>
            {t.input && <div className="text-xs text-[var(--muted)]">Ввод: {t.input}</div>}
            {(t.expected_output || t.output) && (
              <div className="text-xs text-[var(--muted)]">Ожидание: {t.expected_output ?? t.output}</div>
            )}
            {t.actual_output && <div className="text-xs text-[var(--muted)]">Фактический: {t.actual_output}</div>}
            {t.stdout && (
              <div className="text-xs text-[var(--muted)] mt-1">
                stdout: <pre className="whitespace-pre-wrap">{t.stdout}</pre>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs text-[var(--muted)]">stdout</div>
      <pre className="rounded-lg bg-[var(--card)] p-2 text-xs whitespace-pre-wrap">{stdout || "—"}</pre>
      <div className="mt-2 text-xs text-[var(--muted)]">stderr</div>
      <pre className="rounded-lg bg-[var(--card)] p-2 text-xs whitespace-pre-wrap text-rose-500">{stderr || "—"}</pre>
    </Card>
  );
}

function ChatPanel({
  chat,
  input,
  setInput,
  onSend,
  streaming,
  disabled = false,
}: {
  chat: { role: "assistant" | "user"; content: string; id?: string }[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  streaming: boolean;
  disabled?: boolean;
}) {
  return (
    <Card title="Чат с интервьюером">
      <div className="flex flex-col gap-2">
        <div className="max-h-[360px] overflow-auto rounded-lg border border-[var(--border)] bg-[var(--card)] p-3 space-y-2">
          {chat.map((m, idx) => (
            <div
              key={m.id || idx}
              className={`rounded-xl px-3 py-2 text-sm ${
                m.role === "assistant" ? "bg-vibe-50 text-vibe-900 dark:bg-white/10 dark:text-white" : "bg-[var(--card)] border border-[var(--border)]"
              }`}
            >
              <div className="text-xs text-[var(--muted)] mb-1">{m.role === "assistant" ? "AI" : "Кандидат"}</div>
              <div className="whitespace-pre-wrap">{m.content}</div>
            </div>
          ))}
          {streaming && <div className="text-xs text-[var(--muted)]">AI печатает...</div>}
        </div>
        <div className="flex gap-2">
          <TextArea
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Задайте вопрос интервьюеру"
            disabled={disabled}
          />
          <Button onClick={onSend} disabled={!input.trim() || streaming || disabled}>
            Отправить
          </Button>
        </div>
      </div>
    </Card>
  );
}
