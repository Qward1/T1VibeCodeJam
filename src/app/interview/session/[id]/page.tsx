"use client";
import { cn } from "@/utils";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/services/api";
import { useSessionStore } from "@/stores/session";
import { EditorPane } from "@/components/Interview/EditorPane";
import { ChatPane } from "@/components/Interview/ChatPane";
import { TaskCard } from "@/components/Interview/TaskCard";
import { Button } from "@/components/UI/Button";
import { Card } from "@/components/UI/Card";
import { TestResults } from "@/components/Interview/TestResults";
import { useRouter } from "next/navigation";
import { TextArea } from "@/components/UI/TextArea";
import { useState, useEffect, useRef, useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/stores/auth";

export default function InterviewSessionPage({ params }: { params: { id: string } }) {
  const { data: session } = useQuery({
    queryKey: ["session", params.id],
    queryFn: () => api.getInterviewSession(params.id),
  });
  const {
    session: storedSession,
    interviewId,
    setSession,
    setInterviewId,
    setCode,
    code,
    testResult,
    setTestResult,
    reset,
  } = useSessionStore();
  const router = useRouter();
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);

  useEffect(() => {
    if (!session) return;
    // Не затираем время старта, если бэкенд вернул неполные данные
    const merged = {
      ...session,
      startedAt: session.startedAt ?? storedSession?.startedAt,
      timer: session.timer ?? storedSession?.timer,
    };
    setSession(merged);
    if (merged?.id) setInterviewId(merged.id);
  }, [session, storedSession?.startedAt, storedSession?.timer, setSession, setInterviewId]);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!current?.id || !current?.questionId || !current?.codeTaskId || !user?.id) throw new Error("missing_params");
      return api.checkCode({
        sessionId: current.id,
        questionId: current.questionId,
        taskId: current.codeTaskId,
        code,
        language: current.language || "python",
        ownerId: user?.id,
      });
    },
    onSuccess: (result) => {
      if (current?.questionId && result?.attempt) {
        setAttemptsByQuestion((prev) => ({ ...prev, [current.questionId!]: result.attempt }));
      }
      if (result?.hasError) {
        queryClient.invalidateQueries({ queryKey: ["chat", current.id, current.questionId] });
      }
      const passed = Boolean(result?.hiddenPassed && (result?.publicTests || []).every((t: any) => t.status === "passed"));
      const cases = (result?.publicTests || []).map((t: any) => ({
        name: t.name,
        passed: t.status === "passed",
        details: t.status === "failed" ? `ожидалось ${JSON.stringify(t.expected)}, получено ${JSON.stringify(t.actual)}` : undefined,
      }));
      const summary = result?.hasError
        ? "Ошибка выполнения кода"
        : passed
          ? "Все тесты пройдены"
          : "Часть тестов не пройдена";
      setTestResult({ passed, summary, cases });
      setLastTestKind("check");
      const now = Date.now();
      if (lastCodeChange.current && now - lastCodeChange.current < 10000 && current?.id) {
        api.sendAntiCheat({
          sessionId: current.id,
          eventType: "suspicious_solve_time",
          payload: { deltaMs: now - lastCodeChange.current },
          risk: "high",
        });
      }
      lastTestAt.current = now;
      // Если бэкенд вернул новый вопрос — добавляем кнопку
      if (result?.finished && result?.nextQuestion) {
        const nextQ = result.nextQuestion;
        setSession((prev) => {
          if (!prev) return prev;
          const used = prev.usedQuestions ?? [];
          const already = used.find((q) => q.id === nextQ.id);
          const updatedUsed = already
            ? used
            : [...used, { id: nextQ.id, title: nextQ.title, qType: nextQ.qType, codeTaskId: nextQ.codeTaskId, position: nextQ.position }];
          const total = Math.max(prev.total ?? updatedUsed.length, updatedUsed.length);
          return {
            ...prev,
            usedQuestions: updatedUsed,
            total,
            questionId: nextQ.id,
            questionTitle: nextQ.title,
            useIDE: nextQ.qType === "coding",
            codeTaskId: nextQ.codeTaskId ?? prev.codeTaskId,
            starterCode: nextQ.starterCode ?? prev.starterCode,
          };
        });
        const starter = nextQ.starterCode as string | undefined;
        setCodeByQuestion((prev) => {
          const updated = { ...prev };
          if (starter && !(nextQ.id in updated)) {
            updated[nextQ.id] = starter;
          }
          if (typeof window !== "undefined") {
            try {
              localStorage.setItem("vibe-code-by-question", JSON.stringify(updated));
            } catch {
              // ignore
            }
          }
          return updated;
        });
        // сразу переключаем редактор на новую задачу
        if (nextQ.qType === "coding") {
          const fromCache = codeByQuestion[nextQ.id];
          const initialCode = fromCache ?? starter ?? "";
          setCode(initialCode);
        } else {
          setCode("");
        }
        // сбрасываем состояния для нового вопроса
        setAnswer("");
        setFollowUpQuestion(null);
        setFollowUpAnswer("");
        setMissingPoints([]);
        setBaseScore(null);
        setFollowLocked(false);
      }
    },
  });
  const runMutation = useMutation({
    mutationFn: async () => {
      if (!current?.id || !current?.questionId || !current?.codeTaskId || !user?.id) throw new Error("missing_params");
      return api.runSamples({
        sessionId: current.id,
        questionId: current.questionId,
        taskId: current.codeTaskId,
        code,
        language: selectedLanguage || current.language || "python",
        ownerId: user?.id,
      });
    },
    onSuccess: (res) => {
      setRunResult(res);
      setLastTestKind("run");
      if (res?.hasError) {
        queryClient.invalidateQueries({ queryKey: ["chat", current?.id, current?.questionId] });
      }
    },
  });

  const [answer, setAnswer] = useState("");
  const [codeByQuestion, setCodeByQuestion] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined") return {};
    try {
      const raw = localStorage.getItem("vibe-code-by-question");
      return raw ? (JSON.parse(raw) as Record<string, string>) : {};
    } catch {
      return {};
    }
  });
  const [answersState, setAnswersState] = useState<Record<string, string>>({});
  const [followUpQuestion, setFollowUpQuestion] = useState<string | null>(null);
  const [followUpAnswer, setFollowUpAnswer] = useState("");
  const [missingPoints, setMissingPoints] = useState<string[]>([]);
  const [baseScore, setBaseScore] = useState<{ score: number; maxScore: number } | null>(null);
  const [followUpState, setFollowUpState] = useState<
    Record<
      string,
      {
        question: string | null;
        answer: string;
        missing: string[];
        base?: { score: number; maxScore: number } | null;
      }
    >
  >({});
  const [answerStatus, setAnswerStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [followStatus, setFollowStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [followLocked, setFollowLocked] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  const lastCodeChange = useRef<number>(Date.now());
  const lastTestAt = useRef<number | null>(null);
  const splitRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const [paneRatio, setPaneRatio] = useState(0.55); // доля редактора
  const [editorHeight, setEditorHeight] = useState(600);
  const [runResult, setRunResult] = useState<{
    tests?: { name: string; status: string; expected: any; actual: any }[];
    hasError?: boolean;
  } | null>(null);
  const hintMutation = useMutation({
    mutationFn: async () => {
      if (!current?.id || !current?.questionId || !current?.codeTaskId || !user?.id) throw new Error("missing_params");
      return api.codeHint({
        sessionId: current.id,
        questionId: current.questionId,
        taskId: current.codeTaskId,
        language: selectedLanguage || current.language || "python",
        ownerId: user.id,
        userCode: code,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", current?.id, current?.questionId] });
    },
  });
  const [supportOpen, setSupportOpen] = useState(false);
  const [supportMessages, setSupportMessages] = useState<
    { id: string; role: string; content: string; createdAt: string }[]
  >([]);
  const [supportInput, setSupportInput] = useState("");
  const [lastTestKind, setLastTestKind] = useState<"check" | "run" | null>(null);
  const [attemptsByQuestion, setAttemptsByQuestion] = useState<Record<string, number>>({});
  const [finishConfirm, setFinishConfirm] = useState(false);
  const [scoreModal, setScoreModal] = useState<{ open: boolean; score: number | null }>({ open: false, score: null });
  const current = storedSession ?? session ?? null;
const currentIndex =
    current && current.usedQuestions
      ? Math.max(
          0,
          current.usedQuestions.findIndex((q) => q?.id === current.questionId)
        )
      : 0;
  const questionButtons =
    current && current.usedQuestions
      ? current.usedQuestions.map((q, i) => q ?? { id: `placeholder-${i}`, title: `Вопрос ${i + 1}` })
      : [];
  const usedCount = current?.usedQuestions?.length ?? 0;
  const totalCount = current?.total ?? questionButtons.length ?? usedCount ?? 0;
  const [selectedLanguage, setSelectedLanguage] = useState<string>(current?.language || "python");
  const languageOptions = useMemo(() => {
    const base = [current?.language || "python", "python", "javascript", "cpp"];
    return Array.from(new Set(base.filter(Boolean)));
  }, [current?.language]);

  useEffect(() => {
    if (current?.language) {
      setSelectedLanguage(current.language);
    }
  }, [current?.language]);

  useEffect(() => {
    const load = async () => {
      if (current?.id && current.questionId) {
        const res = await api.getAnswer(current.id, current.questionId);
        setAnswer(res.content);
        setAnswersState((prev) => ({ ...prev, [current.questionId]: res.content ?? "" }));
        // Кодовый ответ: используем сохранённый код или starter
        if (current.useIDE) {
          const savedLocal = codeByQuestion[current.questionId];
          const savedBackend = res.content ?? "";
          const finalCode = savedLocal ?? (savedBackend || current.starterCode || "");
          setCode(finalCode);
          setCodeByQuestion((prev) => ({ ...prev, [current.questionId]: finalCode }));
        }
        if (res.decision && res.decision !== "clarify") {
          setFollowLocked(true);
        } else {
          setFollowLocked(false);
        }
        const saved = followUpState[current.questionId];
        if (saved) {
          setFollowUpQuestion(saved.question);
          setFollowUpAnswer(saved.answer);
          setMissingPoints(saved.missing);
          setBaseScore(saved.base || null);
          setFollowLocked(saved.question === null && saved.base != null);
        } else {
          setFollowUpQuestion(null);
          setFollowUpAnswer("");
          setMissingPoints([]);
          setBaseScore(null);
          setFollowLocked(false);
        }
      }
    };
    load();
  }, [current?.id, current?.questionId]);

  const loadNextQuestion = async () => {
    if (!current?.id) return;
    try {
      const next = await api.nextQuestion(current.id);
      const merged = { ...next, startedAt: next.startedAt ?? current.startedAt, timer: next.timer ?? current.timer };
      // Обновляем usedQuestions локально, если бэкенд не прислал новый список
      setSession((prev) => {
        const base = { ...(merged || prev || {}) };
        const prevUsed = base.usedQuestions ?? prev?.usedQuestions ?? [];
        const exists = base.questionId ? prevUsed.find((q) => q.id === base.questionId) : undefined;
        const updatedUsed = exists || !base.questionId ? prevUsed : [...prevUsed, { id: base.questionId, title: base.questionTitle, qType: base.useIDE ? "coding" : "theory", codeTaskId: base.codeTaskId, position: prevUsed.length }];
        return { ...base, usedQuestions: updatedUsed, total: base.total ?? updatedUsed.length };
      });
      setInterviewId(merged.id);
      if (merged.questionId && answersState[merged.questionId] !== undefined) {
        setAnswer(answersState[merged.questionId]);
      } else {
        setAnswer("");
      }
      if (merged.useIDE) {
        const cached = codeByQuestion[merged.questionId ?? ""] ?? merged.starterCode ?? "";
        setCodeByQuestion((prev) => {
          const updated = { ...prev };
          if (merged.questionId && !(merged.questionId in updated) && merged.starterCode) {
            updated[merged.questionId] = merged.starterCode;
          }
          if (typeof window !== "undefined") {
            try {
              localStorage.setItem("vibe-code-by-question", JSON.stringify(updated));
            } catch {
              // ignore
            }
          }
          return updated;
        });
        setCode(cached);
      }
      setFollowUpQuestion(null);
      setFollowUpAnswer("");
      setMissingPoints([]);
      setBaseScore(null);
      setFollowStatus("idle");
      setAnswerStatus("idle");
      setFollowLocked(false);
    } catch (e) {
      console.warn("nextQuestion failed", e);
    }
  };

  // Drag-resize для редактора/чата
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current || !splitRef.current) return;
      const rect = splitRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = Math.min(0.8, Math.max(0.2, x / rect.width));
      setPaneRatio(ratio);
    };
    const stop = () => {
      isDragging.current = false;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", stop);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", stop);
    };
  }, []);

  // Анти-чит: события вкладки, правый клик и вставки вне редактора
  useEffect(() => {
    if (!current?.id) return;
    const record = (eventType: string, payload?: any, risk = "medium") =>
      api.sendAntiCheat({ sessionId: current.id, eventType, payload, risk }).catch(() => undefined);

    const onVisibility = () => {
      record("tab_switch", { state: document.visibilityState });
    };
    const onBlur = () => record("tab_switch", { state: "blur" }, "low");
    const onFocus = () => record("tab_switch", { state: "focus" }, "low");
    const onContext = (e: MouseEvent) => {
      e.preventDefault();
      record("context_blocked", { x: e.clientX, y: e.clientY }, "low");
    };
    const pasteLimit = 800;
    const onPaste = (e: ClipboardEvent) => {
      const text = e.clipboardData?.getData("text") ?? "";
      const target = e.target as HTMLElement | null;
      const insideEditor = target?.closest?.(".ide-allowed");
      if (!insideEditor) {
        e.preventDefault();
        record("external_clipboard_use", { len: text.length }, "high");
        return;
      }
      if (text.length > pasteLimit) {
        e.preventDefault();
        record("paste_limit_exceeded", { len: text.length }, "high");
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    document.addEventListener("contextmenu", onContext);
    document.addEventListener("paste", onPaste, { capture: true });
    // Запрет выделения вне IDE
    const prevSelect = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("contextmenu", onContext);
      document.removeEventListener("paste", onPaste, { capture: true } as any);
      document.body.style.userSelect = prevSelect;
    };
  }, [current?.id]);

  useEffect(() => {
    if (!current) return;
    const fallbackTotal = 45 * 60;
    const totalRaw = Number(current.timer ?? fallbackTotal);
    const total = Number.isFinite(totalRaw) ? totalRaw : fallbackTotal;
    // Если нет корректного startedAt — ведём локальный отсчёт
    let parsed = Date.now();
    if (current.startedAt) {
      // Умеем парсить строки с +00:00, Z или без суффикса
      const raw = current.startedAt;
      const hasZone = raw.includes("+") || raw.endsWith("Z");
      const candidate = hasZone ? raw : `${raw}Z`;
      const ts = Date.parse(candidate);
      parsed = Number.isFinite(ts) ? ts : Date.parse(raw) || Date.now();
    }
    const started = Number.isFinite(parsed) ? parsed : Date.now();
    const calc = () => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setTimeLeft(Math.max(0, total - elapsed));
    };
    calc();
    const id = setInterval(calc, 1000);
    return () => clearInterval(id);
  }, [current]);

  // Поддержка: подтягиваем историю и включаем polling
  useEffect(() => {
    let timer: NodeJS.Timeout | undefined;
    const fetchMessages = async () => {
      if (!supportOpen || !user?.id) return;
      try {
        const msgs = await api.getSupportMessages(user.id);
        setSupportMessages(msgs);
      } catch {
        // ignore
      }
    };
    if (supportOpen && user?.id) {
      fetchMessages();
      timer = setInterval(fetchMessages, 5000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [supportOpen, user?.id]);

  if (!current) return <div>Загрузка сессии...</div>;

  return (
    <>
    <main className="space-y-3 select-none px-0">
      <TaskCard
        description={current.description ?? ""}
        title={current.questionTitle}
        timer={timeLeft}
        currentIndex={currentIndex}
        total={totalCount}
      />
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap items-center gap-2 flex-1">
          {questionButtons.map((q, idx) => (
            <button
              key={q.id}
              className={`rounded-full px-3 py-2 text-sm ${
                q.id === current.questionId ? "bg-vibe-600 text-white" : "border border-[var(--border)] text-[var(--muted)]"
              }`}
              onClick={async () => {
                if (!q.id || q.id.startsWith("placeholder")) {
                  if (totalCount && usedCount >= totalCount) return;
                  const next = await api.nextQuestion(current.id);
                  const merged = { ...next, startedAt: next.startedAt ?? current.startedAt, timer: next.timer ?? current.timer };
                  const updatedUsed = (merged.usedQuestions ?? current.usedQuestions ?? []).length
                    ? merged.usedQuestions ?? current.usedQuestions ?? []
                    : [
                        ...(current.usedQuestions ?? []),
                        ...(merged.questionId ? [{ id: merged.questionId, title: merged.questionTitle, qType: merged.useIDE ? "coding" : "theory", codeTaskId: merged.codeTaskId, position: (current.usedQuestions?.length ?? 0) }] : []),
                      ];
                  setSession({ ...merged, usedQuestions: updatedUsed, total: merged.total ?? updatedUsed.length });
                  setInterviewId(merged.id);
                  setFollowUpQuestion(null);
                  setFollowUpAnswer("");
                  setMissingPoints([]);
                  setBaseScore(null);
                  setFollowLocked(false);
                  setAnswer("");
                  if (merged.useIDE) {
                    const initial = codeByQuestion[merged.questionId ?? ""] ?? merged.starterCode ?? "";
                    setCode(initial);
                    setCodeByQuestion((prev) => {
                      const updated = { ...prev };
                      if (merged.questionId && !(merged.questionId in updated) && merged.starterCode) {
                        updated[merged.questionId] = merged.starterCode;
                      }
                      if (typeof window !== "undefined") {
                        try {
                          localStorage.setItem("vibe-code-by-question", JSON.stringify(updated));
                        } catch {
                          // ignore
                        }
                      }
                      return updated;
                    });
                  } else {
                    setCode("");
                  }
                } else {
                  const s = await api.getInterviewSession(current.id, q.id);
                  const merged = { ...s, startedAt: s.startedAt ?? current.startedAt, timer: s.timer ?? current.timer };
                  setSession(merged);
                  setInterviewId(merged.id);
                  const content = await api.getAnswer(merged.id, merged.questionId ?? "");
                  setAnswer(content.content ?? "");
                  setAnswersState((prev) => ({ ...prev, [merged.questionId ?? ""]: content.content ?? "" }));
                  setFollowUpQuestion(null);
                  setFollowUpAnswer("");
                  setMissingPoints([]);
                  setBaseScore(null);
                  setFollowLocked(false);
                }
              }}
            >
              Вопрос {idx + 1}
            </button>
          ))}
        </div>
        <Button
          variant="outline"
          className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg border-rose-500 bg-rose-50 text-rose-700 shadow-sm hover:bg-rose-100 dark:border-rose-500/70 dark:bg-rose-900/30 dark:text-rose-100 dark:hover:bg-rose-900/50"
          onClick={() => setFinishConfirm(true)}
          title="Завершить собеседование"
        >
          <span className="text-2xl font-bold leading-none">✕</span>
        </Button>
      </div>
      <div
        className={cn(
          "items-start gap-3",
          current.useIDE ? "grid grid-cols-[auto_6px_1fr]" : "grid grid-cols-1"
        )}
        style={
          current.useIDE
            ? { gridTemplateColumns: `${(paneRatio * 100).toFixed(1)}% 6px 1fr` }
            : undefined
        }
        ref={splitRef}
      >
        <div className="space-y-3 min-w-[260px]">
          {!current.useIDE && (
            <Card title="Ответ на задание">
              <TextArea
                className="select-text"
                value={answersState[current.questionId ?? ""] ?? answer}
                onChange={(e) => {
                  if (current?.questionId) {
                    setAnswersState((prev) => ({ ...prev, [current.questionId]: e.target.value }));
                  }
                  setAnswer(e.target.value);
                }}
                placeholder="Опишите решение или вставьте код, если нет IDE"
              />
              <div className="mt-2 flex justify-start">
              <Button
                onClick={async () => {
                  if (!current.id || !current.questionId) return;
                  if (followUpQuestion || followLocked) {
                    // Пока открыт follow-up или ответ уже зафиксирован — основная кнопка не активна
                    return;
                  }
                  try {
                    setAnswerStatus("saving");
                    // Если уже есть follow-up вопрос — оцениваем уточнение
                    const res = await api.evalTheoryAnswer({
                      sessionId: current.id,
                      questionId: current.questionId,
                      ownerId: user?.id ?? "",
                      answer,
                    });
                    setAnswersState((prev) => ({ ...prev, [current.questionId]: answer }));
                    setBaseScore({ score: res.score, maxScore: res.maxScore });
                    setFollowUpState((prev) => ({
                      ...prev,
                      [current.questionId]: {
                        question: res.followUp?.question || null,
                          answer: "",
                          missing: res.missingPoints || [],
                          base: { score: res.score, maxScore: res.maxScore },
                        },
                      }));
                    if (res.followUp?.question) {
                      setFollowUpQuestion(res.followUp.question);
                      setMissingPoints(res.missingPoints || []);
                      setFollowUpAnswer("");
                    } else {
                      setFollowLocked(true);
                      await loadNextQuestion();
                    }
                  setAnswerStatus("saved");
                  setTimeout(() => setAnswerStatus("idle"), 1500);
                  } catch (e) {
                    setAnswerStatus("error");
                  }
                }}
                size="md"
                className="w-1/3 min-w-[180px]"
              >
                {answerStatus === "saving" ? "Отправляем..." : answerStatus === "saved" ? "Отправлено" : "Отправить ответ"}
              </Button>
              </div>
              {followUpQuestion && (
                <div className="mt-4 space-y-2">
                  <div className="text-sm font-semibold">Уточняющий вопрос</div>
                  <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-3 text-sm">
                    {followUpQuestion}
                  </div>
                  <TextArea
                    className="select-text"
                    value={followUpAnswer}
                    onChange={(e) => setFollowUpAnswer(e.target.value)}
                    placeholder="Ответьте на уточняющий вопрос"
                  />
                  <Button
                    onClick={async () => {
                      if (!current?.id || !current.questionId) return;
                      try {
                        setFollowStatus("saving");
                      const res = await api.evalTheoryFollowup({
                        sessionId: current.id,
                        questionId: current.questionId,
                        ownerId: user?.id ?? "",
                        answer: followUpAnswer,
                        followupQuestion: followUpQuestion || "",
                        missingPoints,
                      });
                      // усредняем балл основного и уточнения
                        if (baseScore) {
                          const combinedScore = Math.round((baseScore.score + (res.score ?? 0)) / 2);
                          const maxScore = Math.max(baseScore.maxScore, res.maxScore ?? baseScore.maxScore);
                          const persistedAnswer = answersState[current.questionId] ?? answer;
                          await api.saveAnswer(current.id, current.questionId, persistedAnswer); // чтобы обновить контент
                          // заглушка: бэкенд уже сохраняет score, но здесь только фиксируем UI
                          setBaseScore({ score: combinedScore, maxScore });
                          setAnswersState((prev) => ({ ...prev, [current.questionId]: answer }));
                          setFollowUpState((prev) => ({
                            ...prev,
                            [current.questionId]: {
                              question: null,
                              answer: "",
                              missing: [],
                              base: { score: combinedScore, maxScore },
                            },
                          }));
                        }
                        setFollowUpQuestion(null);
                        setFollowUpAnswer("");
                        setMissingPoints([]);
                        setFollowLocked(true);
                        await loadNextQuestion();
                        setFollowStatus("saved");
                        setTimeout(() => setFollowStatus("idle"), 1500);
                      } catch (e) {
                        setFollowStatus("error");
                      }
                    }}
                    size="md"
                    className="w-full bg-vibe-100 text-vibe-700"
                  >
                    {followStatus === "saving" ? "Оцениваем..." : "Отправить ответ"}
                  </Button>
                </div>
              )}
            </Card>
          )}
          {current.useIDE && (
            <Card className="relative">
              <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-md border border-[var(--border)] bg-white/90 px-2 py-1 text-sm dark:bg-black/50">
                <span className="text-[var(--muted)]">Язык:</span>
                <select
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="rounded border border-[var(--border)] bg-white px-2 py-1 text-sm text-black focus:outline-none dark:bg-slate-800 dark:text-white"
                >
                  {languageOptions.map((lang) => (
                    <option key={lang} value={lang}>
                      {lang}
                    </option>
                  ))}
                </select>
              </div>
              <EditorPane
                sessionId={current.id}
                questionId={current.questionId}
                starterCode={current.starterCode}
                savedCode={codeByQuestion[current.questionId ?? ""] ?? code}
                height={editorHeight}
                onCodeChange={(value) => {
                  lastCodeChange.current = Date.now();
                  if (current?.questionId) {
                    const updated = { ...codeByQuestion, [current.questionId]: value ?? "" };
                    setCodeByQuestion(updated);
                    if (typeof window !== "undefined") {
                      try {
                        localStorage.setItem("vibe-code-by-question", JSON.stringify(updated));
                      } catch {
                        // ignore
                      }
                    }
                    setCode(value ?? "");
                  }
                }}
                onHeavyPaste={(len) => {
                  api.sendAntiCheat({
                    sessionId: current.id!,
                    eventType: "mass_paste",
                    payload: { len },
                    risk: len > 1000 ? "high" : "medium",
                  });
                }}
              />
              <div className="mt-4 flex flex-wrap gap-2">
                <Button
                  onClick={() => runMutation.mutate()}
                  variant="outline"
                  disabled={
                    runMutation.isPending ||
                    !current?.codeTaskId ||
                    (current?.questionId && (attemptsByQuestion[current.questionId] ?? 0) >= 3)
                  }
                  className="flex-1 min-w-[140px] bg-[rgba(109,65,128,0.25)] text-[rgb(109,65,128)] border border-[rgba(109,65,128,0.55)] hover:bg-[rgba(109,65,128,0.35)] shadow-sm"
                >
                  {runMutation.isPending ? "Выполняем примеры..." : "Запустить примеры"}
                </Button>
                <Button
                  onClick={() => mutation.mutate()}
                  disabled={
                    mutation.isPending ||
                    !current?.codeTaskId ||
                    (current?.questionId && (attemptsByQuestion[current.questionId] ?? 0) >= 3)
                  }
                  size="lg"
                  className="flex-1 min-w-[140px] bg-gradient-to-r from-vibe-500 to-vibe-700 text-white hover:brightness-110"
                >
                  {mutation.isPending ? "Проверяем..." : "Проверить решение"}
                </Button>
              </div>
              {lastTestKind === "check" && <TestResults result={testResult} />}
              {lastTestKind === "run" && runResult && (
                <div className="mt-3 rounded-2xl border border-[var(--border)] bg-[var(--card)] p-3">
                  <div className="mb-2 font-semibold">Результаты примеров</div>
                  {runResult.hasError && (
                    <div className="text-sm text-rose-500">
                      Ошибка выполнения кода{runResult.error ? `: ${runResult.error}` : ""}
                    </div>
                  )}
                  {!runResult.hasError && (
                    <div className="space-y-2 text-sm">
                      {(runResult.tests || []).map((t) => (
                        <div key={t.name} className="rounded-xl border border-[var(--border)] px-3 py-2">
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{t.name}</span>
                            <span className={t.status === "passed" ? "text-emerald-500" : t.status === "failed" ? "text-amber-500" : "text-rose-500"}>
                              {t.status === "passed" ? "OK" : t.status === "failed" ? "Fail" : "Error"}
                            </span>
                          </div>
                          <div className="text-xs text-[var(--muted)]">
                            <div>Ожидалось: {JSON.stringify(t.expected)}</div>
                            <div>Получено: {JSON.stringify(t.actual)}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </Card>
          )}
          {!current.useIDE && lastTestKind === "check" && <TestResults result={testResult} />}
        </div>
        {current.useIDE && (
          <div
            className="h-full cursor-col-resize self-stretch rounded-full bg-[var(--border)] transition hover:bg-vibe-400"
            onMouseDown={() => {
              isDragging.current = true;
            }}
          />
        )}
        <div className="relative flex flex-1 min-w-[520px] w-full">
          <ChatPane
            sessionId={current.id}
            questionId={current.questionId}
            taskId={current.codeTaskId}
            ownerId={user?.id}
            language={selectedLanguage || current.language}
            userCode={codeByQuestion[current.questionId ?? ""] ?? code}
          />
        </div>
      </div>
    </main>
    {finishConfirm && (
      <div className="fixed inset-0 z-[3000] flex items-center justify-center bg-black/40 p-4">
        <div className="w-full max-w-md rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-2xl">
          <div className="mb-3 text-lg font-semibold">Вы точно хотите завершить?</div>
          <p className="text-sm text-[var(--muted)] mb-4">Все несохранённые ответы будут потеряны.</p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setFinishConfirm(false)}>
              Нет
            </Button>
            <Button
              className="bg-rose-500 text-white hover:bg-rose-600"
              onClick={async () => {
                if (!current?.id) return;
                const sc = await api.finishInterview(current.id);
                queryClient.setQueryData(["admin-events"], []);
                reset();
                setFinishConfirm(false);
                setScoreModal({ open: true, score: sc ?? null });
              }}
            >
              Да
            </Button>
          </div>
        </div>
      </div>
    )}
    {/* Кнопка поддержки */}
    <button
      aria-label="Открыть поддержку"
      onClick={() => setSupportOpen((prev) => !prev)}
      className="fixed bottom-6 right-6 z-[2000] h-14 w-14 rounded-full bg-gradient-to-br from-[#2F80ED] to-[#0F5AD8] text-white shadow-xl shadow-blue-500/30 transition hover:scale-105 active:scale-95"
    >
      <span className="flex h-full w-full items-center justify-center text-xl">🎧</span>
    </button>
    {supportOpen && (
      <div className="fixed bottom-24 right-6 z-[1999] h-[430px] w-[360px] rounded-2xl border border-[var(--border)] bg-[var(--card)] p-4 shadow-2xl">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-sm font-semibold">Чат поддержки</div>
          <button
            className="text-xs text-[var(--muted)] hover:text-vibe-600"
            onClick={() => setSupportOpen(false)}
          >
            Закрыть
          </button>
        </div>
        {!user && <div className="text-sm text-rose-500">Войдите, чтобы написать в поддержку.</div>}
        <div className="flex h-[350px] flex-col gap-2 text-sm">
          <div className="flex-1 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--card)] p-2">
            {supportMessages.length === 0 && (
              <div className="text-center text-xs text-[var(--muted)]">Сообщений пока нет</div>
            )}
            <div className="space-y-2">
              {supportMessages.map((m) => (
                <div key={m.id} className={`flex ${m.role === "admin" ? "justify-start" : "justify-end"}`}>
                  <div
                    className={`max-w-[90%] rounded-2xl px-3 py-2 text-xs ${
                      m.role === "admin"
                        ? "bg-vibe-50 text-vibe-900 dark:bg-white/10 dark:text-white"
                        : "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white shadow"
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={supportInput}
              onChange={(e) => setSupportInput(e.target.value)}
              placeholder="Напишите сообщение..."
              className="flex-1 rounded-full border border-[var(--border)] bg-transparent px-3 py-2 text-sm"
            />
            <Button
              size="sm"
              onClick={async () => {
                if (!user?.id || !supportInput.trim()) return;
                const text = supportInput.trim();
                setSupportInput("");
                setSupportMessages((prev) => [
                  ...prev,
                  { id: Math.random().toString(36).slice(2), role: "user", content: text, createdAt: new Date().toISOString() },
                ]);
                try {
                  await api.sendSupport(text, user.id);
                  const msgs = await api.getSupportMessages(user.id);
                  setSupportMessages(msgs);
                } catch {
                  // ignore
                }
              }}
            >
              Отправить
            </Button>
          </div>
        </div>
      </div>
    )}
    {scoreModal.open && (
      <div className="fixed inset-0 z-[3500] flex items-center justify-center bg-black/40 p-4">
        <div className="w-full max-w-sm rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-2xl text-center">
          <div className="text-lg font-semibold mb-2">Итоговый Score</div>
          <div className="text-3xl font-bold text-vibe-600 mb-3">
            {scoreModal.score !== null ? scoreModal.score : "—"}
          </div>
          <p className="text-sm text-[var(--muted)] mb-4">Спасибо за прохождение собеседования!</p>
          <div className="flex justify-center gap-2">
            <Button
              className="bg-vibe-600 text-white hover:bg-vibe-700"
              onClick={() => {
                setScoreModal({ open: false, score: null });
                router.push("/profile");
              }}
            >
              Закрыть
            </Button>
          </div>
        </div>
      </div>
    )}
  </>
  );
}
