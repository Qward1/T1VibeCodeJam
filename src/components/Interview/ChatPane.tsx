"use client";
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Button } from "@/components/UI/Button";
import { formatDate } from "@/utils";
import { Message } from "@/types";

type Props = { sessionId: string; questionId?: string; taskId?: string; ownerId?: string; language?: string; userCode?: string };

export const ChatPane = ({ sessionId, questionId, taskId, ownerId, language, userCode }: Props) => {
  const listRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>(600);
  const dragState = useRef<{ startY: number; startH: number } | null>(null);
  const queryClient = useQueryClient();
  const hintMutation = useMutation({
    mutationFn: () =>
      api.codeHint({
        sessionId,
        questionId: questionId ?? "",
        taskId: taskId ?? "",
        language: language ?? "python",
        ownerId,
        userCode,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", sessionId, questionId] });
    },
  });

  const { data } = useQuery({
    queryKey: ["chat", sessionId, questionId],
    queryFn: async () => {
      const res = await api.getChat(sessionId, questionId);
      return res;
    },
  });
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    setMessages(data?.chat ?? []);
  }, [data?.chat]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendHint = () => {
    if ((hintMutation.data?.hintsUsed ?? 0) >= 3) return;
    const msg = "Нужна подсказка";
    const now = new Date().toISOString();
    setMessages((prev) => [...prev, { id: now, role: "user", content: msg, createdAt: now }]);
    hintMutation.mutate();
  };

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragState.current) return;
      const delta = e.clientY - dragState.current.startY;
      const next = Math.min(window.innerHeight * 0.9, Math.max(320, dragState.current.startH + delta));
      setHeight(next);
      e.preventDefault();
    };
    const onUp = () => {
      dragState.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  return (
    <div className="relative flex flex-col gap-3 select-text" style={{ height, minHeight: 320, maxHeight: "90vh", width: "100%" }}>
      <div
        ref={listRef}
        className="relative flex-1 space-y-3 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--card)] p-4 w-full"
        style={{ minWidth: "100%" }}
      >
        {messages.length === 0 && (
          <div className="text-center text-sm text-[var(--muted)]">Сообщений пока нет — задайте вопрос!</div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col gap-1 ${msg.role === "assistant" ? "items-start" : "items-end"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "assistant"
                  ? "bg-vibe-50 text-vibe-950 shadow-sm dark:bg-white/10 dark:text-white"
                  : "bg-gradient-to-r from-vibe-500 to-vibe-700 text-white shadow"
              }`}
            >
              {msg.content}
            </div>
            <span className="text-[11px] text-[var(--muted)]">{formatDate(msg.createdAt)}</span>
          </div>
        ))}
      </div>
      <div className="flex justify-center">
        <Button
          onClick={sendHint}
          disabled={hintMutation.isPending || !taskId || !ownerId || (hintMutation.data?.hintsUsed ?? 0) >= 3}
          className="min-w-[200px] scale-105 px-6 py-3 text-base flex items-center justify-center"
        >
          {hintMutation.isPending
            ? "Готовим подсказку..."
            : (hintMutation.data?.hintsUsed ?? 0) >= 3
              ? "Лимит подсказок"
              : "Нужна подсказка"}
        </Button>
      </div>
      <div
        className="absolute bottom-3 right-3 z-10 flex h-7 w-7 cursor-row-resize items-center justify-center rounded-full border border-[var(--border)] bg-[var(--card)] text-[12px] text-[var(--muted)] shadow-sm"
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
          dragState.current = { startY: e.clientY, startH: height };
        }}
        title="Тянуть, чтобы изменить высоту"
      >
        ↕︎
      </div>
    </div>
  );
};
