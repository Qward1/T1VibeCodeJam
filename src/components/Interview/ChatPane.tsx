"use client";
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import { Button } from "@/components/UI/Button";
import { formatDate } from "@/utils";
import { Message } from "@/types";

type Props = { sessionId: string; questionId?: string };

export const ChatPane = ({ sessionId, questionId }: Props) => {
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

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

  const mutation = useMutation({
    mutationFn: (msg: string) => api.sendMessage(sessionId, msg, questionId),
    onSuccess: (aiMsg) => setMessages((prev) => [...prev, aiMsg]),
  });

  const send = () => {
    if (!input.trim()) return;
    const now = new Date().toISOString();
    setMessages((prev) => [...prev, { id: now, role: "user", content: input, createdAt: now }]);
    mutation.mutate(input);
    setInput("");
  };

  return (
    <div
      className="flex h-full min-h-[720px] flex-col gap-2 select-text"
      style={{ resize: "vertical", minHeight: "400px", maxHeight: "90vh", overflow: "hidden" }}
    >
      <div
        ref={listRef}
        className="flex-1 space-y-3 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--card)] p-4"
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
      <div className="flex flex-col gap-2 rounded-xl border border-[var(--border)] bg-[var(--card)] p-3">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Спросите про задачу, попросите подсказку или повтор вопроса"
            className="flex-1 rounded-full border border-[var(--border)] bg-transparent px-4 py-3 text-sm focus:border-vibe-400 focus:outline-none"
          />
          <Button onClick={send} disabled={mutation.isPending}>
            {mutation.isPending ? "Ждём..." : "Отправить"}
          </Button>
        </div>
      </div>
    </div>
  );
};
