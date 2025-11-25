"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useSessionStore } from "@/stores/session";
import { api } from "@/services/api";
import { useAuthStore } from "@/stores/auth";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <textarea
      className="h-full min-h-[520px] w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-4"
      placeholder="Загрузка редактора..."
      readOnly
    />
  ),
});

type Props = {
  sessionId?: string;
  questionId?: string;
  onHeavyPaste?: (len: number) => void;
  onCodeChange?: () => void;
  height?: number;
};

export const EditorPane = ({ sessionId, questionId, onHeavyPaste, onCodeChange, height = 600 }: Props) => {
  const { code, setCode } = useSessionStore();
  const user = useAuthStore((s) => s.user);
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);

  return (
    <div className="flex h-full flex-col gap-2">
      {ready ? (
        <div
          className="ide-allowed overflow-auto rounded-xl border border-[var(--border)] select-text"
          style={{ height, minHeight: 360, maxHeight: "80vh", resize: "vertical" }}
        >
          <MonacoEditor
            height="100%"
            defaultLanguage="typescript"
            theme="vs-dark"
            value={code}
            onMount={(editor) => {
              // Отслеживаем вставку больших фрагментов
              const pasteListener = editor.onDidPaste?.((data: any) => {
                const text = data?.range?.length ?? data?.text?.length ?? 0;
                const len = typeof text === "number" ? text : 0;
                if (len > 300 && onHeavyPaste) onHeavyPaste(len);
              });
              editor.onDidChangeModelContent((e) => {
                const largeInsert = e.changes?.some((c) => (c.text?.length ?? 0) > 500);
                if (largeInsert && onHeavyPaste) onHeavyPaste(e.changes[0].text.length);
                onCodeChange?.();
              });
              return () => pasteListener?.dispose();
            }}
            onChange={(value) => setCode(value ?? "")}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              automaticLayout: true,
              scrollBeyondLastLine: false,
              renderValidationDecorations: "off",
            }}
          />
        </div>
      ) : (
        <textarea
          className="h-[600px] w-full rounded-xl border border-[var(--border)] bg-[var(--card)] p-4"
          defaultValue={code}
          onChange={(e) => setCode(e.target.value)}
        />
      )}
    </div>
  );
};
