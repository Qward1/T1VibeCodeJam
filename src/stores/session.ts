import { create } from "zustand";
import { persist } from "zustand/middleware";
import { Message, InterviewSession, TestResult } from "@/types";

type SessionState = {
  session?: InterviewSession;
  interviewId?: string;
  code: string;
  testResult?: TestResult;
  messages: Message[];
  setSession: (session?: InterviewSession) => void;
  setInterviewId: (id?: string) => void;
  setCode: (code: string) => void;
  setTestResult: (result?: TestResult) => void;
  setMessages: (messages: Message[]) => void;
  pushMessage: (message: Message) => void;
  reset: () => void;
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      session: undefined,
      interviewId: undefined,
      code: "",
      messages: [],
      testResult: undefined,
      setSession: (session) => set({ session, code: "" }), // редактор стартует пустым
      setInterviewId: (id) => set({ interviewId: id }),
      setCode: (code) => set({ code }),
      setTestResult: (testResult) => set({ testResult }),
      setMessages: (messages) => set({ messages }),
      pushMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
      reset: () => set({ session: undefined, interviewId: undefined, code: "", messages: [], testResult: undefined }),
    }),
    { name: "vibe-session" }
  )
);
