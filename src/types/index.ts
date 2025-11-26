export type Level = "Junior" | "Middle" | "Senior";

export type User = {
  id: string;
  email: string;
  name: string;
  level: Level;
  admin?: boolean;
  role?: "user" | "admin" | "superadmin";
  lang?: "ru" | "en";
};

export type RegisterPayload = { email: string; password: string; name?: string; lang?: "ru" | "en" };
export type LoginPayload = { email: string; password: string };

export type SkillStat = { label: string; value: number };
export type ErrorHeat = { bucket: string; count: number };

export type InterviewHistoryItem = {
  id: string;
  topic: string;
  direction: string;
  level: Level;
  score: number;
  date: string;
};

export type Profile = {
  user: User;
  stats: {
    skillMap: SkillStat[];
    avgSolveTime: number;
    errorHeatmap: ErrorHeat[];
  };
};

export type StartInterviewPayload = {
  direction: string;
  level: Level;
  format: string;
  tasks: string[];
};

export type InterviewSession = {
  id: string;
  ownerId?: string;
  questionId?: string;
  questionTitle?: string;
  useIDE?: boolean;
  usedQuestions?: { id: string; title?: string; qType?: string; codeTaskId?: string }[];
  codeTaskId?: string;
  language?: string;
  direction: string;
  level: Level;
  format: string;
  tasks: string[];
  description: string;
  starterCode: string;
  functionSignature?: string;
  timer: number;
  startedAt?: string;
  solved: number;
  total: number;
  status?: "active" | "completed";
  is_active?: number;
  is_finished?: number;
};

export type MessageRole = "user" | "assistant" | "system";

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
};

export type TestResult = {
  passed: boolean;
  summary: string;
  cases: { name: string; passed: boolean; details?: string; input?: any; expected?: any; actual?: any }[];
};

export type InterviewReport = {
  id: string;
  ownerId?: string;
  sessionId?: string;
  score: number;
  level: Level;
  summary: string;
  timeline: { label: string; at: string }[];
  solutions: { title: string; code: string; errors?: string; tests: TestResult }[];
  analytics: {
    skillMap: SkillStat[];
    errorHeatmap: ErrorHeat[];
    speed: { label: string; value: number }[];
  };
};

export type Candidate = {
  id: string;
  name: string;
  email: string;
  level: Level;
  lastScore: number;
  lastTopic: string;
  admin?: boolean;
  role?: "user" | "admin" | "superadmin";
  hasFlags?: boolean;
  flagsCount?: number;
};

export type FlaggedEvent = {
  id: string;
  candidateId: string;
  type: string;
  at: string;
  details: string;
};

export type AdminOverview = {
  candidates: Candidate[];
  flagged: FlaggedEvent[];
  analytics: {
    hardestTopics: { name: string; score: number }[];
    completionRate: number;
    avgScore: number;
  };
};

export type AssignedInterview = {
  id: string;
  candidateId: string;
  adminId: string;
  direction: string;
  level: Level;
  format: string;
  tasks: string[];
  duration?: number | null;
  status: "pending" | "active" | "completed";
  sessionId?: string | null;
  createdAt?: string;
};
