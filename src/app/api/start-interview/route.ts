import { NextResponse } from "next/server";
import { addSession, saveReport } from "@/server/db";
import { InterviewSession, Level } from "@/types";

const uid = () => Math.random().toString(36).slice(2, 10);

export async function POST(req: Request) {
  const body = await req.json();
  const { userId, direction, level, format, tasks } = body as {
    userId: string;
    direction: string;
    level: Level;
    format: string;
    tasks: string[];
  };
  if (!userId || !direction || !level) return NextResponse.json({ error: "invalid_payload" }, { status: 400 });

  const session: InterviewSession = {
    id: uid(),
    ownerId: userId,
    direction,
    level,
    format,
    tasks,
    description: "Реализуйте функцию поиска оптимального маршрута доставки...",
    starterCode: `function twoSum(nums, target) {\n  const map = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const c = target - nums[i];\n    if (map.has(c)) return [map.get(c), i];\n    map.set(nums[i], i);\n  }\n  return [];\n}`,
    timer: 45 * 60,
    startedAt: new Date().toISOString(),
    solved: 0,
    total: 3,
  };
  await addSession(userId, session);
  // Создаём черновик отчёта, чтобы он был доступен после сессии
  await saveReport({
    id: session.id,
    sessionId: session.id,
    ownerId: userId,
    score: 0,
    level,
    summary: "Отчёт будет сформирован по завершении сессии.",
    timeline: [{ label: "task_start", at: new Date().toISOString() }],
    solutions: [],
    analytics: {
      skillMap: [],
      errorHeatmap: [],
      speed: [],
    },
  });
  return NextResponse.json({ session });
}
