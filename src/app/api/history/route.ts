import { NextResponse } from "next/server";
import { listHistory } from "@/server/db";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");
  if (!userId) return NextResponse.json({ error: "missing_user" }, { status: 400 });
  const sessions = await listHistory(userId);
  const history = sessions.map((s) => ({
    id: s.id,
    topic: s.description.slice(0, 32) + "...",
    direction: s.direction,
    level: s.level,
    score: Math.round(Math.random() * 30 + 60),
    date: new Date().toISOString(),
  }));
  return NextResponse.json({ history });
}
