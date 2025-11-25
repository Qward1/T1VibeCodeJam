import { NextResponse } from "next/server";
import { getUserById, listUsers } from "@/server/db";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const adminId = searchParams.get("adminId");
  if (!adminId) return NextResponse.json({ error: "missing_admin" }, { status: 400 });
  const admin = await getUserById(adminId);
  if (!admin || (!admin.admin && admin.role !== "superadmin")) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const users = await listUsers();
  const candidates = users.map((u) => ({
    id: u.id,
    name: u.name,
    email: u.email,
    level: u.level,
    lastScore: Math.round(Math.random() * 30 + 60),
    lastTopic: "System Design",
  }));

  const flagged = [] as any[];
  const analytics = {
    hardestTopics: [
      { name: "Concurrency", score: 62 },
      { name: "Graph", score: 58 },
      { name: "API design", score: 64 },
    ],
    completionRate: 0.78,
    avgScore: 0.81,
  };

  return NextResponse.json({ candidates, flagged, analytics });
}
