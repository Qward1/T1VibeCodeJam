import { NextResponse } from "next/server";
import { getUserById } from "@/server/db";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");
  if (!userId) return NextResponse.json({ error: "missing_user" }, { status: 400 });
  const user = await getUserById(userId);
  if (!user) return NextResponse.json({ error: "not_found" }, { status: 404 });
  const { passwordHash, ...safe } = user;
  return NextResponse.json({
    user: safe,
    stats: {
      skillMap: [
        { label: "JS/TS", value: 80 },
        { label: "React", value: 85 },
        { label: "Architecture", value: 72 },
        { label: "Algorithms", value: 65 },
        { label: "Debug", value: 75 },
      ],
      avgSolveTime: 14,
      errorHeatmap: [
        { bucket: "Off-by-one", count: 3 },
        { bucket: "Types", count: 2 },
        { bucket: "Edge cases", count: 4 },
      ],
    },
  });
}
