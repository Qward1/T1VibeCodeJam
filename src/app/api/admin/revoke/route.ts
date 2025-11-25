import { NextResponse } from "next/server";
import { getUserById, setAdmin } from "@/server/db";

export async function POST(req: Request) {
  const body = await req.json();
  const { superId, targetUserId } = body as { superId: string; targetUserId: string };
  if (!superId || !targetUserId) return NextResponse.json({ error: "invalid_payload" }, { status: 400 });
  const superUser = await getUserById(superId);
  if (!superUser || superUser.role !== "superadmin") return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const updated = await setAdmin(targetUserId, false);
  return NextResponse.json({ user: updated });
}
