import { NextResponse } from "next/server";
import { authenticate } from "@/server/db";

export async function POST(req: Request) {
  const body = await req.json();
  const { email, password } = body;
  if (!email || !password) return NextResponse.json({ error: "invalid_payload" }, { status: 400 });
  const user = await authenticate(email, password);
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { passwordHash, ...safe } = user;
  return NextResponse.json({ user: safe });
}
