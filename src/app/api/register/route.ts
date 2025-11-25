import { NextResponse } from "next/server";
import { createUser } from "@/server/db";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { email, password, name } = body;
    if (!email || !password) return NextResponse.json({ error: "invalid_payload" }, { status: 400 });
    const user = await createUser(email, password, name ?? "Кандидат");
    const { passwordHash, ...safe } = user;
    return NextResponse.json({ user: safe });
  } catch (e: any) {
    if (e.message === "already_exists") return NextResponse.json({ error: "exists" }, { status: 409 });
    return NextResponse.json({ error: "internal" }, { status: 500 });
  }
}
