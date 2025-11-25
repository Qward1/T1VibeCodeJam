import { NextResponse } from "next/server";
import Database from "better-sqlite3";
import path from "path";

export async function POST(req: Request) {
  try {
    const { userId, lang } = (await req.json()) as { userId?: string; lang?: string };
    if (!userId || !lang) return NextResponse.json({ detail: "invalid_payload" }, { status: 400 });
    const dbPath = path.join(process.cwd(), "data", "api.sqlite");
    const db = new Database(dbPath);
    const user = db.prepare("SELECT * FROM users WHERE id=?").get(userId);
    if (!user) return NextResponse.json({ detail: "not_found" }, { status: 404 });
    db.prepare("UPDATE users SET lang=? WHERE id=?").run(lang, userId);
    const updated = db.prepare("SELECT * FROM users WHERE id=?").get(userId);
    return NextResponse.json({ user: updated });
  } catch (e) {
    return NextResponse.json({ detail: "internal_error" }, { status: 500 });
  }
}
