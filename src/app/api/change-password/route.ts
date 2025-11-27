import { NextResponse } from "next/server";
import Database from "better-sqlite3";
import crypto from "crypto";
import path from "path";

// Локальный маршрут смены пароля через SQLite (использует тот же файл, что и Python-API)
export async function POST(req: Request) {
  let db: Database | null = null;
  try {
    const { userId, oldPassword, newPassword } = (await req.json()) as {
      userId?: string;
      oldPassword?: string;
      newPassword?: string;
    };
    if (!userId || !oldPassword || !newPassword) {
      return NextResponse.json({ detail: "invalid_payload" }, { status: 400 });
    }

    const dbPath = path.join(process.cwd(), "data", "api.sqlite");
    db = new Database(dbPath);
    db.pragma("busy_timeout = 5000");
    db.pragma("journal_mode = WAL");

    const sha1 = (val: string) => crypto.createHash("sha1").update(val).digest("hex");

    const user = db.prepare("SELECT * FROM users WHERE id=?").get(userId);
    if (!user) return NextResponse.json({ detail: "not_found" }, { status: 404 });
    if (user.password !== sha1(oldPassword)) {
      return NextResponse.json({ detail: "wrong_old_password" }, { status: 400 });
    }

    db.prepare("UPDATE users SET password=? WHERE id=?").run(sha1(newPassword), userId);
    return NextResponse.json({ status: "ok" });
  } catch (e: any) {
    return NextResponse.json({ detail: "internal_error" }, { status: 500 });
  } finally {
    try {
      db?.close();
    } catch {}
  }
}
