import { NextResponse } from "next/server";
import Database from "better-sqlite3";
import path from "path";

// Фолбэк-роут для завершения собеседования через Next API, пишет в ту же SQLite
export async function POST(req: Request) {
  let db: Database | null = null;
  try {
    const { sessionId, ownerId } = (await req.json()) as { sessionId?: string; ownerId?: string };
    if (!sessionId || !ownerId) return NextResponse.json({ detail: "invalid_payload" }, { status: 400 });
    const dbPath = path.join(process.cwd(), "data", "api.sqlite");
    db = new Database(dbPath);
    db.pragma("busy_timeout = 5000");
    db.pragma("journal_mode = WAL");
    db.exec(`
      CREATE TABLE IF NOT EXISTS interview_results (
        id TEXT PRIMARY KEY,
        sessionId TEXT NOT NULL,
        ownerId TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed',
        finishedAt TEXT DEFAULT CURRENT_TIMESTAMP
      );
    `);
    db.prepare(
      "INSERT INTO interview_results (id, sessionId, ownerId, status, finishedAt) VALUES (?, ?, ?, ?, datetime('now'))"
    ).run(Math.random().toString(36).slice(2, 10), sessionId, ownerId, "completed");
    return NextResponse.json({ status: "ok" });
  } catch (e) {
    return NextResponse.json({ detail: "internal_error" }, { status: 500 });
  } finally {
    try {
      db?.close();
    } catch {}
  }
}
