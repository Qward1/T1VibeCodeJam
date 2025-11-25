import Database from "better-sqlite3";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import {
  InterviewReport,
  InterviewSession,
  Message,
  User,
} from "@/types";

export type UserRecord = User & {
  passwordHash: string;
  role: "user" | "admin" | "superadmin";
  admin: boolean;
};

const dataDir = path.join(process.cwd(), "data");
const dbPath = path.join(dataDir, "db.sqlite");

if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

const db = new Database(dbPath);
db.pragma("journal_mode = WAL");

// Создаём таблицы, если их нет
const bootstrap = () => {
  db.transaction(() => {
    db.prepare(`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        passwordHash TEXT NOT NULL,
        level TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        admin INTEGER NOT NULL DEFAULT 0
      )
    `).run();

    db.prepare(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        ownerId TEXT NOT NULL,
        direction TEXT,
        level TEXT,
        format TEXT,
        tasks TEXT,
        description TEXT,
        starterCode TEXT,
        timer INTEGER,
        startedAt TEXT,
        solved INTEGER,
        total INTEGER,
        createdAt TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(ownerId) REFERENCES users(id)
      )
    `).run();

    db.prepare(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        sessionId TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        createdAt TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(sessionId) REFERENCES sessions(id)
      )
    `).run();

    db.prepare(`
      CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        sessionId TEXT NOT NULL,
        ownerId TEXT NOT NULL,
        score INTEGER,
        level TEXT,
        summary TEXT,
        timeline TEXT,
        solutions TEXT,
        analytics TEXT,
        createdAt TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(sessionId) REFERENCES sessions(id),
        FOREIGN KEY(ownerId) REFERENCES users(id)
      )
    `).run();

    // Супер-админ
    const exists = db.prepare("SELECT 1 FROM users WHERE role='superadmin' LIMIT 1").get();
    if (!exists) {
      db.prepare(
        `INSERT INTO users (id, email, name, passwordHash, level, role, admin)
         VALUES (@id, @email, @name, @passwordHash, @level, @role, @admin)`
      ).run({
        id: "super-1",
        email: "root@vibe.dev",
        name: "Главный админ",
        passwordHash: hash("admin"),
        level: "Senior",
        role: "superadmin",
        admin: 1,
      });
    }
  })();

  const colsUsers = db.prepare("PRAGMA table_info(users)").all().map((c: any) => c.name);
  const colsSessions = db.prepare("PRAGMA table_info(sessions)").all().map((c: any) => c.name);
  if (!colsUsers.includes("lang")) {
    db.prepare("ALTER TABLE users ADD COLUMN lang TEXT DEFAULT 'ru'").run();
  }
  if (!colsSessions.includes("startedAt")) {
    db.prepare("ALTER TABLE sessions ADD COLUMN startedAt TEXT").run();
  }
};

bootstrap();

const hash = (val: string) => crypto.createHash("sha1").update(val).digest("hex");
const uid = () => Math.random().toString(36).slice(2, 10);

const mapUser = (row: any): UserRecord => ({
  id: row.id,
  email: row.email,
  name: row.name,
  level: row.level,
  passwordHash: row.passwordHash,
  role: row.role,
  admin: Boolean(row.admin),
});

const mapSession = (row: any): InterviewSession => ({
  id: row.id,
  ownerId: row.ownerId,
  direction: row.direction,
  level: row.level,
  format: row.format,
  tasks: row.tasks ? JSON.parse(row.tasks) : [],
  description: row.description,
  starterCode: row.starterCode,
  timer: row.timer,
  startedAt: row.startedAt,
  solved: row.solved,
  total: row.total,
});

export async function createUser(email: string, password: string, name: string): Promise<UserRecord> {
  const exists = db.prepare("SELECT id FROM users WHERE email=@email").get({ email });
  if (exists) throw new Error("already_exists");
  const user: UserRecord = {
    id: uid(),
    email,
    name,
    level: "Junior",
    passwordHash: hash(password),
    role: "user",
    admin: false,
  };
  db.prepare(
    `INSERT INTO users (id, email, name, passwordHash, level, role, admin)
     VALUES (@id, @email, @name, @passwordHash, @level, @role, @admin)`
  ).run(user);
  return user;
}

export async function authenticate(email: string, password: string): Promise<UserRecord | null> {
  const row = db
    .prepare("SELECT * FROM users WHERE email=@email AND passwordHash=@hash LIMIT 1")
    .get({ email, hash: hash(password) });
  return row ? mapUser(row) : null;
}

export async function getUserById(id: string): Promise<UserRecord | undefined> {
  const row = db.prepare("SELECT * FROM users WHERE id=@id").get({ id });
  return row ? mapUser(row) : undefined;
}

export async function listHistory(userId: string) {
  const rows = db
    .prepare("SELECT * FROM sessions WHERE ownerId=@userId ORDER BY datetime(createdAt) DESC")
    .all({ userId });
  return rows.map(mapSession);
}

export async function listUsers() {
  const rows = db.prepare("SELECT * FROM users ORDER BY datetime(id)").all();
  return rows.map(mapUser);
}

export async function addSession(userId: string, session: InterviewSession) {
  db.prepare(
    `INSERT INTO sessions (id, ownerId, direction, level, format, tasks, description, starterCode, timer, startedAt, solved, total)
     VALUES (@id, @ownerId, @direction, @level, @format, @tasks, @description, @starterCode, @timer, @startedAt, @solved, @total)`
  ).run({
    ...session,
    ownerId: userId,
    tasks: JSON.stringify(session.tasks ?? []),
    startedAt: session.startedAt ?? new Date().toISOString(),
  });
}

export async function getSession(id: string) {
  const row = db.prepare("SELECT * FROM sessions WHERE id=@id").get({ id });
  if (!row) return undefined;
  let remaining = row.timer;
  if (row.startedAt) {
    const started = Date.parse(row.startedAt);
    if (!Number.isNaN(started)) {
      const elapsed = (Date.now() - started) / 1000;
      remaining = Math.max(0, Math.floor(row.timer - elapsed));
    }
  }
  return mapSession({ ...row, timer: remaining });
}

export async function saveChat(sessionId: string, msg: Message) {
  db.prepare(
    `INSERT INTO messages (id, sessionId, role, content, createdAt)
     VALUES (@id, @sessionId, @role, @content, @createdAt)`
  ).run({ ...msg, sessionId });
}

export async function getChat(sessionId: string) {
  const rows = db
    .prepare("SELECT * FROM messages WHERE sessionId=@sessionId ORDER BY datetime(createdAt) ASC")
    .all({ sessionId });
  return rows.map((r) => ({
    id: r.id,
    role: r.role,
    content: r.content,
    createdAt: r.createdAt,
  } as Message));
}

export async function saveReport(report: InterviewReport) {
  const sessionId = report.sessionId ?? report.id;
  const session = await getSession(sessionId);
  const ownerId = report.ownerId ?? session?.ownerId ?? "super-1";
  db.prepare(
    `INSERT INTO reports (id, sessionId, ownerId, score, level, summary, timeline, solutions, analytics)
     VALUES (@id, @sessionId, @ownerId, @score, @level, @summary, @timeline, @solutions, @analytics)
     ON CONFLICT(id) DO UPDATE SET score=excluded.score, level=excluded.level, summary=excluded.summary,
       timeline=excluded.timeline, solutions=excluded.solutions, analytics=excluded.analytics`
  ).run({
    ...report,
    sessionId,
    ownerId,
    timeline: JSON.stringify(report.timeline ?? []),
    solutions: JSON.stringify(report.solutions ?? []),
    analytics: JSON.stringify(report.analytics ?? {}),
  });
}

export async function getReport(id: string) {
  const row = db.prepare("SELECT * FROM reports WHERE id=@id").get({ id });
  if (!row) return undefined;
  return {
    id: row.id,
    score: row.score,
    level: row.level,
    summary: row.summary,
    timeline: JSON.parse(row.timeline || "[]"),
    solutions: JSON.parse(row.solutions || "[]"),
    analytics: JSON.parse(row.analytics || "{}"),
  } as InterviewReport;
}

export async function setAdmin(targetUserId: string, flag: boolean) {
  const result = db
    .prepare("UPDATE users SET admin=@flag, role=@role WHERE id=@id")
    .run({ id: targetUserId, flag: flag ? 1 : 0, role: flag ? "admin" : "user" });
  if (!result.changes) throw new Error("not_found");
  const updated = await getUserById(targetUserId);
  if (!updated) throw new Error("not_found");
  return updated;
}
