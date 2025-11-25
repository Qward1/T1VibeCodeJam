import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from pathlib import Path
import sys
import os
import subprocess
import tempfile

# Делаем путь к server_py доступным для импортов при запуске как скрипта
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
  sys.path.append(str(CURRENT_DIR))

from db import ROOT, DATA_DIR, DB_PATH, sha1, fetchone_dict, fetchall_dicts, ensure_schema, seed_questions
from api_next_question import router as next_router
import random

MAX_QUESTIONS = 3  # легко менять при необходимости
TIMER_SECONDS = 45 * 60

app = FastAPI(title="Interview Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(next_router)

@app.get("/")
def root():
    return {"status": "ok", "service": "interview-api"}

# --------- Pydantic схемы
class UserPayload(BaseModel):
    email: str
    password: str
    name: Optional[str] = "Кандидат"
    lang: Optional[str] = "ru"

class LoginPayload(BaseModel):
    email: str
    password: str
    lang: Optional[str] = "ru"

class StartPayload(BaseModel):
    userId: str
    direction: str
    level: str
    format: str
    tasks: List[str]

class ChatPayload(BaseModel):
    message: str
    questionId: Optional[str] = None

class AdminAction(BaseModel):
    superId: str
    targetUserId: str

class ChangePasswordPayload(BaseModel):
    userId: str
    oldPassword: str
    newPassword: str
    confirmPassword: Optional[str] = None

class ChangeLanguagePayload(BaseModel):
    userId: str
    lang: str

class FinishInterviewPayload(BaseModel):
    sessionId: str
    ownerId: str

class AnswerPayload(BaseModel):
    sessionId: str
    questionId: str
    ownerId: str
    content: str

class ClearEventsPayload(BaseModel):
    adminId: str

class RunCodePayload(BaseModel):
    sessionId: str
    code: str
    language: str = "javascript"

class SupportSendPayload(BaseModel):
    userId: str
    message: str

class SupportReplyPayload(BaseModel):
    userId: str
    message: str
    adminId: str

class SupportClosePayload(BaseModel):
    userId: str
    adminId: str

class AntiCheatEvent(BaseModel):
    sessionId: str
    ownerId: str | None = None
    eventType: str
    payload: str | None = None
    risk: str | None = "medium"

# --------- Инициализация БД

def init_db():
    ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Супер-админ: гарантируем единственный аккаунт super-1
    supers = cur.execute("SELECT id FROM users WHERE role='superadmin'").fetchall()
    if not supers:
        cur.execute(
            "INSERT INTO users (id, email, name, password, level, role, admin, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("super-1", "root@vibe.dev", "Главный админ", sha1("admin"), "Senior", "superadmin", 1, "ru"),
        )
    else:
        # Все супер-админы кроме super-1 понижаются до admin, чтобы остался один
        cur.execute("UPDATE users SET role='admin', admin=1 WHERE role='superadmin' AND id!='super-1'")
        exists_root = cur.execute("SELECT 1 FROM users WHERE id='super-1' AND role='superadmin'").fetchone()
        if not exists_root:
            cur.execute(
                "INSERT OR REPLACE INTO users (id, email, name, password, level, role, admin, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("super-1", "root@vibe.dev", "Главный админ", sha1("admin"), "Senior", "superadmin", 1, "ru"),
            )
    conn.commit()
    conn.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


ensure_schema()

# --------- Утилиты

def user_safe(row):
    keys = ["id", "email", "name", "level", "role", "admin", "lang"]
    return {k: row[k] for k in keys if k in row}


def fetchone_dict(cursor):
    row = cursor.fetchone()
    if not row:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def fetchall_dicts(cursor):
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, r)) for r in rows]


# --------- Эндпоинты

@app.post("/api/register")
def register(payload: UserPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if cur.execute("SELECT 1 FROM users WHERE email=?", (payload.email,)).fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="exists")
    # Только роль user при регистрации
    user_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO users (id, email, name, password, level, role, admin, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, payload.email, payload.name, sha1(payload.password), "Junior", "user", 0, payload.lang or "ru"),
    )
    conn.commit()
    row = cur.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    columns = [col[0] for col in cur.description] if cur.description else []
    result = dict(zip(columns, row)) if row else {}
    return {"user": user_safe(result)}


@app.post("/api/login")
def login(payload: LoginPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (payload.email, sha1(payload.password)),
    )
    row = fetchone_dict(cur)
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="unauthorized")
    return {"user": user_safe(row)}


@app.get("/api/profile")
def profile(userId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (userId,))
    row = fetchone_dict(cur)
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return {
        "user": {**user_safe(row), "lang": row.get("lang", "ru")},
        "stats": {
          "skillMap": [
            {"label": "JS/TS", "value": 80},
            {"label": "React", "value": 85},
            {"label": "Architecture", "value": 72},
            {"label": "Algorithms", "value": 65},
            {"label": "Debug", "value": 75},
          ],
          "avgSolveTime": 14,
          "errorHeatmap": [
            {"bucket": "Off-by-one", "count": 3},
            {"bucket": "Types", "count": 2},
            {"bucket": "Edge cases", "count": 4},
          ],
        }
    }


@app.get("/api/history")
def history(userId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE ownerId=? ORDER BY datetime(createdAt) DESC", (userId,))
    sessions = fetchall_dicts(cur)
    # Подтягиваем время завершения из результатов
    result_rows = fetchall_dicts(
        cur.execute("SELECT sessionId, finishedAt FROM interview_results WHERE ownerId=?", (userId,))
    )
    finished_map = {r["sessionId"]: r.get("finishedAt") for r in result_rows if r.get("sessionId")}
    conn.close()
    history = []
    for s in sessions:
        finished = finished_map.get(s["id"])
        # Если нет зафиксированного завершения — используем createdAt
        date_val = finished or s.get("createdAt") or now_iso()
        history.append(
            {
                "id": s["id"],
                "topic": (s.get("description") or "Тема")[0:32] + "...",
                "direction": s.get("direction"),
                "level": s.get("level"),
                "score": 70,
                "date": date_val,
            }
        )
    return {"history": history}


@app.post("/api/start-interview")
def start_interview(payload: StartPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM sessions
        WHERE ownerId=?
          AND status='active'
        ORDER BY datetime(createdAt) DESC
        LIMIT 1
        """,
        (payload.userId,),
    )
    existing = fetchone_dict(cur)
    if existing:
        # Если у старой сессии не заполнен таймер/старт — фиксируем их
        updated = False
        if not existing.get("timer"):
            existing["timer"] = TIMER_SECONDS
            cur.execute("UPDATE sessions SET timer=? WHERE id=?", (TIMER_SECONDS, existing["id"]))
            updated = True
        if not existing.get("startedAt"):
            started_at = now_iso()
            existing["startedAt"] = started_at
            cur.execute("UPDATE sessions SET startedAt=? WHERE id=?", (started_at, existing["id"]))
            updated = True
        used_questions = get_used_questions(cur, existing["id"])
        if updated:
            conn.commit()
        conn.close()
        return {
            "session": {
                "id": existing["id"],
                "direction": existing.get("direction"),
                "level": existing.get("level"),
                "format": existing.get("format"),
                "tasks": existing.get("tasks", "").split(",") if existing.get("tasks") else [],
                "questionTitle": existing.get("questionTitle"),
                "questionId": existing.get("questionId"),
                "useIDE": bool(existing.get("useIDE", 1)),
                "description": existing.get("description"),
                "starterCode": existing.get("starterCode"),
                "timer": existing.get("timer"),
                "startedAt": existing.get("startedAt"),
                "solved": existing.get("solved"),
                "total": existing.get("total"),
                "usedQuestions": used_questions,
                "status": existing.get("status", "active"),
                "is_active": existing.get("is_active", 1),
                "is_finished": existing.get("is_finished", 0),
            }
        }
    cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT ?", (MAX_QUESTIONS,))
    selected = fetchall_dicts(cur)
    if not selected:
        seed_questions(cur, force=True)
        cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT ?", (MAX_QUESTIONS,))
        selected = fetchall_dicts(cur)
    if not selected:
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")
    total_questions = min(len(selected), MAX_QUESTIONS)
    question = selected[0]
    session_id = str(uuid.uuid4())
    description = question["body"]
    starter = """function twoSum(nums, target) {\n  const map = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const c = target - nums[i];\n    if (map.has(c)) return [map.get(c), i];\n    map.set(nums[i], i);\n  }\n  return [];\n}"""
    started_at = now_iso()
    cur.execute(
        """
        INSERT INTO sessions (id, ownerId, questionId, questionTitle, useIDE, direction, level, format, tasks, description, starterCode, timer, solved, total, startedAt, status, is_active, is_finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            payload.userId,
            question["id"],
            question["title"],
            int(question.get("useIDE", 1)),
            payload.direction,
            payload.level,
            payload.format,
            ",".join(payload.tasks),
            description,
            starter,
            TIMER_SECONDS,
            0,
            total_questions,
            started_at,
            "active",
            1,
            0,
        ),
    )
    # Добавляем выбранные вопросы в историю с порядком
    for idx, q in enumerate(selected, start=1):
        cur.execute(
            "INSERT INTO session_questions (sessionId, questionId, questionTitle, position) VALUES (?, ?, ?, ?)",
            (session_id, q["id"], q["title"], idx),
        )
    # Черновик отчёта
    cur.execute(
        """
        INSERT OR REPLACE INTO reports (id, sessionId, ownerId, score, level, summary, timeline, solutions, analytics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            session_id,
            payload.userId,
            0,
            payload.level,
            "Отчёт формируется",
            "[]",
            "[]",
            "{}",
        ),
    )
    conn.commit()
    used_questions = get_used_questions(cur, session_id)
    conn.close()
    return {
        "session": {
            "id": session_id,
            "direction": payload.direction,
            "level": payload.level,
            "format": payload.format,
            "tasks": payload.tasks,
            "questionTitle": question["title"],
            "questionId": question["id"],
            "useIDE": bool(question.get("useIDE")),
            "description": description,
            "starterCode": starter,
            "timer": TIMER_SECONDS,
            "solved": 0,
            "total": total_questions,
            "startedAt": started_at,
            "usedQuestions": used_questions,
            "status": "active",
            "is_active": 1,
            "is_finished": 0,
        }
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str, questionId: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    row = fetchone_dict(cur)
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    # Позволяем открыть завершённые сессии только для реплея, иначе 404
    # Здесь продолжаем отдавать только активные
    if row.get("is_finished") == 1:
        conn.close()
        raise HTTPException(status_code=404, detail="session_completed")
    # Подстраховка на таймер/старт
    updated = False
    if not row.get("timer"):
        row["timer"] = TIMER_SECONDS
        cur.execute("UPDATE sessions SET timer=? WHERE id=?", (TIMER_SECONDS, session_id))
        updated = True
    if not row.get("startedAt"):
        row["startedAt"] = now_iso()
        cur.execute("UPDATE sessions SET startedAt=? WHERE id=?", (row["startedAt"], session_id))
        updated = True
    # Если запрашивается конкретный вопрос — подгружаем его
    if questionId:
        cur.execute("SELECT * FROM questions WHERE id=?", (questionId,))
        q = fetchone_dict(cur)
        if q:
            row["questionId"] = q["id"]
            row["questionTitle"] = q["title"]
            row["description"] = q["body"]
            row["useIDE"] = q.get("useIDE", 1)
    # История выданных вопросов
    used_questions = fetchall_dicts(
        cur.execute("SELECT questionId as id, questionTitle as title FROM session_questions WHERE sessionId=? ORDER BY position", (session_id,))
    )
    if updated:
        conn.commit()
    conn.close()
    return {
        "session": {
            "id": row["id"],
            "direction": row.get("direction"),
            "level": row.get("level"),
            "format": row.get("format"),
            "tasks": row.get("tasks", "").split(",") if row.get("tasks") else [],
            "description": row.get("description"),
            "starterCode": row.get("starterCode"),
            "timer": row.get("timer"),
            "startedAt": row.get("startedAt"),
            "solved": row.get("solved"),
            "total": row.get("total"),
            "questionTitle": row.get("questionTitle"),
            "questionId": row.get("questionId"),
            "useIDE": bool(row.get("useIDE", 1)),
            "usedQuestions": used_questions,
            "status": row.get("status", "active"),
            "is_active": row.get("is_active", 1),
            "is_finished": row.get("is_finished", 0),
        }
    }


@app.post("/api/session/{session_id}/chat")
def chat(session_id: str, payload: ChatPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, session_id, payload.questionId, "user", payload.message, now_iso()),
    )
    reply = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": "Проверьте крайние случаи: пустые входные данные и большие объёмы.",
        "createdAt": now_iso(),
    }
    cur.execute(
        "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
        (reply["id"], session_id, payload.questionId, reply["role"], reply["content"], reply["createdAt"]),
    )
    conn.commit()
    conn.close()
    return {"reply": reply}


@app.get("/api/session/{session_id}/chat")
def get_chat(session_id: str, questionId: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if questionId:
        rows = cur.execute(
            "SELECT * FROM messages WHERE sessionId=? AND questionId=? ORDER BY datetime(createdAt) ASC",
            (session_id, questionId),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT * FROM messages WHERE sessionId=? ORDER BY datetime(createdAt) ASC",
            (session_id,),
        ).fetchall()
    cols = [c[0] for c in cur.description]
    chat = [dict(zip(cols, r)) for r in rows]
    conn.close()
    return {"chat": chat}


@app.post("/api/session/{session_id}/check")
def check(session_id: str):
    result = {
        "passed": False,
        "summary": "Тест №3 падает на кейсе с пустым массивом",
        "cases": [
            {"name": "Возвращает индексы", "passed": True},
            {"name": "Работает с дубликатами", "passed": True},
            {"name": "Пустой массив", "passed": False, "details": "ожидалось []"},
        ],
    }
    return {"result": result}


@app.get("/api/report/{report_id}")
def report(report_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports WHERE id=?", (report_id,))
    row = fetchone_dict(cur)
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return {
        "report": {
            "id": row["id"],
            "score": row.get("score", 0),
            "level": row.get("level"),
            "summary": row.get("summary", ""),
            "timeline": [],
            "solutions": [],
            "analytics": {},
        }
    }


@app.get("/api/admin/overview")
def admin_overview(adminId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    # Подсветка кандидатов с анти-чит событиями
    flag_rows = fetchall_dicts(
        cur.execute(
            "SELECT ownerId, COUNT(*) as cnt FROM interview_events WHERE ownerId IS NOT NULL GROUP BY ownerId"
        )
    )
    flags = {row["ownerId"]: row["cnt"] for row in flag_rows if row.get("ownerId")}
    cur.execute("SELECT * FROM users")
    users = fetchall_dicts(cur)
    conn.close()
    candidates = [
        {
            "id": u["id"],
            "name": u["name"],
            "email": u["email"],
            "level": u["level"],
            "admin": bool(u.get("admin")),
            "role": u.get("role"),
            "hasFlags": bool(flags.get(u["id"])),
            "flagsCount": flags.get(u["id"], 0),
            "lastScore": 70,
            "lastTopic": "System Design",
        }
        for u in users
    ]
    analytics = {
        "hardestTopics": [
            {"name": "Concurrency", "score": 62},
            {"name": "Graph", "score": 58},
            {"name": "API design", "score": 64},
        ],
        "completionRate": 0.78,
        "avgScore": 0.81,
    }
    return {"candidates": candidates, "flagged": [], "analytics": analytics}


@app.get("/api/admin/events")
def admin_events(adminId: str, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    cur.execute(
        """
        SELECT id, sessionId, ownerId, event_type as eventType, payload, risk_level as risk, createdAt
        FROM interview_events
        ORDER BY datetime(createdAt) DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = fetchall_dicts(cur)
    conn.close()
    return {"events": rows}


@app.post("/api/admin/events/clear")
def admin_events_clear(payload: ClearEventsPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    cur.execute("DELETE FROM interview_events")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


@app.post("/api/run-code")
def run_code(payload: RunCodePayload):
    # Пока поддерживаем только python для безопасного исполнения
    lang = (payload.language or "").lower()
    if lang not in ["python", "py"]:
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "executionTimeMs": 0,
            "errorType": "unsupported_language",
        }
    start = datetime.utcnow()
    stdout = ""
    stderr = ""
    error_type = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            f.write(payload.code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, "-u", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        error_type = "timeout"
        stderr = "Время выполнения превышено (5 секунд)."
    except Exception as exc:  # noqa: BLE001
        error_type = "runtime_error"
        stderr = str(exc)
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
    exec_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
    status = "ok" if not error_type and stderr == "" else "error"
    return {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "executionTimeMs": exec_ms,
        "errorType": error_type,
    }


@app.post("/api/support/send")
def support_send(payload: SupportSendPayload):
    if not payload.userId or not payload.message.strip():
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO support_messages (id, userId, role, content, createdAt) VALUES (?, ?, ?, ?, ?)",
        (msg_id, payload.userId, "user", payload.message, now_iso()),
    )
    cur.execute(
        "INSERT OR REPLACE INTO support_dialogs (userId, status) VALUES (?, 'open')",
        (payload.userId,),
    )
    # Простейший автоответ админа, чтобы чат не был пуст
    auto_reply = {
        "id": str(uuid.uuid4()),
        "userId": payload.userId,
        "role": "admin",
        "content": "Спасибо за сообщение! Мы скоро ответим.",
        "createdAt": now_iso(),
    }
    cur.execute(
        "INSERT INTO support_messages (id, userId, role, content, createdAt) VALUES (?, ?, ?, ?, ?)",
        (auto_reply["id"], payload.userId, auto_reply["role"], auto_reply["content"], auto_reply["createdAt"]),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/support/messages")
def support_messages(userId: str):
    if not userId:
        raise HTTPException(status_code=400, detail="invalid_user")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, userId, role, content, createdAt FROM support_messages WHERE userId=? ORDER BY datetime(createdAt) ASC",
        (userId,),
    ).fetchall()
    cols = [c[0] for c in cur.description]
    msgs = [dict(zip(cols, r)) for r in rows]
    conn.close()
    return {"messages": msgs}


@app.get("/api/support/history")
def support_history(userId: str, adminId: str):
    # Проверяем, что это админ
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    rows = cur.execute(
        "SELECT id, userId, role as sender, content as message, createdAt as timestamp FROM support_messages WHERE userId=? ORDER BY datetime(createdAt) ASC",
        (userId,),
    ).fetchall()
    cols = [c[0] for c in cur.description]
    history = [dict(zip(cols, r)) for r in rows]
    conn.close()
    return {"messages": history}


@app.post("/api/support/reply")
def support_reply(payload: SupportReplyPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO support_messages (id, userId, role, content, createdAt) VALUES (?, ?, ?, ?, ?)",
        (msg_id, payload.userId, "admin", payload.message, now_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/support/close")
def support_close(payload: SupportClosePayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    cur.execute("INSERT OR REPLACE INTO support_dialogs (userId, status) VALUES (?, 'closed')", (payload.userId,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/support/inbox")
def support_inbox(adminId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (adminId,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    rows = cur.execute(
        """
        SELECT sm.userId,
               u.name as userName,
               sm.content as lastMessage,
               sm.role as lastSender,
               sm.createdAt
        FROM support_messages sm
        JOIN (
          SELECT userId, MAX(datetime(createdAt)) as maxCreated
          FROM support_messages
          WHERE role='user'
          GROUP BY userId
        ) latest
        ON sm.userId = latest.userId AND datetime(sm.createdAt) = latest.maxCreated
        LEFT JOIN support_dialogs sd ON sd.userId = sm.userId
        LEFT JOIN users u ON u.id = sm.userId
        WHERE (sd.status IS NULL OR sd.status!='closed') AND sm.role='user'
        ORDER BY datetime(sm.createdAt) DESC
        """
    ).fetchall()
    cols = [c[0] for c in cur.description]
    items = [dict(zip(cols, r)) for r in rows]
    # Статус: новое, если последний отправитель user
    for it in items:
        it["status"] = "new" if it.get("lastSender") == "user" else "open"
        if not it.get("userName"):
            it["userName"] = it.get("userId", "Неизвестный")
    conn.close()
    return {"items": items}


@app.get("/api/session/active")
def active_session(ownerId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM sessions WHERE ownerId=? AND status='active' ORDER BY datetime(createdAt) DESC LIMIT 1",
        (ownerId,),
    )
    row = fetchone_dict(cur)
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    # Подстраховка: если нет timer/startedAt — ставим дефолт, чтобы таймер работал
    updated = False
    if not row.get("timer"):
        row["timer"] = TIMER_SECONDS
        cur.execute("UPDATE sessions SET timer=? WHERE id=?", (TIMER_SECONDS, row["id"]))
        updated = True
    if not row.get("startedAt"):
        row["startedAt"] = now_iso()
        cur.execute("UPDATE sessions SET startedAt=? WHERE id=?", (row["startedAt"], row["id"]))
        updated = True
    if updated:
        conn.commit()
    used_questions = fetchall_dicts(
        cur.execute("SELECT questionId as id, questionTitle as title FROM session_questions WHERE sessionId=? ORDER BY position", (row["id"],))
    )
    conn.close()
    return {
        "session": {
            "id": row["id"],
            "direction": row.get("direction"),
            "level": row.get("level"),
            "format": row.get("format"),
            "tasks": row.get("tasks", "").split(",") if row.get("tasks") else [],
            "description": row.get("description"),
            "starterCode": row.get("starterCode"),
            "timer": row.get("timer"),
            "startedAt": row.get("startedAt"),
            "solved": row.get("solved"),
            "total": row.get("total"),
            "questionTitle": row.get("questionTitle"),
            "questionId": row.get("questionId"),
            "useIDE": bool(row.get("useIDE", 1)),
            "usedQuestions": used_questions,
            "status": row.get("status", "active"),
            "is_active": row.get("is_active", 1),
            "is_finished": row.get("is_finished", 0),
        }
    }


@app.post("/api/admin/grant")
def grant_admin(action: AdminAction):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (action.superId,))
    super_user = fetchone_dict(cur)
    if not super_user or super_user.get("role") != "superadmin":
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    # Супер-админа нельзя менять/создавать
    if action.targetUserId == "super-1":
        conn.close()
        raise HTTPException(status_code=400, detail="cannot_modify_superadmin")
    cur.execute("UPDATE users SET admin=1, role='admin' WHERE id=?", (action.targetUserId,))
    conn.commit()
    cur.execute("SELECT * FROM users WHERE id=?", (action.targetUserId,))
    user = fetchone_dict(cur)
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="not_found")
    return {"user": user_safe(user)}


@app.post("/api/admin/revoke")
def revoke_admin(action: AdminAction):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (action.superId,))
    super_user = fetchone_dict(cur)
    if not super_user or super_user.get("role") != "superadmin":
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    if action.targetUserId == "super-1":
        conn.close()
        raise HTTPException(status_code=400, detail="cannot_modify_superadmin")
    cur.execute("UPDATE users SET admin=0, role='user' WHERE id=?", (action.targetUserId,))
    conn.commit()
    cur.execute("SELECT * FROM users WHERE id=?", (action.targetUserId,))
    user = fetchone_dict(cur)
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="not_found")
    return {"user": user_safe(user)}


@app.post("/api/change-password")
def change_password(payload: ChangePasswordPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.userId,))
    user = fetchone_dict(cur)
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    if user.get("password") != sha1(payload.oldPassword):
        conn.close()
        raise HTTPException(status_code=400, detail="wrong_old_password")
    cur.execute("UPDATE users SET password=? WHERE id=?", (sha1(payload.newPassword), payload.userId))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/profile/language")
def change_language(payload: ChangeLanguagePayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.userId,))
    user = fetchone_dict(cur)
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    cur.execute("UPDATE users SET lang=? WHERE id=?", (payload.lang, payload.userId))
    conn.commit()
    cur.execute("SELECT * FROM users WHERE id=?", (payload.userId,))
    updated = fetchone_dict(cur)
    conn.close()
    return {"user": {**user_safe(updated), "lang": updated.get("lang", "ru")}}

def get_used_questions(cur, session_id: str):
    return fetchall_dicts(
        cur.execute(
            "SELECT questionId as id, questionTitle as title FROM session_questions WHERE sessionId=? ORDER BY position",
            (session_id,),
        )
    )


@app.post("/api/anticheat/event")
def add_anticheat_event(payload: AntiCheatEvent):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, ownerId, event_type, payload, risk_level, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            payload.sessionId,
            payload.ownerId,
            payload.eventType,
            payload.payload,
            payload.risk or "medium",
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/interview/finish")
def finish_interview(payload: FinishInterviewPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (payload.sessionId,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        return {"status": "missing"}
    cur.execute("UPDATE sessions SET status='completed', is_active=0, is_finished=1 WHERE id=?", (payload.sessionId,))
    finished_at = now_iso()
    cur.execute(
        "INSERT OR REPLACE INTO interview_results (id, sessionId, ownerId, status, finishedAt) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), payload.sessionId, payload.ownerId, "completed", finished_at),
    )
    # Чистим историю поддержки пользователя при завершении собеседования
    if session.get("ownerId"):
        cur.execute("DELETE FROM support_messages WHERE userId=?", (session.get("ownerId"),))
        cur.execute("DELETE FROM support_dialogs WHERE userId=?", (session.get("ownerId"),))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/session/finish")
def finish_interview_alias(payload: FinishInterviewPayload):
    return finish_interview(payload)


@app.post("/api/answer")
def upsert_answer(payload: AnswerPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Проверяем, что сессия принадлежит пользователю
    cur.execute("SELECT * FROM sessions WHERE id=? AND ownerId=?", (payload.sessionId, payload.ownerId))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    cur.execute(
        """
        INSERT INTO answers (id, sessionId, questionId, ownerId, content, updatedAt)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(sessionId, questionId) DO UPDATE SET content=excluded.content, updatedAt=excluded.updatedAt
        """,
        (str(uuid.uuid4()), payload.sessionId, payload.questionId, payload.ownerId, payload.content, now_iso()),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/answer")
def get_answer(sessionId: str, questionId: str, ownerId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM answers WHERE sessionId=? AND questionId=? AND ownerId=?",
        (sessionId, questionId, ownerId),
    )
    row = fetchone_dict(cur)
    conn.close()
    if not row:
        return {"content": ""}
    return {"content": row.get("content", "")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
