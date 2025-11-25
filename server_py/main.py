import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import json

# Делаем путь к server_py доступным для импортов при запуске как скрипта
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
  sys.path.append(str(CURRENT_DIR))

from db import ROOT, DATA_DIR, DB_PATH, sha1, fetchone_dict, fetchall_dicts, ensure_schema, seed_questions, connect as db_connect
from api_next_question import router as next_router
import random
from llm_client import get_llm_client, LLMError
from code_runner import run_code, CodeRunError
from metrics_service import recalculate_session_metrics
from embedding_service import index_answer, index_question, find_similar_answers, find_similar_questions

MAX_QUESTIONS = 3  # легко менять при необходимости
TIMER_SECONDS = 45 * 60


def get_conn():
    """Создаём соединение с SQLite c таймаутом и WAL, чтобы уменьшить конфликты блокировок."""
    conn = db_connect()
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


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

class StartLLMPayload(BaseModel):
    direction: str
    language: str
    preferred_level: Optional[str] = None
    userId: Optional[str] = None

class CodeRunPayload(BaseModel):
    session_question_id: Optional[int] = None
    sessionQuestionId: Optional[int] = None
    sessionId: Optional[str] = None
    ownerId: Optional[str] = None
    code: str
    language: Optional[str] = None

class CodeSubmitPayload(CodeRunPayload):
    pass

class ChatPayload(BaseModel):
    sessionId: str
    questionId: Optional[str] = None
    message: str

class EventPayload(BaseModel):
    session_id: str
    event_type: str
    payload: Optional[dict] = None
    ownerId: Optional[str] = None

class StreamChatPayload(BaseModel):
    sessionId: str
    questionId: Optional[str] = None
    message: str

class AntiCheatEvent(BaseModel):
    sessionId: str
    ownerId: str | None = None
    eventType: str
    payload: str | None = None
    risk: str | None = "medium"


# --------- Генерация вопроса из шаблонов (promt.md)
class TemplateItem(BaseModel):
    id: str
    direction: str
    level: Optional[str] = None
    language: Optional[str] = None
    title: str
    statement: Optional[str] = None
    hints: Optional[list[str]] = None
    constraints: Optional[str] = None
    io_format: Optional[dict] = None
    examples: Optional[list[dict]] = None


class TemplateRequest(BaseModel):
    candidate_direction: str
    candidate_level: Optional[str] = None
    candidate_language: Optional[str] = None
    templates: list[TemplateItem]

# --------- Инициализация БД

def init_db():
    ensure_schema()
    conn = get_conn()
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
# Подгружаем LLM клиент при необходимости
llm_client = None
def get_client():
    global llm_client
    if llm_client is None:
        llm_client = get_llm_client()
    return llm_client


def parse_tests(raw_json: str):
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def require_admin(cur, admin_id: str):
    cur.execute("SELECT * FROM users WHERE id=?", (admin_id,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        raise HTTPException(status_code=403, detail="forbidden")
    return admin
init_db()

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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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


@app.get("/api/interview/metrics/{session_id}")
def get_metrics(session_id: str):
    metrics = recalculate_session_metrics(session_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="not_found")
    return {"metrics": metrics}


@app.get("/api/interview/state/{session_id}")
def get_interview_state(session_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    # Текущий вопрос
    cur.execute(
        """
        SELECT q.*, sq.id as session_question_id, sq.status as sq_status, sq.order_index
        FROM session_questions sq
        JOIN questions q ON q.id = sq.questionId
        WHERE sq.sessionId=?
        ORDER BY sq.order_index ASC
        """,
        (session_id,),
    )
    sq_rows = fetchall_dicts(cur)
    current_question = None
    if session.get("questionId"):
        current_question = next((r for r in sq_rows if r.get("id") == session.get("questionId")), None)
    if not current_question and sq_rows:
        current_question = sq_rows[0]
    # Ответы
    answers = fetchall_dicts(cur.execute("SELECT * FROM answers WHERE sessionId=?", (session_id,)))
    # Метрики
    metrics = recalculate_session_metrics(session_id)
    # История чата
    messages = fetchall_dicts(
        cur.execute(
            "SELECT id, role, content, createdAt, questionId FROM messages WHERE sessionId=? ORDER BY datetime(createdAt) ASC",
            (session_id,),
        )
    )
    def parse_tests(raw):
        try:
            return json.loads(raw) if raw else []
        except Exception:
            return []
    # Оставшееся время
    total_sec = int(session.get("timer") or TIMER_SECONDS)
    started_raw = session.get("startedAt") or session.get("started_at")
    started_ts = int(datetime.now(timezone.utc).timestamp())
    try:
        if started_raw:
            parsed = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
            started_ts = int(parsed.timestamp())
    except Exception:
        pass
    elapsed = max(0, int(datetime.now(timezone.utc).timestamp()) - started_ts)
    remaining = max(0, total_sec - elapsed)
    conn.close()
    return {
        "session": {**session, "remaining_seconds": remaining},
        "questions": sq_rows,
        "current_question": current_question,
        "visible_tests": parse_tests((current_question or {}).get("visible_tests_json")),
        "hidden_tests": parse_tests((current_question or {}).get("hidden_tests_json")),
        "answers": answers,
        "metrics": metrics,
        "messages": messages,
    }


@app.post("/api/interview/event")
def interview_event(payload: EventPayload):
    session_id = payload.session_id
    if not session_id or not payload.event_type:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")

    weights = {
        "devtools": 3,
        "ide_paste": 2,
        "paste": 2,
        "tab_switch": 1,
        "blur": 1,
        "window_minimize": 1,
    }
    weight = weights.get(payload.event_type, 0)
    now = now_iso()
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            session_id,
            payload.ownerId or session.get("ownerId"),
            payload.event_type,
            json.dumps(payload.payload or {}, ensure_ascii=False),
            now,
            now,
        ),
    )
    if weight:
        cur.execute(
            "UPDATE sessions SET cheat_score = COALESCE(cheat_score, 0) + ? WHERE id=?",
            (weight, session_id),
        )
    conn.commit()
    conn.close()
    metrics = recalculate_session_metrics(session_id)
    return {"status": "ok", "cheat_score": metrics.get("cheat_score") if metrics else None}


@app.get("/api/session/{session_id}/chat")
def get_chat(session_id: str, questionId: Optional[str] = None):
    conn = get_conn()
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
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports WHERE id=? OR sessionId=?", (report_id, report_id))
    row = fetchone_dict(cur)
    # Загружаем метрики, ответы, события
    cur.execute("SELECT * FROM answers WHERE sessionId=?", (report_id,))
    answers = fetchall_dicts(cur)
    cur.execute("SELECT * FROM interview_events WHERE sessionId=? OR session_id=?", (report_id, report_id))
    events = fetchall_dicts(cur)
    metrics_row = fetchone_dict(cur.execute("SELECT * FROM interview_results WHERE sessionId=?", (report_id,)))
    conn.close()
    if not row and not metrics_row:
        raise HTTPException(status_code=404, detail="not_found")
    metrics_json = {}
    try:
        metrics_json = json.loads(metrics_row.get("metrics_json") or "{}") if metrics_row else {}
    except Exception:
        metrics_json = {}
    return {
        "report": {
            "id": (row or metrics_row).get("id", report_id),
            "score": (row or metrics_row or {}).get("score", 0),
            "level": (row or metrics_row or {}).get("level"),
            "summary": (row or metrics_row or {}).get("summary", ""),
            "timeline": events,
            "solutions": answers,
            "analytics": metrics_json,
        }
    }


@app.get("/api/admin/overview")
def admin_overview(adminId: str):
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
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
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
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
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, payload.adminId)
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
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
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, payload.adminId)
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
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, payload.adminId)
    cur.execute("INSERT OR REPLACE INTO support_dialogs (userId, status) VALUES (?, 'closed')", (payload.userId,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/code/run")
def code_run(payload: CodeRunPayload):
    sq_id = payload.session_question_id or payload.sessionQuestionId
    if not sq_id:
        raise HTTPException(status_code=400, detail="session_question_id required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sq.*, q.visible_tests_json, q.language
        FROM session_questions sq
        JOIN questions q ON q.id = sq.questionId
        WHERE sq.id=?
        """,
        (sq_id,),
    )
    row = fetchone_dict(cur)
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    tests = parse_tests(row.get("visible_tests_json") or "[]")
    lang = (payload.language or row.get("language") or "python").lower()
    try:
        results, agg_out, agg_err = run_code(lang, payload.code, tests)
    except CodeRunError as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))
    now = now_iso()
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            row.get("sessionId"),
            row.get("sessionId"),
            payload.ownerId,
            "code_run",
            None,
            json.dumps({"language": lang, "tests": results}, ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "stdout": agg_out,
        "stderr": agg_err,
        "tests": results,
    }


@app.post("/api/code/submit")
def code_submit(payload: CodeSubmitPayload):
    sq_id = payload.session_question_id or payload.sessionQuestionId
    if not sq_id:
        raise HTTPException(status_code=400, detail="session_question_id required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sq.*, q.visible_tests_json, q.hidden_tests_json, q.language, q.statement, q.title, s.started_at, s.startedAt, s.ownerId, s.id as session_id
        FROM session_questions sq
        JOIN questions q ON q.id = sq.questionId
        JOIN sessions s ON s.id = sq.sessionId
        WHERE sq.id=?
        """,
        (sq_id,),
    )
    row = fetchone_dict(cur)
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")

    lang = (payload.language or row.get("language") or "python").lower()
    visible = parse_tests(row.get("visible_tests_json") or "[]")
    hidden = parse_tests(row.get("hidden_tests_json") or "[]")

    try:
        res_vis, agg_out_vis, agg_err_vis = run_code(lang, payload.code, visible)
        res_hid, agg_out_hid, agg_err_hid = run_code(lang, payload.code, hidden)
    except CodeRunError as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=str(exc))

    passed_visible = all(r.get("status") == "passed" for r in res_vis) if res_vis else False
    passed_hidden = all(r.get("status") == "passed" for r in res_hid) if res_hid else False

    cur.execute(
        "SELECT MAX(attempt_number) FROM answers WHERE sessionId=? AND questionId=?",
        (row.get("sessionId"), row.get("questionId")),
    )
    max_attempt = cur.fetchone()[0] or 0
    attempt_number = max_attempt + 1

    start_ts_raw = row.get("started_at") or row.get("startedAt")
    try:
        start_ts = datetime.fromisoformat(start_ts_raw.replace("Z", "+00:00")) if start_ts_raw else None
    except Exception:
        start_ts = None
    duration_ms = None
    if start_ts:
        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)

    now = now_iso()
    cur.execute(
        """
        INSERT INTO answers (id, sessionId, questionId, ownerId, content, code, language, passed_visible, passed_hidden, attempt_number, duration_ms, updatedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sessionId, questionId) DO UPDATE SET
            content=excluded.content,
            code=excluded.code,
            language=excluded.language,
            passed_visible=excluded.passed_visible,
            passed_hidden=excluded.passed_hidden,
            attempt_number=excluded.attempt_number,
            duration_ms=excluded.duration_ms,
            updatedAt=excluded.updatedAt
        """,
        (
            str(uuid.uuid4()),
            row.get("sessionId"),
            row.get("questionId"),
            payload.ownerId or row.get("ownerId"),
            payload.code,
            payload.code,
            lang,
            int(passed_visible),
            int(passed_hidden),
            attempt_number,
            duration_ms,
            now,
        ),
    )

    # Готовим данные для LLM-оценки
    tests_for_llm = []
    for r in res_vis:
        tests_for_llm.append(
            {
                "kind": "visible",
                "status": r.get("status"),
                "input": visible[r["test_id"] - 1].get("input") if r["test_id"] <= len(visible) else "",
                "expected": visible[r["test_id"] - 1].get("output") if r["test_id"] <= len(visible) else "",
                "actual": r.get("actual_output", ""),
                "stderr": r.get("stderr", ""),
            }
        )
    for r in res_hid:
        tests_for_llm.append(
            {
                "kind": "hidden",
                "status": r.get("status"),
                "input": "",
                "expected": "",
                "actual": r.get("actual_output", ""),
                "stderr": r.get("stderr", ""),
            }
        )
    llm_payload = {
        "task_statement": row.get("statement") or row.get("title"),
        "language": lang,
        "code": payload.code,
        "tests": tests_for_llm,
    }
    metrics = None
    try:
        client = get_client()
        raw = client.chat(
            model="qwen3-32b-awq",
            messages=[
                {
                    "role": "system",
                    "content": "/no_think Ты опытный техлид. Оцени решение кандидата по корректности, асимптотической сложности, читаемости и обработке краевых случаев. Верни строго JSON.",
                },
                {"role": "user", "content": json.dumps(llm_payload, ensure_ascii=False)},
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=600,
            stream=False,
        )
        metrics = json.loads(raw)
    except Exception:
        metrics = None

    if metrics:
        cur.execute(
            "UPDATE answers SET metrics_json=? WHERE sessionId=? AND questionId=?",
            (json.dumps(metrics, ensure_ascii=False), row.get("sessionId"), row.get("questionId")),
        )
        # Сохраняем в interview_results для отчёта
        cur.execute(
            """
            INSERT OR REPLACE INTO interview_results (id, sessionId, ownerId, status, finishedAt, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("sessionId"),
                row.get("sessionId"),
                row.get("ownerId") or payload.ownerId,
                "active",
                now,
                json.dumps(metrics, ensure_ascii=False),
            ),
        )

    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            row.get("sessionId"),
            row.get("sessionId"),
            payload.ownerId or row.get("ownerId"),
            "code_submit",
            json.dumps({"passed_visible": passed_visible, "passed_hidden": passed_hidden, "attempt": attempt_number}, ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    # Индексация ответа
    try:
        cur_ans = get_conn().cursor()
        cur_ans.execute("SELECT id FROM answers WHERE sessionId=? AND questionId=?", (row.get("sessionId"), row.get("questionId")))
        ans_row = cur_ans.fetchone()
        if ans_row:
            index_answer(ans_row[0])
    except Exception:
        pass
    metrics_summary = recalculate_session_metrics(row.get("sessionId"))
    return {
        "passed_visible": passed_visible,
        "passed_hidden": passed_hidden,
        "visible_tests": res_vis,
        "hidden_tests": res_hid,
        "stdout": agg_out_vis + agg_out_hid,
        "stderr": agg_err_vis + agg_err_hid,
        "metrics": metrics,
        "session_metrics": metrics_summary,
    }


@app.post("/api/interview/chat")
def interview_chat(payload: ChatPayload):
    session_id = payload.sessionId
    user_msg = payload.message.strip()
    if not session_id or not user_msg:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    # Текущий вопрос
    question_id = payload.questionId or session.get("questionId")
    cur.execute("SELECT * FROM questions WHERE id=?", (question_id,))
    question = fetchone_dict(cur)
    # История сообщений
    history = fetchall_dicts(
        cur.execute(
            "SELECT role, content, createdAt FROM messages WHERE sessionId=? ORDER BY datetime(createdAt) ASC",
            (session_id,),
        )
    )
    # Последний ответ/код
    cur.execute(
        "SELECT * FROM answers WHERE sessionId=? ORDER BY datetime(updatedAt) DESC LIMIT 1",
        (session_id,),
    )
    last_answer = fetchone_dict(cur)
    conn.close()

    context_snippets = []
    if question:
        context_snippets.append(f"Задача: {question.get('title')}. Условие: {question.get('statement') or question.get('body')}")
    if last_answer:
        context_snippets.append(f"Последний код кандидата:\n{last_answer.get('code') or last_answer.get('content')}")
        if last_answer.get("passed_hidden") == 0:
            context_snippets.append("Скрытые тесты не пройдены.")
    system_prompt = (
        "/no_think Ты технический интервьюер на живом собеседовании. "
        "Помогай кандидату понять задачу и исправить ошибки, но не пиши полностью решение. "
        "Фокусируйся на наводящих вопросах и объяснениях, учитывай, что кандидат видит условие задачи и тесты."
    )
    msgs = [{"role": "system", "content": system_prompt}]
    if context_snippets:
        msgs.append({"role": "assistant", "content": "\n".join(context_snippets)})
    for m in history[-15:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_msg})

    # Выбор модели
    model = "qwen3-32b-awq"
    if last_answer and last_answer.get("code"):
        model = "qwen3-coder-30b-a3b-instruct-fp8"

    try:
        client = get_client()
        reply = client.chat(
            model=model,
            messages=msgs,
            temperature=0.5,
            top_p=0.9,
            max_tokens=600,
            stream=False,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM chat failed: {exc}")

    now = now_iso()
    conn = get_conn()
    cur = conn.cursor()
    # Сохраняем реплики
    user_msg_id = str(uuid.uuid4())
    bot_msg_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
        (user_msg_id, session_id, question_id, "user", user_msg, now),
    )
    cur.execute(
        "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
        (bot_msg_id, session_id, question_id, "assistant", reply, now),
    )
    # Логируем события
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            session_id,
            session.get("ownerId"),
            "chat_user",
            json.dumps({"message": user_msg}, ensure_ascii=False),
            now,
            now,
        ),
    )
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            session_id,
            session.get("ownerId"),
            "chat_assistant",
            json.dumps({"message": reply}, ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return {"reply": reply}


def _sse_gen(text_stream):
    try:
        for chunk in text_stream:
            if chunk:
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: {\"delta\": \"[END]\"}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")


@app.post("/api/interview/chat/stream")
def interview_chat_stream(payload: StreamChatPayload):
    session_id = payload.sessionId
    user_msg = payload.message.strip()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    question_id = payload.questionId or session.get("questionId")
    cur.execute("SELECT * FROM questions WHERE id=?", (question_id,))
    question = fetchone_dict(cur)
    history = fetchall_dicts(
        cur.execute(
            "SELECT role, content, createdAt FROM messages WHERE sessionId=? ORDER BY datetime(createdAt) ASC",
            (session_id,),
        )
    )
    cur.execute(
        "INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            session_id,
            session_id,
            session.get("ownerId"),
            "llm_stream_start",
            json.dumps({"type": "chat"}, ensure_ascii=False),
            now_iso(),
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()

    context_snippets = []
    if question:
        context_snippets.append(f"Задача: {question.get('title')}. Условие: {question.get('statement') or question.get('body')}")
    system_prompt = (
        "/no_think Ты технический интервьюер на живом собеседовании. "
        "Помогай кандидату понять задачу и исправить ошибки, но не пиши полностью решение. "
        "Фокусируйся на наводящих вопросах и объяснениях, учитывай, что кандидат видит условие задачи и тесты."
    )
    msgs = [{"role": "system", "content": system_prompt}]
    if context_snippets:
        msgs.append({"role": "assistant", "content": "\n".join(context_snippets)})
    for m in history[-15:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_msg})

    client = get_client()
    stream = client.chat(
        model="qwen3-32b-awq",
        messages=msgs,
        temperature=0.5,
        top_p=0.9,
        max_tokens=600,
        stream=True,
    )

    def streamer():
        full_text = ""
        for chunk in stream:
            full_text += chunk
            yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: {\"delta\": \"[END]\"}\n\n"
        # сохраняем после потока
        now = now_iso()
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, question_id, "user", user_msg, now),
        )
        cur.execute(
            "INSERT INTO messages (id, sessionId, questionId, role, content, createdAt) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, question_id, "assistant", full_text, now),
        )
        cur.execute(
            "INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                session_id,
                session.get("ownerId"),
                "llm_stream_end",
                json.dumps({"type": "chat"}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/api/interview/next/stream")
def interview_next_stream(payload: dict):
    session_id = payload.get("sessionId")
    ownerId = payload.get("ownerId")
    if not session_id or not ownerId:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    issued_rows = fetchall_dicts(cur.execute("SELECT questionId as id, status FROM session_questions WHERE sessionId=?", (session_id,)))
    current_level = (session.get("current_level") or session.get("level") or "junior").lower()
    level_order = ["junior", "middle", "senior"]
    def promote(level: str) -> str:
        idx = level_order.index(level) if level in level_order else 0
        return level_order[min(idx + 1, len(level_order) - 1)]
    def demote(level: str) -> str:
        idx = level_order.index(level) if level in level_order else 0
        return level_order[max(idx - 1, 0)]
    cur.execute("SELECT * FROM answers WHERE sessionId=? ORDER BY datetime(updatedAt) DESC LIMIT 1", (session_id,))
    last_answer = fetchone_dict(cur)
    target_level = current_level
    if last_answer:
        passed_hidden = bool(last_answer.get("passed_hidden"))
        attempts = last_answer.get("attempt_number") or 0
        ev_rows = fetchall_dicts(cur.execute("SELECT event_type FROM interview_events WHERE sessionId=? OR session_id=?", (session_id, session_id)))
        runs = sum(1 for e in ev_rows if e.get("event_type") in ["code_run", "code_submit"])
        hints = sum(1 for e in ev_rows if e.get("event_type") in ["chat_user", "chat_assistant"])
        if passed_hidden and attempts <= 2:
            target_level = promote(current_level)
        elif (not passed_hidden) and (runs > 5 or hints > 3):
            target_level = demote(current_level)
        else:
            target_level = current_level
    history_rows = fetchall_dicts(
        cur.execute(
            """
            SELECT a.*, q.level as q_level
            FROM answers a
            LEFT JOIN questions q ON q.id = a.questionId
            WHERE a.sessionId=?
            ORDER BY datetime(a.updatedAt) ASC
            """,
            (session_id,),
        )
    )
    prev_tasks = []
    for row in history_rows:
        res = "passed" if row.get("passed_hidden") else "failed"
        prev_tasks.append(
            {
                "level": row.get("q_level") or current_level,
                "result": res,
                "attempts": row.get("attempt_number") or 0,
                "main_error": "",
            }
        )
    conn.close()

    system_prompt = (
        "/no_think Ты ИИ-интервьюер. Подбирай следующую задачу так, чтобы за ограниченное время интервью объективно оценить кандидата. "
        "Сложность задачи должна соответствовать полю target_level. Отвечай строго JSON как в предыдущем шаге."
    )
    user_prompt = json.dumps(
        {
            "direction": session.get("direction") or "general",
            "language": payload.get("language") or session.get("language") or "python",
            "target_level": target_level,
            "prev_tasks": prev_tasks,
        },
        ensure_ascii=False,
    )
    client = get_client()
    stream = client.chat(
        model="qwen3-coder-30b-a3b-instruct-fp8",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        top_p=0.9,
        max_tokens=900,
        stream=True,
    )
    def streamer():
        full_text = ""
        for chunk in stream:
            full_text += chunk
            yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: {\"delta\": \"[END]\"}\n\n"
        now = now_iso()
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                session_id,
                ownerId,
                "llm_stream_end",
                json.dumps({"type": "next", "raw": full_text}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.get("/api/support/inbox")
def support_inbox(adminId: str):
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
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


@app.get("/api/admin/sessions")
def admin_sessions(adminId: str, level: Optional[str] = None, status: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
    query = "SELECT s.*, u.name as candidate_name, u.email as candidate_email FROM sessions s LEFT JOIN users u ON u.id = s.ownerId WHERE 1=1"
    params = []
    if level:
        query += " AND (s.level=? OR s.current_level=?)"
        params.extend([level, level])
    if status:
        query += " AND s.status=?"
        params.append(status)
    if date_from:
        query += " AND datetime(s.createdAt) >= datetime(?)"
        params.append(date_from)
    if date_to:
        query += " AND datetime(s.createdAt) <= datetime(?)"
        params.append(date_to)
    query += " ORDER BY datetime(s.createdAt) DESC"
    rows = fetchall_dicts(cur.execute(query, params))
    sessions = []
    for r in rows:
        sessions.append(
            {
                "id": r["id"],
                "candidate": r.get("candidate_name") or r.get("candidate_email") or r.get("ownerId"),
                "createdAt": r.get("createdAt"),
                "direction": r.get("direction"),
                "level": r.get("level") or r.get("current_level"),
                "cheat_score": r.get("cheat_score", 0),
                "status": r.get("status"),
                "progress_percent": r.get("progress_percent"),
            }
        )
    conn.close()
    return {"sessions": sessions}


@app.get("/api/admin/session/{session_id}")
def admin_session_detail(session_id: str, adminId: str):
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    questions = fetchall_dicts(
        cur.execute(
            "SELECT sq.*, q.* FROM session_questions sq JOIN questions q ON q.id = sq.questionId WHERE sq.sessionId=? ORDER BY sq.order_index",
            (session_id,),
        )
    )
    answers = fetchall_dicts(cur.execute("SELECT * FROM answers WHERE sessionId=?", (session_id,)))
    events = fetchall_dicts(cur.execute("SELECT * FROM interview_events WHERE sessionId=? OR session_id=?", (session_id, session_id)))
    messages = fetchall_dicts(cur.execute("SELECT * FROM messages WHERE sessionId=?", (session_id,)))
    metrics = recalculate_session_metrics(session_id)
    conn.close()
    return {
        "session": session,
        "questions": questions,
        "answers": answers,
        "events": events,
        "messages": messages,
        "metrics": metrics,
    }


@app.get("/api/admin/config")
def admin_config_get(adminId: str):
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
    rows = fetchall_dicts(cur.execute("SELECT key, value FROM admin_config"))
    cfg = {}
    for r in rows:
        cfg[r["key"]] = r.get("value")
    conn.close()
    return {"config": cfg}


@app.post("/api/admin/config")
def admin_config_set(payload: dict):
    adminId = payload.get("adminId")
    config = payload.get("config") or {}
    if not adminId:
        raise HTTPException(status_code=400, detail="adminId_required")
    conn = get_conn()
    cur = conn.cursor()
    require_admin(cur, adminId)
    for k, v in config.items():
        cur.execute(
            "INSERT OR REPLACE INTO admin_config (key, value) VALUES (?, ?)",
            (k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)),
        )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/session/active")
def active_session(ownerId: str):
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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
    conn = get_conn()
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


@app.post("/api/interview/template-question")
def generate_from_templates(payload: TemplateRequest):
    """
    Генератор вопроса из переданных шаблонов (promt.md).
    Выбирает случайный подходящий шаблон по direction/level/language и формирует JSON вопроса.
    """
    direction = payload.candidate_direction
    level = (payload.candidate_level or "").lower() or None
    language = (payload.candidate_language or "").lower() or None

    def matches(t: TemplateItem) -> bool:
        if t.direction != direction:
            return False
        if level and t.level and t.level.lower() != level:
            return False
        if language and t.language and t.language.lower() != language:
            return False
        return True

    candidates = [t for t in payload.templates if matches(t)]
    import random

    if not candidates:
        question_text = "Не найден шаблон для выбранного направления. Уточните шаблон или добавьте задачи для этого направления."
        return {"type": "clarification", "question": question_text}

    tmpl = random.choice(candidates)
    io_format = tmpl.io_format or {}
    examples = tmpl.examples or []
    statement_parts = []
    if tmpl.statement:
        statement_parts.append(tmpl.statement)
    if tmpl.constraints:
        statement_parts.append(f"Ограничения: {tmpl.constraints}")
    full_statement = "\n".join(statement_parts) if statement_parts else tmpl.title

    # Гарантируем минимум один пример
    if not examples:
        examples = [{"input": "1\n2\n3", "output": "пример вывода"}]

    return {
        "type": "question",
        "template_id": tmpl.id,
        "direction": tmpl.direction,
        "level": tmpl.level or level or "unspecified",
        "language": tmpl.language or language or "unspecified",
        "title": tmpl.title,
        "statement": full_statement,
        "io_format": {
          "input_description": io_format.get("input_description", "См. условие задачи"),
          "output_description": io_format.get("output_description", "См. условие задачи"),
        },
        "examples": [
          {"input": ex.get("input", ""), "output": ex.get("output", "")} for ex in examples
        ],
    }


@app.post("/api/anticheat/event")
def add_anticheat_event(payload: AntiCheatEvent):
    conn = get_conn()
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
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (payload.sessionId,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        return {"status": "missing"}
    if session.get("status") != "active":
        conn.close()
        return {"status": "already_finished"}

    now = now_iso()
    # логируем запрос на завершение
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            payload.sessionId,
            payload.sessionId,
            payload.ownerId or session.get("ownerId"),
            "interview_finish_request",
            json.dumps({"source": "manual_button"}),
            now,
            now,
        ),
    )
    # Помечаем незавершённые вопросы
    cur.execute(
        "UPDATE session_questions SET status='incomplete' WHERE sessionId=? AND (status IS NULL OR status='current')",
        (payload.sessionId,),
    )
    # Обновляем статус сессии
    cur.execute(
        "UPDATE sessions SET status='finished', is_active=0, is_finished=1, finished_at=? WHERE id=?",
        (now, payload.sessionId),
    )
    # Пересчитываем метрики
    metrics = recalculate_session_metrics(payload.sessionId)
    # Сохраняем результат
    cur.execute(
        """
        INSERT OR REPLACE INTO interview_results (id, sessionId, ownerId, status, finishedAt, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.sessionId,
            payload.sessionId,
            payload.ownerId or session.get("ownerId"),
            "finished",
            now,
            json.dumps(metrics or {}, ensure_ascii=False),
        ),
    )
    # Создаём/обновляем отчёт (пока простой pending)
    cur.execute(
        """
        INSERT OR REPLACE INTO reports (id, sessionId, ownerId, score, level, summary, timeline, solutions, analytics, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.sessionId,
            payload.sessionId,
            payload.ownerId or session.get("ownerId") or "unknown",
            None,
            session.get("level") or session.get("current_level"),
            "Формируем отчёт...",
            "[]",
            "[]",
            json.dumps(metrics or {}, ensure_ascii=False),
            now,
        ),
    )
    # Чистим историю поддержки пользователя при завершении собеседования
    if session.get("ownerId"):
        cur.execute("DELETE FROM support_messages WHERE userId=?", (session.get("ownerId"),))
        cur.execute("DELETE FROM support_dialogs WHERE userId=?", (session.get("ownerId"),))
    conn.commit()
    conn.close()
    return {"status": "ok", "metrics": metrics}


@app.post("/api/session/finish")
def finish_interview_alias(payload: FinishInterviewPayload):
    return finish_interview(payload)


@app.post("/api/interview/start")
def interview_start(payload: StartLLMPayload):
    """
    Старт сессии: берём первую задачу из БД/шаблонов (без LLM), создаём запись sessions и выдаём её.
    """
    owner_id = payload.userId or "anonymous"
    direction = payload.direction
    language = payload.language
    level = (payload.preferred_level or "junior").lower()
    if level not in ["junior", "middle", "senior"]:
        level = "junior"

    conn = get_conn()
    cur = conn.cursor()
    # Берём случайную задачу по направлению/уровню, если нет — любую
    cur.execute(
        """
        SELECT * FROM questions
        WHERE (level IS NULL OR level=?)
          AND (direction IS NULL OR direction=? OR tags LIKE ?)
        ORDER BY RANDOM() LIMIT 1
        """,
        (level, direction, f"%{direction}%"),
    )
    row = fetchone_dict(cur)
    if not row:
        cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
        row = fetchone_dict(cur)
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")

    def _norm_tests(tests):
        arr = []
        for t in tests:
            if isinstance(t, dict) and "input" in t and "output" in t:
                arr.append({"input": str(t["input"]), "output": str(t["output"])})
        return arr

    visible_tests = _norm_tests(parse_tests(row.get("visible_tests_json") or "[]"))
    hidden_tests = _norm_tests(parse_tests(row.get("hidden_tests_json") or "[]"))
    if not visible_tests:
        conn.close()
        raise HTTPException(status_code=422, detail="Видимые тесты пустые")

    data = {
        "title": row.get("title"),
        "statement": row.get("statement") or row.get("body") or row.get("title"),
        "language": row.get("language") or language,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "canonical_solution": row.get("canonical_solution") or row.get("answer") or "",
        "id": row.get("id") or str(uuid.uuid4()),
        "source": "db",
    }
    question_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    now = now_iso()
    # Сохраняем вопрос (если новый) — только когда его не было id
    if row.get("id"):
        question_id = row.get("id")
    else:
        cur.execute(
            """
            INSERT INTO questions (id, title, body, statement, answer, difficulty, level, language, tags,
                                   visible_tests_json, hidden_tests_json, canonical_solution, source, useIDE, createdAt, updatedAt, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                data["title"],
                data.get("statement") or data["title"],
                data.get("statement") or data["title"],
                data.get("canonical_solution", ""),
                level,
                level,
                language,
                direction,
                json.dumps(visible_tests, ensure_ascii=False),
                json.dumps(hidden_tests, ensure_ascii=False),
                data.get("canonical_solution", ""),
                data.get("source", "db"),
                1,
                now,
                now,
                direction,
            ),
        )
    # Сессия
    cur.execute(
        """
        INSERT INTO sessions (id, ownerId, candidate_id, questionId, questionTitle, useIDE, direction, level, current_level,
                              current_question_index, status, started_at, description, starterCode, timer, solved, total, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            owner_id,
            owner_id,
            question_id,
            data["title"],
            1,
            direction,
            level,
            level,
            0,
            "active",
            now,
            data.get("statement") or data["title"],
            "",
            45 * 60,
            0,
            1,
            now,
        ),
    )
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, session_id, questionId, questionTitle, position, order_index, status, llm_raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            session_id,
            question_id,
            data["title"],
            1,
            0,
            "current",
            raw,
        ),
    )
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, session_id, ownerId, event_type, payload, payload_json, ts, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            session_id,
            owner_id,
            "task_issued",
            None,
            json.dumps(data, ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    # Индексация эмбеддинга вопроса
    index_question(question_id)

    return {
        "session_id": session_id,
        "question_id": question_id,
        "title": data["title"],
        "statement": data.get("statement") or data["title"],
        "visible_tests": visible_tests,
    }


@app.post("/api/answer")
def upsert_answer(payload: AnswerPayload):
    conn = get_conn()
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
    conn = get_conn()
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
