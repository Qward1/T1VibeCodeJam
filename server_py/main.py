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
import json
import logging

# Делаем путь к server_py доступным для импортов при запуске как скрипта
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
  sys.path.append(str(CURRENT_DIR))

from db import ROOT, DATA_DIR, DB_PATH, sha1, fetchone_dict, fetchall_dicts, ensure_schema, seed_questions
from api_next_question import router as next_router
import random
from llm_theory import (
    classify_theory_answer,
    generate_followup_question,
    evaluate_theory_answer,
    evaluate_followup_answer,
    final_decision_after_followup,
    generate_theory_question,
    TheoryDecision,
)
from question_utils import pick_question, normalize_task_types, collect_previous_theory_topics

MAX_QUESTIONS = 3  # легко менять при необходимости
TIMER_SECONDS = 45 * 60
logger = logging.getLogger(__name__)

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

class AssignInterviewPayload(BaseModel):
    adminId: str
    candidateId: str
    direction: str
    level: str
    format: str = "Full interview"
    tasks: List[str] = []
    duration: Optional[int] = None  # minutes

class AssignedStartPayload(BaseModel):
    userId: str

class TheoryEvalPayload(BaseModel):
    sessionId: str
    questionId: str
    ownerId: str
    answer: str
    isFollowup: bool = False
    followupQuestion: Optional[str] = None
    baseQuestionJson: Optional[dict] = None
    missingPoints: Optional[List[str]] = None

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

def ensure_admin_user(cur, admin_id: str):
    cur.execute("SELECT * FROM users WHERE id=?", (admin_id,))
    admin = fetchone_dict(cur)
    if not admin or (not admin.get("admin") and admin.get("role") != "superadmin"):
        return None
    return admin



# --------- Эндпоинты

@app.post("/api/register")
def register(payload: UserPayload):
    email = (payload.email or "").strip().lower()
    password = (payload.password or "").strip()
    name = (payload.name or "Кандидат").strip() or "Кандидат"
    if not email or not password:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    existing = fetchone_dict(cur.execute("SELECT * FROM users WHERE email=?", (email,)))
    if existing:
        # Если аккаунт уже есть и пароль совпадает — ведём себя идемпотентно и возвращаем профиль
        if existing.get("password") == sha1(password):
            conn.close()
            return {"user": user_safe(existing)}
        conn.close()
        raise HTTPException(status_code=409, detail="exists")
    # Только роль user при регистрации
    user_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO users (id, email, name, password, level, role, admin, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, email, name, sha1(password), "Junior", "user", 0, payload.lang or "ru"),
    )
    conn.commit()
    row = cur.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    columns = [col[0] for col in cur.description] if cur.description else []
    result = dict(zip(columns, row)) if row else {}
    return {"user": user_safe(result)}


@app.post("/api/login")
def login(payload: LoginPayload):
    email = (payload.email or "").strip().lower()
    password = (payload.password or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, sha1(password)),
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
    # Всегда генерируем один теоретический вопрос через LLM
    question_id = None
    question_title = "Вопрос 1"
    description = ""
    question_type = "theory"
    use_ide = False
    meta_json = None
    starter = ""

    try:
        # Пока сессия не создана — нет предыдущих вопросов
        prev_topics: List[str] = []
        q_obj = generate_theory_question((payload.direction or "fullstack").lower(), (payload.level or "middle").lower(), prev_topics)
        question_id = f"llm-theory-{uuid.uuid4()}"
        description = q_obj.get("question", "")
        question_title = q_obj.get("title") or question_title
        meta_json = json.dumps(q_obj, ensure_ascii=False)
    except Exception as exc:
        logger.exception("LLM generation failed on start-interview", extra={"direction": payload.direction, "level": payload.level})
        question_id = None

    if not question_id:
        logger.error(
            "No questions available after LLM generation",
            extra={"direction": payload.direction, "level": payload.level, "tasks": payload.tasks},
        )
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")

    total_questions = 1
    session_id = str(uuid.uuid4())
    started_at = now_iso()
    if question_type == "theory":
        starter = ""

    cur.execute(
        """
        INSERT INTO sessions (id, ownerId, questionId, questionTitle, useIDE, direction, level, format, tasks, description, starterCode, timer, solved, total, startedAt, status, is_active, is_finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            payload.userId,
            question_id,
            question_title,
            int(use_ide),
            payload.direction,
            payload.level,
            payload.format,
            ",".join(["theory"]),
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
    # Добавляем выбранный вопрос в историю с порядком
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            question_id,
            question_title,
            1,
            meta_json,
            question_type or "coding",
        ),
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
            "tasks": ["theory"],
            "questionTitle": question_title,
            "questionId": question_id,
            "useIDE": bool(use_ide),
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
            row["description"] = q["statement"]
            row["useIDE"] = q.get("useIDE", 1)
        else:
            # Пробуем достать LLM-вопрос из session_questions.meta_json
            meta = cur.execute(
                "SELECT meta_json FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1",
                (session_id, questionId),
            ).fetchone()
            if meta and meta[0]:
                try:
                    meta_obj = json.loads(meta[0])
                    row["questionId"] = questionId
                    row["questionTitle"] = meta_obj.get("title") or row.get("questionTitle")
                    row["description"] = meta_obj.get("question") or row.get("description")
                    row["useIDE"] = 0
                except Exception:
                    pass
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

# --------- Назначенные интервью (админ)

@app.post("/api/admin/assign-interview")
def assign_interview(payload: AssignInterviewPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    admin = ensure_admin_user(cur, payload.adminId)
    if not admin:
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    assigned_id = str(uuid.uuid4())
    tasks_str = ",".join(payload.tasks or [])
    duration_minutes = payload.duration if payload.duration and payload.duration > 0 else None
    cur.execute(
        """
        INSERT INTO assigned_interviews (id, candidateId, adminId, direction, level, format, tasks, duration, status, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
        (
            assigned_id,
            payload.candidateId,
            payload.adminId,
            payload.direction,
            payload.level,
            payload.format or "Full interview",
            tasks_str,
            duration_minutes,
            now_iso(),
        ),
    )
    # Простое уведомление через таблицу событий
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, ownerId, event_type, payload, risk_level, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            "",
            payload.candidateId,
            "assigned_interview_created",
            assigned_id,
            "info",
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()
    return {
        "assigned": {
            "id": assigned_id,
            "candidateId": payload.candidateId,
            "adminId": payload.adminId,
            "direction": payload.direction,
            "level": payload.level,
            "format": payload.format or "Full interview",
            "tasks": payload.tasks or [],
            "status": "pending",
            "duration": duration_minutes,
            "createdAt": now_iso(),
        }
    }

@app.get("/api/assigned-interview")
def get_assigned_interview(userId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT * FROM assigned_interviews
        WHERE candidateId=? AND status IN ('pending', 'active')
        ORDER BY datetime(createdAt) DESC
        LIMIT 1
        """,
        (userId,),
    ).fetchone()
    columns = [c[0] for c in cur.description] if cur.description else []
    assigned = dict(zip(columns, row)) if row else None
    conn.close()
    if not assigned:
        return {"assigned": None}
    assigned["tasks"] = (assigned.get("tasks") or "").split(",") if assigned.get("tasks") else []
    return {"assigned": assigned}

@app.get("/api/assigned-interviews")
def list_assigned_interviews(userId: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT * FROM assigned_interviews
        WHERE candidateId=? AND status IN ('pending', 'active')
        ORDER BY datetime(createdAt) DESC
        """,
        (userId,),
    ).fetchall()
    cols = [c[0] for c in cur.description] if cur.description else []
    items = [dict(zip(cols, r)) for r in rows]
    conn.close()
    for it in items:
        it["tasks"] = (it.get("tasks") or "").split(",") if it.get("tasks") else []
    return {"assigned": items}

@app.post("/api/assigned-interview/start/{assigned_id}")
def start_assigned_interview(assigned_id: str, payload: AssignedStartPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM assigned_interviews WHERE id=?", (assigned_id,))
    assigned = fetchone_dict(cur)
    if not assigned:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    if assigned.get("candidateId") != payload.userId:
        conn.close()
        raise HTTPException(status_code=403, detail="forbidden")
    # Если уже есть активная сессия по назначению — возвращаем её
    if assigned.get("status") == "active" and assigned.get("sessionId"):
        session_id = assigned["sessionId"]
        conn.close()
        return get_session(session_id)
    # Подготавливаем первый вопрос с учётом фильтров назначения
    task_types = normalize_task_types((assigned.get("tasks") or "").split(",") if assigned.get("tasks") else [])

    question_id = None
    question_title = "Вопрос 1"
    description = ""
    question_type = "theory"
    use_ide = False
    meta_json = None
    starter = ""

    # Пытаемся сгенерировать теоретический вопрос для назначенного интервью
    try:
        prev_topics = collect_previous_theory_topics(cur, assigned.get("sessionId") or "")
        q_obj = generate_theory_question((assigned.get("direction") or "fullstack").lower(), (assigned.get("level") or "middle").lower(), prev_topics)
        question_id = f"llm-theory-{uuid.uuid4()}"
        description = q_obj.get("question", "")
        question_title = q_obj.get("title") or question_title
        meta_json = json.dumps(q_obj, ensure_ascii=False)
    except Exception as exc:
        logger.exception("LLM generation failed on assigned-interview start", extra={"direction": assigned.get("direction"), "level": assigned.get("level")})
        question_id = None

    # Если теорию не удалось — пробуем coding как фолбэк
    if not question_id:
        q = pick_question(cur, assigned.get("direction"), assigned.get("level"), ["coding"], [])
        if q:
            question_id = q["id"]
            question_title = q["title"]
            description = q["statement"]
            question_type = "coding"
            use_ide = bool(q.get("useIDE", 1))

    if not question_id:
        logger.error(
            "No questions available for assigned interview after LLM+coding fallback",
            extra={"direction": assigned.get("direction"), "level": assigned.get("level"), "tasks": assigned.get("tasks")},
        )
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")
    total_questions = 1
    session_id = str(uuid.uuid4())
    starter = starter if question_type != "theory" else ""
    started_at = now_iso()
    timer_val = TIMER_SECONDS
    try:
        if assigned.get("duration"):
            timer_val = int(assigned.get("duration")) * 60
    except Exception:
        timer_val = TIMER_SECONDS
    cur.execute(
        """
        INSERT INTO sessions (id, ownerId, questionId, questionTitle, useIDE, direction, level, format, tasks, description, starterCode, timer, solved, total, startedAt, status, is_active, is_finished, assignedId)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            payload.userId,
            question_id,
            question_title,
            int(use_ide),
            assigned.get("direction"),
            assigned.get("level"),
            assigned.get("format") or "Full interview",
            ",".join(["theory"]),
            description,
            starter,
            timer_val,
            0,
            total_questions,
            started_at,
            "active",
            1,
            0,
            assigned_id,
        ),
    )
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            question_id,
            question_title,
            1,
            meta_json,
            question_type or "coding",
        ),
    )
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
            assigned.get("level"),
            "Отчёт формируется",
            "[]",
            "[]",
            "{}",
        ),
    )
    # Обновляем назначение
    cur.execute(
        "UPDATE assigned_interviews SET status='active', sessionId=? WHERE id=?",
        (session_id, assigned_id),
    )
    cur.execute(
        """
        INSERT INTO interview_events (id, sessionId, ownerId, event_type, payload, risk_level, createdAt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            payload.userId,
            "assigned_interview_started",
            assigned_id,
            "info",
            now_iso(),
        ),
    )
    conn.commit()
    used_questions = get_used_questions(cur, session_id)
    conn.close()
    return {
        "session": {
            "id": session_id,
            "direction": assigned.get("direction"),
            "level": assigned.get("level"),
            "format": assigned.get("format") or "Full interview",
            "tasks": ["theory"],
            "questionTitle": question_title,
            "questionId": question_id,
            "useIDE": bool(use_ide),
            "description": description,
            "starterCode": starter,
            "timer": timer_val,
            "solved": 0,
            "total": total_questions,
            "startedAt": started_at,
            "usedQuestions": used_questions,
            "status": "active",
            "is_active": 1,
            "is_finished": 0,
            "assignedId": assigned_id,
        }
    }


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
            "SELECT questionId as id, questionTitle as title, q_type FROM session_questions WHERE sessionId=? ORDER BY position",
            (session_id,),
        )
    )


@app.post("/api/theory/eval")
def eval_theory(payload: TheoryEvalPayload):
    question_obj = payload.baseQuestionJson or {}
    # Пытаемся взять вопрос из session_questions.meta_json, если не передали
    if not question_obj:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        row = cur.execute(
            "SELECT meta_json FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1",
            (payload.sessionId, payload.questionId),
        ).fetchone()
        if row and row[0]:
            try:
                question_obj = json.loads(row[0])
            except Exception:
                question_obj = {}
        conn.close()
    if not question_obj:
        raise HTTPException(status_code=400, detail="missing_question_data")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Проверяем существующий ответ
    cur.execute(
        "SELECT * FROM answers WHERE sessionId=? AND questionId=? AND ownerId=?",
        (payload.sessionId, payload.questionId, payload.ownerId),
    )
    existing = fetchone_dict(cur)

    if not payload.isFollowup:
        if existing and existing.get("decision") and existing.get("decision") != TheoryDecision.CLARIFY.value:
            conn.close()
            return {
                "decision": existing.get("decision"),
                "score": existing.get("score"),
                "maxScore": existing.get("max_score"),
                "coveredPoints": [],
                "missingPoints": [],
                "feedbackShort": "",
                "feedbackDetailed": "",
                "followUp": None,
            }
        result = evaluate_theory_answer(question_obj, payload.answer)
        decision = classify_theory_answer(result.get("score", 0), result.get("max_score", 1))
        follow = None
        if decision == TheoryDecision.CLARIFY:
            follow_data = generate_followup_question(
                question_obj,
                result.get("missing_points", []),
                payload.answer,
            )
            follow = {"question": follow_data.get("follow_up_question")}
        # Сохраняем результат в answers
        cur.execute(
            """
            INSERT INTO answers (id, sessionId, questionId, ownerId, content, updatedAt, score, max_score, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sessionId, questionId) DO UPDATE SET
              content=excluded.content,
              updatedAt=excluded.updatedAt,
              score=excluded.score,
              max_score=excluded.max_score,
              decision=excluded.decision
            """,
            (
                str(uuid.uuid4()),
                payload.sessionId,
                payload.questionId,
                payload.ownerId,
                payload.answer,
                now_iso(),
                result.get("score", 0),
                result.get("max_score", question_obj.get("max_score", 10)),
                decision.value,
            ),
        )
        conn.commit()
        conn.close()
        return {
            "decision": decision.value,
            "score": result.get("score"),
            "maxScore": result.get("max_score"),
            "verdict": result.get("verdict"),
            "coveredPoints": result.get("covered_points", []),
            "missingPoints": result.get("missing_points", []),
            "feedbackShort": result.get("feedback_short"),
            "feedbackDetailed": result.get("feedback_detailed"),
            "followUp": follow,
        }
    else:
        if existing and existing.get("decision") and existing.get("decision") != TheoryDecision.CLARIFY.value:
            conn.close()
            return {
                "decision": existing.get("decision"),
                "score": existing.get("score"),
                "maxScore": existing.get("max_score"),
                "coveredPoints": [],
                "missingPoints": [],
                "feedbackShort": "",
                "feedbackDetailed": None,
                "followUp": None,
            }
        missing_points = payload.missingPoints or []
        follow_up_question = payload.followupQuestion or ""
        follow_result = evaluate_followup_answer(
            question_obj,
            missing_points,
            follow_up_question,
            payload.answer,
        )
        decision = final_decision_after_followup(
            0,
            0,
            follow_result.get("score", 0),
            follow_result.get("max_score", 0),
        )
        cur.execute(
            """
            INSERT INTO answers (id, sessionId, questionId, ownerId, content, updatedAt, score, max_score, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sessionId, questionId) DO UPDATE SET
              content=excluded.content,
              updatedAt=excluded.updatedAt,
              score=excluded.score,
              max_score=excluded.max_score,
              decision=excluded.decision
            """,
            (
                str(uuid.uuid4()),
                payload.sessionId,
                payload.questionId,
                payload.ownerId,
                payload.answer,
                now_iso(),
                follow_result.get("score", 0),
                follow_result.get("max_score", 0),
                decision.value,
            ),
        )
        conn.commit()
        conn.close()
        return {
            "decision": decision.value,
            "score": follow_result.get("score"),
            "maxScore": follow_result.get("max_score"),
            "coveredPoints": follow_result.get("covered_points", []),
            "missingPoints": follow_result.get("missing_points_still", []),
            "feedbackShort": follow_result.get("feedback_short"),
            "feedbackDetailed": None,
            "followUp": None,
        }


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
    # Обновляем назначенное интервью, если есть связь
    if session.get("assignedId"):
        cur.execute(
            "UPDATE assigned_interviews SET status='completed', sessionId=? WHERE id=?",
            (payload.sessionId, session.get("assignedId")),
        )
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
        return {"content": "", "decision": None, "score": None, "maxScore": None}
    return {
        "content": row.get("content", ""),
        "decision": row.get("decision"),
        "score": row.get("score"),
        "maxScore": row.get("max_score"),
    }


if __name__ == "__main__":
    import uvicorn

    # Use full module path so running `python server_py/main.py` works reliably
    uvicorn.run("server_py.main:app", host="0.0.0.0", port=8000, reload=True)
