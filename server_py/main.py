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
import sys
import re
import time

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
from llm_code import generate_code_task, parse_function_name, build_tests_for_task, explain_code_error, generate_code_hint
from question_utils import (
    pick_question,
    normalize_task_types,
    collect_previous_theory_topics,
    collect_previous_code_topics,
    save_code_task_and_tests,
)
from docker_runner import run_code_in_docker, run_code_with_fallback

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

@app.on_event("startup")
def _log_startup():
    logger.info("FastAPI app startup complete")

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


class RunSamplesPayload(BaseModel):
    sessionId: str
    questionId: str
    taskId: str
    ownerId: str
    code: str
    language: str


class CheckCodePayload(BaseModel):
    sessionId: str
    questionId: str
    taskId: str
    ownerId: str
    code: str
    language: str

class CodeHintPayload(BaseModel):
    sessionId: str
    questionId: str
    taskId: str
    ownerId: str
    language: str
    userCode: Optional[str] = None

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


def get_code_task_description(cur, code_task_id: Optional[str]) -> Optional[str]:
    if not code_task_id:
        return None
    row = fetchone_dict(cur.execute("SELECT description_markdown FROM code_tasks WHERE task_id=?", (code_task_id,)))
    return row.get("description_markdown") if row else None


def calculate_session_score(cur, session_id: str) -> int:
    ans_row = fetchone_dict(cur.execute("SELECT COALESCE(SUM(score),0) as s FROM answers WHERE sessionId=?", (session_id,)))
    code_row = fetchone_dict(cur.execute("SELECT COALESCE(SUM(score),0) as s FROM code_attempts WHERE session_id=?", (session_id,)))
    ans_score = ans_row.get("s") if ans_row else 0
    code_score = code_row.get("s") if code_row else 0
    try:
        return int(ans_score or 0) + int(code_score or 0)
    except Exception:
        return 0


def sanitize_starter_code(task: dict) -> str:
    """
    Возвращаем безопасный каркас функции: только сигнатура + pass/ TODO.
    Игнорируем starter_code из LLM, если оно содержит реализацию.
    """
    signature = task.get("function_signature") or ""
    sig = signature.strip()
    if not sig:
        return ""
    # Убираем возможный завершающий код
    if not sig.endswith(":"):
        sig = sig + ":"
    return f"{sig}\n    # TODO: реализуйте функцию\n    pass\n"


def _param_count(signature: str) -> Optional[int]:
    m = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\((.*?)\)", signature or "", re.S)
    if not m:
        return None
    raw = m.group(1)
    params = []
    for part in raw.split(","):
        p = part.strip()
        if not p or p.startswith("*"):
            continue
        params.append(p)
    return len(params)


def _prepare_runner_input(raw_input, param_count: Optional[int]):
    if param_count == 1:
        return [raw_input]
    if isinstance(raw_input, list):
        return raw_input
    return [raw_input]


def _prepare_runner_tests(rows: List[dict], param_count: Optional[int]) -> List[dict]:
    tests = []
    for idx, row in enumerate(rows):
        name = row.get("name") or f"test_{idx+1}"
        try:
            raw_input = json.loads(row.get("input_json") or "[]")
        except Exception:
            raw_input = []
        try:
            expected = json.loads(row.get("expected_json") or "null")
        except Exception:
            expected = None
        tests.append(
            {
                "name": name,
                "input": _prepare_runner_input(raw_input, param_count),
                "expected": expected,
            }
        )
    return tests


def score_for_attempt(attempt: int, max_score: int = 10) -> int:
    if attempt == 1:
        return max_score
    if attempt == 2:
        return int(max_score * 0.7)
    if attempt == 3:
        return int(max_score * 0.5)
    return 0


def _base_max_score(task: dict) -> int:
    category = (task.get("category") or "").lower()
    if category == "domain":
        return 15
    return 10


def _get_hints_used(cur, session_id: str, question_id: str) -> int:
    row = fetchone_dict(
        cur.execute(
            "SELECT hints_used FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1",
            (session_id, question_id),
        )
    )
    return int(row.get("hints_used") or 0) if row else 0


def _bump_hints_used(cur, session_id: str, question_id: str) -> int:
    cur.execute(
        "UPDATE session_questions SET hints_used=COALESCE(hints_used,0)+1 WHERE sessionId=? AND questionId=?",
        (session_id, question_id),
    )
    return _get_hints_used(cur, session_id, question_id)


def _candidate_languages(task: dict, requested: Optional[str]) -> List[str]:
    langs: List[str] = []
    if requested:
        langs.append(requested.lower())
    try:
        allowed = json.loads(task.get("allowed_languages_json") or "[]")
        if isinstance(allowed, list):
            for l in allowed:
                if isinstance(l, str):
                    ll = l.lower()
                    if ll not in langs:
                        langs.append(ll)
    except Exception:
        pass
    if task.get("language"):
        ll = str(task.get("language")).lower()
        if ll not in langs:
            langs.append(ll)
    return langs or [requested or "python"]


@app.post("/api/code/hint")
def code_hint(payload: CodeHintPayload):
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    cur = conn.cursor()
    try:
        task_id = _resolve_task_id(cur, payload.sessionId, payload.questionId, payload.taskId)
        if not task_id:
            raise HTTPException(status_code=404, detail="task_not_found")
        task, _, _ = _load_task_and_tests(cur, task_id, payload.sessionId, payload.questionId)
        # Берём код из запроса или последней попытки
        user_code = payload.userCode
        if not user_code:
            row = fetchone_dict(
                cur.execute(
                    """
                    SELECT code FROM code_attempts
                    WHERE session_id=? AND question_id=? AND task_id=?
                    ORDER BY attempt_number DESC LIMIT 1
                    """,
                    (payload.sessionId, payload.questionId, task_id),
                )
            )
            user_code = row.get("code") if row else None
        hints_used_before = _get_hints_used(cur, payload.sessionId, payload.questionId)
        hint_level = hints_used_before + 1
        if hint_level > 3:
            return {"hint": None, "hintsUsed": hints_used_before, "effectiveMaxScore": max(_base_max_score(task) - hints_used_before * 2, 0)}
        prev_rows = fetchall_dicts(
            cur.execute(
                """
                SELECT content FROM messages
                WHERE sessionId=? AND questionId=? AND source='hint'
                ORDER BY datetime(createdAt) ASC
                """,
                (payload.sessionId, payload.questionId),
            )
        )
        prev_hints = [r.get("content") for r in prev_rows if r.get("content")]
        hint_text = generate_code_hint(task, user_code, hint_level, prev_hints)
        hints_used = _bump_hints_used(cur, payload.sessionId, payload.questionId)
        base_max = _base_max_score(task)
        effective_max = max(base_max - hints_used_before * 2, 0)
        if hint_text:
            cur.execute(
                """
                INSERT INTO messages (id, sessionId, questionId, ownerId, role, content, source, createdAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    payload.sessionId,
                    payload.questionId,
                    payload.ownerId,
                    "assistant",
                    hint_text,
                    "hint",
                    now_iso(),
                ),
            )
        conn.commit()
        return {"hint": hint_text, "hintsUsed": hints_used, "effectiveMaxScore": effective_max}
    finally:
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()


def choose_next_category(cur, session_id: str, track: str) -> Optional[str]:
    """
    Внутри одного собеседования хотим 2 algo и 2 domain.
    Если лимиты исчерпаны — возвращаем None (новые вопросы не генерируем).
    """
    rows = fetchall_dicts(
        cur.execute(
            "SELECT category FROM session_questions WHERE sessionId=? AND q_type='coding'",
            (session_id,),
        )
    )
    algo_count = sum(1 for r in rows if (r.get("category") or "").lower() == "algo")
    domain_count = sum(1 for r in rows if (r.get("category") or "").lower() == "domain")
    if algo_count < 2:
        return "algo"
    if domain_count < 2:
        return "domain"
    return None


def _is_duplicate_code_task(cur, session_id: str, track: str, category: str, new_task: dict) -> bool:
    track = (track or "").lower()
    category = (category or "").lower()
    new_topic = (new_task.get("topic") or "").strip().lower()
    new_title = (new_task.get("title") or "").strip().lower()
    new_desc_prefix = " ".join((new_task.get("description_markdown") or "").strip().lower().split()[:12])
    rows = fetchall_dicts(
        cur.execute(
            """
            SELECT sq.category, sq.meta_json, sq.questionTitle, ct.topic, ct.title as ct_title, ct.description_markdown as ct_desc, ct.track
            FROM session_questions sq
            LEFT JOIN code_tasks ct ON ct.task_id = sq.code_task_id
            WHERE sq.sessionId=? AND sq.q_type='coding'
            """,
            (session_id,),
        )
    )
    for r in rows:
        r_track = (r.get("track") or "").lower()
        if r_track and r_track != track:
            continue
        r_cat = (r.get("category") or "").lower() or (r.get("meta_json") and json.loads(r.get("meta_json")).get("category", "") or "").lower()
        if r_cat and r_cat != category:
            continue
        topic = (r.get("topic") or r.get("questionTitle") or r.get("ct_title") or "").strip().lower()
        if not topic and r.get("meta_json"):
            try:
                obj = json.loads(r.get("meta_json"))
                topic = (obj.get("topic") or obj.get("title") or "").strip().lower()
            except Exception:
                topic = ""
        desc = (r.get("ct_desc") or "").strip().lower()
        desc_prefix = " ".join(desc.split()[:12])
        if new_topic and topic and new_topic == topic:
            return True
        if new_title and ((topic and new_title == topic) or new_title == (r.get("ct_title") or "").strip().lower()):
            return True
        if new_desc_prefix and desc_prefix and new_desc_prefix == desc_prefix:
            return True
    return False


def generate_unique_code_task(cur, session_id: str, track: str, level: str, category: str, language: str) -> dict:
    prev_algo, prev_domain = collect_previous_code_topics(cur, session_id, track)
    last_task = None
    for _ in range(3):
        task = generate_code_task(track, level, category, language, prev_algo, prev_domain)
        task["track"] = task.get("track") or track
        task["level"] = task.get("level") or level
        task["category"] = task.get("category") or category
        task["language"] = task.get("language") or language
        last_task = task
        if not _is_duplicate_code_task(cur, session_id, track, category, task):
            return task
    return last_task or {}


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
        cur.execute("SELECT sessionId, finishedAt, score FROM interview_results WHERE ownerId=?", (userId,))
    )
    finished_map = {r["sessionId"]: r.get("finishedAt") for r in result_rows if r.get("sessionId")}
    score_map = {r["sessionId"]: r.get("score") for r in result_rows if r.get("sessionId")}
    conn.close()
    history = []
    for s in sessions:
        finished = finished_map.get(s["id"])
        # Если нет зафиксированного завершения — используем createdAt
        date_val = finished or s.get("createdAt") or now_iso()
        score_val = score_map.get(s["id"])
        if score_val is None:
            try:
                cur_score = calculate_session_score(sqlite3.connect(DB_PATH).cursor(), s["id"])
                score_val = cur_score
            except Exception:
                score_val = 0
        history.append(
            {
                "id": s["id"],
                "topic": (s.get("description") or "Тема")[0:32] + "...",
                "direction": s.get("direction"),
                "level": s.get("level"),
                "score": score_val or 0,
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
        # Подхватываем codeTaskId/language из meta текущего вопроса
        if existing.get("questionId"):
            sq = fetchone_dict(
                cur.execute("SELECT meta_json, code_task_id FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1", (existing["id"], existing["questionId"]))
            )
            if sq and sq.get("meta_json"):
                try:
                    meta_obj = json.loads(sq.get("meta_json"))
                    existing["codeTaskId"] = existing.get("codeTaskId") or sq.get("code_task_id")
                    existing["language"] = meta_obj.get("language") or existing.get("language")
                    if not existing.get("description"):
                        existing["description"] = meta_obj.get("description_markdown") or meta_obj.get("question")
                except Exception:
                    pass
            elif sq:
                existing["codeTaskId"] = existing.get("codeTaskId") or sq.get("code_task_id")
        # Если описания нет, пробуем подтянуть из code_tasks по codeTaskId
        if not existing.get("description"):
            desc = get_code_task_description(cur, existing.get("codeTaskId"))
            if desc:
                existing["description"] = desc
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
                "codeTaskId": existing.get("codeTaskId"),
                "language": existing.get("language"),
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
                "ownerId": existing.get("ownerId"),
            }
        }
    # Всегда генерируем один теоретический вопрос через LLM
    task_types = normalize_task_types(payload.tasks)
    dual_mode = "coding" in task_types and "theory" in task_types
    question_id = None
    question_title = "Вопрос 1"
    description = ""
    question_type = "theory"
    use_ide = False
    meta_json = None
    starter = ""
    function_signature = None
    function_signature = None
    q_obj: dict = {}

    # Пробуем сгенерировать coding-задачу, если выбрана и не в дуальном режиме (в дуальном теорию выдаём первой)
    code_task_id = None
    if "coding" in task_types and not dual_mode:
        try:
            q_obj = generate_unique_code_task(cur, "", (payload.direction or "fullstack").lower(), (payload.level or "middle").lower(), "algo", "python")
            code_task_id = q_obj.get("task_id") or f"code-{uuid.uuid4()}"
            public_tests, hidden_tests = build_tests_for_task(q_obj)
            save_code_task_and_tests(cur, code_task_id, q_obj, public_tests, hidden_tests)
            question_id = f"llm-code-{uuid.uuid4()}"
            question_title = q_obj.get("title") or question_title
            description = q_obj.get("description_markdown", "")
            question_type = "coding"
            use_ide = True
            starter = sanitize_starter_code(q_obj)
            function_signature = q_obj.get("function_signature")
            meta_json = json.dumps({**q_obj, "task_id": code_task_id, "starter_code": starter}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("LLM coding generation failed on start-interview", extra={"direction": payload.direction, "level": payload.level})
            question_id = None

    # Если coding не удалось или не выбрано — генерируем theory
    if not question_id and "theory" in task_types:
        try:
            prev_topics: List[str] = []
            q_obj = generate_theory_question((payload.direction or "fullstack").lower(), (payload.level or "middle").lower(), prev_topics)
            question_id = f"llm-theory-{uuid.uuid4()}"
            description = q_obj.get("question", "")
            question_title = q_obj.get("title") or question_title
            question_type = "theory"
            use_ide = False
            starter = ""
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

    total_questions = 10 if dual_mode else 1
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
            int(question_type == "coding"),
            payload.direction,
            payload.level,
            payload.format,
            ",".join(task_types),
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
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type, code_task_id, category, status, is_finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
        """,
        (
            session_id,
            question_id,
            question_title,
            1,
            meta_json,
            question_type or "coding",
            code_task_id,
            q_obj.get("category") if question_type == "coding" else None,
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
            "questionDisplayId": f"Q-1-{question_id[-6:]}" if question_id else None,
            "direction": payload.direction,
            "level": payload.level,
            "format": payload.format,
            "tasks": task_types,
            "questionTitle": question_title,
            "questionId": question_id,
            "useIDE": bool(question_type == "coding"),
            "functionSignature": function_signature,
            "codeTaskId": code_task_id,
            "language": (q_obj or {}).get("language"),
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
            "ownerId": payload.userId,
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
    current_qtype = None
    meta_obj = None
    target_qid = questionId or row.get("questionId")
    if target_qid:
        cur.execute("SELECT * FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1", (session_id, target_qid))
        sq_row = fetchone_dict(cur)
        if sq_row:
            current_qtype = sq_row.get("q_type")
            row["codeTaskId"] = sq_row.get("code_task_id")
            if sq_row.get("meta_json"):
                try:
                    meta_obj = json.loads(sq_row.get("meta_json"))
                except Exception:
                    meta_obj = None
        cur.execute("SELECT * FROM questions WHERE id=?", (target_qid,))
        q = fetchone_dict(cur)
        if q:
            row["questionId"] = q["id"]
            row["questionTitle"] = q["title"]
            row["description"] = q["statement"]
            row["useIDE"] = q.get("useIDE", 1)
        elif meta_obj:
            row["questionId"] = target_qid
            row["questionTitle"] = meta_obj.get("title") or row.get("questionTitle")
            row["description"] = meta_obj.get("description_markdown") or meta_obj.get("question") or row.get("description")
            row["starterCode"] = meta_obj.get("starter_code") or row.get("starterCode")
            row["functionSignature"] = meta_obj.get("function_signature")
            row["useIDE"] = 1 if current_qtype == "coding" else 0
            row["codeTaskId"] = row.get("codeTaskId") or meta_obj.get("task_id")
            row["language"] = meta_obj.get("language") or row.get("language")
    # Если описание пустое и есть привязанный code_task_id — подтащим из code_tasks
    if not row.get("description") and row.get("codeTaskId"):
        desc = get_code_task_description(cur, row.get("codeTaskId"))
        if desc:
            row["description"] = desc
    # История выданных вопросов
    used_questions = fetchall_dicts(
        cur.execute(
            """
            SELECT
              questionId as id,
              questionTitle as title,
              q_type as qType,
              code_task_id as codeTaskId,
              position,
              category,
              status,
              is_finished as isFinished
            FROM session_questions
            WHERE sessionId=?
            ORDER BY position
            """,
            (session_id,),
        )
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
            "functionSignature": row.get("functionSignature"),
            "codeTaskId": row.get("codeTaskId"),
            "language": row.get("language"),
            "timer": row.get("timer"),
            "startedAt": row.get("startedAt"),
            "solved": row.get("solved"),
            "total": row.get("total"),
            "questionTitle": row.get("questionTitle"),
            "questionId": row.get("questionId"),
            "questionDisplayId": f"Q-{row.get('questionId')[-6:]}" if row.get("questionId") else None,
            "ownerId": row.get("ownerId"),
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
    candidates = []
    for u in users:
        # Последний результат пользователя
        res = fetchone_dict(
            cur.execute(
                "SELECT score FROM interview_results WHERE ownerId=? ORDER BY datetime(finishedAt) DESC LIMIT 1",
                (u["id"],),
            )
        )
        last_score = res.get("score") if res else 0
        candidates.append(
            {
                "id": u["id"],
                "name": u["name"],
                "email": u["email"],
                "level": u["level"],
                "admin": bool(u.get("admin")),
                "role": u.get("role"),
                "hasFlags": bool(flags.get(u["id"])),
                "flagsCount": flags.get(u["id"], 0),
                "lastScore": last_score or 0,
                "lastTopic": "Собеседование",
            }
        )
    conn.close()
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


def _resolve_task_id(cur, session_id: Optional[str], question_id: Optional[str], task_id: Optional[str]) -> Optional[str]:
    if task_id:
        return task_id
    if session_id and question_id:
        sq = fetchone_dict(
            cur.execute(
                "SELECT code_task_id, meta_json FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1",
                (session_id, question_id),
            )
        )
        if sq:
            if sq.get("code_task_id"):
                return sq.get("code_task_id")
            if sq.get("meta_json"):
                try:
                    meta = json.loads(sq.get("meta_json"))
                    if meta.get("task_id"):
                        return meta.get("task_id")
                except Exception:
                    pass
    return task_id


def _restore_code_task_from_meta(cur, session_id: Optional[str], question_id: Optional[str], task_id: str):
    if not session_id or not question_id:
        return None
    sq = fetchone_dict(
        cur.execute(
            "SELECT meta_json FROM session_questions WHERE sessionId=? AND questionId=? LIMIT 1",
            (session_id, question_id),
        )
    )
    if not sq or not sq.get("meta_json"):
        return None
    try:
        meta = json.loads(sq.get("meta_json"))
    except Exception:
        return None
    # Минимально проверяем обязательные поля
    if not meta or "reference_solution" not in meta or "sample_inputs" not in meta or "edge_case_inputs" not in meta:
        return None
    try:
        public_tests, hidden_tests = build_tests_for_task(meta)
        save_code_task_and_tests(cur, task_id, meta, public_tests, hidden_tests)
        return fetchone_dict(cur.execute("SELECT * FROM code_tasks WHERE task_id=?", (task_id,)))
    except Exception:
        return None


def _load_task_and_tests(cur, task_id: str, session_id: Optional[str] = None, question_id: Optional[str] = None):
    task = fetchone_dict(cur.execute("SELECT * FROM code_tasks WHERE task_id=?", (task_id,)))
    if not task:
        task = _restore_code_task_from_meta(cur, session_id, question_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_not_found")
    public_tests = fetchall_dicts(
        cur.execute("SELECT * FROM code_tests WHERE task_id=? AND is_public=1 ORDER BY id", (task_id,))
    )
    hidden_tests = fetchall_dicts(
        cur.execute("SELECT * FROM code_tests WHERE task_id=? AND is_public=0 ORDER BY id", (task_id,))
    )
    return task, public_tests, hidden_tests


def _runner_has_error(run_result: dict) -> tuple[bool, str]:
    if not run_result:
        return True, "empty_result"
    if run_result.get("error"):
        return True, str(run_result.get("error"))
    for t in run_result.get("tests", []):
        if t.get("error"):
            return True, str(t.get("error"))
    return False, ""


def _tests_from_runner(run_result: dict) -> List[dict]:
    out = []
    for t in run_result.get("tests", []):
        status = "passed" if t.get("passed") else "failed"
        if t.get("error"):
            status = "error"
        out.append(
            {
                "name": t.get("name"),
                "status": status,
                "expected": t.get("expected"),
                "actual": t.get("output"),
            }
        )
    return out


def _save_error_hint(cur, session_id: str, question_id: str, owner_id: str, hint: str):
    if not hint:
        return
    cur.execute(
        """
        INSERT INTO messages (id, sessionId, ownerId, role, content, source, questionId)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), session_id, owner_id, "assistant", hint, "code_error_hint", question_id),
    )


def _save_attempt(cur, payload: CheckCodePayload, attempt: int, code: str, passed_public: int, passed_hidden: int, score: int, task_id: Optional[str] = None):
    params = (
        payload.sessionId,
        payload.questionId,
        task_id or payload.taskId,
        payload.ownerId,
        attempt,
        code,
        passed_public,
        passed_hidden,
        score,
        now_iso(),
    )
    for i in range(3):
        try:
            cur.execute(
                """
                INSERT INTO code_attempts (session_id, question_id, task_id, owner_id, attempt_number, code, passed_public, passed_hidden, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or i == 2:
                raise
            time.sleep(0.2 * (i + 1))


@app.post("/api/code/run-samples")
def code_run_samples(payload: RunSamplesPayload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        task_id = _resolve_task_id(cur, payload.sessionId, payload.questionId, payload.taskId)
        if not task_id:
            logger.warning("run-samples: task id not resolved", extra={"session": payload.sessionId, "question": payload.questionId})
            return {"tests": [], "hasError": True, "error": "task_not_found"}
        try:
            task, public_tests, _hidden = _load_task_and_tests(cur, task_id, payload.sessionId, payload.questionId)
        except HTTPException as exc:
            logger.warning("run-samples: task/tests missing", extra={"task_id": task_id, "detail": exc.detail})
            return {"tests": [], "hasError": True, "error": str(exc.detail)}
        if not public_tests:
            logger.warning("run-samples: no public tests", extra={"task_id": task_id})
            return {"tests": [], "hasError": True, "error": "no_public_tests"}
        signature = task.get("function_signature") or task.get("starter_code") or ""
        function_name = parse_function_name(signature)
        if not function_name:
            return {"tests": [], "hasError": True, "error": "function_not_found"}
        param_count = _param_count(signature)
        runner_tests = _prepare_runner_tests(public_tests, param_count)
        run_result = None
        last_error = None
        chosen_lang = payload.language
        for lang in _candidate_languages(task, payload.language):
            try:
                tmp_res = run_code_with_fallback(lang, payload.code, function_name, runner_tests)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue
            has_error, err_msg = _runner_has_error(tmp_res)
            if not has_error:
                run_result = tmp_res
                chosen_lang = lang
                break
            last_error = err_msg
            run_result = tmp_res
        if run_result is None:
            hint = explain_code_error(payload.language, signature, payload.code, last_error or "runtime_error") if signature else None
            if hint:
                _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
            conn.commit()
            return {"tests": [], "hasError": True, "error": last_error or "runtime_error"}
        has_error, err_msg = _runner_has_error(run_result)
        if has_error:
            hint = explain_code_error(chosen_lang, signature, payload.code, err_msg) if signature else None
            if hint:
                _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
            conn.commit()
            return {"tests": _tests_from_runner(run_result), "hasError": True, "error": err_msg or "runtime_error"}
        return {"tests": _tests_from_runner(run_result), "hasError": False, "error": None}
    finally:
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()


@app.post("/api/code/check")
def code_check(payload: CheckCodePayload):
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    cur = conn.cursor()
    try:
        session_row = fetchone_dict(cur.execute("SELECT * FROM sessions WHERE id=?", (payload.sessionId,)))
        if not session_row:
            raise HTTPException(status_code=404, detail="session_not_found")
        task_id = _resolve_task_id(cur, payload.sessionId, payload.questionId, payload.taskId)
        task, public_tests, hidden_tests = _load_task_and_tests(cur, task_id, payload.sessionId, payload.questionId)
        signature = task.get("function_signature") or task.get("starter_code") or ""
        function_name = parse_function_name(signature)
        if not function_name:
            raise HTTPException(status_code=400, detail="function_not_found")
        param_count = _param_count(signature)
        runner_public = _prepare_runner_tests(public_tests, param_count)
        lang_candidates = _candidate_languages(task, payload.language)
        hints_used = _get_hints_used(cur, payload.sessionId, payload.questionId)
        base_max = _base_max_score(task)
        effective_max = max(base_max - hints_used * 2, 0)
        cur.execute(
            """
            SELECT COALESCE(MAX(attempt_number), 0) AS max_attempt
            FROM code_attempts
            WHERE session_id=? AND question_id=? AND task_id=? AND owner_id=?
            """,
            (payload.sessionId, payload.questionId, task_id, payload.ownerId),
        )
        row = fetchone_dict(cur)
        attempt = (row.get("max_attempt") or 0) + 1

        finished = False
        next_question = None
        response: dict = {}
        public_result = None
        chosen_lang = lang_candidates[0]
        last_err = None
        public_tests_resp: list[dict] = []
        for lang in lang_candidates:
            try:
                tmp = run_code_with_fallback(lang, payload.code, function_name, runner_public)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                continue
            has_error_public, err_msg = _runner_has_error(tmp)
            public_tests_resp = _tests_from_runner(tmp)
            if not has_error_public:
                public_result = tmp
                chosen_lang = lang
                break
            last_err = err_msg
            public_result = tmp
        if public_result is None:
            _save_attempt(cur, payload, attempt, payload.code, 0, 0, 0, task_id)
            hint = explain_code_error(payload.language, signature, payload.code, last_err or "runtime_error") if signature else None
            if hint:
                _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
            response = {
                "solved": False,
                "attempt": attempt,
                "score": 0,
                "maxScore": effective_max,
                "publicTests": [{"name": "runtime", "status": "error", "expected": None, "actual": last_err or "error"}],
                "hiddenPassed": False,
                "hasError": True,
                "error": "docker_unavailable",
                "hintsUsed": hints_used,
            }
        else:
            has_error_public, err_msg = _runner_has_error(public_result)
            if has_error_public:
                _save_attempt(cur, payload, attempt, payload.code, 0, 0, 0, task_id)
                hint = explain_code_error(chosen_lang, signature, payload.code, err_msg) if signature else None
                if hint:
                    _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
                response = {
                    "solved": False,
                    "attempt": attempt,
                    "score": 0,
                    "maxScore": effective_max,
                    "publicTests": [{"name": "runtime", "status": "error", "expected": None, "actual": err_msg or "error"}],
                    "hiddenPassed": False,
                    "hasError": True,
                    "error": err_msg or "runtime_error",
                    "hintsUsed": hints_used,
                }
            else:
                public_all_passed = all(t.get("status") == "passed" for t in public_tests_resp)
                if not public_all_passed:
                    _save_attempt(cur, payload, attempt, payload.code, 0, 0, 0, task_id)
                    response = {
                        "solved": False,
                        "attempt": attempt,
                        "score": 0,
                        "maxScore": effective_max,
                        "publicTests": public_tests_resp,
                        "hiddenPassed": False,
                        "hasError": False,
                        "hintsUsed": hints_used,
                    }
                else:
                    runner_hidden = _prepare_runner_tests(hidden_tests, param_count)
                    hidden_result = None
                    last_hidden_err = None
                    for lang in lang_candidates:
                        try:
                            tmp_hid = run_code_with_fallback(lang, payload.code, function_name, runner_hidden)
                        except Exception as exc:  # noqa: BLE001
                            last_hidden_err = str(exc)
                            continue
                        has_error_hidden, err_h = _runner_has_error(tmp_hid)
                        hidden_tests_resp = _tests_from_runner(tmp_hid)
                        if not has_error_hidden:
                            hidden_result = tmp_hid
                            chosen_lang = lang
                            break
                        last_hidden_err = err_h
                        hidden_result = tmp_hid
                    if hidden_result is None:
                        _save_attempt(cur, payload, attempt, payload.code, 1, 0, 0, task_id)
                        hint = explain_code_error(payload.language, signature, payload.code, last_hidden_err or "runtime_error") if signature else None
                        if hint:
                            _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
                        response = {
                            "solved": False,
                            "attempt": attempt,
                            "score": 0,
                            "maxScore": effective_max,
                            "publicTests": public_tests_resp,
                            "hiddenPassed": False,
                            "hasError": True,
                            "error": last_hidden_err or "runtime_error",
                            "hintsUsed": hints_used,
                        }
                    else:
                        has_error_hidden, _ = _runner_has_error(hidden_result)
                        hidden_tests_resp = _tests_from_runner(hidden_result)
                        hidden_passed = not has_error_hidden and all(t.get("status") == "passed" for t in hidden_tests_resp)
                        if not hidden_passed:
                            _save_attempt(cur, payload, attempt, payload.code, 1, 0, 0, task_id)
                            hint = explain_code_error(chosen_lang, signature, payload.code, "hidden tests failed") if signature else None
                            if hint:
                                _save_error_hint(cur, payload.sessionId, payload.questionId, payload.ownerId, hint)
                            response = {
                                "solved": False,
                                "attempt": attempt,
                                "score": 0,
                                "maxScore": effective_max,
                                "publicTests": public_tests_resp,
                                "hiddenPassed": False,
                                "hasError": has_error_hidden,
                                "error": "runtime_error" if has_error_hidden else None,
                                "hintsUsed": hints_used,
                            }
                        else:
                            score = score_for_attempt(attempt, effective_max)
                            _save_attempt(cur, payload, attempt, payload.code, 1, 1, score, task_id)
                            response = {
                                "solved": True,
                                "attempt": attempt,
                                "score": score,
                                "maxScore": effective_max,
                                "publicTests": public_tests_resp,
                                "hiddenPassed": True,
                                "hasError": False,
                                "error": None,
                                "hintsUsed": hints_used,
                            }
                            finished = True

        # Если три попытки — завершаем даже при неудаче
        if not finished and attempt >= 3:
            finished = True

        # Помечаем вопрос завершённым и генерируем следующий, если нужно
        if finished:
            cur.execute(
                "UPDATE session_questions SET status='done', is_finished=1 WHERE sessionId=? AND questionId=?",
                (payload.sessionId, payload.questionId),
            )
            try:
                track = (session_row.get("direction") or "fullstack").lower()
                level = (session_row.get("level") or "middle").lower()
                lang = (session_row.get("language") or task.get("language") or "python").lower()
                category = choose_next_category(cur, payload.sessionId, track)
                if category:
                    next_q_obj = generate_unique_code_task(cur, payload.sessionId, track, level, category, lang)
                    next_code_task_id = next_q_obj.get("task_id") or f"code-{uuid.uuid4()}"
                    public_tests2, hidden_tests2 = build_tests_for_task(next_q_obj)
                    save_code_task_and_tests(cur, next_code_task_id, next_q_obj, public_tests2, hidden_tests2)
                    next_question_id = f"llm-code-{uuid.uuid4()}"
                    pos = cur.execute("SELECT COUNT(*) FROM session_questions WHERE sessionId=?", (payload.sessionId,)).fetchone()[0] + 1
                    starter_next = sanitize_starter_code(next_q_obj)
                    meta_json = json.dumps({**next_q_obj, "task_id": next_code_task_id, "starter_code": starter_next}, ensure_ascii=False)
                    # Обновляем сессию на новый текущий вопрос
                    cur.execute(
                        """
                        UPDATE sessions
                        SET questionId=?, questionTitle=?, description=?, starterCode=?, useIDE=1
                        WHERE id=?
                        """,
                        (
                            next_question_id,
                            next_q_obj.get("title") or f"Вопрос {pos}",
                            next_q_obj.get("description_markdown"),
                            starter_next,
                            payload.sessionId,
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type, code_task_id, category, status, is_finished)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
                        """,
                        (
                            payload.sessionId,
                            next_question_id,
                            next_q_obj.get("title") or f"Вопрос {pos}",
                            pos,
                            meta_json,
                            "coding",
                            next_code_task_id,
                            category,
                        ),
                    )
                    next_question = {
                        "id": next_question_id,
                        "title": next_q_obj.get("title"),
                        "description": next_q_obj.get("description_markdown"),
                        "starterCode": starter_next,
                        "functionSignature": next_q_obj.get("function_signature"),
                        "codeTaskId": next_code_task_id,
                        "qType": "coding",
                        "category": category,
                        "position": pos,
                    }
            except Exception as exc:  # noqa: BLE001
                logger.exception("failed to generate next coding question", extra={"session": payload.sessionId, "question": payload.questionId, "task_id": task_id})
        conn.commit()
        response["finished"] = finished
        response["nextQuestion"] = next_question
        return response
    finally:
        conn.close()


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
    q_obj: dict = {}
    code_task_id = None
    function_signature = None

    task_types = normalize_task_types((assigned.get("tasks") or "").split(",") if assigned.get("tasks") else [])

    # Сначала пробуем coding, если выбран
    if "coding" in task_types:
        try:
            q_obj = generate_unique_code_task(cur, assigned.get("sessionId") or session_id, (assigned.get("direction") or "fullstack").lower(), (assigned.get("level") or "middle").lower(), "algo", "python")
            code_task_id = q_obj.get("task_id") or f"code-{uuid.uuid4()}"
            public_tests, hidden_tests = build_tests_for_task(q_obj)
            save_code_task_and_tests(cur, code_task_id, q_obj, public_tests, hidden_tests)
            question_id = f"llm-code-{uuid.uuid4()}"
            question_title = q_obj.get("title") or question_title
            description = q_obj.get("description_markdown", "")
            question_type = "coding"
            use_ide = True
            starter = q_obj.get("starter_code", "")
            function_signature = q_obj.get("function_signature")
            meta_json = json.dumps({**q_obj, "task_id": code_task_id}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("LLM coding generation failed on assigned-interview start", extra={"direction": assigned.get("direction"), "level": assigned.get("level")})
            question_id = None

    # Пытаемся сгенерировать теоретический вопрос для назначенного интервью
    if not question_id and "theory" in task_types:
        try:
            prev_topics = collect_previous_theory_topics(cur, assigned.get("sessionId") or "")
            q_obj = generate_theory_question((assigned.get("direction") or "fullstack").lower(), (assigned.get("level") or "middle").lower(), prev_topics)
            question_id = f"llm-theory-{uuid.uuid4()}"
            description = q_obj.get("question", "")
            question_title = q_obj.get("title") or question_title
            question_type = "theory"
            use_ide = False
            starter = ""
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
    total_questions = 9 if dual_mode else 1
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
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type, code_task_id, category, status, is_finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
        """,
        (
            session_id,
            question_id,
            question_title,
            1,
            meta_json,
            question_type or "coding",
            code_task_id,
            code_task_id and (q_obj.get("category") if question_type == "coding" else None),
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
            "tasks": task_types,
            "questionTitle": question_title,
            "questionId": question_id,
            "useIDE": bool(use_ide),
            "functionSignature": function_signature,
            "codeTaskId": code_task_id,
            "language": (meta_json and json.loads(meta_json).get("language")) if meta_json else None,
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
            "ownerId": payload.userId,
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
        cur.execute(
            """
            SELECT
              questionId as id,
              questionTitle as title,
              q_type as qType,
              code_task_id as codeTaskId
            FROM session_questions
            WHERE sessionId=?
            ORDER BY position
            """,
            (row["id"],),
        )
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
            """
            SELECT
              questionId as id,
              questionTitle as title,
              q_type as qType,
              code_task_id as codeTaskId,
              position,
              category,
              status,
              is_finished as isFinished
            FROM session_questions
            WHERE sessionId=?
            ORDER BY position
            """,
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
    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
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
    except sqlite3.OperationalError as exc:  # noqa: BLE001
        logger.warning("anticheat event skipped due to db lock", extra={"err": str(exc)})
        return {"status": "ignored"}


@app.post("/api/interview/finish")
def finish_interview(payload: FinishInterviewPayload):
    last_err = None
    for _ in range(3):
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
            conn.execute("PRAGMA busy_timeout=5000")
            cur = conn.cursor()
            cur.execute("SELECT * FROM sessions WHERE id=?", (payload.sessionId,))
            session = fetchone_dict(cur)
            if not session:
                return {"status": "missing"}
            cur.execute("UPDATE sessions SET status='completed', is_active=0, is_finished=1 WHERE id=?", (payload.sessionId,))
            finished_at = now_iso()
            total_score = calculate_session_score(cur, payload.sessionId)
            cur.execute(
                "INSERT OR REPLACE INTO interview_results (id, sessionId, ownerId, status, finishedAt, score) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), payload.sessionId, payload.ownerId, "completed", finished_at, total_score),
            )
            if session.get("ownerId"):
                cur.execute("DELETE FROM support_messages WHERE userId=?", (session.get("ownerId"),))
                cur.execute("DELETE FROM support_dialogs WHERE userId=?", (session.get("ownerId"),))
            if session.get("assignedId"):
                cur.execute(
                    "UPDATE assigned_interviews SET status='completed', sessionId=? WHERE id=?",
                    (payload.sessionId, session.get("assignedId")),
                )
            return {"status": "ok", "score": total_score}
        except sqlite3.OperationalError as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.3)
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    logger.warning("finish_interview skipped due to db lock", extra={"err": str(last_err)})
    raise HTTPException(status_code=500, detail="db_locked")


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
    uvicorn.run(
        "server_py.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        log_config=uvicorn.config.LOGGING_CONFIG,  # вернём стандартный uvicorn-логгер
    )
