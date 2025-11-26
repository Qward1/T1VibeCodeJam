from fastapi import APIRouter, HTTPException
import sqlite3
import uuid
import json
from datetime import datetime, timezone
import logging

from db import DB_PATH, fetchone_dict, fetchall_dicts
from llm_theory import generate_theory_question
from question_utils import pick_question, normalize_task_types, collect_previous_theory_topics, save_code_task_and_tests
from llm_code import generate_code_task, build_tests_for_task, _fallback_code_task
import uuid as _uuid


def now_iso():
    return datetime.now(timezone.utc).isoformat()


router = APIRouter()
logger = logging.getLogger(__name__)


def sanitize_starter_code(task: dict) -> str:
    signature = task.get("function_signature") or ""
    sig = signature.strip()
    if not sig:
        return ""
    if not sig.endswith(":"):
        sig = sig + ":"
    return f"{sig}\n    # TODO: реализуйте функцию\n    pass\n"


def next_question_legacy(payload: dict):
    """Старый вариант выдачи из БД."""
    sessionId = payload.get("sessionId")
    ownerId = payload.get("ownerId")
    if not sessionId or not ownerId:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (sessionId,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")
    issued_rows = fetchall_dicts(cur.execute("SELECT questionId as id FROM session_questions WHERE sessionId=?", (sessionId,)))
    issued = [r["id"] for r in issued_rows]
    cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
    question = fetchone_dict(cur)
    if not question:
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")
    pos = cur.execute("SELECT COUNT(*) FROM session_questions WHERE sessionId=?", (sessionId,)).fetchone()[0] + 1
    cur.execute(
        "INSERT INTO session_questions (sessionId, questionId, questionTitle, position) VALUES (?, ?, ?, ?)",
        (sessionId, question["id"], question["title"], pos),
    )
    conn.commit()
    conn.close()
    return question


@router.post("/api/interview/next")
def next_question(payload: dict):
    sessionId = payload.get("sessionId")
    ownerId = payload.get("ownerId")
    if not sessionId or not ownerId:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (sessionId,))
    session = fetchone_dict(cur)
    if not session:
        cur.execute(
            "SELECT * FROM sessions WHERE ownerId=? AND status='active' ORDER BY datetime(createdAt) DESC LIMIT 1",
            (ownerId,),
        )
        session = fetchone_dict(cur)
        if session:
            sessionId = session["id"]
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="not_found")

    task_types = normalize_task_types(session.get("tasks", "").split(",") if session.get("tasks") else [])
    issued_rows = fetchall_dicts(
        cur.execute(
            "SELECT questionId as id, questionTitle as title, q_type, meta_json FROM session_questions WHERE sessionId=? ORDER BY position",
            (sessionId,),
        )
    )
    try:
        limit = int(session.get("total") or 1)
    except Exception:
        limit = 1
    # Если выбраны теория+код — целимся в 10 вопросов (6 theory + 4 coding)
    if "theory" in task_types and "coding" in task_types and limit < 10:
        limit = 10
        try:
            cur.execute("UPDATE sessions SET total=? WHERE id=?", (limit, sessionId))
            conn.commit()
        except Exception:
            pass
    # Если лимит исчерпан, увеличиваем total и продолжаем, чтобы сформировать следующий вопрос
    if len(issued_rows) >= limit:
        limit = len(issued_rows) + 1
        try:
            cur.execute("UPDATE sessions SET total=? WHERE id=?", (limit, sessionId))
            conn.commit()
        except Exception:
            pass

    # В режиме теории+кода: сначала 6 теоретических, затем 4 кодовых
    theory_count = sum(1 for r in issued_rows if (r.get("q_type") or "").lower() == "theory")
    coding_count = sum(1 for r in issued_rows if (r.get("q_type") or "").lower() == "coding")
    if "theory" in task_types and "coding" in task_types:
        if theory_count < 6:
            desired_type = "theory"
        elif coding_count < 4:
            desired_type = "coding"
        else:
            conn.close()
            raise HTTPException(status_code=404, detail="no_questions")
    elif "theory" in task_types:
        desired_type = "theory"
    else:
        desired_type = "coding"

    question = None
    question_id = None
    question_title = None
    description = ""
    use_ide = True
    meta_json = None
    used_ids = []
    for r in issued_rows:
        rid = r.get("id")
        if rid is None:
            continue
        # Сохраняем как есть; кодинговые вопросы имеют числовые id, теоретические — строковые
        used_ids.append(rid)

    code_task_id = None
    if desired_type == "coding":
        try:
            # Выбираем категорию с учётом лимита 2 algo + 2 domain
            cat_rows = fetchall_dicts(
                cur.execute(
                    "SELECT category FROM session_questions WHERE sessionId=? AND q_type='coding'",
                    (sessionId,),
                )
            )
            algo_count = sum(1 for r in cat_rows if (r.get("category") or "").lower() == "algo")
            domain_count = sum(1 for r in cat_rows if (r.get("category") or "").lower() == "domain")
            if algo_count >= 2 and domain_count >= 2:
                conn.close()
                raise HTTPException(status_code=404, detail="coding_limit_reached")
            category = "algo" if algo_count < 2 else "domain"
            from question_utils import collect_previous_code_topics  # локальный импорт, чтобы избежать циклов
            prev_algo, prev_domain = collect_previous_code_topics(
                cur, sessionId, (session.get("direction") or "fullstack").lower()
            )
            q_obj = generate_code_task(
                (session.get("direction") or "fullstack").lower(),
                (session.get("level") or "middle").lower(),
                category,
                "python",
                prev_algo,
                prev_domain,
            )
        except Exception as exc:
            logger.exception("LLM generation failed on next_question coding", extra={"direction": session.get("direction"), "level": session.get("level")})
            q_obj = _fallback_code_task(
                (session.get("direction") or "fullstack").lower(),
                (session.get("level") or "middle").lower(),
                category,
                "python",
            )
        try:
            code_task_id = q_obj.get("task_id") or f"code-{_uuid.uuid4()}"
            # Вычисляем expected и сохраняем тесты
            public_tests, hidden_tests = build_tests_for_task(q_obj)
            save_code_task_and_tests(cur, code_task_id, q_obj, public_tests, hidden_tests)
            question_id = f"llm-code-{_uuid.uuid4()}"
            question_title = q_obj.get("title") or "Кодинговая задача"
            description = q_obj.get("description_markdown", "")
            use_ide = True
            starter_code = sanitize_starter_code(q_obj)
            meta_json = json.dumps({**q_obj, "task_id": code_task_id, "starter_code": starter_code}, ensure_ascii=False)
            desired_type = "coding"
            starter_code = starter_code
            function_signature = q_obj.get("function_signature")
        except Exception as exc:
            logger.exception("Failed to build/save coding question", extra={"direction": session.get("direction"), "level": session.get("level")})
            question_id = None
    if desired_type == "theory":
        try:
            prev_topics = collect_previous_theory_topics(cur, sessionId)
            q_obj = generate_theory_question((session.get("direction") or "fullstack").lower(), (session.get("level") or "middle").lower(), prev_topics)
            question_id = f"llm-theory-{uuid.uuid4()}"
            question_title = q_obj.get("title") or (q_obj.get("question") or "Теоретический вопрос")[0:60]
            description = q_obj.get("question", "")
            use_ide = False
            meta_json = json.dumps(q_obj, ensure_ascii=False)
            desired_type = "theory"
        except Exception as exc:
            logger.exception("LLM generation failed on next_question", extra={"direction": session.get("direction"), "level": session.get("level")})
            question_id = None
    if not question_id:
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")

    function_signature = None
    starter_code = None
    if desired_type == "coding" and meta_json:
        try:
            obj = json.loads(meta_json)
            function_signature = obj.get("function_signature")
            starter_code = obj.get("starter_code")
        except Exception:
            pass

    # Обновляем текущую сессию новым вопросом
    cur.execute(
        "UPDATE sessions SET questionId=?, questionTitle=?, description=?, useIDE=? WHERE id=?",
        (question_id, question_title, description, int(desired_type == "coding"), sessionId),
    )
    # Добавляем в историю
    pos = cur.execute("SELECT COUNT(*) FROM session_questions WHERE sessionId=?", (sessionId,)).fetchone()[0] + 1
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type, code_task_id, category, status, is_finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
        """,
        (
            sessionId,
            question_id,
            question_title,
            pos,
            meta_json,
            desired_type,
            code_task_id,
            q_obj.get("category") if desired_type == "coding" else None,
        ),
    )
    conn.commit()
    used_questions = fetchall_dicts(
        cur.execute(
            """
            SELECT questionId as id, questionTitle as title, q_type as qType, code_task_id as codeTaskId
            FROM session_questions WHERE sessionId=?
            ORDER BY position
            """,
            (sessionId,),
        )
    )
    conn.close()
    return {
        "session": {
            "id": sessionId,
            "questionId": question_id,
            "questionTitle": question_title,
            "questionDisplayId": f"Q-{pos}-{question_id[-6:]}" if question_id else None,
            "useIDE": bool(desired_type == "coding"),
            "functionSignature": function_signature,
            "codeTaskId": code_task_id,
            "language": (json.loads(meta_json).get("language") if meta_json else None) or session.get("language"),
            "starterCode": starter_code or session.get("starterCode"),
            "description": description,
            "direction": session.get("direction"),
            "level": session.get("level"),
            "format": session.get("format"),
            "tasks": session.get("tasks", "").split(",") if session.get("tasks") else [],
            "timer": session.get("timer"),
            "startedAt": session.get("startedAt"),
            "solved": session.get("solved"),
            "total": session.get("total"),
            "usedQuestions": used_questions,
        }
    }
