from fastapi import APIRouter, HTTPException
import sqlite3
import uuid
import json
from datetime import datetime, timezone
import logging

from db import DB_PATH, fetchone_dict, fetchall_dicts
from llm_theory import generate_theory_question
from question_utils import pick_question, normalize_task_types, collect_previous_theory_topics


def now_iso():
    return datetime.now(timezone.utc).isoformat()


router = APIRouter()
logger = logging.getLogger(__name__)


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
    # Если лимит исчерпан, увеличиваем total и продолжаем, чтобы сформировать следующий вопрос
    if len(issued_rows) >= limit:
        limit = len(issued_rows) + 1
        try:
            cur.execute("UPDATE sessions SET total=? WHERE id=?", (limit, sessionId))
            conn.commit()
        except Exception:
            pass

    last_type = issued_rows[-1].get("q_type") if issued_rows else None
    if "theory" in task_types and "coding" in task_types:
        desired_type = "theory" if last_type != "theory" else "coding"
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

    if desired_type == "coding":
        question = pick_question(cur, session.get("direction"), session.get("level"), ["coding"], used_ids)
        if question:
            question_id = question["id"]
            question_title = question["title"]
            description = question["statement"]
            use_ide = bool(question.get("useIDE", 1))
    if desired_type == "theory" or (not question and "theory" in task_types):
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

    # Обновляем текущую сессию новым вопросом
    cur.execute(
        "UPDATE sessions SET questionId=?, questionTitle=?, description=?, useIDE=? WHERE id=?",
        (question_id, question_title, description, int(use_ide), sessionId),
    )
    # Добавляем в историю
    pos = cur.execute("SELECT COUNT(*) FROM session_questions WHERE sessionId=?", (sessionId,)).fetchone()[0] + 1
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, questionId, questionTitle, position, meta_json, q_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (sessionId, question_id, question_title, pos, meta_json, desired_type),
    )
    conn.commit()
    used_questions = fetchall_dicts(
        cur.execute("SELECT questionId as id, questionTitle as title, q_type FROM session_questions WHERE sessionId=? ORDER BY position", (sessionId,))
    )
    conn.close()
    return {
        "session": {
            "id": sessionId,
            "questionId": question_id,
            "questionTitle": question_title,
            "useIDE": bool(use_ide),
            "description": description,
            "starterCode": session.get("starterCode"),
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
