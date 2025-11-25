from fastapi import APIRouter, HTTPException
import sqlite3
import uuid
from db import DB_PATH, fetchone_dict, fetchall_dicts, seed_questions
from datetime import datetime, timezone

def now_iso():
    return datetime.now(timezone.utc).isoformat()

router = APIRouter()

@router.post("/api/interview/next")
def next_question(payload: dict):
    sessionId = payload.get("sessionId")
    ownerId = payload.get("ownerId")
    if not sessionId or not ownerId:
        raise HTTPException(status_code=400, detail="invalid_payload")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Ищем сессию по id; если нет — пытаемся взять последнюю активную для пользователя
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
    issued_rows = fetchall_dicts(cur.execute("SELECT questionId as id FROM session_questions WHERE sessionId=?", (sessionId,)))
    issued = [r["id"] for r in issued_rows]
    try:
        limit = int(session.get("total") or 3)
    except Exception:
        limit = 3
    if len(issued) >= limit:
        conn.close()
        raise HTTPException(status_code=400, detail="limit_reached")
    placeholders = ",".join(["?"] * len(issued)) if issued else ""
    question = None
    if issued:
        cur.execute(f"SELECT * FROM questions WHERE id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT 1", issued)
        question = fetchone_dict(cur)
    if not question:
        cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
        question = fetchone_dict(cur)
    if not question:
        seed_questions(cur, force=True)
        cur.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
        question = fetchone_dict(cur)
    if not question:
        conn.close()
        raise HTTPException(status_code=404, detail="no_questions")
    # Обновляем текущую сессию новым вопросом
    cur.execute(
        "UPDATE sessions SET questionId=?, questionTitle=?, description=?, useIDE=? WHERE id=?",
        (question["id"], question["title"], question["body"], int(question.get("useIDE", 1)), sessionId),
    )
    # Добавляем в историю
    pos = cur.execute("SELECT COUNT(*) FROM session_questions WHERE sessionId=?", (sessionId,)).fetchone()[0] + 1
    cur.execute(
        "INSERT INTO session_questions (sessionId, questionId, questionTitle, position) VALUES (?, ?, ?, ?)",
        (sessionId, question["id"], question["title"], pos),
    )
    conn.commit()
    used_questions = fetchall_dicts(
        cur.execute("SELECT questionId as id, questionTitle as title FROM session_questions WHERE sessionId=? ORDER BY position", (sessionId,))
    )
    conn.close()
    return {
        "session": {
            "id": sessionId,
            "questionId": question["id"],
            "questionTitle": question["title"],
            "useIDE": bool(question.get("useIDE", 1)),
            "description": question["body"],
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
