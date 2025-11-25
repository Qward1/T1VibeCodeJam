from fastapi import APIRouter, HTTPException
import sqlite3
import uuid
import json
from datetime import datetime, timezone

from db import DB_PATH, fetchone_dict, fetchall_dicts
from llm_client import get_llm_client


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
    # Ищем сессию
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

    # Ограничение по количеству
    issued_rows = fetchall_dicts(cur.execute("SELECT questionId as id, status FROM session_questions WHERE sessionId=?", (sessionId,)))
    issued = [r["id"] for r in issued_rows]
    limit = int(session.get("total") or 3)
    if len(issued) >= limit:
        conn.close()
        raise HTTPException(status_code=400, detail="limit_reached")

    # Вычисляем таргет-уровень
    current_level = (session.get("current_level") or session.get("level") or "junior").lower()
    level_order = ["junior", "middle", "senior"]

    def promote(level: str) -> str:
        idx = level_order.index(level) if level in level_order else 0
        return level_order[min(idx + 1, len(level_order) - 1)]

    def demote(level: str) -> str:
        idx = level_order.index(level) if level in level_order else 0
        return level_order[max(idx - 1, 0)]

    cur.execute(
        "SELECT * FROM answers WHERE sessionId=? ORDER BY datetime(updatedAt) DESC LIMIT 1",
        (sessionId,),
    )
    last_answer = fetchone_dict(cur)
    target_level = current_level
    if last_answer:
        passed_hidden = bool(last_answer.get("passed_hidden"))
        attempts = last_answer.get("attempt_number") or 0
        ev_rows = fetchall_dicts(
            cur.execute(
                "SELECT event_type FROM interview_events WHERE sessionId=? OR session_id=?",
                (sessionId, sessionId),
            )
        )
        runs = sum(1 for e in ev_rows if e.get("event_type") in ["code_run", "code_submit"])
        hints = sum(1 for e in ev_rows if e.get("event_type") in ["chat_user", "chat_assistant"])
        if passed_hidden and attempts <= 2:
            target_level = promote(current_level)
        elif (not passed_hidden) and (runs > 5 or hints > 3):
            target_level = demote(current_level)
        else:
            target_level = current_level

    # История prev_tasks
    history_rows = fetchall_dicts(
        cur.execute(
            """
            SELECT a.*, q.level as q_level
            FROM answers a
            LEFT JOIN questions q ON q.id = a.questionId
            WHERE a.sessionId=?
            ORDER BY datetime(a.updatedAt) ASC
            """,
            (sessionId,),
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
    client = get_llm_client()
    try:
        raw = client.chat(
            model="qwen3-coder-30b-a3b-instruct-fp8",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            top_p=0.9,
            max_tokens=900,
            stream=False,
        )
    except HTTPException:
        conn.close()
        raise
    except Exception as exc:  # noqa: BLE001
        conn.close()
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

    try:
        data = json.loads(raw)
    except Exception:
        conn.close()
        raise HTTPException(status_code=422, detail="LLM вернул не-JSON")
    required_fields = ["title", "statement", "language", "visible_tests", "hidden_tests", "canonical_solution"]
    for f in required_fields:
        if f not in data:
            conn.close()
            raise HTTPException(status_code=422, detail=f"LLM ответ не содержит {f}")
    if not isinstance(data.get("visible_tests"), list) or not isinstance(data.get("hidden_tests"), list):
        conn.close()
        raise HTTPException(status_code=422, detail="Неверный формат тестов")

    def _norm_tests(tests):
        arr = []
        for t in tests:
            if isinstance(t, dict) and "input" in t and "output" in t:
                arr.append({"input": str(t["input"]), "output": str(t["output"])})
        return arr

    visible_tests = _norm_tests(data.get("visible_tests"))
    hidden_tests = _norm_tests(data.get("hidden_tests"))
    if not visible_tests:
        conn.close()
        raise HTTPException(status_code=422, detail="Видимые тесты пустые")

    question_id = str(uuid.uuid4())
    now = now_iso()
    cur.execute(
        """
        INSERT INTO questions (id, title, body, statement, answer, difficulty, level, language,
                               visible_tests_json, hidden_tests_json, canonical_solution, source, useIDE, createdAt, updatedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question_id,
            data["title"],
            data.get("statement") or data["title"],
            data.get("statement") or data["title"],
            data.get("canonical_solution", ""),
            target_level,
            target_level,
            data.get("language") or payload.get("language") or "python",
            json.dumps(visible_tests, ensure_ascii=False),
            json.dumps(hidden_tests, ensure_ascii=False),
            data.get("canonical_solution", ""),
            "llm",
            1,
            now,
            now,
        ),
    )
    new_index = (session.get("current_question_index") or 0) + 1
    cur.execute(
        "UPDATE sessions SET questionId=?, questionTitle=?, description=?, current_level=?, current_question_index=? WHERE id=?",
        (question_id, data["title"], data.get("statement") or data["title"], target_level, new_index, sessionId),
    )
    cur.execute(
        """
        INSERT INTO session_questions (sessionId, session_id, questionId, questionTitle, position, order_index, status, llm_raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sessionId,
            sessionId,
            question_id,
            data["title"],
            new_index + 1,
            new_index,
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
            sessionId,
            sessionId,
            ownerId,
            "llm_task_generated",
            None,
            json.dumps({"target_level": target_level, "task": data}, ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()

    used_questions = issued + [question_id]
    return {
        "session": {
            "id": sessionId,
            "questionId": question_id,
            "questionTitle": data["title"],
            "useIDE": True,
            "description": data.get("statement") or data["title"],
            "starterCode": "",
            "direction": session.get("direction"),
            "level": session.get("level"),
            "current_level": target_level,
            "format": session.get("format"),
            "tasks": session.get("tasks", "").split(",") if session.get("tasks") else [],
            "timer": session.get("timer"),
            "startedAt": session.get("startedAt") or session.get("started_at"),
            "solved": session.get("solved"),
            "total": session.get("total"),
            "questionTitle": data["title"],
            "questionId": question_id,
            "usedQuestions": [{"id": qid} for qid in used_questions],
            "status": session.get("status", "active"),
            "is_active": session.get("is_active", 1),
            "is_finished": session.get("is_finished", 0),
            "visible_tests": visible_tests,
        }
    }
