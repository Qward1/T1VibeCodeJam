import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any
from db import DB_PATH, fetchall_dicts, fetchone_dict, connect as db_connect


def recalculate_session_metrics(session_id: str) -> Dict[str, Any]:
    conn = db_connect()
    cur = conn.cursor()
    # Быстрая миграция на случай старой схемы
    try:
        cols = [c[1] for c in cur.execute("PRAGMA table_info(interview_results)").fetchall()]
        if "finishedAt" not in cols:
            cur.execute("ALTER TABLE interview_results ADD COLUMN finishedAt TEXT DEFAULT CURRENT_TIMESTAMP")
        if "metrics_json" not in cols:
            cur.execute("ALTER TABLE interview_results ADD COLUMN metrics_json TEXT")
        s_cols = [c[1] for c in cur.execute("PRAGMA table_info(sessions)").fetchall()]
        if "progress_percent" not in s_cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN progress_percent INTEGER DEFAULT 0")
    except Exception:
        pass
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    session = fetchone_dict(cur)
    if not session:
        conn.close()
        return {}

    # Answers
    answers = fetchall_dicts(cur.execute("SELECT * FROM answers WHERE sessionId=?", (session_id,)))
    solved_hidden = sum(1 for a in answers if a.get("passed_hidden"))
    attempts = sum((a.get("attempt_number") or 0) for a in answers)
    durations = [a.get("duration_ms") for a in answers if a.get("duration_ms")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Events
    events = fetchall_dicts(cur.execute("SELECT * FROM interview_events WHERE sessionId=? OR session_id=?", (session_id, session_id)))
    code_runs = sum(1 for e in events if e.get("event_type") in ["code_run", "code_submit"])
    chats = sum(1 for e in events if e.get("event_type") in ["chat_user", "chat_assistant"])
    cheat_score = sum(float(e.get("risk_level") == "high") for e in events if e.get("event_type", "").startswith("anti") or "cheat" in (e.get("event_type") or ""))

    # LLM metrics
    cur.execute("SELECT finishedAt, metrics_json FROM interview_results WHERE sessionId=?", (session_id,))
    res_row = fetchone_dict(cur)
    metrics = {}
    if res_row and res_row.get("metrics_json"):
        try:
            metrics = json.loads(res_row["metrics_json"])
        except Exception:
            metrics = {}
    scores = metrics.get("scores", {}) if isinstance(metrics, dict) else {}
    score_quality = scores.get("correctness") or 0
    score_complexity = scores.get("complexity") or 0
    score_readability = scores.get("readability") or 0
    score_edge = scores.get("edge_cases") or 0

    total_tasks = session.get("total") or len(answers) or 1
    progress_percent = int((solved_hidden / total_tasks) * 100)
    status = session.get("status") or "active"
    if solved_hidden >= total_tasks:
        status = "finished"
    # Обновляем interview_results агрегаты
    finished_at = (res_row or {}).get("finishedAt")
    if not finished_at and status == "finished":
        finished_at = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT OR REPLACE INTO interview_results (id, sessionId, ownerId, status, finishedAt, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            session_id,
            session.get("ownerId"),
            status,
            finished_at or datetime.now(timezone.utc).isoformat(),
            json.dumps(metrics or {}),
        ),
    )
    # Обновляем сессию
    cur.execute(
        "UPDATE sessions SET status=?, progress_percent=? WHERE id=?",
        (status, progress_percent, session_id),
    )
    conn.commit()
    conn.close()
    return {
        "solved_hidden": solved_hidden,
        "attempts": attempts,
        "avg_duration_ms": avg_duration,
        "code_runs": code_runs,
        "chats": chats,
        "score_quality": score_quality,
        "score_complexity": score_complexity,
        "score_readability": score_readability,
        "score_edge_cases": score_edge,
        "cheat_score": cheat_score,
        "progress_percent": progress_percent,
        "status": status,
    }
