from typing import List, Optional
import json
import sqlite3
import time


def _exec_retry(cur, sql: str, params: tuple = (), attempts: int = 4, delay: float = 0.2):
    for attempt in range(attempts):
        try:
            return cur.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise

try:
    from db import fetchone_dict
except Exception:
    from server_py.db import fetchone_dict  # type: ignore
import uuid

LEVEL_ORDER = ["intern", "junior", "middle", "middle+", "senior", "senior+"]
LEVEL_MAX_SCORE = {"intern": 5, "junior": 8, "middle": 10, "middle+": 15, "senior": 20, "senior+": 25}


def _level_idx(name: Optional[str]) -> int:
    if not name:
        return 2
    s = (name or "").strip().lower()
    if s in LEVEL_ORDER:
        return LEVEL_ORDER.index(s)
    if s in {"junior+", "junior plus"}:
        return 1
    if s in {"middleplus", "middle plus"}:
        return 3
    if s in {"seniorplus", "senior plus"}:
        return 5
    return 2


def _level_from_idx(idx: int) -> str:
    idx = max(0, min(idx, len(LEVEL_ORDER) - 1))
    return LEVEL_ORDER[idx]


def normalize_level(level: Optional[str], default: str = "middle") -> str:
    if not level:
        return default
    return LEVEL_ORDER[_level_idx(level)]


def compute_adaptive_level(session_row: dict) -> tuple[str, int]:
    current = normalize_level(
        session_row.get("adaptive_level")
        or session_row.get("level")
        or session_row.get("initial_level")
        or "middle"
    )
    max_val = session_row.get("adaptive_max_score")
    try:
        max_int = int(max_val) if max_val is not None else None
    except Exception:
        max_int = None
    return current, max_int or LEVEL_MAX_SCORE.get(current, 10)


def apply_adaptive_outcome(cur, session_row: dict, outcome: str, fast_success: bool = False) -> tuple[str, int]:
    """
    outcome: 'success' | 'fail'
    fast_success=True значит, что ответ был верный с первой попытки без подсказок.
    """
    initial = normalize_level(session_row.get("initial_level") or session_row.get("level") or "middle")
    current = normalize_level(session_row.get("adaptive_level") or session_row.get("level") or initial)
    cur_idx = _level_idx(current)
    init_idx = _level_idx(initial)
    floor_idx = max(0, init_idx - 1)
    if outcome == "fail":
        new_idx = max(floor_idx, cur_idx - 1)
    else:
        new_idx = cur_idx
        if fast_success:
            new_idx = min(len(LEVEL_ORDER) - 1, cur_idx + 1)
    new_level = _level_from_idx(new_idx)
    new_max = LEVEL_MAX_SCORE.get(new_level, LEVEL_MAX_SCORE.get(initial, 10))
    cur.execute(
        "UPDATE sessions SET adaptive_level=?, adaptive_max_score=? WHERE id=?",
        (new_level, new_max, session_row["id"]),
    )
    session_row["adaptive_level"] = new_level
    session_row["adaptive_max_score"] = new_max
    return new_level, new_max


def infer_last_finished_outcome(cur, session_id: str):
    """
    Возвращает (outcome, fast_success, question_id) для последнего завершённого вопроса,
    по которому ещё не применялся адаптив (adaptive_applied=0).
    """
    row = fetchone_dict(
        cur.execute(
            """
            SELECT questionId, q_type, hints_used
            FROM session_questions
            WHERE sessionId=? AND is_finished=1 AND COALESCE(adaptive_applied, 0)=0
            ORDER BY position DESC
            LIMIT 1
            """,
            (session_id,),
        )
    )
    if not row:
        return None
    qid = row.get("questionId")
    q_type = (row.get("q_type") or "coding").lower()
    hints_used = int(row.get("hints_used") or 0)
    if q_type == "coding":
        stats = fetchone_dict(
            cur.execute(
                """
                SELECT
                  MAX(attempt_number) AS attempts,
                  MAX(passed_hidden) AS passed_hidden,
                  MIN(CASE WHEN passed_hidden=1 THEN attempt_number END) AS first_success_attempt
                FROM code_attempts
                WHERE session_id=? AND question_id=?
                """,
                (session_id, qid),
            )
        )
        if not stats:
            return None
        passed = bool(stats.get("passed_hidden"))
        first_success = stats.get("first_success_attempt") or None
        fast = passed and first_success == 1 and hints_used == 0
        return ("success" if passed else "fail", fast, qid)
    ans = fetchone_dict(
        cur.execute(
            """
            SELECT decision, score, max_score
            FROM answers
            WHERE sessionId=? AND questionId=?
            ORDER BY datetime(updatedAt) DESC
            LIMIT 1
            """,
            (session_id, qid),
        )
    )
    if not ans:
        return None
    decision = (ans.get("decision") or "").lower()
    if decision == "wrong":
        return ("fail", False, qid)
    if decision == "correct":
        score = ans.get("score") or 0
        max_score = ans.get("max_score") or 0
        fast = bool(max_score and score >= max_score)
        return ("success", fast, qid)
    return None


def apply_adaptive_from_history(cur, session_row: dict) -> tuple[str, int]:
    """
    Смотрит на последний завершённый вопрос без adaptive_applied,
    применяет повышение/понижение сложности и помечает вопрос.
    """
    inferred = infer_last_finished_outcome(cur, session_row["id"])
    if not inferred:
        return compute_adaptive_level(session_row)
    outcome, fast_success, qid = inferred
    new_level, new_max = apply_adaptive_outcome(cur, session_row, outcome, fast_success=fast_success)
    cur.execute(
        "UPDATE session_questions SET adaptive_applied=1 WHERE sessionId=? AND questionId=?",
        (session_row["id"], qid),
    )
    return new_level, new_max


def normalize_task_types(tasks: List[str]) -> List[str]:
    """Нормализует типы задач из UI в значения для БД/логики."""
    seen = set()
    result = []
    for t in tasks or []:
        key = (t or "").strip().lower()
        if key in ["theory", "theoretical", "theoretical questions"]:
            key = "theory"
        else:
            key = "coding"
        if key not in seen:
            seen.add(key)
            result.append(key)
    if not result:
        result = ["coding"]
    return result


def pick_question(cur, direction: Optional[str], difficulty: Optional[str], types: Optional[List[str]], exclude_ids: List[str]):
    """
    Возвращает случайный вопрос из таблицы questions с учётом фильтров.
    """
    filters = []
    params = []
    if direction:
        filters.append("direction = ?")
        params.append(direction)
    if difficulty:
        filters.append("difficulty = ?")
        params.append(difficulty)
    if types:
        types_norm = [t.lower() for t in types]
        placeholders = ",".join(["?"] * len(types_norm))
        filters.append(f"type IN ({placeholders})")
        params.extend(types_norm)
    if exclude_ids:
        placeholders = ",".join(["?"] * len(exclude_ids))
        filters.append(f"id NOT IN ({placeholders})")
        params.extend(exclude_ids)
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    query = f"SELECT * FROM questions {where_clause} ORDER BY RANDOM() LIMIT 1"
    cur.execute(query, tuple(params))
    return fetchone_dict(cur)


def collect_previous_theory_topics(cur, session_id: str) -> List[str]:
    """
    Собирает темы/заголовки уже выданных теоретических вопросов в сессии,
    чтобы не повторять их при генерации.
    """
    rows = cur.execute(
        "SELECT questionTitle, meta_json FROM session_questions WHERE sessionId=? AND q_type='theory'",
        (session_id,),
    ).fetchall()
    topics: List[str] = []
    for title, meta in rows:
        if title:
            topics.append(str(title))
        if meta:
            try:
                obj = json.loads(meta)
                for key in ["topic", "title", "question"]:
                    if obj.get(key):
                        topics.append(str(obj[key]))
                        break
            except Exception:
                continue
    # Удаляем дубликаты, сохраняя порядок
    seen = set()
    uniq = []
    for t in topics:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def collect_previous_code_topics(cur, session_id: str, track: str):
    """
    Собирает темы уже выданных кодовых задач для сессии и трека.
    Возвращает (algo_topics, domain_topics) списками строк.
    """
    rows = cur.execute(
        """
        SELECT sq.category, sq.meta_json, sq.questionTitle, ct.topic, ct.title as ct_title, ct.track
        FROM session_questions sq
        LEFT JOIN code_tasks ct ON ct.task_id = sq.code_task_id
        WHERE sq.sessionId=? AND sq.q_type='coding'
        """,
        (session_id,),
    ).fetchall()
    algo = []
    domain = []
    for cat, meta, qtitle, ctopic, ctitle, ctrack in rows:
        # Если track не задан у задачи — считаем, что она относится к текущему треку
        if ctrack and str(ctrack).lower() != str(track).lower():
            continue
        topic_val = ctopic or qtitle or ctitle
        if not topic_val and meta:
            try:
                obj = json.loads(meta)
                topic_val = obj.get("topic") or obj.get("title")
            except Exception:
                topic_val = None
        if not topic_val:
            continue
        if (cat or "").lower() == "domain":
            domain.append(str(topic_val))
        else:
            algo.append(str(topic_val))
    # дедубликация с сохранением порядка
    def dedup(items):
        seen = set()
        res = []
        for t in items:
            if t in seen:
                continue
            seen.add(t)
            res.append(t)
        return res
    return dedup(algo), dedup(domain)



def save_code_task_and_tests(cur, task_id: str, task: dict, public_tests: List[dict], hidden_tests: List[dict]):
    """Сохраняет кодовую задачу и тесты в БД."""
    # очищаем старые тесты для этого task_id, чтобы не смешивались с прошлых сессий
    _exec_retry(cur, "DELETE FROM code_tests WHERE task_id=?", (task_id,))
    _exec_retry(
        """
        INSERT OR REPLACE INTO code_tasks (task_id, track, level, category, language, allowed_languages_json, title, description_markdown, function_signature, starter_code, constraints_json, reference_solution, topic, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            task.get("track"),
            task.get("level"),
            task.get("category"),
            task.get("language"),
            json.dumps(task.get("allowed_languages") or [], ensure_ascii=False),
            task.get("title"),
            task.get("description_markdown"),
            task.get("function_signature"),
            task.get("starter_code"),
            json.dumps(task.get("constraints") or [], ensure_ascii=False),
            task.get("reference_solution"),
            task.get("topic"),
            json.dumps(task, ensure_ascii=False),
        ),
    )
    for t in public_tests:
        _exec_retry(
            """
            INSERT INTO code_tests (task_id, name, is_public, input_json, expected_json)
            VALUES (?, ?, 1, ?, ?)
            """,
            (
                task_id,
                t.get("name"),
                json.dumps(t.get("input"), ensure_ascii=False),
                json.dumps(t.get("expected"), ensure_ascii=False),
            ),
        )
    for t in hidden_tests:
        _exec_retry(
            """
            INSERT INTO code_tests (task_id, name, is_public, input_json, expected_json)
            VALUES (?, ?, 0, ?, ?)
            """,
            (
                task_id,
                t.get("name"),
                json.dumps(t.get("input"), ensure_ascii=False),
                json.dumps(t.get("expected"), ensure_ascii=False),
            ),
        )
