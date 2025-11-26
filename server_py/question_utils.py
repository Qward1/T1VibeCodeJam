from typing import List, Optional
import json

try:
    from db import fetchone_dict
except Exception:
    from server_py.db import fetchone_dict  # type: ignore
import uuid


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
    cur.execute("DELETE FROM code_tests WHERE task_id=?", (task_id,))
    cur.execute(
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
        cur.execute(
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
        cur.execute(
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
