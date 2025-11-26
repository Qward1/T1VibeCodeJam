from typing import List, Optional
import json

try:
    from db import fetchone_dict
except Exception:
    from server_py.db import fetchone_dict  # type: ignore


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
