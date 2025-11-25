import sqlite3
import json
import math
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
from db import DB_PATH, fetchone_dict, fetchall_dicts
from llm_client import get_llm_client


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ensure_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
          id TEXT PRIMARY KEY,
          entity_type TEXT NOT NULL,
          entity_id TEXT NOT NULL,
          vector TEXT NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id)")
    conn.commit()
    conn.close()


def index_question(question_id: str):
    ensure_table()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM questions WHERE id=?", (question_id,))
    q = fetchone_dict(cur)
    if not q:
        conn.close()
        return
    text = f"{q.get('statement') or q.get('body')}\n{q.get('canonical_solution') or ''}"
    client = get_llm_client()
    vecs = client.embed(text, model="bge-m3")
    if not vecs:
        conn.close()
        return
    vjson = json.dumps(vecs[0])
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (id, entity_type, entity_id, vector, created_at) VALUES (?, ?, ?, ?, ?)",
        (f"q-{question_id}", "question", question_id, vjson, now_iso()),
    )
    conn.commit()
    conn.close()


def index_answer(answer_id: str):
    ensure_table()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM answers WHERE id=?", (answer_id,))
    a = fetchone_dict(cur)
    if not a:
        conn.close()
        return
    text = (a.get("code") or a.get("content") or "")[:6000]
    client = get_llm_client()
    vecs = client.embed(text, model="bge-m3")
    if not vecs:
        conn.close()
        return
    vjson = json.dumps(vecs[0])
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (id, entity_type, entity_id, vector, created_at) VALUES (?, ?, ?, ?, ?)",
        (f"a-{answer_id}", "answer", answer_id, vjson, now_iso()),
    )
    conn.commit()
    conn.close()


def find_similar(entity_type: str, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
    ensure_table()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = fetchall_dicts(cur.execute("SELECT entity_id, vector FROM embeddings WHERE entity_type=?", (entity_type,)))
    conn.close()
    scored = []
    for r in rows:
        try:
            v = json.loads(r["vector"])
            score = _cosine(vector, v)
            scored.append((r["entity_id"], score))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def find_similar_questions(vector: List[float], top_k: int = 5):
    return find_similar("question", vector, top_k)


def find_similar_answers(vector: List[float], top_k: int = 5):
    return find_similar("answer", vector, top_k)


def clean_old_embeddings(days: int = 90):
    ensure_table()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM embeddings WHERE datetime(created_at) < datetime('now', ?)", (f"-{days} days",))
    conn.commit()
    conn.close()
