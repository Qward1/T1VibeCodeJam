import sqlite3
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "api.sqlite"

def sha1(val: str) -> str:
    return hashlib.sha1(val.encode()).hexdigest()


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
    except Exception:
        pass
    return conn


def seed_questions(cur, force: bool = False):
    samples = [
        (
            "q1",
            "Два числа",
            "Даны числа и target. Верните индексы двух чисел, сумма которых равна target. Используйте O(n).",
            "[i, j], где nums[i] + nums[j] === target",
            "Подумайте про хеш-таблицу",
            "Junior",
            "arrays,hashmap",
            1,
        ),
        (
            "q2",
            "Минимальный путь",
            "Найдите кратчайший путь в ориентированном графе без отрицательных весов.",
            "Используйте Дейкстру",
            "Очередь с приоритетом ускорит решение",
            "Middle",
            "graph,dijkstra",
            1,
        ),
        (
            "q3",
            "Дедупликация логов",
            "Есть поток логов. Реализуйте структуру, которая выдаёт уникальные записи за последние 5 минут.",
            "Используйте очередь + хешсет с экспирацией",
            "Подумайте про скользящее окно",
            "Middle",
            "queue,sliding-window",
            0,
        ),
    ]
    if force:
        cur.execute("DELETE FROM questions")
    cur.executemany(
        """
        INSERT OR IGNORE INTO questions (id, title, body, answer, hint, difficulty, tags, useIDE)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        samples,
    )


def ensure_schema():
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          id TEXT PRIMARY KEY,
          email TEXT UNIQUE NOT NULL,
          name TEXT NOT NULL,
          password TEXT NOT NULL,
          level TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'user',
          admin INTEGER NOT NULL DEFAULT 0,
          lang TEXT DEFAULT 'ru'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_events (
          id TEXT PRIMARY KEY,
          sessionId TEXT,
          session_id TEXT,
          ownerId TEXT,
          event_type TEXT NOT NULL,
          payload TEXT,
          payload_json TEXT,
          risk_level TEXT DEFAULT 'low',
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
          ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY,
          ownerId TEXT NOT NULL,
          questionId TEXT,
          questionTitle TEXT,
          useIDE INTEGER DEFAULT 1,
          direction TEXT,
          level TEXT,
          candidate_id TEXT,
          current_level TEXT,
          current_question_index INTEGER DEFAULT 0,
          status TEXT DEFAULT 'active',
          started_at TEXT,
          finished_at TEXT,
          cheat_score REAL,
          metadata_json TEXT,
          format TEXT,
          tasks TEXT,
          description TEXT,
          starterCode TEXT,
          timer INTEGER,
          startedAt TEXT,
          solved INTEGER,
          total INTEGER,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          questionId TEXT,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          score INTEGER,
          level TEXT,
          summary TEXT,
          timeline TEXT,
          solutions TEXT,
          analytics TEXT,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_results (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'completed',
          finishedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          metrics_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          questionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          content TEXT NOT NULL,
          code TEXT,
          language TEXT,
          passed_visible INTEGER,
          passed_hidden INTEGER,
          attempt_number INTEGER,
          duration_ms INTEGER,
          metrics_json TEXT,
          updatedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(sessionId, questionId)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_questions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          sessionId TEXT NOT NULL,
          session_id TEXT,
          questionId TEXT NOT NULL,
          questionTitle TEXT,
          position INTEGER,
          order_index INTEGER,
          status TEXT DEFAULT 'current',
          llm_raw_response TEXT
        )
        """
    )
    sq_cols = [c[1] for c in cur.execute("PRAGMA table_info(session_questions)").fetchall()]
    for col, ddl in [
        ("session_id", "ALTER TABLE session_questions ADD COLUMN session_id TEXT"),
        ("order_index", "ALTER TABLE session_questions ADD COLUMN order_index INTEGER"),
        ("status", "ALTER TABLE session_questions ADD COLUMN status TEXT DEFAULT 'current'"),
        ("llm_raw_response", "ALTER TABLE session_questions ADD COLUMN llm_raw_response TEXT"),
    ]:
        if col not in sq_cols:
            cur.execute(ddl)
    # Добавляем колонку lang, если её нет
    cols = [c[1] for c in cur.execute("PRAGMA table_info(users)").fetchall()]
    if "lang" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN lang TEXT DEFAULT 'ru'")
    # interview_events дополнительные поля
    ie_cols = [c[1] for c in cur.execute("PRAGMA table_info(interview_events)").fetchall()]
    if "session_id" not in ie_cols:
        cur.execute("ALTER TABLE interview_events ADD COLUMN session_id TEXT")
    if "payload_json" not in ie_cols:
        cur.execute("ALTER TABLE interview_events ADD COLUMN payload_json TEXT")
    if "ts" not in ie_cols:
        cur.execute("ALTER TABLE interview_events ADD COLUMN ts TEXT")
        cur.execute("UPDATE interview_events SET ts = COALESCE(createdAt, datetime('now')) WHERE ts IS NULL")
    session_cols = [c[1] for c in cur.execute("PRAGMA table_info(sessions)").fetchall()]
    if "startedAt" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN startedAt TEXT")
    if "questionTitle" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN questionTitle TEXT")
    if "useIDE" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN useIDE INTEGER DEFAULT 1")
    if "status" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN status TEXT DEFAULT 'active'")
    if "is_active" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN is_active INTEGER DEFAULT 1")
    if "is_finished" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN is_finished INTEGER DEFAULT 0")
    if "candidate_id" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN candidate_id TEXT")
    if "current_level" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN current_level TEXT")
    if "current_question_index" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN current_question_index INTEGER DEFAULT 0")
    if "started_at" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN started_at TEXT")
    if "finished_at" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN finished_at TEXT")
    if "cheat_score" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN cheat_score REAL")
    if "metadata_json" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN metadata_json TEXT")
    if "direction" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN direction TEXT")
    if "level" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN level TEXT")
    if "progress_percent" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN progress_percent INTEGER DEFAULT 0")
    message_cols = [c[1] for c in cur.execute("PRAGMA table_info(messages)").fetchall()]
    if "questionId" not in message_cols:
        cur.execute("ALTER TABLE messages ADD COLUMN questionId TEXT")
    # Результаты интервью
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_results (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'completed',
          finishedAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Гарантируем наличие finishedAt в интервью-результатах
    ir_cols = [c[1] for c in cur.execute("PRAGMA table_info(interview_results)").fetchall()]
    if "finishedAt" not in ir_cols:
        cur.execute("ALTER TABLE interview_results ADD COLUMN finishedAt TEXT DEFAULT CURRENT_TIMESTAMP")
    if "metrics_json" not in ir_cols:
        cur.execute("ALTER TABLE interview_results ADD COLUMN metrics_json TEXT")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          questionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          content TEXT NOT NULL,
          code TEXT,
          language TEXT,
          passed_visible INTEGER,
          passed_hidden INTEGER,
          attempt_number INTEGER,
          duration_ms INTEGER,
          metrics_json TEXT,
          updatedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(sessionId, questionId)
        )
        """
    )
    ans_cols = [c[1] for c in cur.execute("PRAGMA table_info(answers)").fetchall()]
    for col, ddl in [
        ("code", "ALTER TABLE answers ADD COLUMN code TEXT"),
        ("language", "ALTER TABLE answers ADD COLUMN language TEXT"),
        ("passed_visible", "ALTER TABLE answers ADD COLUMN passed_visible INTEGER"),
        ("passed_hidden", "ALTER TABLE answers ADD COLUMN passed_hidden INTEGER"),
        ("attempt_number", "ALTER TABLE answers ADD COLUMN attempt_number INTEGER"),
        ("duration_ms", "ALTER TABLE answers ADD COLUMN duration_ms INTEGER"),
        ("metrics_json", "ALTER TABLE answers ADD COLUMN metrics_json TEXT"),
    ]:
        if col not in ans_cols:
            cur.execute(ddl)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_questions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          sessionId TEXT NOT NULL,
          questionId TEXT NOT NULL,
          questionTitle TEXT,
          position INTEGER
        )
        """
    )
    if "questionId" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN questionId TEXT")
    # Сообщения поддержки
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS support_messages (
          id TEXT PRIMARY KEY,
          userId TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS support_dialogs (
          userId TEXT PRIMARY KEY,
          status TEXT DEFAULT 'open'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_config (
          key TEXT PRIMARY KEY,
          value TEXT
        )
        """
    )
    question_cols = [c[1] for c in cur.execute("PRAGMA table_info(questions)").fetchall()]
    if "useIDE" not in question_cols or len(question_cols) < 6:
        cur.execute("DROP TABLE IF EXISTS questions")
        cur.execute(
            """
            CREATE TABLE questions (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              body TEXT NOT NULL,
              statement TEXT,
              answer TEXT NOT NULL,
              hint TEXT,
              difficulty TEXT NOT NULL,
              level TEXT,
              language TEXT,
              tags TEXT,
              visible_tests_json TEXT,
              hidden_tests_json TEXT,
              canonical_solution TEXT,
              source TEXT DEFAULT 'manual',
              useIDE INTEGER DEFAULT 0,
              createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
              updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    else:
        if "statement" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN statement TEXT")
        if "level" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN level TEXT")
        if "language" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN language TEXT")
        if "visible_tests_json" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN visible_tests_json TEXT")
        if "hidden_tests_json" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN hidden_tests_json TEXT")
        if "canonical_solution" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN canonical_solution TEXT")
        if "source" not in question_cols:
            cur.execute("ALTER TABLE questions ADD COLUMN source TEXT DEFAULT 'manual'")
    # Сидим примеры, если база пустая
    existing = cur.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    if existing == 0:
        seed_questions(cur)
    conn.commit()
    conn.close()


def fetchone_dict(cursor):
    row = cursor.fetchone()
    if not row:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def fetchall_dicts(cursor):
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, r)) for r in rows]


ensure_schema()
