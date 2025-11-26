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
    return sqlite3.connect(DB_PATH)


def seed_questions(cur, force: bool = False):
    """Заглушка: больше не сидируем вопросы автоматически."""
    if force:
        cur.execute("DELETE FROM questions")


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
          sessionId TEXT NOT NULL,
          ownerId TEXT,
          event_type TEXT NOT NULL,
          payload TEXT,
          risk_level TEXT DEFAULT 'low',
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
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
          finishedAt TEXT DEFAULT CURRENT_TIMESTAMP
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
          updatedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          score INTEGER,
          max_score INTEGER,
          decision TEXT,
          UNIQUE(sessionId, questionId)
        )
        """
    )
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          body TEXT NOT NULL,
          answer TEXT NOT NULL,
          hint TEXT,
          difficulty TEXT NOT NULL,
          tags TEXT,
          useIDE INTEGER DEFAULT 0,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
          updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Добавляем колонку lang, если её нет
    cols = [c[1] for c in cur.execute("PRAGMA table_info(users)").fetchall()]
    if "lang" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN lang TEXT DEFAULT 'ru'")
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
    if "assignedId" not in session_cols:
        cur.execute("ALTER TABLE sessions ADD COLUMN assignedId TEXT")
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answers (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          questionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          content TEXT NOT NULL,
          updatedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          score INTEGER,
          max_score INTEGER,
          decision TEXT,
          UNIQUE(sessionId, questionId)
        )
        """
    )
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
    # Добавляем недостающие поля в answers
    ans_cols = [c[1] for c in cur.execute("PRAGMA table_info(answers)").fetchall()]
    if "score" not in ans_cols:
        cur.execute("ALTER TABLE answers ADD COLUMN score INTEGER")
    if "max_score" not in ans_cols:
        cur.execute("ALTER TABLE answers ADD COLUMN max_score INTEGER")
    if "decision" not in ans_cols:
        cur.execute("ALTER TABLE answers ADD COLUMN decision TEXT")
    # Добавляем недостающие колонки в session_questions
    sq_cols = [c[1] for c in cur.execute("PRAGMA table_info(session_questions)").fetchall()]
    if "meta_json" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN meta_json TEXT")
    if "q_type" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN q_type TEXT DEFAULT 'coding'")
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
    # Создаём таблицу questions под новую схему, не очищая существующие данные
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          direction TEXT,
          difficulty TEXT,
          type TEXT,
          title TEXT NOT NULL,
          statement TEXT NOT NULL,
          language TEXT,
          visible_tests_json TEXT,
          hidden_tests_json TEXT,
          canonical_answer TEXT,
          useIDE INTEGER DEFAULT 0
        )
        """
    )
    # Назначенные интервью
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS assigned_interviews (
          id TEXT PRIMARY KEY,
          candidateId TEXT NOT NULL,
          adminId TEXT NOT NULL,
          direction TEXT,
          level TEXT,
          format TEXT,
          tasks TEXT,
          duration INTEGER,
          status TEXT DEFAULT 'pending',
          sessionId TEXT,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Связь сессии с назначением
    session_cols = [c[1] for c in cur.execute("PRAGMA table_info(sessions)").fetchall()]
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
