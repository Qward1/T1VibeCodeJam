import sqlite3
from pathlib import Path
import hashlib
import shutil
import logging

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "api.sqlite"
logger = logging.getLogger(__name__)


def _healthy_db(path: Path) -> bool:
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA quick_check;")
        conn.execute("SELECT name FROM sqlite_master LIMIT 1;")
        conn.close()
        return True
    except Exception:
        return False


def _force_disable_wal(path: Path):
    if not path.exists():
        return
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.close()
    except Exception:
        # Игнорируем сбои при смене journal_mode, чтобы не шуметь в логах
        pass


# Если основной файл бьётся об I/O (часто на DrvFS из-за WAL), пробуем безопасно
# переключиться на копию, чтобы не падали on-demand генерации отчётов.
if not _healthy_db(DB_PATH):
    fallback = DATA_DIR / "api_repair.sqlite"
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DB_PATH, fallback)
        if _healthy_db(fallback):
            logger.warning("Switching DB_PATH to fallback copy due to disk I/O error", extra={"fallback": str(fallback)})
            DB_PATH = fallback
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create fallback DB copy", extra={"error": str(exc), "source": str(DB_PATH), "fallback": str(fallback)})

_force_disable_wal(DB_PATH)

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
          ownerId TEXT,
          questionId TEXT,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          source TEXT,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    msg_cols = [c[1] for c in cur.execute("PRAGMA table_info(messages)").fetchall()]
    if "ownerId" not in msg_cols:
        cur.execute("ALTER TABLE messages ADD COLUMN ownerId TEXT")
    if "source" not in msg_cols:
        cur.execute("ALTER TABLE messages ADD COLUMN source TEXT")
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
    # Таблицы для кодовых задач (больше не дропаем, чтобы не терять данные)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS code_tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          task_id TEXT UNIQUE,
          track TEXT,
          level TEXT,
          category TEXT,
          language TEXT,
          allowed_languages_json TEXT,
          title TEXT,
          description_markdown TEXT,
          function_signature TEXT,
          starter_code TEXT,
          constraints_json TEXT,
          reference_solution TEXT,
          topic TEXT,
          raw_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS code_tests (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          task_id TEXT,
          name TEXT,
          is_public INTEGER,
          input_json TEXT,
          expected_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS code_attempts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT,
          question_id TEXT,
          task_id TEXT,
          owner_id TEXT,
          attempt_number INTEGER,
          code TEXT,
          passed_public INTEGER,
          passed_hidden INTEGER,
          score INTEGER,
          created_at DATETIME
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
          position INTEGER,
          meta_json TEXT,
          q_type TEXT DEFAULT 'coding',
          code_task_id TEXT,
          category TEXT,
          status TEXT DEFAULT 'active',
          is_finished INTEGER DEFAULT 0
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
          finishedAt TEXT DEFAULT CURRENT_TIMESTAMP,
          score INTEGER
        )
        """
    )
    # Гарантируем наличие finishedAt в интервью-результатах
    ir_cols = [c[1] for c in cur.execute("PRAGMA table_info(interview_results)").fetchall()]
    if "finishedAt" not in ir_cols:
        cur.execute("ALTER TABLE interview_results ADD COLUMN finishedAt TEXT DEFAULT CURRENT_TIMESTAMP")
    if "score" not in ir_cols:
        cur.execute("ALTER TABLE interview_results ADD COLUMN score INTEGER")
    # Итоговые отчёты
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_reports (
          id TEXT PRIMARY KEY,
          sessionId TEXT NOT NULL,
          ownerId TEXT NOT NULL,
          track TEXT,
          level TEXT,
          metrics_json TEXT,
          questions_json TEXT,
          summary_candidate TEXT,
          summary_admin TEXT,
          recommendations_json TEXT,
          createdAt TEXT DEFAULT CURRENT_TIMESTAMP
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
    # Добавляем недостающие поля в session_questions
    sq_cols = [c[1] for c in cur.execute("PRAGMA table_info(session_questions)").fetchall()]
    if "meta_json" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN meta_json TEXT")
    if "q_type" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN q_type TEXT DEFAULT 'coding'")
    if "code_task_id" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN code_task_id TEXT")
    if "category" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN category TEXT")
    if "status" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN status TEXT DEFAULT 'active'")
    if "is_finished" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN is_finished INTEGER DEFAULT 0")
    if "hints_used" not in sq_cols:
        cur.execute("ALTER TABLE session_questions ADD COLUMN hints_used INTEGER DEFAULT 0")
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


def _init_schema_safe():
    global DB_PATH
    try:
        ensure_schema()
        return
    except sqlite3.OperationalError as exc:
        if "disk I/O error" not in str(exc).lower():
            raise
        logger.warning("ensure_schema failed with disk I/O, trying fallback copy", extra={"path": str(DB_PATH)})
        fallback = DATA_DIR / "api_repair.sqlite"
        try:
            shutil.copyfile(DB_PATH, fallback)
            DB_PATH = fallback
            _force_disable_wal(DB_PATH)
            ensure_schema()
            return
        except Exception as inner:  # noqa: BLE001
            logger.error("ensure_schema recovery failed", extra={"error": str(inner), "fallback": str(fallback)})
            raise
    except Exception:
        raise


_init_schema_safe()
