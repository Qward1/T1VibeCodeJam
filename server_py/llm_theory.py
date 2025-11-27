import json
from enum import Enum
from typing import List

try:
    # Когда модуль импортируется как пакет server_py.*
    from .llm_client import chat_completion  # type: ignore
except Exception:
    # Когда запускается из корня, а server_py в sys.path
    from llm_client import chat_completion  # type: ignore

LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.7

TRACK_HINTS = {
    "ds": """
Направление: Data Science.
Фокусируйся на:
- анализе данных, статистике, вероятностях,
- EDA, фичеринге, обработке пропусков и выбросов,
- работе с pandas, SQL, визуализацией,
- оценке качества моделей, A/B-тестах и интерпретации результатов.
Важно: вопросы по DS не должны быть про архитектуру нейросетей или тонкости обучения сложных ML-моделей, а именно про анализ данных и статистику.
""",
    "ml": """
Направление: Machine Learning.
Фокусируйся на:
- базовых алгоритмах ML (линейные модели, деревья, бустинг, SVM и т.п.),
- переобучении, регуляризации, кросс-валидации,
- метриках качества (ROC-AUC, F1, Precision/Recall и т.д.),
- пайплайнах обучения, разделении train/val/test, выборе модели.
Важно: вопросы по ML должны быть именно про обучение моделей и обобщающую способность, а не только про общий анализ данных.
""",
    "frontend": "... при желании заполни позже ...",
    "backend": "...",
    "fullstack": "..."
}



class TheoryDecision(str, Enum):
    WRONG = "wrong"
    CORRECT = "correct"
    CLARIFY = "clarify"


def _safe_chat(model: str, messages: list, temperature: float, max_tokens: int, fallback_json: str | None = None, response_format=None):
    last_err = None
    for attempt in range(8):
        try:
            return chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            try:
                import time

                time.sleep(0.4 * (attempt + 1))
            except Exception:
                pass
    if fallback_json is not None:
        return fallback_json
    if last_err:
        raise last_err
    raise RuntimeError("LLM chat failed without exception")


def _extract_or_fallback(raw, fallback_json: str):
    if isinstance(raw, dict):
        return raw
    try:
        return extract_json(raw)
    except Exception:
        try:
            return json.loads(fallback_json)
        except Exception:
            return {}


def extract_json(raw: str) -> dict:
    """
    Вырезает первый JSON-объект из строки, обрезая <think> и markdown-обёртки.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON not found in LLM response")
    json_str = raw[start : end + 1]
    return json.loads(json_str)


def classify_theory_answer(score: int, max_score: int) -> TheoryDecision:
    if max_score <= 0:
        return TheoryDecision.WRONG
    ratio = score / max_score
    if ratio < LOW_THRESHOLD:
        return TheoryDecision.WRONG
    elif ratio >= HIGH_THRESHOLD:
        return TheoryDecision.CORRECT
    else:
        return TheoryDecision.CLARIFY


SYSTEM_PROMPT_GENERATE_THEORY = """
/no_think Ты — ИИ-интервьюер, который генерирует короткие теоретические вопросы для технических собеседований.

Твоя задача:
- придумать ОДИН теоретический вопрос для кандидата,
- сформировать КРАТКИЙ заголовок задания (title),
- сформировать список ключевых пунктов, по которым будет оцениваться ответ,
- задать максимальное количество баллов.

Требования:
- Все текстовые поля для человека (question, title, key_points) должны быть на русском языке.
- Вопрос должен быть таким, чтобы на него можно было ответить за 2–8 минут.
- Вопрос должен проверять понимание концепции и умение объяснить её словами.
- Вопрос должен соответствовать направлению (track) и уровню (level).
- НЕЛЬЗЯ повторять тему вопросов, которые уже задавались в рамках текущей сессии (см. список previous_questions).

Требования к title:
- Краткое название задания, до 8 слов.
- Без лишней пунктуации и вводных фраз.
- Должно отражать суть вопроса, например: "Кеширование и инвалидация данных", "Переобучение и регуляризация моделей".

Формат ответа:
- Верни СТРОГО один объект JSON без пояснений и без Markdown.

Структура JSON:
{
  "title": "краткий заголовок задания до 8 слов",
  "question": "строка с текстом вопроса",
  "track": "frontend | backend | ds | ml | fullstack",
  "level": "junior | middle | senior",
  "estimated_answer_time_min": целое число,  // от 2 до 5
  "key_points": [
    "3–6 коротких ключевых идей, которые должен затронуть хороший ответ"
  ],
  "max_score": целое число,  // например 10
  "topic": "краткое название темы вопроса, например 'переобучение и регуляризация'"
}

Ограничения:
- Поля track и level ДОЛЖНЫ совпадать с теми значениями, которые передал пользователь.
- В key_points не пиши большие абзацы, только короткие формулировки.
- В поле topic отрази тему вопроса так, чтобы по нему можно было понять, о чём был вопрос и избежать повторов.
"""




def build_user_prompt_generate_theory(track: str, level: str, previous_topics: list[str] | None = None) -> str:
    hints = TRACK_HINTS.get(track, "")
    prev_block = ""
    if previous_topics:
        joined = "\\n- ".join(previous_topics)
        prev_block = f"""
Ранее в этой сессии уже задавались вопросы (нельзя задавать вопросы на те же темы):
- {joined}
"""
    return f"""Сгенерируй один теоретический вопрос для технического собеседования.

Параметры кандидата:
- направление (track): {track}
- уровень (level): {level}

Контекст направления:
{hints}

{prev_block}

Требования к вопросу:
- Вопрос должен относиться к направлению track.
- Уровень должен соответствовать level (не слишком простой и не чрезмерно сложный).
- Допускается упоминание реальных технологий и типичных задач из этого направления.
- Вопрос должен проверять понимание принципов и умение объяснить их словами.

Верни только ОДИН JSON-объект в формате, описанном в system-подсказке."""



def generate_theory_question(track: str, level: str, previous_topics: list[str] | None = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE_THEORY},
        {"role": "user", "content": build_user_prompt_generate_theory(track, level, previous_topics)},
    ]
    fallback_json = json.dumps(
        {
            "title": "HTTP/REST основы",
            "question": "Резервный вопрос: расскажите, как работает HTTP/REST и чем отличаются GET/POST.",
            "track": "fullstack",
            "level": "middle",
            "estimated_answer_time_min": 3,
            "key_points": [
                "HTTP методы и идемпотентность",
                "Статусы ответа",
                "Заголовки и тело запроса",
                "Что такое REST и ресурсы",
                "Кэширование и безопасность",
            ],
            "max_score": 10,
        },
        ensure_ascii=False,
    )
    raw = _safe_chat(
        model="qwen3-32b-awq",
        messages=messages,
        temperature=0.4,
        max_tokens=800,
        fallback_json=fallback_json,
        response_format={"type": "json_object"},
    )
    data = _extract_or_fallback(raw, fallback_json)
    if not data.get("title"):
        data["title"] = "Теоретический вопрос"
    return data



SYSTEM_PROMPT_EVAL_THEORY = """
/no_think Ты — ИИ-оценщик ответов на теоретические вопросы на техническом собеседовании.

Ты получаешь:
- формулировку вопроса,
- список ключевых пунктов (key_points),
- максимальный балл (max_score),
- текст ответа кандидата.

Твоя задача:
- сравнить ответ с ключевыми пунктами,
- оценить полноту и точность объяснения,
- выставить итоговый балл от 0 до max_score,
- дать короткий и развёрнутый фидбек.

Оценивай мягко по следующим правилам:
- Если ответ совсем не по теме и почти не затрагивает key_points — ставь 0–2 балла.
- Если затронут хотя бы 1 ключевой пункт, но ответ поверхностный — ставь 3–5 баллов.
- Если затронута примерно половина key_points и объяснение в целом верное — ставь 6–8 баллов.
- Если затронуто большинство или все key_points и объяснение чёткое — ставь 8–10 баллов.

Вердикт:
- ratio = score / max_score;
- ratio < 0.25 → verdict = "fail";
- 0.25 ≤ ratio < 0.6 → verdict = "partial";
- 0.6 ≤ ratio < 0.85 → verdict = "good";
- ratio ≥ 0.85 → verdict = "excellent".

Требования:
- Пиши на русском языке.
- Не раскрывай никаких внутренних рассуждений и не используй теги <think>.
- Не придумывай новых ключевых пунктов, опирайся на переданный список key_points.

Формат ответа:
- Верни СТРОГО один объект JSON без пояснений и без Markdown.

Структура JSON:
{
  "score": число от 0 до max_score,
  "max_score": число,
  "verdict": "fail" | "partial" | "good" | "excellent",
  "covered_points": ["какие ключевые пункты кандидат раскрыл хорошо"],
  "missing_points": ["какие ключевые пункты кандидат не раскрыл или раскрыл слабо"],
  "feedback_short": "1–2 предложения краткого фидбека для кандидата",
  "feedback_detailed": "3–6 предложений более развёрнутого объяснения",
  "suggested_next_difficulty": "decrease" | "same" | "increase"
}
"""



def build_user_prompt_eval_theory(question_obj: dict, candidate_answer: str) -> str:
    return f"""
Оцени теоретический ответ кандидата.

Вопрос:
{question_obj["question"]}

Ключевые пункты (key_points), по которым нужно оценивать:
{json.dumps(question_obj.get("key_points", []), ensure_ascii=False)}

Максимальный балл (max_score): {question_obj.get("max_score", 10)}

Ответ кандидата:
{candidate_answer}

Верни JSON-объект с полями:
score, max_score, verdict, covered_points, missing_points,
feedback_short, feedback_detailed, suggested_next_difficulty.
"""


def evaluate_theory_answer(question_obj: dict, candidate_answer: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVAL_THEORY},
        {"role": "user", "content": build_user_prompt_eval_theory(question_obj, candidate_answer)},
    ]
    fallback_json = json.dumps(
        {
            "score": 0,
            "max_score": question_obj.get("max_score", 10),
            "verdict": "fail",
            "covered_points": [],
            "missing_points": [],
            "feedback_short": "",
            "feedback_detailed": "",
            "suggested_next_difficulty": "same",
        },
        ensure_ascii=False,
    )
    raw = _safe_chat(
        model="qwen3-32b-awq",
        messages=messages,
        temperature=0.1,
        max_tokens=800,
        fallback_json=fallback_json,
        response_format={"type": "json_object"},
    )
    data = _extract_or_fallback(raw, fallback_json)
    data["score"] = int(data.get("score", 0))
    data["max_score"] = int(data.get("max_score", question_obj.get("max_score", 10)))
    return data


SYSTEM_PROMPT_FOLLOWUP = """
/no_think Ты — ИИ-интервьюер на техническом собеседовании.

Ты получаешь:
- исходный теоретический вопрос,
- список ключевых пунктов, которые кандидат раскрыл слабо или не раскрыл (missing_points),
- текст ответа кандидата.

Твоя задача:
- Сформулировать ОДИН короткий уточняющий вопрос, который помогает проверить недостающие аспекты.
- Фокусироваться именно на missing_points, а не спрашивать всё заново.
- Писать на русском языке.

Формат ответа:
- Верни СТРОГО один объект JSON вида:
{
  "follow_up_question": "текст уточняющего вопроса"
}

Ограничения:
- Вопрос должен быть понятным и отвечаемым за 1–2 минуты.
- Не повторяй дословно исходный вопрос, переформулируй его узко под конкретные пробелы.
"""


def build_user_prompt_followup(question_obj: dict, missing_points: List[str], candidate_answer: str) -> str:
    return f"""
Сгенерируй уточняющий вопрос к теоретическому ответу кандидата.

Исходный вопрос:
{question_obj["question"]}

Пункты, которые кандидат раскрыл слабо или не раскрыл (missing_points):
{json.dumps(missing_points, ensure_ascii=False)}

Краткое содержание ответа кандидата:
{candidate_answer}

Верни JSON:
{{
  "follow_up_question": "..."
}}
"""


def generate_followup_question(question_obj: dict, missing_points: List[str], candidate_answer: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
        {"role": "user", "content": build_user_prompt_followup(question_obj, missing_points, candidate_answer)},
    ]
    fallback_json = json.dumps({"follow_up_question": ""}, ensure_ascii=False)
    raw = _safe_chat(
        model="qwen3-32b-awq",
        messages=messages,
        temperature=0.3,
        max_tokens=400,
        fallback_json=fallback_json,
        response_format={"type": "json_object"},
    )
    return _extract_or_fallback(raw, fallback_json)


SYSTEM_PROMPT_EVAL_FOLLOWUP = """
/no_think Ты — ИИ-оценщик уточняющих ответов на теоретические вопросы.

Ты получаешь:
- исходный вопрос собеседования,
- список недостающих пунктов (missing_points),
- текст уточняющего вопроса,
- ответ кандидата на уточняющий вопрос,
- максимальный балл за уточнение (обычно от 2 до 4).

Твоя задача:
- Оценить, насколько кандидат раскрыл именно missing_points,
- Выставить балл от 0 до max_score_followup,
- Дать короткий комментарий.

Формат ответа:
Верни СТРОГО один объект JSON:
{
  "score": число от 0 до max_score_followup,
  "max_score": число,
  "covered_points": ["какие missing_points кандидат теперь раскрыл"],
  "missing_points_still": ["какие missing_points остались нераскрытыми"],
  "feedback_short": "краткий комментарий по уточнению"
}
"""


def build_user_prompt_eval_followup(question_obj: dict, missing_points: List[str], follow_up_question: str, follow_up_answer: str, max_score_followup: int) -> str:
    return f"""
Оцени ответ кандидата на уточняющий вопрос.

Исходный вопрос собеседования:
{question_obj["question"]}

Недостающие ключевые пункты (missing_points), которые нужно проверить:
{json.dumps(missing_points, ensure_ascii=False)}

Уточняющий вопрос:
{follow_up_question}

Ответ кандидата на уточняющий вопрос:
{follow_up_answer}

Максимальный балл за уточнение: {max_score_followup}

Верни JSON-объект в формате:
{{
  "score": 0-{max_score_followup},
  "max_score": {max_score_followup},
  "covered_points": [],
  "missing_points_still": [],
  "feedback_short": ""
}}
"""


def evaluate_followup_answer(question_obj: dict, missing_points: List[str], follow_up_question: str, follow_up_answer: str, max_score_followup: int = 4) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVAL_FOLLOWUP},
        {"role": "user", "content": build_user_prompt_eval_followup(question_obj, missing_points, follow_up_question, follow_up_answer, max_score_followup)},
    ]
    fallback_json = json.dumps(
        {
            "score": 0,
            "max_score": max_score_followup,
            "covered_points": [],
            "missing_points_still": missing_points,
            "feedback_short": "",
        },
        ensure_ascii=False,
    )
    raw = _safe_chat(
        model="qwen3-32b-awq",
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        fallback_json=fallback_json,
        response_format={"type": "json_object"},
    )
    data = _extract_or_fallback(raw, fallback_json)
    data["score"] = int(data.get("score", 0))
    data["max_score"] = int(data.get("max_score", max_score_followup))
    return data


def final_decision_after_followup(base_score: int, base_max_score: int, follow_score: int, follow_max_score: int) -> TheoryDecision:
    if follow_max_score <= 0:
        return TheoryDecision.WRONG
    follow_ratio = follow_score / follow_max_score
    if follow_ratio >= HIGH_THRESHOLD:
        return TheoryDecision.CORRECT
    else:
        return TheoryDecision.WRONG
