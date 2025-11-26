1. Какие данные у тебя уже есть от LLM

Учитывай, что:

Теория — генерация вопроса

LLM уже отдаёт примерно такой JSON:

{
  "question": "Объясните, как вы реализуете атомарность...",
  "track": "backend",
  "level": "middle",
  "estimated_answer_time_min": 3,
  "key_points": [
    "Понимание принципа атомарности и транзакций",
    "Использование механизмов компенсации или idempotency",
    "Пример реализации (например, Saga)",
    "Как избежать несогласованности при частичных сбоях"
  ],
  "max_score": 10
}

Теория — оценка ответа

LLM даёт:

{
  "score": 4,
  "max_score": 10,
  "verdict": "partial",  // incorrect | partial | good | excellent
  "covered_points": [...],
  "missing_points": [...],
  "feedback_short": "Ответ частично охватывает тему...",
  "feedback_detailed": "Кандидат упомянул..., но не рассказал про...",
  "suggested_next_difficulty": "same"  // decrease | same | increase
}

Код — генерация задачи

LLM даёт:

{
  "task_id": "...",
  "title": "...",
  "description_markdown": "...",
  "track": "backend",
  "level": "middle",
  "category": "algo" | "domain",
  "language": "python",
  "function_signature": "...",
  "starter_code": "...",
  "reference_solution": "...",
  "sample_inputs": [...],
  "edge_case_inputs": [...],
  "topic": "алгоритм Кадане, ..."
}

Код — оценка решения

Это уже не LLM, а твоя логика:

code_attempts:

attempt_number,

passed_public, passed_hidden,

score, max_score,

time_spent,

hints_used,

кол-во ошибок исполнения.

Плюс LLM-подсказки:

текст подсказок (hint_level 1/2/3),

объяснения ошибок (ментор по исключениям).

2. Нормализуй всё в единую модель для отчёта

Реализуй модуль, который после завершения собеседования собирает сырые данные и переводит их в унифицированный формат, не зависящий от конкретного LLM.

2.1. Структура нормализованного вопроса в отчёте
{
  "question_id": "uuid",
  "order": 1,
  "q_type": "theory",      // theory | coding
  "track": "backend",
  "level": "middle",

  "title": "Кеширование в backend",
  "category": "algo" | "domain" | null,

  "raw_prompt": "...",     // опционально
  "raw_llm_question": {...},   // JSON от генерации (для debug)

  "metrics": {
    // общие
    "time_spent_sec": 180,
    "started_at": "...",
    "finished_at": "...",

    // теория
    "score": 8,
    "max_score": 10,
    "score_ratio": 0.8,
    "verdict": "good",
    "key_points": [
      {
        "text": "Использование in-memory кэша",
        "status": "covered"      // covered | missing | partial
      },
      {
        "text": "Стратегия TTL / LRU",
        "status": "missing"
      }
    ],
    "llm_feedback_short": "...",
    "llm_feedback_detailed": "...",

    // код
    "attempts": 2,
    "score_code": 7,
    "max_score_code": 10,
    "score_ratio_code": 0.7,
    "public_tests_total": 5,
    "public_tests_passed": 5,
    "hidden_tests_total": 5,
    "hidden_tests_passed": 4,
    "hints_used": 1,
    "errors_count": 2,
    "first_success_attempt": 2
  },

  "llm_evaluation": { ... },       // исходный JSON оценки теории
  "code_task": { ... },            // исходный JSON задачи по коду
  "code_attempts": [ ... ],        // массив попыток
  "hints": [ ... ],                // тексты подсказок
  "error_explanations": [ ... ]    // объяснения ошибок от LLM
}


Сделай функцию:

def build_normalized_question_report(session_id: str, question_id: str) -> dict:
    ...


Она:

для q_type = 'theory':

берёт question JSON и evaluation JSON,

мапит key_points + covered_points + missing_points → статусы,

считает score_ratio,

для q_type = 'coding':

собирает все code_attempts,

считает:

attempts,

final_score, max_score, score_ratio,

hints_used (из отдельного поля или попыток),

статистику тестов,

first_success_attempt (если есть).

3. Сводные метрики по собеседованию

Реализуй функцию:

def build_session_metrics(session_id: str) -> dict:
    questions = [build_normalized_question_report(session_id, qid) for qid in ...]
    ...


Она должна считать:

{
  "overall": {
    "total_questions": 8,
    "theory_questions": 4,
    "coding_questions": 4,

    "total_score": 60,
    "max_total_score": 80,
    "score_ratio": 0.75,

    "time_total_sec": 3600,
    "avg_time_per_question_sec": 450
  },
  "by_type": {
    "theory": {
      "score": 30,
      "max_score": 40,
      "score_ratio": 0.75,
      "avg_score_per_question": 7.5
    },
    "coding": {
      "score": 30,
      "max_score": 40,
      "score_ratio": 0.75,
      "avg_attempts": 1.8,
      "avg_hints_used": 0.5
    }
  },
  "by_track": {
    "backend": { ... },
    "ds": { ... }
  },
  "anti_cheat": {
    "suspicious_events": 0,
    "notes": "Подозрительных действий не зафиксировано"
  }
}

4. Структура итогового отчёта (что отдавать фронту / сохранять в БД)

Реализуй таблицу interview_reports и JSON-структуру:

{
  "session_id": "uuid",
  "owner_id": "candidate-id",
  "track": "backend",
  "level": "middle",
  "created_at": "...",

  "metrics": { ... },           // результат build_session_metrics
  "questions": [ ... ],         // массив normalized_question_report
  "llm_summary_candidate": "...", // текстовый отчёт для кандидата
  "llm_summary_admin": "...",     // текстовый отчёт для рекрутера/компании
  "next_recommendations": {       // структурированные рекомендации
    "difficulty": "increase",     
    "tracks": ["backend", "system design"],
    "topics_to_improve": [
      "Транзакции и паттерн Saga",
      "Стратегии кэширования и инвалидации"
    ]
  }
}

5. LLM для красивого текстового отчёта

Тут самое вкусное: у тебя уже есть структурированные данные, поэтому отчёт лучше генерировать одним LLM-вызовом, а не городить «магический» текст из кода.

5.1. System-промт для отчёта

Реализуй:

SYSTEM_PROMPT_INTERVIEW_REPORT = """
/no_think Ты — опытный технический тимлид и наставник.
Твоя задача — по структурированным данным о собеседовании:

- кратко описать общий уровень кандидата,
- выделить сильные стороны,
- честно, но аккуратно описать зоны роста,
- дать рекомендации, куда двигаться дальше.

Требования:
- Пиши на русском языке.
- Раздели отчёт на блоки с заголовками: 
  1) Итоговая оценка 
  2) Сильные стороны 
  3) Зоны роста 
  4) Рекомендации по развитию.
- Не повторяй дословно тексты вопросов; пересказывай своими словами.
- Используй данные о баллах, попытках, подсказках и покрытии key_points, чтобы делать выводы об уровне.
- Отдельно отметь разницу между теорией и практикой (код).
- Не придумывай факты, которых нет в данных (если чего-то не хватает, просто не упоминай это).
"""

5.2. User-промт для отчёта

Реализуй:

def build_user_prompt_interview_report(session_summary: dict) -> str:
    """
    session_summary — dict с:
      - metrics
      - questions (уже нормализованные)
    """
    # Лучше передавать JSON текстом, но явно обозначить формат
    return f"""
Ниже приведены структурированные данные о собеседовании в формате JSON.

Данные:
```json
{json.dumps(session_summary, ensure_ascii=False)}


На основе этих данных:

Сформируй подробный, но компактный отчёт о кандидате.

Соблюдай структуру: 'Итоговая оценка', 'Сильные стороны', 'Зоны роста', 'Рекомендации по развитию'.

Особое внимание:

какие темы кандидат раскрывал хорошо (по key_points и score_ratio),

где часто не хватало ключевых пунктов,

как он справлялся с задачами по коду (сколько попыток, сколько подсказок).
"""


### 5.3. Генерация двух версий отчёта

Реализуй:

```python
def generate_interview_reports_text(session_id: str) -> tuple[str, str]:
    summary = build_full_session_summary(session_id)  # metrics + questions + anti_cheat

    # Отчёт для кандидата
    messages_candidate = [
        {"role": "system", "content": SYSTEM_PROMPT_INTERVIEW_REPORT},
        {"role": "user", "content": build_user_prompt_interview_report(summary)},
    ]
    resp_cand = client.chat.completions.create(
        model="qwen3-32b-awq",
        messages=messages_candidate,
        temperature=0.4,
        max_tokens=1500,
    )
    text_candidate = resp_cand.choices[0].message.content.strip()

    # Отчёт для админа/компании — тот же summary, но доп. акцент
    SYSTEM_PROMPT_INTERVIEW_REPORT_ADMIN = SYSTEM_PROMPT_INTERVIEW_REPORT + """
Дополнительно:
- Пиши чуть более формально.
- Можешь упомянуть, подходит ли кандидат для middle-уровня, junior+ или senior-, исходя из данных.
- Отдельно оцени риски: нестабильность знаний, слабые места, которые критичны для продакшн-разработки.
"""
    messages_admin = [
        {"role": "system", "content": SYSTEM_PROMPT_INTERVIEW_REPORT_ADMIN},
        {"role": "user", "content": build_user_prompt_interview_report(summary)},
    ]
    resp_admin = client.chat.completions.create(
        model="qwen3-32b-awq",
        messages=messages_admin,
        temperature=0.3,
        max_tokens=1500,
    )
    text_admin = resp_admin.choices[0].message.content.strip()

    return text_candidate, text_admin

6. Когда генерировать отчёт

Реализуй формирование отчёта в момент, когда:

пользователь нажимает кнопку «Завершить собеседование»,
или

все вопросы отмечены как status = 'done'.

Пайплайн:

Закрыть сессию (sessions.status = 'finished').

Вызвать:

summary = build_full_session_summary(session_id)

text_candidate, text_admin = generate_interview_reports_text(session_id)

Сохранить в interview_reports:

metrics = summary["metrics"],

questions = summary["questions"],

llm_summary_candidate = text_candidate,

llm_summary_admin = text_admin,

next_recommendations — можешь вытащить как отдельное поле из LLM (например, попросить LLM вернуть JSON с рекомендациями в отдельном вызове).

Отдать на фронт:

короткий summary (оценка, общие баллы),

id отчёта для детального просмотра.

7. Что в итоге получится

Для кандидата: читаемый, структурный отчёт «где я молодец, что подтянуть, какие темы учить», основанный НЕ на чувствах интервьюера, а на:

покрытии key_points,

баллах за код,

количестве подсказок,

попытках и времени.

Для компании: формальный JSON + текст, где:

видно баллы по блокам (теория/код/направления),

видны слабые зоны (например, «кандидат часто проваливал domain-задачи по backend»),

есть прозрачная история: почему итоговый verdict именно такой.

На старнице админ реализуй кнопку отчет , нажав по которой будет скачиваться txt файл с отчетом от данного участника(последнего его собеседования).
На странице профиль , на панели "Истории собеседований" уже есть кнопки "Отчет" , реализуй чтобы при нажатии на неё скачивался отчет по конкртному собеседованию