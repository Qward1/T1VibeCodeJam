# file: test_generate_code_task.py
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.t1v.scibox.tech/v1",
    api_key='sk-bJoOnZIeHtnYoiEjq897YA',
)

def extract_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON not found")
    return json.loads(raw[start:end+1])

SYSTEM_PROMPT_GENERATE_CODE_TASK = """
/no_think Ты — генератор задач по программированию для технических собеседований.

Твоя задача:
- сгенерировать ОДНУ задачу по коду в формате JSON,
- сделать её выполнимой за 10–25 минут для кандидата указанного уровня,
- выдать краткий заголовок (title),
- сформировать каркас решения (starter_code) для редактора,
- выдать эталонное решение (reference_solution),
- предложить набор входных данных для открытых и скрытых тестов (БЕЗ ожидаемых результатов).

ОБЯЗАТЕЛЬНАЯ структура ответа:
Ты ДОЛЖЕН вернуть СТРОГО один JSON-объект следующего вида
(ключи и типы полей должны совпадать):

{
  "task_id": "строка id задачи",
  "title": "краткий заголовок до 8 слов",
  "description_markdown": "подробное описание задачи в Markdown на русском",
  "track": "frontend | backend | ds | ml | fullstack",
  "level": "junior | middle | senior",
  "category": "algo" | "domain",
  "language": "python | javascript | ...",
  "allowed_languages": ["python", "javascript"],

  "function_signature": "строка сигнатуры, например: def solve(arr: list[int]) -> int:",
  "starter_code": "каркас функции/класса с TODO/пустым телом, который можно вставить в редактор",
  "constraints": [
    "список читаемых ограничений на входные данные"
  ],

  "reference_solution": "полный код решения на указанном языке одной строкой (с \\n внутри)",

  "sample_inputs": [
    {
      "name": "пример_1",
      "input": [ <аргументы функции в виде JSON-значений> ]
    }
  ],
  "edge_case_inputs": [
    {
      "name": "крайний_случай_1",
      "input": [ ... ]
    }
  ],
  "topic": "краткое текстовое описание темы задачи, например 'скользящее окно и подмассивы'"
}

СТРОГИЕ ограничения:
- Разрешено использовать ТОЛЬКО перечисленные выше поля.
- НЕЛЬЗЯ добавлять поля: input, output, explanation, hint, difficulty или любые другие.
- Если ты добавишь лишние поля, ответ будет считаться некорректным.
- Поле description_markdown должно содержать полное условие задачи (можно с примером) на русском.
- Поле starter_code должно быть компилируемым каркасом: правильный синтаксис, но тело решения не реализовано (например, pass или TODO).
- Поле reference_solution должно содержать полностью рабочее решение, которое проходит все описанные тесты.
- Поля sample_inputs и edge_case_inputs должны содержать ТОЛЬКО входные данные (input), без ожидаемых ответов.
- Аргументы в input должны строго соответствовать сигнатуре функции по порядку.

Требования к сложности:
- Задача должна решаться за время O(n) или O(n log n), если иное не оговорено в описании.
- Не используй внешние библиотеки, только стандартный язык.

Формат ответа:
- Верни СТРОГО один JSON-объект без пояснений, без комментариев и без Markdown-обёрток.
- Не оборачивай JSON в ```json или другие кавычки.
"""


def build_user_prompt_generate_code_task(track, level, category, language, previous_topics=None):
    track_hint = {
        "backend": """
Направление: backend.
- Для category = "algo": классические алгоритмы и структуры данных (массивы, строки, хэш-таблицы, очереди, стеки, деревья, графы, поиск, сортировка).
- Для category = "domain": задачи, связанные с обработкой строк, JSON, логов, HTTP-параметров, простых запросов к псевдо-БД (без реальных сетевых вызовов и фреймворков).
""",
        "frontend": """
Направление: frontend.
- Для category = "algo": задачи на работу со строками, массивами, форматированием данных.
- Для category = "domain": задачи на обработку данных, которые типично приходят во фронтенд (JSON, формы, валидация, простые трансформации перед отправкой на сервер).
""",
        "ds": """
Направление: Data Science.
- Для category = "algo": алгоритмы обработки массивов и матриц, сортировки, базовые статистические вычисления.
- Для category = "domain": задачи на агрегацию и фильтрацию данных, вычисление статистик (среднее, медиана, квантили), подготовку признаков.
""",
        "ml": """
Направление: Machine Learning.
- Для category = "algo": функции работы с векторами и матрицами, вычисление метрик (accuracy, precision, recall, F1 и т.п.).
- Для category = "domain": задачи на расчёт метрик качества, разбиение выборки на train/val/test, простые пред- и пост-обработки предсказаний.
""",
        "fullstack": """
Направление: fullstack.
- Для category = "algo": универсальные алгоритмические задачи (строки, массивы, графы, динамическое программирование).
- Для category = "domain": задачи, сочетающие обработку данных, простую бизнес-логику и подготовку результата, который мог бы быть возвращён из API или отображён во фронтенде.
"""
    }.get(track, "")

    prev_block = ""
    if previous_topics:
        joined = "\n- ".join(previous_topics)
        prev_block = f"""
Ранее в этой сессии уже были задачи на темы (их НЕЛЬЗЯ повторять):
- {joined}
"""

    return f"""Сгенерируй одну задачу по программированию.

Параметры:
- направление (track): {track}
- уровень (level): {level}
- категория (category): {category}
- язык реализации (language): {language}

Контекст направления:
{track_hint}
{prev_block}

Требования к задаче:
- Задача должна быть решаема за ~10–25 минут.
- Не должна быть тривиальной (не 'сумма двух чисел').
- Для category = "algo": сосредоточься на алгоритмах и структурах данных.
- Для category = "domain": сосредоточься на задачах, типичных для данного направления в реальной разработке.
- Используй один основной вход (или несколько аргументов) и один возвращаемый результат.

Верни только ОДИН JSON-объект в формате, описанном в system-подсказке."""


def generate_code_task():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE_CODE_TASK},
        {"role": "user", "content": build_user_prompt_generate_code_task(
            track="backend",
            level="middle",
            category="algo",
            language="python",
            previous_topics=[]
        )},
    ]
    resp = client.chat.completions.create(
        model="qwen3-coder-30b-a3b-instruct-fp8",
        messages=messages,
        temperature=0.4,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content
    print("RAW RESPONSE:\n", raw)
    try:
        data = extract_json(raw)
        print("\nPARSED JSON KEYS:", data.keys())
    except Exception as e:
        print("JSON parse error:", e)

if __name__ == "__main__":
    generate_code_task()
