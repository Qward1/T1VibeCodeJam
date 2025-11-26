Подзадача 0. Общие требования и смена логики показа редактора

Перестань использовать колонку useIDE из таблицы questions для решения, показывать ли редактор кода.

Сделай так, чтобы вывод редактора на экран собеседования теперь определялся по колонке q_type из таблицы session_questions:

если session_questions.q_type = 'coding' → показывать редактор кода;

если q_type = 'theory' → показывать только текстовое поле / чат без код-редактора;

остальные типы (system, info, и т.п.) при необходимости обработать по умолчанию (без редактора).

Пройди по всем местам в коде, где сейчас логика опирается на useIDE (например, при формировании ответа /api/interview/next или /api/session/...) и замени её на проверку q_type из session_questions.

Подзадача 1. Формат LLM-задачи по коду

Реализуй генерацию задач по коду через LLM в следующем формате JSON:

{
  "task_id": "backend_middle_algo_001",
  "title": "Максимальная сумма подмассива",
  "description_markdown": "Текст условия в Markdown на русском...",
  "track": "backend",
  "level": "middle",
  "category": "algo",
  "language": "python",
  "allowed_languages": ["python", "javascript"],
  "function_signature": "def max_subarray_sum(arr: list[int]) -> int:",
  "starter_code": "def max_subarray_sum(arr: list[int]) -> int:\n    # TODO: реализуйте алгоритм\n    pass",
  "constraints": [
    "Массив может содержать от 1 до 10^5 элементов",
    "Элементы массива могут быть от -10^4 до 10^4"
  ],
  "reference_solution": "def max_subarray_sum(...):\n    ...",
  "sample_inputs": [
    { "name": "пример_1", "input": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]] }
  ],
  "edge_case_inputs": [
    { "name": "крайний_случай_1", "input": [[-1, -2, -3]] }
  ],
  "topic": "алгоритм Кадане, максимальная сумма подмассива"
}


Обрати внимание:

sample_inputs и edge_case_inputs содержат только входы (input), без expected.

starter_code — каркас функции/класса, который компилируется, но ещё не решает задачу.

reference_solution — корректное эталонное решение, которое будет использовано только на сервере для генерации тестов.

Подзадача 2. Промты и клиент для генерации задач по коду
2.1. Клиент LLM (если ещё не сделан)

Используй существующий llm_client.py (или создай его, если ещё нет) с обёрткой chat_completion.

2.2. System-промт для генерации задач по коду

Добавь в модуль, например llm_code.py, константу:

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

2.3. User-промт генерации задачи

Добавь функцию:

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

2.4. Функция генерации задачи

Добавь функцию:

from .llm_client import chat_completion
from .llm_theory import extract_json  # если утилита уже есть

def generate_code_task(track: str, level: str, category: str, language: str, previous_topics=None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE_CODE_TASK},
        {"role": "user", "content": build_user_prompt_generate_code_task(track, level, category, language, previous_topics)},
    ]
    raw = chat_completion(
        model="qwen3-coder-30b-a3b-instruct-fp8",
        messages=messages,
        temperature=0.4,
        max_tokens=2000,
    )
    data = extract_json(raw)
    return data

Подзадача 3. Распределение тем задач и интеграция с интервью

Введи поле category в логике интервью для кодинга: "algo" и "domain".

Для backend реализуй паттерн распределения на интервью из 4 задач:

def choose_code_category(track: str, index: int) -> str:
    if track == "backend":
        pattern = ["algo", "domain", "algo", "domain"]
        return pattern[index % len(pattern)]
    # можно добавить паттерны для других треков
    return "algo"


В логике формирования списка вопросов для сессии (при старте интервью или при выдаче следующего вопроса) добавь ветку для q_type = 'coding', где:

выбирается category = choose_code_category(track, index),

вызывается generate_code_task(...),

результат сохраняется в БД.

Подзадача 4. Сохранение задач и тестов в БД
4.1. Таблицы

Добавь/обнови таблицы (через миграцию или чистый SQL):

code_tasks:

id INTEGER PK,

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

raw_json TEXT.

code_tests:

id INTEGER PK,

task_id TEXT,

name TEXT,

is_public INTEGER, -- 1 для открытых тестов, 0 для скрытых

input_json TEXT,

expected_json TEXT.

code_attempts:

id INTEGER PK,

session_id TEXT,

question_id TEXT,

task_id TEXT,

owner_id TEXT,

attempt_number INTEGER,

code TEXT,

passed_public INTEGER,

passed_hidden INTEGER,

score INTEGER,

created_at DATETIME.

4.2. Генерация expected-результатов из reference_solution

Реализуй функцию:

def build_tests_for_task(task: dict) -> tuple[list[dict], list[dict]]:
    """
    На основе sample_inputs и edge_case_inputs вызывает reference_solution в Docker
    и возвращает (public_tests, hidden_tests) с полями name, input, expected.
    """


Алгоритм:

Собери all_inputs = sample_inputs + edge_case_inputs.

Для каждого элемента:

запусти reference_solution в Docker-контейнере (по task["language"]),

передай аргументы из input в функцию,

получи expected (сериализуй в JSON).

public_tests — те, что соответствуют sample_inputs, hidden_tests — edge_case_inputs.

Сохрани тесты в code_tests с флагом is_public.

Подзадача 5. Замена выдачи задач на код

В эндпоинте, отвечающем за выдачу следующего вопроса (/api/interview/next или аналог), замени логику кодинговых задач, которая сейчас берёт вопросы из таблицы questions, на следующую:

Для текущей сессии найди список уже заданных session_questions с q_type = 'coding' и их task_id.

Если нужно выдать новую кодинговую задачу:

Определи индекс кодинговой задачи (index).

Выбери category = choose_code_category(track, index).

Вызови generate_code_task(track, level, category, language="python", previous_topics=[topics...] ).

Сохрани задачу в code_tasks (если ещё нет записи по task_id).

Сгенерируй и сохрани тесты (public/hidden) через build_tests_for_task.

Создай запись в session_questions:

q_type = 'coding',

questionId = <generated task_id or внутренний id>,

questionTitle = task["title"],

description = task["description_markdown"] (или отдельное поле),

position — номер вопроса.

В ответ API добавь:

q_type: "coding",

title и description,

starterCode для фронта,

language / allowedLanguages.

Убедись, что фронт теперь показывает редактор кода по q_type = 'coding' и берёт starterCode из ответа бэкенда.

Подзадача 6. Эндпоинты «Запустить примеры» и «Проверить решение»
6.1. Кнопка «Запустить примеры»

Реализуй эндпоинт POST /api/code/run-samples:

Вход (Pydantic-модель):

class RunSamplesPayload(BaseModel):
    sessionId: str
    questionId: str
    taskId: str
    ownerId: str
    code: str
    language: str


Логика:

Найди все code_tests с task_id и is_public = 1.

Для каждого теста:

запусти код пользователя в Docker,

подставь аргументы input в функцию (согласно function_signature),

собери actual.

Верни:

{
  "tests": [
    {"name": "пример_1", "status": "passed", "expected": 6, "actual": 6},
    {"name": "пример_2", "status": "failed", "expected": -1, "actual": 0}
  ],
  "hasError": false
}


В случае ошибки исполнения (исключение, timeout) вызови LLM-ментор (см. Подзадачу 7) и верни:

{
  "tests": [],
  "hasError": true
}


Не увеличивай счётчик попыток и не ставь баллы.

6.2. Кнопка «Проверить решение»

Реализуй эндпоинт POST /api/code/check:

Вход (Pydantic):

class CheckCodePayload(BaseModel):
    sessionId: str
    questionId: str
    taskId: str
    ownerId: str
    code: str
    language: str


Логика:

Определи номер попытки для этой (sessionId, questionId, taskId, ownerId) как attempt_number = предыдущий_max + 1.

Запусти все public-тесты.

Если есть ошибка исполнения → см. Подзадачу 7 (LLM-объяснение), увеличь attempt, создай запись в code_attempts c:

passed_public = 0, passed_hidden = 0, score = 0,

верни solved = false, publicTests с status = "error", hasError = true.

Если просто часть тестов упала:

не запускай скрытые тесты,

создай запись code_attempts с passed_public = 0, passed_hidden = 0, score = 0,

верни solved = false, publicTests с passed/failed, hiddenPassed = false.

Если все public-тесты пройдены:

Запусти все hidden-тесты.

Если скрытые тесты не пройдены:

создай запись code_attempts с passed_public = 1, passed_hidden = 0, score = 0,

верни solved = false, hiddenPassed = false.

Если все hidden-тесты пройдены:

вычисли score по номеру попытки:

def score_for_attempt(attempt: int, max_score: int = 10) -> int:
    if attempt == 1:
        return max_score
    if attempt == 2:
        return int(max_score * 0.7)
    if attempt == 3:
        return int(max_score * 0.5)
    return 0


создай запись code_attempts с passed_public = 1, passed_hidden = 1, score = score_for_attempt(attempt),

верни:

{
  "solved": true,
  "attempt": <номер_попытки>,
  "score": <балл>,
  "maxScore": 10,
  "publicTests": [...],
  "hiddenPassed": true
}

Подзадача 7. Обработка ошибок кода и LLM-объяснение в чат

Реализуй логику «умного объяснения ошибок» при падении кода пользователя.

При запуске кода (в /run-samples и /check) если:

Docker возвращает ненулевой exit-code,

время выполнения превышено,

либо парсинг результата завершился исключением,

то:

собери:

язык (language),

function_signature,

фрагмент кода пользователя (полностью или обрезанный до разумного размера),

текст ошибки/traceback.

Вызови qwen3-coder-30b-a3b-instruct-fp8 с отдельным промтом-ментором.

System-промт ментора по ошибкам
SYSTEM_PROMPT_CODE_ERROR_MENTOR = """
/no_think Ты — доброжелательный ментор по программированию.

Твоя задача:
- по коду пользователя и тексту ошибки кратко объяснить, в чём суть проблемы,
- мягко направить пользователя в сторону верного решения,
- НЕ давать готового полного решения и НЕ переписывать всю функцию за него.

Требования:
- Пиши на русском языке.
- Объясняй простыми словами (2–5 предложений).
- Не раскрывай полный правильный код.
- Не используй фразы вида 'вот исправленное решение', 'замени весь код на...' и т.п.
- Даём только направление: на что обратить внимание (индексы, границы циклов, типы данных, граничные случаи, пустые входы и т.д.).
"""

User-промт для ментора
def build_user_prompt_code_error(language: str, function_signature: str, user_code: str, error_text: str) -> str:
    return f"""
Проанализируй ошибку в коде.

Язык: {language}
Сигнатура функции: {function_signature}

Код пользователя:
```{language}
{user_code}


Текст ошибки/traceback:
{error_text}

Объясни кратко, в чём проблема, и подскажи, что нужно проверить или поправить, но не давай готового решения и не пиши полный исправленный код.
"""


3. Полученный от LLM текст (ответ) сохрани как новое сообщение в таблице чата текущей сессии:
   - `role = 'assistant'` или `'interviewer'`,
   - `source = 'code_error_hint'` (если есть такое поле),
   - `message = <ответ LLM>`.

4. Обеспечь, чтобы фронтенд показывал это сообщение в панели чата рядом с задачей, когда пользователь получает ошибку при запуске кода.

---

## Подзадача 8. Проверка целостности и интеграция

1. Убедись, что:

   - выдача вопросов теперь различает `q_type = 'coding'` и `q_type = 'theory'`,
   - редактор кода на фронте показывается **по `q_type`, а не по `useIDE`**,
   - новые эндпоинты `/api/code/run-samples` и `/api/code/check` доступны и возвращают корректный JSON.

2. Проверь полный цикл:

   - старт интервью с кодинговыми задачами,
   - выдача первой кодинговой задачи (LLM-генерация → сохранение в БД → ответ на фронт),
   - запуск «Запустить примеры»:
     - успешный случай (видно результаты тестов),
     - случай с ошибкой (появляется сообщение в чате).
   - запуск «Проверить решение»:
     - провал по открытым тестам,
     - провал по скрытым,
     - успешное решение с 1, 2 и 3 попытки (проверить начисление баллов и запись в `code_attempts`).

3. Убедись, что логика подсчёта баллов за кодинговые задачи интегрируется с общей системой баллов инт