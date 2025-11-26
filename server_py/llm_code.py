import json
import re
import logging
import uuid
from typing import List, Optional
import tempfile
import subprocess
import sys
import ast

try:
    from .llm_client import chat_completion  # type: ignore
    from .llm_theory import extract_json  # type: ignore
except Exception:
    from llm_client import chat_completion  # type: ignore
    from llm_theory import extract_json  # type: ignore

logger = logging.getLogger(__name__)

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

Ограничения для тестов:
- Все наборы входных данных из поля "sample_inputs" и поля "edge_case_inputs" должны быть КОРРЕКТНЫМИ для правильно реализованной функции.
- При запуске reference_solution на ЛЮБОМ из этих входов не должно возникать ошибок выполнения (деление на ноль, выход за границы массива, обращение к несуществующему ключу и т.п.), если только обработка такой ситуации явно не описана в условии задачи.
- Не генерируй edge_case_inputs, которые приводят к выбросу исключений в корректной реализации (например, деление на ноль, пустой список там, где по условию должен быть хотя бы один элемент, и т.д.).
- Граничные случаи в edge_case_inputs должны проверять сложность алгоритма и корректность логики, а не провоцировать аварийное падение программы.

СТРОГИЕ ограничения:
- Разрешено использовать ТОЛЬКО перечисленные выше поля.
- НЕЛЬЗЯ добавлять поля: input, output, explanation, hint, difficulty или любые другие.
- Поле description_markdown должно содержать полное условие задачи (можно с примером) на русском.
- Поле starter_code должно быть компилируемым каркасом: правильный синтаксис, но тело решения не реализовано (например, pass или TODO).
- Поле reference_solution должно содержать полностью рабочее решение, которое проходит все описанные тесты.
- Поля sample_inputs и edge_case_inputs должны содержать ТОЛЬКО входные данные (input), без ожидаемых ответов.
- Аргументы в input должны строго соответствовать сигнатуре функции по порядку.
- Нельзя генерировать задачу на ту же тему, что уже была в этой сессии по тому же track+category, даже если слегка перефразировать. topic и title новой задачи должны отражать новую концепцию.

Требования к сложности:
- Задача должна решаться за время O(n) или O(n log n), если иное не оговорено в описании.
- Не используй внешние библиотеки, только стандартный язык.

Формат ответа:
- Верни СТРОГО один JSON-объект без пояснений, без комментариев и без Markdown-обёрток.
- Не оборачивай JSON в ```json или другие кавычки.
"""


def build_user_prompt_generate_code_task(
    track: str,
    level: str,
    category: str,
    language: str,
    previous_algo_topics: Optional[List[str]] = None,
    previous_domain_topics: Optional[List[str]] = None,
):
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
    if category == "algo" and previous_algo_topics:
        joined = "\n- ".join(previous_algo_topics)
        prev_block = f"""
Ранее в этой сессии уже были алгоритмические задачи на темы (их НЕЛЬЗЯ повторять и близко перефразировать):
- {joined}
"""
    elif category == "domain" and previous_domain_topics:
        joined = "\n- ".join(previous_domain_topics)
        prev_block = f"""
Ранее в этой сессии уже были профильные задачи на темы (их НЕЛЬЗЯ повторять и близко перефразировать):
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



def generate_code_task(
    track: str,
    level: str,
    category: str,
    language: str,
    previous_algo_topics: Optional[List[str]] = None,
    previous_domain_topics: Optional[List[str]] = None,
) -> dict:
    def _parse_llm_json(raw_str: str) -> dict:
        text = raw_str.strip()
        fence = re.match(r"```(?:json)?(.*)```", text, re.S)
        if fence:
            text = fence.group(1).strip()

        def _loads_safe(snippet: str):
            cleaned = re.sub(r",(\s*[}\]])", r"\1", snippet)  # убираем висячие запятые
            try:
                return json.loads(cleaned)
            except Exception:
                pass
            # Расширенный парсер: переводим JSON-подобную строку в python-литерал
            try:
                py_like = (
                    cleaned.replace("null", "None")
                    .replace("true", "True")
                    .replace("false", "False")
                )
                return ast.literal_eval(py_like)
            except Exception as exc:
                raise ValueError("cannot_parse_llm_json") from exc

        try:
            return extract_json(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return _loads_safe(m.group(0))
        if "{" in text and "}" in text:
            first, last = text.find("{"), text.rfind("}")
            return _loads_safe(text[first : last + 1])
        raise ValueError("cannot_parse_llm_json")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATE_CODE_TASK},
        {
            "role": "user",
            "content": build_user_prompt_generate_code_task(
                track, level, category, language, previous_algo_topics, previous_domain_topics
            ),
        },
    ]
    try:
        raw = chat_completion(
            model="qwen3-32b-awq",
            messages=messages,
            temperature=0.3,
            max_tokens=900,
        )
        logger.info("LLM code task raw response", extra={"track": track, "level": level, "category": category})
        if not raw or not isinstance(raw, str):
            raise ValueError("empty_llm_response")
        data = _parse_llm_json(raw)
        if not data:
            raise ValueError("cannot_parse_llm_json")
        logger.info("LLM code task parsed", extra={"task_id": data.get("task_id"), "title": data.get("title")})
        return data
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM code task generation failed, using fallback", extra={"track": track, "level": level, "category": category, "error": str(exc)})
        return _fallback_code_task(track, level, category, language)


def _fallback_code_task(track: str, level: str, category: str, language: str) -> dict:
    track_norm = (track or "fullstack").lower()
    category_norm = (category or "algo").lower()
    lang = language or "python"
    tid = f"fallback-{uuid.uuid4()}"

    if category_norm == "algo":
        title = "Проверка анаграмм"
        function_signature = "def are_anagrams(a: str, b: str) -> bool:"
        starter_code = f"""{function_signature}
    # TODO: верните True, если строки являются анаграммами
    pass
"""
        reference_solution = """def are_anagrams(a: str, b: str) -> bool:
    from collections import Counter
    return Counter(a) == Counter(b)
"""
        topic = "anagrams"
        description = "Даны две строки. Определите, являются ли они анаграммами (одинаковые символы с одинаковой кратностью). Верните True/False."
        sample_inputs = [
            {"name": "s1", "input": ["listen", "silent"]},
            {"name": "s2", "input": ["abc", "cbaa"]},
        ]
        edge_inputs = [
            {"name": "edge_empty", "input": ["", ""]},
            {"name": "edge_case", "input": ["aabbcc", "abcabc"]},
        ]
    else:
        if track_norm in ("ml", "ds"):
            title = "Расчёт F1-метрики"
            function_signature = "def f1_score(tp: int, fp: int, fn: int) -> float:"
            starter_code = f"""{function_signature}
    # TODO: вернуть F1-метрику, аккуратно обрабатывая нулевые суммы
    pass
"""
            reference_solution = """def f1_score(tp: int, fp: int, fn: int) -> float:
    precision_den = tp + fp
    recall_den = tp + fn
    if precision_den == 0 or recall_den == 0:
        return 0.0
    precision = tp / precision_den
    recall = tp / recall_den
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
"""
            topic = "ml_metrics_f1"
            description = "Реализуйте функцию для расчёта F1-метрики по tp, fp, fn. При нулевых знаменателях верните 0.0."
            sample_inputs = [
                {"name": "sample_balanced", "input": [8, 2, 2]},
                {"name": "sample_zero_fp", "input": [5, 0, 3]},
            ]
            edge_inputs = [
                {"name": "edge_zero_tp", "input": [0, 5, 5]},
                {"name": "edge_all_zero", "input": [0, 0, 0]},
            ]
        else:
            title = "Агрегация статусов заказов"
            function_signature = "def aggregate_orders(orders: list[dict]) -> dict:"
            starter_code = f"""{function_signature}
    # TODO: посчитайте количество заказов по статусам
    pass
"""
            reference_solution = """def aggregate_orders(orders: list[dict]) -> dict:
    result = {}
    for o in orders:
        status = o.get("status", "unknown")
        result[status] = result.get(status, 0) + 1
    return result
"""
            topic = "orders_status_aggregation"
            description = "Дан список заказов в виде словарей со статусом. Верните словарь с количеством заказов по каждому статусу."
            sample_inputs = [
                {"name": "sample_backend", "input": [[{"id": 1, "status": "new"}, {"id": 2, "status": "done"}, {"id": 3, "status": "new"}]]},
                {"name": "sample_mixed", "input": [[{"status": "done"}, {"status": "cancelled"}, {"status": "done"}]]},
            ]
            edge_inputs = [
                {"name": "edge_empty", "input": [[]]},
                {"name": "edge_missing", "input": [[{"id": 1}, {"id": 2, "status": "new"}]]},
            ]

    return {
        "task_id": tid,
        "title": title,
        "description_markdown": description,
        "track": track_norm,
        "level": level,
        "category": category_norm,
        "language": lang,
        "allowed_languages": [lang],
        "function_signature": function_signature,
        "starter_code": starter_code,
        "constraints": [],
        "reference_solution": reference_solution,
        "topic": topic,
        "sample_inputs": sample_inputs,
        "edge_case_inputs": edge_inputs,
    }


def parse_function_name(signature: str) -> Optional[str]:
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", signature or "")
    return m.group(1) if m else None


def _param_count(signature: str) -> Optional[int]:
    m = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\((.*?)\)", signature or "", re.S)
    if not m:
        return None
    raw_params = m.group(1)
    params = []
    for part in raw_params.split(","):
        p = part.strip()
        if not p or p.startswith("*"):
            continue
        params.append(p)
    return len(params)


def _build_call_expr(fn_name: str, args: object, param_count: Optional[int]) -> str:
    if isinstance(args, dict):
        return f"{fn_name}(**{json.dumps(args, ensure_ascii=False)})"
    if param_count == 1:
        val = args[0] if isinstance(args, (list, tuple)) and len(args) == 1 else args
        return f"{fn_name}({json.dumps(val, ensure_ascii=False)})"
    if not isinstance(args, (list, tuple)):
        args = [args]
    return f"{fn_name}(*{json.dumps(list(args), ensure_ascii=False)})"


def _run_python_reference(task: dict, args: list):
    fn_name = parse_function_name(task.get("function_signature") or "")
    if not fn_name:
        raise ValueError("Cannot parse function name from signature")
    param_count = _param_count(task.get("function_signature") or "")
    call_expr = _build_call_expr(fn_name, args, param_count)
    code = task.get("reference_solution") or ""
    wrapper = f"""
import json
{code}
res = {call_expr}
print(json.dumps(res, ensure_ascii=False))
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
        f.write(wrapper)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        out = result.stdout.strip()
        return json.loads(out) if out else None
    finally:
        try:
            subprocess.run(["rm", "-f", path], check=False)
        except Exception:
            pass


def build_tests_for_task(task: dict) -> tuple[list[dict], list[dict]]:
    language = (task.get("language") or "python").lower()
    if language not in ("python",):
        raise NotImplementedError(f"Language {language} not supported for test generation")

    sample_inputs = task.get("sample_inputs") or []
    edge_inputs = task.get("edge_case_inputs") or []
    all_inputs = sample_inputs + edge_inputs

    public_tests: list[dict] = []
    hidden_tests: list[dict] = []

    for idx, inp in enumerate(all_inputs):
        args = inp.get("input") or []
        expected = None
        if language == "python":
            try:
                expected = _run_python_reference(task, args)
                logger.info("Computed expected for test", extra={"task_id": task.get("task_id"), "test_name": inp.get("name"), "expected": expected})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to compute expected for test; leaving expected=None", extra={"task_id": task.get("task_id"), "test_name": inp.get("name"), "error": str(exc)})
                expected = None
        test_obj = {"name": inp.get("name") or f"test_{idx+1}", "input": args, "expected": expected}
        if idx < len(sample_inputs):
            public_tests.append(test_obj)
        else:
            hidden_tests.append(test_obj)
    return public_tests, hidden_tests
