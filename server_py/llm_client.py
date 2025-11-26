from openai import OpenAI
import os
import logging
import json
from pathlib import Path


BASE_URL = "https://llm.t1v.scibox.tech/v1"
DEFAULT_TIMEOUT = 30

def load_env_key():
    """
    Подхватываем SCIBOX_API_KEY из окружения или .env/.env.local.
    """
    if os.environ.get("SCIBOX_API_KEY"):
        return os.environ["SCIBOX_API_KEY"]
    for fname in [".env.local", ".env"]:
        path = Path(__file__).resolve().parent.parent / fname
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "SCIBOX_API_KEY":
                    os.environ.setdefault("SCIBOX_API_KEY", v.strip())
                    return v.strip()
    return None

api_key = load_env_key() or "dev-token"

try:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )
except Exception as exc:
    # Если установлен старый openai-клиент — не падаем, логируем и используем заглушку
    logging.exception("Failed to init OpenAI client", extra={"base_url": BASE_URL})
    client = None

logger = logging.getLogger(__name__)


def chat_completion(model: str, messages: list, **kwargs):
    """
    Унифицированный хелпер для вызова чат-моделей.
    По умолчанию выключаем streaming.
    """
    params = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.2),
        "max_tokens": kwargs.get("max_tokens", 800),
        "stream": kwargs.get("stream", False),
    }
    if kwargs.get("response_format"):
        params["response_format"] = kwargs["response_format"]
    try:
        if client is None:
            raise RuntimeError("LLM client is not initialized")
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        return content
    except Exception as exc:
        logger.exception(
            "LLM chat_completion failed, using fallback stub",
            extra={"model": model, "base_url": BASE_URL, "api_key_present": bool(api_key and api_key != 'dev-token')},
        )
        fallback = {
            "question": "Резервный вопрос: расскажите, как работает HTTP/REST и чем отличаются GET/POST.",
            "title": "HTTP/REST основы",
            "track": "fullstack",
            "level": "middle",
            "estimated_answer_time_min": 3,
            "key_points": [
                "HTTP методы и их идемпотентность",
                "Статусы ответа",
                "Заголовки и тело запроса",
                "Что такое REST и ресурсы",
                "Кэширование и безопасность на уровне HTTP",
            ],
            "max_score": 10,
        }
        fb = kwargs.get("fallback_json")
        if fb:
            return fb
        return json.dumps(fallback, ensure_ascii=False)
