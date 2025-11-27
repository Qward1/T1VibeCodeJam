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
    ?????????????????? ????? ???-?????? ??? ????????.
    ?? ????????? ????????? streaming.
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
        msg = resp.choices[0].message
        if kwargs.get("response_format") and getattr(msg, "parsed", None) is not None:
            try:
                return json.dumps(msg.parsed, ensure_ascii=False)
            except Exception:
                return msg.parsed
        content = msg.content
        if isinstance(content, list):
            content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
        return content
    except Exception as exc:
        fb = kwargs.get("fallback_json")
        if fb is not None:
            return fb
        raise
