import os
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from fastapi import HTTPException, status

_ENV_LOADED = False


def _load_env_files() -> None:
    """
    Простейшая подгрузка .env/.env.local без сторонних зависимостей.
    Нужна, чтобы LLM_API_KEY/LLM_BASE_URL можно было задать в env-файле.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for fname in [".env.local", ".env"]:
        p = Path(__file__).resolve().parent.parent / fname
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    _ENV_LOADED = True


# Поддерживаем обе версии SDK: новый клиент openai>=1.x и старый openai<1.x
try:
    from openai import (  # type: ignore
        OpenAI,
        APIError,
        APIConnectionError,
        RateLimitError,
        BadRequestError,
        AuthenticationError,
        NotFoundError,
    )
    _LEGACY = False
except Exception:  # noqa: BLE001
    import openai  # type: ignore

    OpenAI = None  # type: ignore
    APIConnectionError = AuthenticationError = RateLimitError = APIError = BadRequestError = NotFoundError = None  # type: ignore
    try:
        from openai.error import (  # type: ignore
            APIConnectionError,
            AuthenticationError,
            RateLimitError,
            InvalidRequestError as BadRequestError,
            APIError,
        )
        try:
            from openai.error import NotFoundError  # type: ignore
        except Exception:  # noqa: BLE001
            NotFoundError = BadRequestError  # type: ignore
    except Exception:  # noqa: BLE001
        pass
    _LEGACY = True
    openai  # silence linter

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Единая ошибка для работы с LLM/SciBox."""


def _raise_http(detail: str, code: int) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def _handle_openai_error(exc: Exception) -> HTTPException:
    request_id = getattr(exc, "request_id", None)
    base_detail = f"LLM error: {exc}"
    if request_id:
        base_detail += f" (request_id={request_id})"
    if isinstance(exc, AuthenticationError):
        return _raise_http("Неверный API ключ LLM", status.HTTP_401_UNAUTHORIZED)
    if isinstance(exc, RateLimitError):
        return _raise_http("Превышен лимит запросов к LLM, попробуйте позже", status.HTTP_429_TOO_MANY_REQUESTS)
    if isinstance(exc, BadRequestError):
        return _raise_http(f"Некорректный запрос к LLM: {exc}", status.HTTP_400_BAD_REQUEST)
    if isinstance(exc, NotFoundError):
        return _raise_http("Ресурс LLM не найден", status.HTTP_404_NOT_FOUND)
    if isinstance(exc, APIConnectionError):
        return _raise_http("Не удалось подключиться к LLM сервису", status.HTTP_503_SERVICE_UNAVAILABLE)
    if isinstance(exc, APIError):
        return _raise_http("Ошибка LLM сервиса, попробуйте позже", status.HTTP_502_BAD_GATEWAY)
    return _raise_http(base_detail, status.HTTP_500_INTERNAL_SERVER_ERROR)


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        _load_env_files()
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://llm.t1v.scibox.tech/v1")
        if not api_key:
            raise LLMError("LLM_API_KEY не задан")
        self.api_key = api_key
        self.base_url = base_url
        if not _LEGACY:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            import openai  # type: ignore

            # Настраиваем старый SDK
            openai.api_key = api_key
            if base_url:
                openai.api_base = base_url
            self.client = openai

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        try:
            if not _LEGACY:
                if stream:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                    )

                    def _gen() -> Generator[str, None, None]:
                        try:
                            for chunk in resp:  # type: ignore[operator]
                                delta = chunk.choices[0].delta
                                text = delta.content or ""
                                if text:
                                    yield text
                        except Exception as exc:  # noqa: BLE001
                            raise _handle_openai_error(exc)

                    return _gen()
                else:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=False,
                    )
                    return resp.choices[0].message.content or ""
            else:
                import openai  # type: ignore

                if stream:
                    # Старый SDK не давал простого стриминга через генератор в том же виде,
                    # поэтому используем обычный вызов и вернём единственный chunk.
                    resp = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=False,
                    )
                    text = resp["choices"][0]["message"]["content"]

                    def _gen() -> Generator[str, None, None]:
                        if text:
                            yield text

                    return _gen()
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
                return resp["choices"][0]["message"]["content"]
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _handle_openai_error(exc)

    def embed(self, input: Union[str, Iterable[str]], model: str = "bge-m3") -> List[List[float]]:
        try:
            if not _LEGACY:
                resp = self.client.embeddings.create(
                    model=model,
                    input=input,
                )
                return [item.embedding for item in resp.data]
            else:
                import openai  # type: ignore

                resp = openai.Embedding.create(model=model, input=input)
                data = resp.get("data", [])
                return [item["embedding"] for item in data]
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _handle_openai_error(exc)


# Готовый клиент для переиспользования
llm_client = None
def get_llm_client() -> LLMClient:
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client
