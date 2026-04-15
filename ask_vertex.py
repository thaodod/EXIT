import os
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "savefiles-185217")
DEFAULT_VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
_VERTEX_CLIENTS: Dict[Tuple[str, str, int], Any] = {}
_THINK_BLOCK_RE = re.compile(
    r"^\s*<(?:think|thinking|reasoning|analysis)>\s*.*?\s*</(?:think|thinking|reasoning|analysis)>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_FENCE_RE = re.compile(
    r"^\s*```(?:think|thinking|reasoning|analysis)\s*\n.*?\n```\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_INLINE_ANSWER_RE = re.compile(
    r"^\s*(?:[-*]\s+|#+\s*)?(?:\*\*|__)?(?:final\s+answer|answer)(?:\*\*|__)?[ \t]*[:\-][ \t]*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ANSWER_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s+|#+\s*)?(?:\*\*|__)?(?:final\s+answer|answer)(?:\*\*|__)?[ \t]*[:\-]?[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_LEADING_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:[-*]\s+|#+\s*)?(?:\*\*|__)?(?:final\s+answer|answer)(?:\*\*|__)?[ \t]*[:\-]?[ \t]*",
    flags=re.IGNORECASE,
)
_WRAPPED_FENCE_RE = re.compile(
    r"^\s*```(?:text|plaintext|markdown)?\s*\n?(.*?)\n?```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _read_api_key_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _load_openrouter_api_key() -> str:
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    cache_dir = os.path.expanduser("./cache")
    api_key_file = os.path.join(cache_dir, "openrouter_api_key.txt")
    file_key = _read_api_key_file(api_key_file)
    if file_key:
        return file_key

    raise FileNotFoundError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY or put it in cache/openrouter_api_key.txt."
    )


def _resolve_openai_compatible_api_key(explicit_key: Optional[str]) -> str:
    if explicit_key is not None:
        return explicit_key.strip()

    for env_name in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value

    return ""


def _build_openai_compatible_chat_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("api_base_url cannot be empty.")

    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"

    if normalized.endswith("/v1/chat/completions") or normalized.endswith("/chat/completions"):
        return normalized

    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"

    return f"{normalized}/v1/chat/completions"


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _normalize_timeout_ms(request_timeout: Any) -> int:
    if isinstance(request_timeout, (tuple, list)):
        numeric_values = [float(value) for value in request_timeout if value is not None]
        timeout_seconds = max(numeric_values) if numeric_values else 120.0
        return max(1, int(timeout_seconds * 1000))
    if request_timeout is None:
        return 120000
    return max(1, int(float(request_timeout) * 1000))


def _get_vertex_client(timeout_ms: int):
    if genai is None or types is None:
        raise ImportError(
            "google-genai is required for Vertex AI calls. Install it before using Gemini models."
        )

    cache_key = (DEFAULT_VERTEX_PROJECT, DEFAULT_VERTEX_LOCATION, timeout_ms)
    client = _VERTEX_CLIENTS.get(cache_key)
    if client is None:
        client = genai.Client(
            vertexai=True,
            project=DEFAULT_VERTEX_PROJECT,
            location=DEFAULT_VERTEX_LOCATION,
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=timeout_ms,
            ),
        )
        _VERTEX_CLIENTS[cache_key] = client
    return client


def _sanitize_model_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    def strip_leading_reasoning_artifacts(value: str) -> str:
        cleaned_value = value.strip()
        while True:
            updated = _THINK_BLOCK_RE.sub("", cleaned_value, count=1).strip()
            updated = _THINK_FENCE_RE.sub("", updated, count=1).strip()
            if updated == cleaned_value:
                return cleaned_value
            cleaned_value = updated

    def extract_explicit_answer(value: str) -> Optional[str]:
        inline_matches = [
            match.group(1).strip()
            for match in _INLINE_ANSWER_RE.finditer(value)
            if match.group(1).strip()
        ]
        if inline_matches:
            return inline_matches[-1]

        label_matches = list(_ANSWER_LABEL_RE.finditer(value))
        if not label_matches:
            return None

        trailing_text = value[label_matches[-1].end():].strip()
        return trailing_text or None

    cleaned = strip_leading_reasoning_artifacts(text)
    explicit_answer = extract_explicit_answer(cleaned)
    if explicit_answer:
        cleaned = explicit_answer

    cleaned = strip_leading_reasoning_artifacts(cleaned)
    cleaned = _LEADING_ANSWER_PREFIX_RE.sub("", cleaned, count=1).strip()
    wrapped_match = _WRAPPED_FENCE_RE.fullmatch(cleaned)
    if wrapped_match:
        cleaned = wrapped_match.group(1).strip()
    return cleaned


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        item_type = content.get("type")
        if item_type in {"text", "output_text"}:
            text_value = content.get("text", "")
            if isinstance(text_value, dict):
                text_value = text_value.get("value", "")
            return text_value if isinstance(text_value, str) else ""
        return ""

    if isinstance(content, list):
        text_parts = []
        for item in content:
            part_text = _extract_content_text(item)
            if part_text:
                text_parts.append(part_text)
        return "".join(text_parts)

    return ""


def _extract_chat_completion_text(response_data: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    choices = response_data.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected chat completion response: {response_data}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    finish_reason = choices[0].get("finish_reason")
    text = _extract_content_text(content)
    if not text and not isinstance(content, (str, list, dict)):
        text = str(content)
    return _sanitize_model_answer(text), finish_reason


def _extract_vertex_response_text(response: Any) -> Tuple[str, Optional[str]]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = _enum_name(getattr(prompt_feedback, "block_reason", None))
            return "", block_reason or "PROMPT_BLOCKED"
        raise RuntimeError(f"Unexpected Vertex response without candidates: {response}")

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    text_parts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            text_parts.append(text)

    finish_reason = _enum_name(getattr(candidate, "finish_reason", None))
    return _sanitize_model_answer("".join(text_parts)), finish_reason


def ask_vertex(prompt,
               model=DEFAULT_MODEL,
               max_retries=10,
               retry_delay=2,
               mock=False,
               temperature=0.0001,
               top_p=1.0,
               max_output_tokens=360,
               request_timeout=(10, 120),
               api_base_url: Optional[str] = None,
               api_key: Optional[str] = None,
               return_metadata: bool = False):
    """ 
    Returns:
        Generated text response or None if failed
    """
    
    retries = 0
    provider = "vertex" if model and "gemini" in model.lower() else "other"
    if mock:
        mock_result = {
            "text": "This is a mock response.",
            "ok": True,
            "finish_reason": "MOCK",
            "provider": provider,
            "error": None,
        }
        return mock_result if return_metadata else mock_result["text"]

    # Routing priority:
    # 1. Explicit OpenAI-compatible server (for vLLM or OpenAI-compatible gateways)
    # 2. Vertex AI for Gemini models
    # 3. OpenRouter fallback for the legacy non-Vertex API path
    use_openai_compatible = bool(api_base_url and api_base_url.strip())
    use_vertex = bool(model) and ("gemini" in model.lower())
    provider = "openai_compatible" if use_openai_compatible else ("vertex" if use_vertex else "openrouter")
    timeout_ms = _normalize_timeout_ms(request_timeout)
    
    while retries <= max_retries:
        try:
            if use_openai_compatible:
                request_url = _build_openai_compatible_chat_url(api_base_url)
                resolved_api_key = _resolve_openai_compatible_api_key(api_key)

                headers = {
                    "Content-Type": "application/json",
                }
                if resolved_api_key:
                    headers["Authorization"] = f"Bearer {resolved_api_key}"

                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": max_output_tokens,
                }
                if temperature is not None:
                    payload["temperature"] = temperature
                if top_p is not None:
                    payload["top_p"] = top_p

                response = requests.post(
                    request_url,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
                if response.status_code >= 400:
                    try:
                        error_payload = response.json()
                    except Exception:
                        error_payload = response.text
                    raise RuntimeError(
                        f"OpenAI-compatible request failed ({response.status_code}) at {request_url}: {error_payload}"
                    )

                response_data = response.json()
                text, finish_reason = _extract_chat_completion_text(response_data)
                result = {
                    "text": text,
                    "ok": True,
                    "finish_reason": finish_reason,
                    "provider": provider,
                    "error": None,
                }
                return result if return_metadata else result["text"]

            if use_vertex:
                client = _get_vertex_client(timeout_ms)

                # Gemini 3 reasoning models should use thinking_level rather than the legacy budget.
                generate_content_config = types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW,
                    ),
                    response_mime_type="text/plain",
                )
                if temperature not in (None, 0.0001):
                    generate_content_config.temperature = temperature
                if top_p not in (None, 1.0):
                    generate_content_config.top_p = top_p

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=generate_content_config,
                )
                text, finish_reason = _extract_vertex_response_text(response)
                result = {
                    "text": text,
                    "ok": True,
                    "finish_reason": finish_reason,
                    "provider": provider,
                    "error": None,
                }
                return result if return_metadata else result["text"]
            else:
                # Use OpenRouter for other models
                openrouter_api_key = _load_openrouter_api_key()
                headers = {
                    'Authorization': f'Bearer {openrouter_api_key}',
                    'Content-Type': 'application/json',
                }
                
                payload = {
                    'model': model,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'max_tokens': max_output_tokens,
                    'reasoning': {
                        'exclude': True,
                    },
                    'provider': {
                        'quantizations': ['bf16', 'fp16', 'fp8']
                    }
                }
                if temperature is not None:
                    payload['temperature'] = temperature
                if top_p is not None:
                    payload['top_p'] = top_p
                
                response = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=request_timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                text, finish_reason = _extract_chat_completion_text(response_data)
                result = {
                    "text": text,
                    "ok": True,
                    "finish_reason": finish_reason,
                    "provider": provider,
                    "error": None,
                }
                return result if return_metadata else result["text"]

        except Exception as e:
            retries += 1
            if retries > max_retries:
                error_message = f"Failed after {max_retries} attempts. Error: {str(e)}"
                print(error_message)
                if return_metadata:
                    return {
                        "text": "",
                        "ok": False,
                        "finish_reason": None,
                        "provider": provider,
                        "error": str(e),
                    }
                return None
            sleep_seconds = min(float(retry_delay) * (2 ** (retries - 1)), 30.0)
            time.sleep(sleep_seconds * random.uniform(0.8, 1.2))


if __name__ == "__main__":
    # Example with Vertex AI (Gemini model)
    response = ask_vertex(
        "How close are dogs and wolves related?",
        model="gemini-3-flash-preview",
    )
    print("Vertex AI Response:", response)
    
    # Example with OpenRouter (non-Gemini/Llama model)
    response2 = ask_vertex(
        "What is the capital of France?",
        model="openai/gpt-oss-120b",
    )
    print("OpenRouter Response:", response2)
