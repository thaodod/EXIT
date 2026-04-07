import os
import time
from typing import Any, Dict, Optional

import requests

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEFAULT_MODEL = "gemini-2.5-flash"
CLIENT = (
    genai.Client(
        vertexai=True,
        project="savefiles-185217",
        location="us-central1",  # for llama 3 series
        # location="us-east5",  # for llama 4 series
    )
    if genai is not None
    else None
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


def _extract_chat_completion_text(response_data: Dict[str, Any]) -> str:
    choices = response_data.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected chat completion response: {response_data}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts).strip()

    return str(content).strip()


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
               api_key: Optional[str] = None):
    """ 
    Returns:
        Generated text response or None if failed
    """
    
    retries = 0
    if mock:
        return "This is a mock response."

    # Routing priority:
    # 1. Explicit OpenAI-compatible server (for vLLM or OpenAI-compatible gateways)
    # 2. Vertex AI for Gemini / MaaS models
    # 3. OpenRouter fallback for the legacy non-Vertex API path
    use_openai_compatible = bool(api_base_url and api_base_url.strip())
    use_vertex = bool(model) and ("gemini" in model.lower() or "maas" in model.lower())
    
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
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_output_tokens,
                }

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
                return _extract_chat_completion_text(response_data)

            if use_vertex:
                # Use Vertex AI for Gemini models
                if CLIENT is None or types is None:
                    raise ImportError(
                        "google-genai is required for Vertex AI calls. Install it before using Gemini/MaaS models."
                    )

                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                    response_mime_type="text/plain",
                )
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt)
                        ]
                    ),
                ]
                
                response_stream = CLIENT.models.generate_content_stream(
                    model=model, 
                    contents=contents, 
                    config=generate_content_config
                )
                full_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                return full_response.strip()
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
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_output_tokens,
                    'provider': {
                        'quantizations': ['bf16', 'fp16', 'fp8']
                    }
                }
                
                response = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=request_timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                return response_data['choices'][0]['message']['content'].strip()

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                return None
            time.sleep(retry_delay)
            retry_delay *= 1.05


if __name__ == "__main__":
    # Example with Vertex AI (Gemini model)
    response = ask_vertex(
        "How close are dogs and wolves related?",
        model="gemini-2.5-flash-lite",
    )
    print("Vertex AI Response:", response)
    
    # Example with OpenRouter (non-Gemini/Llama model)
    response2 = ask_vertex(
        "What is the capital of France?",
        model="openai/gpt-oss-120b",
    )
    print("OpenRouter Response:", response2)
