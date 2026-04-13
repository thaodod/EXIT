import os
import time
from typing import Optional
from openai import OpenAI

# Initialize the OpenAI client lazily
_CLIENT: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        # Load API key from .vscode/openrouter_api_key.txt
        api_key_file = os.path.join(os.path.dirname(__file__), '.vscode', 'openrouter_api_key.txt')
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"OpenRouter API key not found at {api_key_file}")
            
        _CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _CLIENT

def call_qwen(prompt: str,
              max_retries: int = 10,
              retry_delay: float = 2.0,
              mock: bool = False,
              temperature: float = 0.0001) -> Optional[str]:
    """
    Utility function to call `qwen3.5-flash-02-23` via OpenRouter.
    Returns only the final answer string, disposing of the reasoning details.
    """
    if mock:
        return "This is a mock response from qwen3.5-flash-02-23."

    client = _get_client()
    retries = 0
    
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.5-flash-02-23",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                # Enable thinking/reasoning as requested
                extra_body={"reasoning": {"enabled": True}}
            )
            
            # Dispose of the reasoning metadata, simply take the final answer
            return response.choices[0].message.content.strip()

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Failed after {max_retries} attempts calling qwen3.5-flash-02-23. Error: {str(e)}")
                return None
            time.sleep(retry_delay)
            retry_delay *= 1.05

if __name__ == "__main__":
    # Test the utility
    ans = call_qwen("Think carefully, then give only the final count of 'r's in the word 'strawberry'.")
    print("Final Answer:")
    print(ans)