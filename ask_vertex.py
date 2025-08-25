import time
import requests
from google import genai
from google.genai import types
import os

DEFAULT_MODEL = "gemini-2.5-flash"
CLIENT = genai.Client(
    vertexai=True,
    project="savefiles-185217",
    location="us-central1", ## for llama 3 series
    # location="us-east5", ## for llama 4 series
)

# Load OpenRouter API key from file
cache_dir = os.path.expanduser("./cache")
api_key_file = os.path.join(cache_dir, "openrouter_api_key.txt")

try:
    with open(api_key_file, 'r') as f:
        OPENROUTER_API_KEY = f.read().strip()
except Exception as e:
    print(f"Error reading OpenRouter API key: {e}")


def ask_vertex(prompt, 
               model=DEFAULT_MODEL,
               max_retries=4, 
               retry_delay=8, 
               mock=False,
               temperature=0.0001,
               top_p=1.0,
               max_output_tokens=360):
    """ 
    Returns:
        Generated text response or None if failed
    """
    
    retries = 0
    if mock:
        return "This is a mock response."

    # Check if model should use Vertex AI or OpenRouter
    use_vertex = "gemini" in model.lower() or "maas" in model.lower()
    
    while retries <= max_retries:
        try:
            if use_vertex:
                # Use Vertex AI for Gemini models
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
                headers = {
                    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
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
                        'quantizations': ['bf16', 'fp16']
                    }
                }
                
                response = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers=headers,
                    json=payload
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
            retry_delay *= 2


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