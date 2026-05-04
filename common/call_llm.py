import requests

def call_llm(messages):
    data = {
        "model": "Qwen/Qwen3.5-4B",
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.2,
        "stream": False,
        "max_tokens": 4096
    }
    
    response = requests.post("http://127.0.0.1:8299/v1/chat/completions", json=data)
    response = response.json()
    response_text = response["choices"][0]["message"]
    return response_text