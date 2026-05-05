import requests

def call_llm(messages, model="Qwen/Qwen3.5-2B"):
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.2,
        "stream": False,
        "max_tokens": 4096
    }
    
    response = requests.post("http://127.0.0.1:8299/v1/chat/completions", json=data)
    res_data = response.json()
    
    # 💡 加入防呆機制：如果 vLLM 退件，把真實的錯誤原因印出來讓我們看見
    if "choices" not in res_data:
        print(f"\n[vLLM 拒絕處理]: {res_data}\n")
        return ""
        
    response_text = res_data["choices"][0]["message"]["content"]
    return response_text