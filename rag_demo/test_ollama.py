import requests
import json
import time

def test_ollama():
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": "qwen3:4b",
        "prompt": "微信支付结算周期"
    }
    
    print(f"🚀 Sending EMBEDDING request to Ollama ({payload['model']})...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        duration = time.time() - start_time
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"⏱️ Time Taken: {duration:.2f}s")
        
        embedding = data.get("embedding")
        if embedding:
            print(f"✅ Embedding generated! Dimension: {len(embedding)}")
            print(f"SAMPLE: {embedding[:5]}...")
        else:
            print("❌ No embedding found in response")
            print(json.dumps(data, indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ollama()
