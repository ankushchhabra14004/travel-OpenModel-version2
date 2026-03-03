#!/usr/bin/env python3
"""
Simple Llama 3.1 8B connectivity test
"""
import subprocess
import json

def test_llama():
    print("🦙 Testing Llama 3.1 8B at http://192.168.2.5:8000...")
    
    try:
        result = subprocess.run([
            'curl', '-s', '--connect-timeout', '5', '--max-time', '30',
            'http://192.168.2.5:8000/v1/completions',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps({
                "model": "./llama",
                "prompt": "Hello, what is a visa?",
                "max_tokens": 100,
                "temperature": 0.3
            })
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                if 'choices' in response and response['choices']:
                    text = response['choices'][0]['text'].strip()
                    print(f"✅ SUCCESS! Llama responded:")
                    print(f"📝 {text}")
                    return True
            except:
                pass
        
        print(f"❌ FAILED - Connection error or invalid response")
        print(f"Status: {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ FAILED - {e}")
        return False

if __name__ == "__main__":
    test_llama()