import requests

BASE_URL = "http://127.0.0.1:8001/api/v1/platform"

def test_integration():
    print("Testing Remote Run from Web UI Simulation...")
    # Simulate what Web UI does
    payload = {"project_id": "prj-web", "prompt": "Can you analyze this local dataset?"}
    headers = {"X-API-Key": "YOUR_API_KEY_HERE"}
    
    res = requests.post(f"{BASE_URL}/runs", params=payload, headers=headers)
    print(res.status_code, res.json())

if __name__ == "__main__":
    test_integration()
