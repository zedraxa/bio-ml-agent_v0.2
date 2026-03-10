import requests

BASE_URL = "http://127.0.0.1:8001/api/v1/platform"

def test_middleware():
    print("Testing with NO API KEY (Should Fail)")
    res = requests.post(f"{BASE_URL}/runs", params={"project_id": "p1", "prompt": "hi"})
    print("Response NO KEY:", res.status_code)

    print("\nTesting with VALID API KEY (Should Pass)")
    headers = {"X-API-Key": "YOUR_API_KEY_HERE"}
    res = requests.post(f"{BASE_URL}/runs", params={"project_id": "p1", "prompt": "hi"}, headers=headers)
    print("Response VALID KEY:", res.status_code)

if __name__ == "__main__":
    test_middleware()
