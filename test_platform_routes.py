import requests

BASE_URL = "http://127.0.0.1:8001/api/v1/platform"

def test_routes():
    print("Testing Auth Login...")
    res = requests.post(f"{BASE_URL}/auth/login")
    print(res.status_code, res.json())
    
    print("\nTesting Project Creation...")
    res = requests.post(f"{BASE_URL}/projects?name=TestProject")
    print(res.status_code, res.json())
    
    print("\nTesting Dashboard Summary...")
    res = requests.get(f"{BASE_URL}/dashboard/summary")
    print(res.status_code, res.json())

if __name__ == "__main__":
    test_routes()
