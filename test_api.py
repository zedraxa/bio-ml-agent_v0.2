from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def run_tests():
    print("🚀 API Testleri Başlıyor...")
    
    # Kök dizin testi
    response = client.get("/")
    assert response.status_code == 200
    print("✅ GET / ->", response.json())
    
    # Hızlı trigger testi
    payload = {
        "dataset_path": "workspace/dl_demo/data/raw/brain_mri",
        "preset": "brain_mri",
        "architecture": "resnet18",
        "epochs": 1
    }
    
    # Veri seti gerçekten varsa asenkron test edelim
    response = client.post("/api/v1/agent/train_cnn", json=payload)
    assert response.status_code == 202
    res_data = response.json()
    print("✅ POST /api/v1/agent/train_cnn ->", res_data)
    
    task_id = res_data["task_id"]
    
    # Status endpoint testi
    status_res = client.get(f"/api/v1/agent/status/{task_id}")
    assert status_res.status_code == 200
    print("✅ GET /api/v1/agent/status/{task_id} ->", status_res.json()["status"])

if __name__ == "__main__":
    run_tests()
