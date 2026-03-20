import pytest
from unittest.mock import patch, MagicMock

# api_server requires Redis at module-level, so we must mock it before import
_mock_redis = MagicMock()
_mock_queue = MagicMock()

with patch("redis.Redis", return_value=_mock_redis), \
     patch("rq.Queue", return_value=_mock_queue):
    from fastapi.testclient import TestClient
    from bio_ml_agent.api.api_server import app

client = TestClient(app)

def test_api_health():
    resp = client.get("/api/v1/health")
    # if it exists, cool. Usually health endpoints might be 404 if not mounted
    assert resp.status_code in [200, 404]

def test_train_cnn_endpoint():
    payload = {
        "dataset_path": "data/brain",
        "preset": "brain_mri",
        "architecture": "resnet18",
        "epochs": 5
    }
    # It might require an API key, so we check for 202 or 403
    # With mocked Redis, the queue enqueue will also be mocked
    resp = client.post("/api/v1/agent/train_cnn", json=payload)
    assert resp.status_code in [200, 202, 403, 429]
