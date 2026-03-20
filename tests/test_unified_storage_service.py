import pytest
from datetime import datetime
from bio_ml_agent.services.unified_storage_service import storage_service
from bio_ml_agent.models.unified_storage import (
    CacheItemStatus, ArtifactTier,
    SecretScope, AuditLogLevel
)

def test_unified_storage_service_flow():
    # 1. Cache
    cache = storage_service.sync_local_cache("proj-1", "/dataset", 1000)
    assert cache.project_id == "proj-1"
    assert len(storage_service.get_cache_by_project("proj-1")) == 1

    # 2. Artifact
    art = storage_service.register_artifact("proj-1", "model.pt", "s3-bucket")
    assert art.bucket_name == "s3-bucket"
    assert art.tier == ArtifactTier.HOT

    # 3. Experiment
    exp = storage_service.create_experiment_run("run-A", "CNN", {"lr": 0.01})
    assert exp.model_architecture == "CNN"

    # 4. Metadata
    meta = storage_service.upsert_metadata("user", "usr-1", {"role": "admin"})
    fetched_meta = storage_service.get_metadata("user", "usr-1")
    assert fetched_meta.attributes["role"] == "admin"

    # 5. Audit
    event = storage_service.log_audit_event("usr-1", "login", "system", AuditLogLevel.INFO)
    logs = storage_service.list_audit_logs()
    assert len(logs) > 0
    assert logs[0].actor_id == "usr-1"

    # 6. Secret
    secret = storage_service.store_secret(SecretScope.USER, "usr-1", "GITHUB_TOKEN", "ghp_123")
    val = storage_service.get_secret("usr-1", "GITHUB_TOKEN")
    assert val == "ghp_123"
