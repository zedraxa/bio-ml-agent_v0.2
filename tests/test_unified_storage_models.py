import pytest
from datetime import datetime, timezone
from models.unified_storage import (
    LocalWorkspaceCache, CacheItemStatus,
    ObjectStorageArtifact, ArtifactTier,
    ExperimentRegistryMeta,
    VectorMemoryIndex,
    RelationalMetadataRecord,
    AuditTrailEvent, AuditLogLevel,
    EncryptedSecretVault, SecretScope
)

def test_local_workspace_cache():
    cache = LocalWorkspaceCache(
        cache_id="c-123",
        project_id="p-1",
        local_path="/tmp/cache/f1",
        size_bytes=1024,
        last_accessed=datetime.now(timezone.utc),
        status=CacheItemStatus.DIRTY
    )
    assert cache.cache_id == "c-123"
    assert cache.status == CacheItemStatus.DIRTY

def test_object_storage_artifact():
    artifact = ObjectStorageArtifact(
        artifact_id="art-456",
        project_id="p-1",
        object_key="models/v1.pt",
        bucket_name="ml-bucket",
        created_at=datetime.now(timezone.utc)
    )
    assert artifact.tier == ArtifactTier.HOT
    assert artifact.bucket_name == "ml-bucket"

def test_experiment_registry_meta():
    experiment = ExperimentRegistryMeta(
        experiment_id="exp-789",
        run_name="resnet-train-1",
        model_architecture="resnet50",
        hyperparameters={"lr": 0.001, "batch_size": 32},
        metrics={"loss": 0.5},
        created_at=datetime.now(timezone.utc)
    )
    assert experiment.model_architecture == "resnet50"
    assert experiment.hyperparameters["lr"] == 0.001

def test_vector_memory_index():
    index = VectorMemoryIndex(
        index_id="idx-001",
        collection_name="papers",
        embedding_model="text-embedding-3",
        dimension_size=1536,
        last_updated=datetime.now(timezone.utc)
    )
    assert index.dimension_size == 1536

def test_relational_metadata_record():
    record = RelationalMetadataRecord(
        record_id="rec-002",
        entity_type="user",
        entity_id="u-99",
        attributes={"role": "admin"},
        updated_at=datetime.now(timezone.utc)
    )
    assert record.attributes["role"] == "admin"
    assert record.schema_version == "1.0"

def test_audit_trail_event():
    event = AuditTrailEvent(
        event_id="evt-003",
        timestamp=datetime.now(timezone.utc),
        actor_id="u-99",
        action="delete_project",
        target_resource="p-1",
        level=AuditLogLevel.WARNING
    )
    assert event.level == AuditLogLevel.WARNING

def test_encrypted_secret_vault():
    vault = EncryptedSecretVault(
        secret_id="sec-004",
        scope=SecretScope.PROJECT,
        owner_id="p-1",
        key_name="AWS_ACCESS_KEY",
        encrypted_value="enc_v1_dGhpcyBpcyBhIHNlY3JldA==",
        created_at=datetime.now(timezone.utc)
    )
    assert vault.scope == SecretScope.PROJECT
    assert vault.key_name == "AWS_ACCESS_KEY"
