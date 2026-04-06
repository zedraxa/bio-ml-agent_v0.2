from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
import logging

from bio_ml_agent.models.unified_storage import (
    LocalWorkspaceCache, CacheItemStatus,
    ObjectStorageArtifact, ArtifactTier,
    ExperimentRegistryMeta,
    VectorMemoryIndex,
    RelationalMetadataRecord,
    AuditTrailEvent, AuditLogLevel,
    EncryptedSecretVault, SecretScope
)
from bio_ml_agent.db.session import SessionLocal
from bio_ml_agent.db.models import ArtifactDB, TimelineEventDB, ProjectDB
from bio_ml_agent.models.workspace_ux import TimelineEventType
import time

logger = logging.getLogger(__name__)

class UnifiedStorageService:
    def __init__(self):
        # 1. Local Workspace Cache
        self._cache_store: Dict[str, LocalWorkspaceCache] = {}
        # 2. Object Storage Metadata
        self._artifact_store: Dict[str, ObjectStorageArtifact] = {}
        # 3. Experiment Registry
        self._experiment_store: Dict[str, ExperimentRegistryMeta] = {}
        # 4. Vector Memory Store Meta
        self._vector_store: Dict[str, VectorMemoryIndex] = {}
        # 5. Relational Metadata
        self._metadata_store: Dict[str, RelationalMetadataRecord] = {}
        # 6. Observability & Audit
        self._audit_store: List[AuditTrailEvent] = []
        # 7. Secret Vault
        self._secret_store: Dict[str, EncryptedSecretVault] = {}

    # --- 1. Local Workspace Cache ---
    def sync_local_cache(self, project_id: str, local_path: str, size: int) -> LocalWorkspaceCache:
        cache_id = f"cache-{uuid.uuid4().hex[:8]}"
        cache_item = LocalWorkspaceCache(
            cache_id=cache_id,
            project_id=project_id,
            local_path=local_path,
            size_bytes=size,
            last_accessed=datetime.now(timezone.utc),
            status=CacheItemStatus.SYNCED
        )
        self._cache_store[cache_id] = cache_item
        return cache_item

    def get_cache_by_project(self, project_id: str) -> List[LocalWorkspaceCache]:
        return [c for c in self._cache_store.values() if c.project_id == project_id]

    # --- 2. Object Storage Artifact ---
    def register_artifact(self, project_id: str, key: str, bucket: str, run_id: Optional[str] = None, title: str = "Unnamed Artifact", category: str = "DATA") -> ObjectStorageArtifact:
        art_id = f"art-{uuid.uuid4().hex[:8]}"
        artifact_pydantic = ObjectStorageArtifact(
            artifact_id=art_id,
            project_id=project_id,
            run_id=run_id,
            object_key=key,
            bucket_name=bucket,
            created_at=datetime.now(timezone.utc)
        )
        self._artifact_store[art_id] = artifact_pydantic

        # Sync to DB Truth Layer
        db = SessionLocal()
        try:
            db_art = ArtifactDB(
                artifact_id=art_id,
                project_id=project_id,
                mission_id=run_id,
                title=title,
                category=category,
                file_type=key.split(".")[-1] if "." in key else "unknown",
                created_by="agent-system",
                content_uri=f"s3://{bucket}/{key}",
                created_at=time.time(),
                updated_at=time.time()
            )
            db.add(db_art)
            db.commit()
            logger.info(f"[Storage] Artifact {art_id} registered in DB.")
        except Exception as e:
            logger.error(f"[Storage] Failed to sync artifact {art_id} to DB: {e}")
            db.rollback()
        finally:
            db.close()

        return artifact_pydantic

    # --- 3. Experiment Registry ---
    def create_experiment_run(self, run_name: str, arch: str, params: dict) -> ExperimentRegistryMeta:
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        exp = ExperimentRegistryMeta(
            experiment_id=exp_id,
            run_name=run_name,
            model_architecture=arch,
            hyperparameters=params,
            created_at=datetime.now(timezone.utc)
        )
        self._experiment_store[exp_id] = exp
        return exp

    # --- 4. Relational Metadata Store ---
    def upsert_metadata(self, entity_type: str, entity_id: str, attributes: dict) -> RelationalMetadataRecord:
        record_id = f"meta-{entity_type}-{entity_id}"
        record = RelationalMetadataRecord(
            record_id=record_id,
            entity_type=entity_type,
            entity_id=entity_id,
            attributes=attributes,
            updated_at=datetime.now(timezone.utc)
        )
        self._metadata_store[record_id] = record
        return record

    def get_metadata(self, entity_type: str, entity_id: str) -> Optional[RelationalMetadataRecord]:
        return self._metadata_store.get(f"meta-{entity_type}-{entity_id}")

    # --- 5. Observability & Audit Trail ---
    def log_audit_event(self, actor: str, action: str, resource: str, level: AuditLogLevel = AuditLogLevel.INFO, details: dict = None, project_id: Optional[str] = None) -> AuditTrailEvent:
        event_id = f"evt-{uuid.uuid4().hex[:8]}"
        event = AuditTrailEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            actor_id=actor,
            action=action,
            target_resource=resource,
            level=level,
            details=details or {}
        )
        self._audit_store.append(event)

        # Sync to Timeline if linked to a project
        if project_id:
            db = SessionLocal()
            try:
                db_event = TimelineEventDB(
                    event_id=event_id,
                    project_id=project_id,
                    event_type=TimelineEventType.INFO,
                    message=f"{actor} performed {action} on {resource}",
                    timestamp=time.time(),
                    metadata_json=details or {},
                    agent_name=actor if "agent" in actor.lower() else None
                )
                db.add(db_event)
                db.commit()
            except Exception as e:
                logger.error(f"[Storage] Failed to sync audit event to timeline: {e}")
                db.rollback()
            finally:
                db.close()

        logger.info(f"[AUDIT] {actor} -> {action} on {resource} [{level.value}]")
        return event

    def list_audit_logs(self, limit: int = 50) -> List[AuditTrailEvent]:
        # Return most recent
        return sorted(self._audit_store, key=lambda x: x.timestamp, reverse=True)[:limit]

    # --- 6. Secret Management ---
    def store_secret(self, scope: SecretScope, owner: str, name: str, value: str) -> EncryptedSecretVault:
        sec_id = f"sec-{uuid.uuid4().hex[:8]}"
        vault = EncryptedSecretVault(
            secret_id=sec_id,
            scope=scope,
            owner_id=owner,
            key_name=name,
            encrypted_value=f"enc_mock_{value}", # Mock encryption wrapper
            created_at=datetime.now(timezone.utc)
        )
        self._secret_store[sec_id] = vault
        return vault

    def get_secret(self, owner: str, name: str) -> Optional[str]:
        # Only simulate finding internal key value logic
        for vault in self._secret_store.values():
            if vault.owner_id == owner and vault.key_name == name:
                # Mock decryption logic
                return str(vault.encrypted_value).replace("enc_mock_", "")
        return None

# Global singleton storage service dispatcher
storage_service = UnifiedStorageService()
