import logging
from typing import Dict, Any, List

from temporalio import activity

log = logging.getLogger("bio_ml_agent")


@activity.defn
async def index_workspace_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    S3-3: Workspace'i otonom olarak tarar ve Qdrant (RAG) indeksini günceller.
    """
    from bio_ml_agent.utils.config import get_config
    from bio_ml_agent.ultra_agent.rag.ingestion import FileParser
    from bio_ml_agent.ultra_agent.memory.qdrant_store import QdrantMemoryStore
    from bio_ml_agent.ultra_agent.memory.schema import MemoryEntry
    from pathlib import Path

    app_config = get_config()
    workspace_path = Path(params.get("workspace", app_config.workspace.base_dir))
    project_name = params.get("project", "default")

    log.info(f"Temporal Activity: RAG İndeksleme Başlatıldı -> {workspace_path}")

    if not workspace_path.exists():
        return {"status": "error", "reason": f"Workspace dizini bulunamadı: {workspace_path}"}

    parser = FileParser()
    store = QdrantMemoryStore(
        host=app_config.memory.qdrant.host,
        port=app_config.memory.qdrant.port,
        collection_name=app_config.memory.qdrant.collection
    )

    indexed_count = 0
    try:
        # Desteklenen tüm dosyaları tara
        for ext in parser.SUPPORTED_EXTENSIONS:
            for file_path in workspace_path.glob(f"**/*{ext}"):
                if "__pycache__" in str(file_path) or ".git" in str(file_path):
                    continue

                try:
                    chunks = parser.parse_file(file_path)
                    for chunk in chunks:
                        entry = MemoryEntry(
                            content=chunk.text,
                            memory_type="document_chunk",
                            project=project_name,
                            tags=[file_path.suffix[1:], "temporal_sync"],
                            metadata=chunk.metadata
                        )
                        store.upsert_memory(entry)
                    indexed_count += 1
                except Exception as e:
                    log.warning(f"Dosya indeksleme hatası ({file_path.name}): {e}")
    except Exception as ex:
        log.error(f"Global indexing error: {ex}")
        return {"status": "error", "reason": str(ex)}

    return {
        "status": "success",
        "indexed_files": indexed_count,
        "workspace": str(workspace_path),
        "project": project_name
    }


@activity.defn
async def run_virtual_screening_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pillar 4-1: Sanal Tarama (Virtual Screening) Activity.
    
    Bir hedef protein ve SMILES kütüphanesi verildiğinde,
    rdkit ile Lipinski kurallarını uygulayarak en iyi adayları raporlar.
    """
    target = params.get("target_protein", "Unknown Target")
    smiles_library = params.get("smiles_library", [])
    max_candidates = params.get("max_candidates", 10)

    log.info(f"Virtual Screening başladı: {target}, {len(smiles_library)} bileşik")

    candidates: List[Dict[str, Any]] = []

    for smiles in smiles_library:
        try:
            from bio_ml_agent.ml.bioeng_toolkit import DrugDiscoveryHelper
            helper = DrugDiscoveryHelper(smiles)
            lipinski = helper.lipinski_rule_of_five()

            if lipinski.get("passes_rule", False):
                score = 0.0
                # Basit skorlama: Lipinski parametrelerine yakınlık
                mw = lipinski.get("molecular_weight", 500)
                logp = lipinski.get("logp", 5)
                hbd = lipinski.get("h_bond_donors", 5)
                hba = lipinski.get("h_bond_acceptors", 10)

                # İdeal değerlere yakınlık skoru (0-1 arası)
                score += max(0, 1 - abs(mw - 300) / 200)  # İdeal MW ~300
                score += max(0, 1 - abs(logp - 2.5) / 2.5)  # İdeal LogP ~2.5
                score += max(0, 1 - hbd / 5)
                score += max(0, 1 - hba / 10)
                score = round(score / 4, 4)  # Normalize

                candidates.append({
                    "smiles": smiles,
                    "lipinski": lipinski,
                    "drug_likeness_score": score,
                })
        except Exception as e:
            log.warning(f"SMILES işleme hatası ({smiles[:20]}...): {e}")
            continue

    # Skora göre sırala ve top N al
    candidates.sort(key=lambda c: c["drug_likeness_score"], reverse=True)
    top_candidates = candidates[:max_candidates]

    result = {
        "status": "success",
        "target_protein": target,
        "total_screened": len(smiles_library),
        "passed_lipinski": len(candidates),
        "top_candidates": top_candidates,
    }

    log.info(f"Virtual Screening tamamlandı: {len(candidates)}/{len(smiles_library)} Lipinski geçti")
    return result
