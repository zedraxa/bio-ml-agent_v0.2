import logging
from typing import Dict, Any, List

from temporalio import activity

log = logging.getLogger("bio_ml_agent")


@activity.defn
async def index_workspace_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Önceden RQ ile asenkron yapılan dosya indeksleme işlemini Temporal üzerinden yapar.
    S3-3 (RQ Bridge) görevi için hazırlanmıştır.
    """
    log.info(f"Temporal Activity: workspace indeksleniyor: {params.get('workspace_name', 'default')}")
    # Gerçek indeksleme mantığı (rag_engine) ileride buraya bağlanacak.

    # Şimdilik dummy dönüş:
    return {"status": "success", "indexed_files": 12, "workspace": params.get('workspace_name')}


@activity.defn
async def run_virtual_screening_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pillar 4-1: Sanal Tarama (Virtual Screening) Activity.

    Bir hedef protein ve SMILES kütüphanesi verildiğinde,
    rdkit ile Lipinski kurallarını uygulayarak en iyi adayları raporlar.

    Params:
        target_protein: Hedef protein adı veya PDB ID.
        smiles_library: SMILES kodları listesi.
        max_candidates: Raporlanacak maksimum aday sayısı.
    """
    target = params.get("target_protein", "Unknown Target")
    smiles_library = params.get("smiles_library", [])
    max_candidates = params.get("max_candidates", 10)

    log.info(f"Virtual Screening başladı: {target}, {len(smiles_library)} bileşik")

    candidates: List[Dict[str, Any]] = []

    for smiles in smiles_library:
        try:
            from bioeng_toolkit import DrugDiscoveryHelper
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
