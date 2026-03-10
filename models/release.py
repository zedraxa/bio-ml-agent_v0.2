from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1, 3, 6: Release Manifest ---
class ReleaseType(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    RELEASE_CANDIDATE = "rc"

class ReleaseManifest(BaseModel):
    """Ürün sürümünün nihai kalifikasyon belgesi."""
    version_tag: str # "v1.0.0-rc1"
    release_type: ReleaseType
    is_public_roadmap_v1: bool = True
    deprecated_apis_removed: bool
    technical_debt_score: float # 0.0 (Clean) - 10.0 (High Debt)
    migration_guide_url: Optional[str] = None
    release_notes: List[str] = Field(default_factory=list)

# --- GÖREV 4 & 5 & 7: Templates & Deploy Presets ---
class StarterTemplate(BaseModel):
    """Yeni başlayanlar için hazır proje yapılandırması."""
    template_id: str
    name: str # "protein_folding_starter"
    description: str
    preloaded_data_refs: List[str]
    default_agent_roles: List[str]

class PresetConfig(BaseModel):
    """Tek tıkla kurulum konfigürasyon paketi."""
    preset_id: str
    target_environment: str # "local_docker", "aws_eks", "gcp_run"
    is_one_click_deployable: bool = True
    included_helm_charts: List[str]

# --- GÖREV 8: Security and License Scan ---
class SecurityScanReport(BaseModel):
    """Son çıkış öncesi güvenlik tarama raporu."""
    scan_id: str
    target_version: str
    critical_vulnerabilities_count: int
    high_vulnerabilities_count: int
    all_licenses_compliant: bool
    scan_passed: bool
    report_url: Optional[str] = None
