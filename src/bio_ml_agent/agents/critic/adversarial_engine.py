import logging
import re
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import Evidence

log = logging.getLogger("critic.adversarial_engine")

class AdversarialEngine:
    """
    AdversarialEngine: Ajanların iddialarını ve sundukları kanıtları çapraz sorgular.
    - Contradiction Detection: İddia ile metin parçası arasındaki mantıksal zıtlıkları bulur.
    - Numeric Mismatch: İddia edilen sayılar ile kanıttaki sayıları karşılaştırır (Örn: %5 vs %50).
    - Hallucination Guard: Modelin uydurduğu referansları tespit eder.
    """

    @staticmethod
    def detect_contradictions(claim: str, evidence: List[Evidence]) -> List[Dict[str, Any]]:
        issues = []
        for ev in evidence:
            content_lower = ev.content_snippet.lower()
            claim_lower = claim.lower()

            # 1. Negation Conflict
            if " not " in content_lower and " not " not in claim_lower:
                issues.append({"type": "negation_conflict", "evidence": ev.content_snippet, "claim": claim})

            # 2. Numeric Mismatch (En ince detay)
            claim_nums = re.findall(r'\d+', claim)
            ev_nums = re.findall(r'\d+', ev.content_snippet)

            for num in claim_nums:
                if num not in ev_nums and len(num) > 1: # Tek basamaklıları gürültü diye atla
                    issues.append({
                        "type": "numeric_mismatch",
                        "missing_number": num,
                        "evidence_context": ev.content_snippet
                    })

        return issues

    @staticmethod
    def find_gaps(plan: List[str], outcomes: List[Any]) -> List[str]:
        gaps = []
        if len(outcomes) < len(plan):
            gaps.append(f"Missing outcomes for {len(plan) - len(outcomes)} planned steps.")
        return gaps

    @staticmethod
    def audit_provenance(evidence: List[Evidence]) -> bool:
        for ev in evidence:
            if not ev.source.startswith(("http", "/", "file")):
                log.warning(f"⚠️ Suspicious evidence source: {ev.source}")
                return False
        return True
