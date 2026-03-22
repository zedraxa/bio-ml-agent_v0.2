import logging
from typing import Dict, Any, Optional, List
import time
from collections import defaultdict
from datetime import datetime

log = logging.getLogger("bio_ml_agent")


class OTelMetricTracker:
    """
    S8-1 & S8-2: OpenTelemetry ve Prometheus Metricleri (Low Cardinality).

    v1.1 Eklentiler:
    - Live Telemetry: Workflow ve simülasyon ilerleme takibi
    - Cost Tracking: Model/proje bazlı maliyet raporlama
    """
    def __init__(self):
        self._enabled = True
        # Sayaçlar
        self._counters = {
            "agent_runs_total": 0,
            "tool_calls_total": 0,
            "llm_fallbacks_total": 0,
            "llm_requests_total": 0,
        }
        # Live Telemetry: aktif iş akışları
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        # Cost Tracking: model başına kullanım
        self._cost_records: List[Dict[str, Any]] = []
        # Model fiyatlandırma ($/1M token, yaklaşık)
        self._model_pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
            "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
            "qwen2.5": {"input": 0.0, "output": 0.0},  # Lokal, bedava
        }

    def increment_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        if metric_name in self._counters:
            self._counters[metric_name] += 1
            log_str = f"Metric Inc: {metric_name}"
            if labels:
                log_str += f" {labels}"
            log.debug(log_str)

    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        log.debug(f"Metric Histogram: {metric_name} = {value:.3f}s {labels or ''}")

    # ── Live Telemetry ────────────────────────────────

    def register_workflow(self, workflow_id: str, workflow_type: str, metadata: Optional[Dict] = None):
        """Aktif bir iş akışını telemetry'ye kaydet."""
        self._active_workflows[workflow_id] = {
            "type": workflow_type,
            "status": "running",
            "progress_pct": 0,
            "started_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def update_workflow_progress(self, workflow_id: str, pct: int, phase: str = ""):
        """İş akışı ilerleme bilgisini güncelle."""
        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id]["progress_pct"] = pct
            if phase:
                self._active_workflows[workflow_id]["phase"] = phase

    def complete_workflow(self, workflow_id: str, status: str = "completed"):
        """İş akışını tamamlandı olarak işaretle."""
        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id]["status"] = status
            self._active_workflows[workflow_id]["progress_pct"] = 100
            self._active_workflows[workflow_id]["completed_at"] = datetime.now().isoformat()

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Aktif iş akışlarını döndür (Live Telemetry Dashboard için)."""
        return {wid: info for wid, info in self._active_workflows.items()
                if info["status"] == "running"}

    # ── Cost Tracking ────────────────────────────────

    def record_llm_usage(self, model: str, input_tokens: int, output_tokens: int,
                         project: str = "", session_id: str = ""):
        """LLM kullanım maliyetini kaydet."""
        # Model fiyatını bul (prefix eşleşmesi)
        pricing = {"input": 0.0, "output": 0.0}
        for model_key, price in self._model_pricing.items():
            if model.startswith(model_key):
                pricing = price
                break

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # OpenTelemetry Attribution
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.input_tokens", input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.cost_usd", total_cost)
            if project:
                span.set_attribute("llm.project", project)
            if session_id:
                span.set_attribute("llm.session_id", session_id)

        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "project": project,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._cost_records.append(record)
        self.increment_counter("llm_requests_total", {"model": model})

    def get_cost_report(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Maliyet raporunu döndür. Proje filtresi opsiyonel.

        Returns:
            {
                "total_cost_usd": float,
                "by_model": {model: {cost, requests, tokens}},
                "by_project": {project: {cost, requests}},
                "recent": [son 20 kayıt],
            }
        """
        records = self._cost_records
        if project:
            records = [r for r in records if r.get("project") == project]

        by_model: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0})
        by_project: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "requests": 0})
        total = 0.0

        for r in records:
            cost = r["total_cost_usd"]
            total += cost
            m = r["model"]
            by_model[m]["cost"] += cost
            by_model[m]["requests"] += 1
            by_model[m]["tokens"] += r["input_tokens"] + r["output_tokens"]
            p = r.get("project", "unknown")
            by_project[p]["cost"] += cost
            by_project[p]["requests"] += 1

        # Round values
        for v in by_model.values():
            v["cost"] = round(v["cost"], 6)
        for v in by_project.values():
            v["cost"] = round(v["cost"], 6)

        return {
            "total_cost_usd": round(total, 6),
            "total_requests": len(records),
            "by_model": dict(by_model),
            "by_project": dict(by_project),
            "recent": records[-20:],
        }


metrics = OTelMetricTracker()
