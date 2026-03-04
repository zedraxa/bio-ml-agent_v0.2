import logging
from typing import Dict, Any, Optional
import time

log = logging.getLogger("bio_ml_agent")

class OTelMetricTracker:
    """
    S8-1 & S8-2: OpenTelemetry ve Prometheus Metricleri (Low Cardinality).
    Gerçek otel kütüphanesi yerine mock wrapper. 
    İleride export_prometheus kullanılacak.
    """
    def __init__(self):
        self._enabled = True
        # Sayaçlar
        self._counters = {
            "agent_runs_total": 0,
            "tool_calls_total": 0,
            "llm_fallbacks_total": 0
        }
        # Metrik PII-free low-cardinality etiketleri (labels) olacak
    
    def increment_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        if metric_name in self._counters:
            self._counters[metric_name] += 1
            log_str = f"Metric Inc: {metric_name}"
            if labels:
                # low cardinality label example: status=success, tool=bash
                log_str += f" {labels}"
            log.debug(log_str)
            
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        # Örnek: agent_task_duration_seconds
        log.debug(f"Metric Histogram: {metric_name} = {value:.3f}s {labels or ''}")

metrics = OTelMetricTracker()
