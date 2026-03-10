import unittest
from datetime import datetime
from models.observability import (
    SpanType, ObservabilitySpan, TraceTree, RunComparison, AlertSeverity, SystemAlert
)

class TestObservabilityModels(unittest.TestCase):
    def test_observability_span(self):
        span = ObservabilitySpan(
            span_id="s-001",
            trace_id="t-123",
            name="Tool Execution",
            span_type=SpanType.TOOL,
            start_time=datetime.utcnow().isoformat()
        )
        self.assertEqual(span.span_type, SpanType.TOOL)
        self.assertEqual(span.status, "unset")

    def test_trace_tree(self):
        root = ObservabilitySpan(
            span_id="s-root",
            trace_id="t-123",
            name="Main Run",
            span_type=SpanType.PLAN,
            start_time=datetime.utcnow().isoformat()
        )
        tree = TraceTree(
            root_span=root,
            total_latency_ms=1500.0,
            total_cost=0.05
        )
        self.assertEqual(tree.total_latency_ms, 1500.0)
        self.assertEqual(len(tree.children), 0)

    def test_run_comparison(self):
        comp = RunComparison(
            comparison_id="comp-001",
            run_id_a="run-1",
            run_id_b="run-2",
            diff_summary="Run B is faster",
            metric_deltas={"latency": -500.0},
            changed_spans=["s-002"],
            findings=["Caching worked"]
        )
        self.assertEqual(comp.metric_deltas["latency"], -500.0)

    def test_system_alert(self):
        alert = SystemAlert(
            alert_id="alt-001",
            severity=AlertSeverity.CRITICAL,
            category="cost",
            message="Budget exceeded",
            timestamp=datetime.utcnow().isoformat()
        )
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertFalse(alert.is_resolved)

if __name__ == '__main__':
    unittest.main()
