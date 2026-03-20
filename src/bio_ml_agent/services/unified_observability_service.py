from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
import logging

from bio_ml_agent.models.unified_observability import (
    TraceSpan, SpanType, RunTimeline,
    BrowserTimelineEvent, FailureReplaySession,
    CloudCostEntry, CostCategory, ProjectBillingTrace,
    UserAuditLog, SystemActionType,
    SecretAccessReceipt
)

logger = logging.getLogger(__name__)

class UnifiedObservabilityService:
    def __init__(self):
        # 1. Timelines
        self._timelines: Dict[str, RunTimeline] = {}
        # 2. Replay Sessions
        self._replays: Dict[str, FailureReplaySession] = {}
        # 3. Cost Tracing
        self._cost_entries: List[CloudCostEntry] = []
        # 4. User Audits
        self._user_audits: List[UserAuditLog] = []
        # 5. Secret Receipts
        self._secret_receipts: List[SecretAccessReceipt] = []

    # --- 1. Run Timeline & Spans ---
    def start_run_timeline(self, run_id: str) -> RunTimeline:
        timeline_id = f"tl-{uuid.uuid4().hex[:6]}"
        tl = RunTimeline(
            timeline_id=timeline_id,
            run_id=run_id,
            created_at=datetime.now(timezone.utc)
        )
        self._timelines[run_id] = tl
        return tl

    def add_span(self, run_id: str, span_type: SpanType, name: str, inputs: dict = None) -> TraceSpan:
        if run_id not in self._timelines:
            self.start_run_timeline(run_id)
            
        span = TraceSpan(
            span_id=f"span-{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            span_type=span_type,
            name=name,
            inputs=inputs or {},
            started_at=datetime.now(timezone.utc)
        )
        self._timelines[run_id].spans.append(span)
        logger.debug(f"[Trace] Run {run_id} started span: {name} ({span_type.value})")
        return span

    def end_span(self, run_id: str, span_id: str, outputs: dict = None, error: str = None):
        if run_id not in self._timelines:
            return
        
        for span in self._timelines[run_id].spans:
            if span.span_id == span_id:
                span.ended_at = datetime.now(timezone.utc)
                span.outputs = outputs or {}
                if error:
                    span.status = "error"
                    span.error_message = error
                
                # Update total duration
                delta = span.ended_at - span.started_at
                self._timelines[run_id].total_duration_ms += int(delta.total_seconds() * 1000)
                break

    # --- 2. Failure Replay ---
    def record_browser_failure(self, run_id: str, reason: str, events: List[BrowserTimelineEvent]) -> FailureReplaySession:
        replay_id = f"replay-{uuid.uuid4().hex[:6]}"
        session = FailureReplaySession(
            replay_id=replay_id,
            run_id=run_id,
            failure_reason=reason,
            events=events,
            recorded_at=datetime.now(timezone.utc)
        )
        self._replays[replay_id] = session
        logger.warning(f"Recorded failure replay {replay_id} for run {run_id} (Reason: {reason})")
        return session

    # --- 3. Cloud Cost ---
    def track_cost(self, run_id: str, category: CostCategory, provider: str, amount: float, metric: str) -> CloudCostEntry:
        entry = CloudCostEntry(
            cost_id=f"cost-{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            category=category,
            resource_provider=provider,
            amount_usd=amount,
            usage_metric=metric,
            timestamp=datetime.now(timezone.utc)
        )
        self._cost_entries.append(entry)
        return entry

    def generate_project_billing(self, project_id: str, run_ids: List[str]) -> ProjectBillingTrace:
        relevant_entries = [c for c in self._cost_entries if c.run_id in run_ids]
        total = sum(c.amount_usd for c in relevant_entries)
        
        trace = ProjectBillingTrace(
            trace_id=f"bill-{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            total_cost_usd=total,
            entries=relevant_entries,
            generated_at=datetime.now(timezone.utc)
        )
        return trace

    # --- 4. Audits & Receipts ---
    def log_user_action(self, user_id: str, action: SystemActionType, target_id: str, context: dict = None) -> UserAuditLog:
        audit = UserAuditLog(
            audit_id=f"audit-{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            action_type=action,
            target_resource_id=target_id,
            context_snapshot=context or {},
            timestamp=datetime.now(timezone.utc)
        )
        self._user_audits.append(audit)
        return audit

    def generate_secret_receipt(self, secret_name: str, run_id: str, justification: str, agent: str = None, user: str = None) -> SecretAccessReceipt:
        receipt = SecretAccessReceipt(
            receipt_id=f"rec-{uuid.uuid4().hex[:8]}",
            secret_name=secret_name,
            accessed_by_user=user,
            accessed_by_agent=agent,
            run_id=run_id,
            justification=justification,
            accessed_at=datetime.now(timezone.utc)
        )
        self._secret_receipts.append(receipt)
        logger.info(f"Secret {secret_name} accessed via receipt {receipt.receipt_id}")
        return receipt

# Singleton Dispatcher
observability_service = UnifiedObservabilityService()
