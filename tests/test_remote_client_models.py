import unittest
from models.remote_client import (
    DashboardModule, DashboardSummaryTemplate, MobileActionType,
    MobileClientAction, SharedReadonlyView, HandoffState, DeviceHandoffEvent
)

class TestRemoteClientModels(unittest.TestCase):
    def test_dashboard_summary(self):
        summary = DashboardSummaryTemplate(
            dashboard_id="dash-01",
            user_id="usr-123",
            active_modules=[DashboardModule.ACTIVE_RUNS, DashboardModule.COST_SUMMARY],
            total_cost_mtd=15.5,
            pending_approvals_count=2
        )
        self.assertEqual(summary.pending_approvals_count, 2)
        self.assertEqual(len(summary.active_modules), 2)

    def test_mobile_action(self):
        action = MobileClientAction(
            action_id="act-01",
            session_id="sess-abc",
            action_type=MobileActionType.APPROVE_STEP,
            target_run_id="run-456",
            device_fingerprint="ios-safari-1234"
        )
        self.assertEqual(action.action_type, MobileActionType.APPROVE_STEP)

    def test_readonly_view(self):
        view = SharedReadonlyView(
            share_id="share-uuid",
            target_project_id="proj-789",
            target_run_id="run-456",
            created_by_user_id="usr-123",
            allowed_guest_emails=["colleague@lab.com"],
            secret_share_token="token_abc_123"
        )
        self.assertFalse(view.is_live_tracking_enabled) # Default var
        self.assertEqual(len(view.allowed_guest_emails), 1)

    def test_device_handoff(self):
        handoff = DeviceHandoffEvent(
            handoff_id="hand-xyz",
            source_device_id="desktop-mac",
            user_id="usr-123",
            current_run_id="run-456",
            pending_approval_id="appr-777",
            created_at="2026-03-10T20:45:00",
            expires_at="2026-03-10T20:50:00"
        )
        self.assertEqual(handoff.state, HandoffState.INITIATED)

if __name__ == '__main__':
    unittest.main()
