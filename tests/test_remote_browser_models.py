import unittest
from models.remote_browser import (
    LiveBrowserStreamFrame, TakeoverStatus, BrowserTakeoverEvent,
    RiskLevel, RiskActionApprovalState, TimelineSnapshot, SessionReplayTimeline
)

class TestRemoteBrowserModels(unittest.TestCase):
    def test_live_browser_stream(self):
        frame = LiveBrowserStreamFrame(
            frame_id="frm-01",
            session_id="sess-abc",
            step_number=5,
            screenshot_base64_or_url="/assets/scr5.png",
            action_type="click",
            timestamp="2026-03-10T21:15:00"
        )
        self.assertEqual(frame.step_number, 5)

    def test_browser_takeover(self):
        event = BrowserTakeoverEvent(
            event_id="tko-123",
            session_id="sess-abc",
            requested_by_user_id="usr-99",
            status=TakeoverStatus.HUMAN_IN_CONTROL,
            timestamp="2026-03-10T21:16:00"
        )
        self.assertEqual(event.status, TakeoverStatus.HUMAN_IN_CONTROL)

    def test_risk_action_approval(self):
        approval = RiskActionApprovalState(
            state_id="appr-01",
            session_id="sess-abc",
            detected_action="payment_click",
            dom_target_context={"button_text": "Pay $50"},
            risk_level=RiskLevel.CRITICAL
        )
        self.assertEqual(approval.risk_level, RiskLevel.CRITICAL)
        self.assertIsNone(approval.is_approved)

    def test_session_replay_timeline(self):
        snapshot = TimelineSnapshot(
            snapshot_id="snp-01",
            step_index=1,
            url="https://google.com",
            screenshot_ref="img-google.png",
            dom_changes_diff="<div>Loaded</div>",
            agent_reasoning="Loaded initial search page.",
            timestamp_offset_ms=1050
        )
        timeline = SessionReplayTimeline(
            timeline_id="tl-01",
            session_id="sess-abc",
            total_duration_ms=65000,
            snapshots=[snapshot]
        )
        self.assertEqual(len(timeline.snapshots), 1)

if __name__ == '__main__':
    unittest.main()
