import unittest
from models.remote_gateway import (
    GatewayRequestLog, AuthMethod, DeviceSession, AuthSession,
    RemoteSessionRegistry, EventStreamType, StreamEvent
)

class TestRemoteGatewayModels(unittest.TestCase):
    def test_gateway_request_log(self):
        log = GatewayRequestLog(
            request_id="req-123",
            ip_address="192.168.1.1",
            endpoint="/api/v1/run",
            is_signed=True,
            timestamp="2026-03-10T20:00:00"
        )
        self.assertTrue(log.is_signed)
        self.assertEqual(log.rate_limit_hits, 0)

    def test_auth_session(self):
        device = DeviceSession(
            device_id="dev-ios",
            device_name="Yusuf's iPhone",
            last_active="2026-03-10T20:01:00",
            ip_address="192.168.1.5",
            is_current=True
        )
        session = AuthSession(
            session_id="sess-abc",
            user_id="usr-123",
            auth_method=AuthMethod.MAGIC_LINK,
            access_token="acc-tok-123",
            refresh_token="ref-tok-123",
            expires_at="2026-03-11T20:00:00",
            device_sessions=[device]
        )
        self.assertEqual(len(session.device_sessions), 1)
        self.assertEqual(session.auth_method, AuthMethod.MAGIC_LINK)

    def test_stream_event(self):
        event = StreamEvent(
            event_id="evt-001",
            session_id="sess-abc",
            event_type=EventStreamType.ARTIFACT_READY,
            payload={"artifact_id": "art-999"},
            timestamp="2026-03-10T20:05:00"
        )
        self.assertEqual(event.event_type, EventStreamType.ARTIFACT_READY)
        self.assertEqual(event.payload["artifact_id"], "art-999")

if __name__ == '__main__':
    unittest.main()
