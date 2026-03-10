import unittest
from models.omnichannel import (
    ChannelType, ChannelAdapterConfig, ChannelPolicy,
    NotificationType, NotificationEvent
)

class TestOmnichannelModels(unittest.TestCase):
    def test_channel_adapter(self):
        adapter = ChannelAdapterConfig(
            adapter_id="adp-tg-01",
            channel_type=ChannelType.TELEGRAM,
            credentials_secret_ref="sec-tg-token"
        )
        self.assertEqual(adapter.channel_type, ChannelType.TELEGRAM)
        self.assertTrue(adapter.is_active)

    def test_channel_policy(self):
        policy = ChannelPolicy(
            policy_id="pol-tg",
            channel_type=ChannelType.TELEGRAM,
            send_only_summaries=True,
            allow_sensitive_artifacts=False
        )
        self.assertFalse(policy.allow_sensitive_artifacts)
        self.assertTrue(policy.send_only_summaries)

    def test_notification_event(self):
        event = NotificationEvent(
            notification_id="notif-123",
            target_user_id="usr-abc",
            notification_type=NotificationType.APPROVAL_PENDING,
            message_title="Human Approval Required",
            message_body="The browser agent requests permission to click 'Buy'.",
            delivered_channels=[ChannelType.WHATSAPP, ChannelType.EMAIL],
            timestamp="2026-03-10T21:00:00"
        )
        self.assertEqual(len(event.delivered_channels), 2)
        self.assertEqual(event.notification_type, NotificationType.APPROVAL_PENDING)

if __name__ == '__main__':
    unittest.main()
