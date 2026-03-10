import unittest
from models.browser import (
    DOMElement, PerceptionState,
    SelectorType, SelectorCandidate,
    BrowserActionType, BrowserAction,
    DeltaStatus, DOMDelta,
    TransferDirection, BrowserTransferEvent,
    BrowserStepTimeline,
    WaitCondition, SmartWaitConfig,
    InputFieldType, FormField, FormValidationState
)

class TestBrowserModels(unittest.TestCase):
    def test_dom_element_perception(self):
        element = DOMElement(
            bio_id="btn-login",
            tag="BUTTON",
            label="Giriş Yap",
            vision_score=0.9
        )
        self.assertEqual(element.tag, "BUTTON")
        self.assertTrue(element.is_visible)
        
        perception = PerceptionState(
            url="https://example.com/login",
            title="Login",
            interactive_elements=[element]
        )
        self.assertEqual(len(perception.interactive_elements), 1)

    def test_browser_action_planner(self):
        selector = SelectorCandidate(
            type=SelectorType.CSS,
            value="#submit",
            confidence_score=0.95
        )
        action = BrowserAction(
            type=BrowserActionType.CLICK,
            target_selector=selector,
            reason="Formu Gönder"
        )
        self.assertEqual(action.type, BrowserActionType.CLICK)
        self.assertEqual(action.target_selector.value, "#submit")

    def test_state_delta(self):
        delta = DOMDelta(
            added_nodes_count=5,
            visual_change_percent=15.5,
            status=DeltaStatus.SUCCESS
        )
        self.assertEqual(delta.status, DeltaStatus.SUCCESS)

    def test_download_upload_manager(self):
        event = BrowserTransferEvent(
            direction=TransferDirection.DOWNLOAD,
            file_name="dataset.csv",
            workspace_path="/data/dataset.csv",
            size_bytes=1024
        )
        self.assertEqual(event.direction, TransferDirection.DOWNLOAD)

    def test_smart_wait_config(self):
        config = SmartWaitConfig(
            condition=WaitCondition.MUTATION_STOP,
            timeout_ms=5000
        )
        self.assertEqual(config.condition, WaitCondition.MUTATION_STOP)
        self.assertTrue(config.require_visual_stability)

    def test_form_inference(self):
        field = FormField(
            bio_id="input-password",
            inferred_type=InputFieldType.PASSWORD,
            is_required=True
        )
        self.assertEqual(field.inferred_type, InputFieldType.PASSWORD)
        self.assertTrue(field.is_required)

        state = FormValidationState(
            has_validation_error=True,
            error_messages=["Bu alan boş bırakılamaz"]
        )
        self.assertTrue(state.has_validation_error)
        self.assertEqual(len(state.error_messages), 1)

if __name__ == '__main__':
    unittest.main()
