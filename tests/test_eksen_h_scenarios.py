import pytest
import json
from unittest.mock import MagicMock, patch
from bio_ml_agent.brain.models import (
    MissionPlan, MissionStep, StepStatus, WhatsAppCardType, Comment, CommentStatus, ReviewerRole
)
from bio_ml_agent.brain.comment_manager import CommentManager
from bio_ml_agent.brain.mission_brain import MissionBrain
from bio_ml_agent.whatsapp_connector import app as wa_app

@pytest.fixture
def mock_brain():
    brain = MagicMock(spec=MissionBrain)
    # Configure mock brain with a mock project
    mock_project = MagicMock()
    mock_project.project_id = "test_project"
    mock_project.active_mission_id = "test_mission"
    brain.active_project = mock_project
    
    mock_plan = MissionPlan(
        mission_id="test_mission",
        title="Test Mission",
        steps=[
            MissionStep(step_id="step_1", description="Waiting approval", requires_approval=True, status=StepStatus.WAITING_APPROVAL)
        ]
    )
    brain._find_mission.return_value = mock_plan
    return brain

@pytest.fixture
def wa_client():
    wa_app.config["TESTING"] = True
    with wa_app.test_client() as client:
        yield client

# H1: Comment-to-revision scenario
def test_comment_to_revision_h1():
    manager = CommentManager("proj_1", "iter_1")
    
    # User adds a comment to a report paragraph
    comment = manager.add_comment(
        content="Rewrite this paragraph to be shorter",
        target_uri="file://report.md#L10-15",
        author="User",
        role=ReviewerRole.USER
    )
    
    assert comment.status == CommentStatus.OPEN
    
    # Simulate agent processing the feedback and logging a resolution
    manager.resolve_comment(comment.comment_id, resolution_note="Paragraph rewritten and shortened by 30%.")
    
    resolved_comment = manager.get_comment(comment.comment_id)
    assert resolved_comment.status == CommentStatus.RESOLVED
    assert "shortened" in resolved_comment.resolution_note
    
    # Assert thread context returns properly
    threads = manager.get_active_threads()
    assert len(threads) == 0 # All resolved

# H2: WhatsApp approval scenario
@patch("bio_ml_agent.whatsapp_connector.brain")
@patch("bio_ml_agent.whatsapp_connector.PROJECT_STORE")
@patch("bio_ml_agent.whatsapp_connector.MISSION_STORE")
def test_whatsapp_approval_h2(mock_mission_store, mock_project_store, mock_brain_import, wa_client, mock_brain):
    # Map the whatsapp_connector's brain to our mocked brain
    mock_brain_import.resolve_pending_approval = mock_brain.resolve_pending_approval
    mock_brain_import.active_project = mock_brain.active_project
    mock_project_store.load.return_value = mock_brain.active_project
    
    payload = {
        "text": "ONAYLA",
        "from": "+1234567890"
    }
    
    response = wa_client.post("/whatsapp-local", json=payload)
    assert response.status_code == 200
    
    # Ensure resolve_pending_approval was called
    mock_brain_import.resolve_pending_approval.assert_called_once_with(True, feedback="WhatsApp üzerinden onaylandı.")

# H3: WhatsApp file intake scenario
@patch("bio_ml_agent.whatsapp_connector.brain")
@patch("bio_ml_agent.whatsapp_connector.PROJECT_STORE")
def test_whatsapp_file_intake_h3(mock_project_store, mock_brain_import, wa_client):
    """Simulate a Twilio Webhook delivering an image."""
    payload = {
        "From": "whatsapp:+1234567890",
        "Body": "",
        "MediaUrl0": "https://example.com/microscopy.jpg",
        "MediaContentType0": "image/jpeg",
        "NumMedia": "1"
    }
    
    response = wa_client.post("/whatsapp-webhook", data=payload)
    assert response.status_code == 200
    
    # The response should be a TwiML string containing the MEDIA_RECEIVED ack
    xml_data = response.data.decode("utf-8")
    assert "Dosya alındı" in xml_data
    assert "Mikroskop Analizi" in xml_data

# H4: Comment conflict scenario
def test_comment_conflict_h4():
    manager = CommentManager("proj_2", "iter_1")
    
    # User asks to shorten
    c1 = manager.add_comment("Kısalt", target_uri="file://abstract.md", role=ReviewerRole.USER)
    # Critic agent asks to add details
    c2 = manager.add_comment("Detay ekle ve literatür koy", target_uri="file://abstract.md", role=ReviewerRole.CRITIC_AGENT)
    
    bundle = manager.prepare_review_bundle()
    
    # Revisions should contain both, creating a conflict for the agent to resolve
    assert bundle["total_comments"] == 2
    assert "Kısalt" in bundle["summary"]
    assert "Detay ekle" in bundle["summary"]

# H5: Multi-review consistency test
def test_multi_review_consistency_h5():
    manager = CommentManager("proj_3", "iter_1")
    
    c_user = manager.add_comment("Fix the typo", role=ReviewerRole.USER)
    c_critic = manager.add_comment("Methodology is flawed", role=ReviewerRole.CRITIC_AGENT)
    c_writer = manager.add_comment("Transition sentence is weak", role=ReviewerRole.WRITING_AGENT)
    
    threads = manager.get_active_threads()
    assert len(threads) == 3
    
    # Resolve one
    manager.resolve_comment(c_user.comment_id, "Typo fixed")
    
    # Export state to check consistency
    state = manager.export_state()
    assert len(state["comments"]) == 3
    
    # Verify the remaining active threads are exactly the unresolved ones
    active_now = manager.get_active_threads()
    assert len(active_now) == 2
    
    roles_active = [c.role for c in active_now]
    assert ReviewerRole.CRITIC_AGENT in roles_active
    assert ReviewerRole.WRITING_AGENT in roles_active
    assert ReviewerRole.USER not in roles_active
