"""
bio_ml_agent.whatsapp_connector — backward-compatible shim.

The WhatsApp connector now lives at bio_ml_agent.services.whatsapp_connector.
This module re-exports the Flask ``app`` object so existing imports continue to work.
"""
try:
    from bio_ml_agent.services.whatsapp_connector import app
    from bio_ml_agent.brain.persistence import PROJECT_STORE, MISSION_STORE
    from bio_ml_agent.brain.mission_brain import MissionBrain as brain  # noqa: N811

    __all__ = ["app", "PROJECT_STORE", "MISSION_STORE", "brain"]
except Exception:  # noqa: BLE001
    # If optional deps (flask, twilio, etc.) are not installed, expose a stub.
    import logging
    logging.getLogger("bio_ml_agent").warning(
        "whatsapp_connector could not be imported — optional deps missing."
    )

    class _StubApp:
        def __init__(self):
            self.config = {}

        def test_client(self):
            raise RuntimeError(
                "WhatsApp connector optional dependencies not installed. "
                "Install flask and twilio to use this feature."
            )

    app = _StubApp()
    PROJECT_STORE = None
    MISSION_STORE = None
    brain = None
    __all__ = ["app", "PROJECT_STORE", "MISSION_STORE", "brain"]
