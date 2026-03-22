# Phase R8 — Real-World Integration & Deployment (Part VIII)

Goal: Consolidate the massive multi-agent system into a single, cohesive, physical-world runnable state. We need to ensure that everything from the WhatsApp connector to the web UI and agent core starts up smoothly and talks to physical external APIs.

## Proposed Changes

### [Execution & Bootstrapping]
#### [NEW] [start_bio_ml.py](file:///media/yusuf/Data/bio-ml-agent_v0.2/start_bio_ml.py)
A unified bootstrapper that:
- Spawns [web_ui.py](file:///media/yusuf/Data/bio-ml-agent_v0.2/tests/test_web_ui.py) (Gradio Dashboard).
- Spawns [whatsapp_connector.py](file:///media/yusuf/Data/bio-ml-agent_v0.2/src/bio_ml_agent/whatsapp_connector.py) (Flask Webhook & Router) on a dedicated port.
- Initializes the [MissionBrain](file:///media/yusuf/Data/bio-ml-agent_v0.2/src/bio_ml_agent/brain/mission_brain.py#299-1052) background workers.
- Automatically handles `.env` loading and checks for required keys (OpenAI/Gemini/Twilio API keys).

#### [NEW] [utils/ngrok_manager.py](file:///media/yusuf/Data/bio-ml-agent_v0.2/src/bio_ml_agent/utils/ngrok_manager.py)
Automated local tunneling for WhatsApp testing:
- Programatically launches `ngrok` on the Flask port.
- Prints the exact Twilio webhook URL (e.g., `https://xxxx.ngrok.io/whatsapp-webhook`) to the console in bold green for easy copy-pasting into the Twilio console.

### [System Health & Verification]
#### [NEW] [verify_system_health.py](file:///media/yusuf/Data/bio-ml-agent_v0.2/verify_system_health.py)
A self-diagnostic CLI command that executes real-world integration sanity checks:
- Calls LLM endpoint with a ping ("say hello").
- Checks artifact database writability.
- Pre-checks Twilio and WhatsApp configuration variables.

## Verification Plan
- Run `python start_bio_ml.py` and verify all processes boot cleanly.
- Watch console output for the `ngrok` URL and verify LLM connections.
