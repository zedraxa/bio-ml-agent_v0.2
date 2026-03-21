import logging
from typing import Dict, Any

from bio_ml_agent.kernel.message_bus import MessageBus

log = logging.getLogger("mission.base")

class BaseMission:
    """
    Core fundamental class for end-to-end scientific missions.
    Provides generic setup, execution, and state persistence patterns.
    """
    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.status = "initialized"
        log.info(f"Mission '{self.name}' initialized.")
        
    def setup(self) -> None:
        """Prepare context, register listeners, allocate subsystems."""
        self.status = "setup_complete"
        log.info(f"Mission '{self.name}' setup complete.")
        
    def execute(self) -> Dict[str, Any]:
        """Execute the primary mission graph."""
        self.status = "executing"
        raise NotImplementedError("Missions must implement execute()")
