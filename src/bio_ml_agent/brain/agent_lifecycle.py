import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from .agent_contract import (
    UnifiedAgentContract, AgentState, AgentRole, AgentStateRecord,
    AgentStateTransition, VALID_STATE_TRANSITIONS, AgentCapabilityCard
)

logger = logging.getLogger("bio_ml_agent.brain.lifecycle")

class AgentLifecycleManager:
    """
    B3/B4: Manages the registration, state, and health of all agents.
    Now loads from agent_registry.yaml for R6-2.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self.active_agents: Dict[str, UnifiedAgentContract] = {}
        self.agent_states: Dict[str, AgentStateRecord] = {}
        self.registry: Dict[str, Any] = {}

        # Load registry if provided
        if not registry_path:
            # Default to peer directory
            registry_path = str(Path(__file__).parent / "agent_registry.yaml")

        self.load_registry(registry_path)

    def load_registry(self, path: str):
        """Loads agent definitions from YAML."""
        try:
            with open(path, 'r') as f:
                self.registry = yaml.safe_load(f)
            logger.info(f"[Axis B] Loaded agent registry from {path}")
        except Exception as e:
            logger.error(f"[Axis B] Failed to load agent registry: {e}")
            self.registry = {"agents": {}}

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Returns IDs of agents that declare a specific capability."""
        matches = []
        for agent_id, info in self.registry.get("agents", {}).items():
            if capability in info.get("capabilities", []):
                matches.append(agent_id)
        return matches

    def get_agent_for_role(self, role: AgentRole) -> Optional[str]:
        """Finds the first registered agent that matches a role."""
        for agent_id, info in self.registry.get("agents", {}).items():
            if info.get("role") == role or info.get("role") == role.value:
                return agent_id
        return None

    def register_agent(self, agent: UnifiedAgentContract):
        """Registers an agent and initializes its state track."""
        role_key = agent.agent_role
        self.active_agents[role_key] = agent
        self.agent_states[role_key] = AgentStateRecord(agent_name=agent.agent_name)
        logger.info(f"[LifecycleManager] Registered Agent: {agent.agent_name} ({role_key})")

    def get_agent(self, role: str) -> Optional[UnifiedAgentContract]:
        """Retrieves an agent by its role."""
        return self.active_agents.get(role)

    def get_status(self, role: str) -> AgentState:
        """Returns the current state of an agent."""
        if role in self.agent_states:
            return self.agent_states[role].current_state
        return AgentState.FAILED # Default to failed if unknown

    def update_state(self, role: str, new_state: AgentState, reason: str = ""):
        """Transitions an agent to a new state."""
        if role in self.agent_states:
            self.agent_states[role].transition_to(new_state, reason=reason)
            logger.debug(f"[LifecycleManager] {role} → {new_state}")

    def list_capabilities(self) -> List[AgentCapabilityCard]:
        """Returns the capability cards for all registered agents."""
        return [a.capability_card for a in self.active_agents.values()]

    def health_check(self) -> Dict[str, bool]:
        """Performs a basic connectivity/health check on all agents."""
        health = {}
        for role, agent in self.active_agents.items():
            # In a real system, this might call a heartbeat or a simple perceive()
            try:
                # Mock health check
                is_healthy = self.get_status(role) != AgentState.FAILED
                health[role] = is_healthy
            except Exception:
                health[role] = False
        return health
