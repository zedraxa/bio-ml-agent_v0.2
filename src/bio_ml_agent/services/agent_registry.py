import logging
from typing import Dict, List, Optional
from bio_ml_agent.models.lifecycle import AgentRegistryEntry, AgentTier, AgentCapability
from bio_ml_agent.models.workspace_ux import AgentRole, TaskType
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all agent families in the bio-ml-agent ecosystem."""
    
    def __init__(self, registry_path: str = "src/bio_ml_agent/resources/agent_registry.yaml"):
        self._entries: Dict[str, AgentRegistryEntry] = {}
        self.registry_path = Path(registry_path)
        self._initialize_from_yaml()
        self._discover_unregistered_agents()

    def _initialize_from_yaml(self):
        """Initial population of the registry from the YAML configuration."""
        import yaml
        if not self.registry_path.exists():
            logger.warning(f"[Registry] YAML not found at {self.registry_path}. Creating default empty registry.")
            return

        try:
            with open(self.registry_path, "r") as f:
                config = yaml.safe_load(f)
                
            for agent_data in config.get("agents", []):
                # Map capability string to Enum if needed (Pydantic handles this)
                entry = AgentRegistryEntry(**agent_data, description=agent_data.get("description", f"Specialized {agent_data['family']} agent."))
                self.register(entry)
            logger.info(f"[Registry] Loaded {len(self._entries)} agents from YAML.")
        except Exception as e:
            logger.error(f"[Registry] Failed to load agent_registry.yaml: {e}")

    def _discover_unregistered_agents(self):
        """Phase 2: Scans the agents/ directory and registers any untracked families as EXPERIMENTAL."""
        agents_root = Path(__file__).resolve().parent.parent / "agents"
        families = ["academic", "biology", "browser", "coder", "critic", "dataset", "document", "microscopy", "omics", "structural", "structural_coding"]
        
        # Mapping families to default roles
        family_role_map = {
            "academic": AgentRole.ACADEMIC_EXPERT,
            "biology": AgentRole.BIOINFORMATICIAN,
            "browser": AgentRole.BROWSER_AGENT,
            "coder": AgentRole.CODING_AGENT,
            "critic": AgentRole.CRITIC,
            "dataset": AgentRole.DATA_ENGINEER,
            "document": AgentRole.ACADEMIC_EXPERT,
            "microscopy": AgentRole.MICROSCOPY_AGENT,
            "omics": AgentRole.BIOINFORMATICIAN,
            "structural": AgentRole.IN_SILICO_EXPERT,
            "structural_coding": AgentRole.CODING_AGENT
        }

        discovered_count = 0
        for family in families:
            family_dir = agents_root / family
            if not family_dir.exists(): continue
            
            for agent_file in family_dir.glob("*.py"):
                if agent_file.name.startswith("__") or "abstract" in agent_file.name: continue
                
                agent_id = agent_file.stem
                if agent_id not in self._entries:
                    # Register as experimental discovery
                    entry = AgentRegistryEntry(
                        agent_id=agent_id,
                        role=family_role_map.get(family, AgentRole.RESEARCHER),
                        tier=AgentTier.EXPERIMENTAL,
                        path=f"agents/{family}/{agent_file.name}",
                        mission_pack=f"experimental_{family}",
                        description=f"Auto-discovered {family} agent: {agent_id}",
                        is_enabled=True # Visible but experimental
                    )
                    self.register(entry)
                    discovered_count += 1
        
        if discovered_count > 0:
            logger.info(f"[Registry] Auto-discovered {discovered_count} new subagents into EXPERIMENTAL tier.")

    def register(self, entry: AgentRegistryEntry):
        """Register a new agent family."""
        self._entries[entry.agent_id] = entry
        logger.info(f"[Registry] Registered agent: {entry.agent_id} ({entry.role.value}) - Tier: {entry.tier.value}")

    def get_agent(self, agent_id: str) -> Optional[AgentRegistryEntry]:
        """Retrieve agent metadata by ID."""
        return self._entries.get(agent_id)

    def find_by_role(self, role: AgentRole, min_tier: AgentTier = AgentTier.EXPERIMENTAL) -> List[AgentRegistryEntry]:
        """Find agents matching a role and minimum maturity tier."""
        tier_values = {
            AgentTier.STABLE: 3,
            AgentTier.ACTIVE: 2,
            AgentTier.BETA: 1,
            AgentTier.EXPERIMENTAL: 0
        }
        min_val = tier_values.get(min_tier, 0)
        
        results = []
        for entry in self._entries.values():
            if entry.role == role and tier_values.get(entry.tier, 0) >= min_val:
                if entry.is_enabled:
                    results.append(entry)
                    
        # Sort by tier descending (Stable first)
        results.sort(key=lambda x: tier_values.get(x.tier, 0), reverse=True)
        return results

    def list_all(self) -> List[AgentRegistryEntry]:
        """Return all registered agent entries."""
        return list(self._entries.values())

    def get_agent_instance(self, agent_id: str, config, **kwargs):
        """Phase 3: Instantiates a standardized agent class by its registered ID."""
        import importlib
        import importlib.util
        import sys
        
        entry = self.get_agent(agent_id)
        if not entry:
            raise ValueError(f"[Registry] Agent {agent_id} not found in registry.")

        # Ensure absolute path
        agents_root = Path(__file__).resolve().parent.parent / "agents"
        # The entry.path is relative to agents/ root in discovery, or full path in YAML if absolute
        agent_path = Path(entry.path)
        if not agent_path.is_absolute():
            agent_path = agents_root / entry.path.replace("agents/", "")

        if not agent_path.exists():
            raise FileNotFoundError(f"[Registry] Agent file not found at {agent_path}")

        # Dynamic import
        module_name = f"bio_ml_agent.agents_dynamic.{agent_id}"
        spec = importlib.util.spec_from_file_location(module_name, agent_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find the agent class (usually camel case of agent_id or first class inheriting from BaseSubAgent)
        from bio_ml_agent.core.agent_base import BaseSubAgent
        agent_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, BaseSubAgent) and obj != BaseSubAgent:
                agent_class = obj
                break
        
        if not agent_class:
            raise TypeError(f"[Registry] No BaseSubAgent subclass found in {agent_path}")

        return agent_class(config=config, **kwargs)

# Global singleton
agent_registry = AgentRegistry()
