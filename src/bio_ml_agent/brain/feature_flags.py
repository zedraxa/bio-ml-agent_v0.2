from typing import Dict, Any, List, Optional
import logging
from .models import FeatureFlag, FeatureStatus

logger = logging.getLogger(__name__)

# ─── Default Feature Flags ────────────────────────────────────────────────────

DEFAULT_FLAGS = {
    # Agents
    "agent.microscopy": FeatureFlag(name="agent.microscopy", status=FeatureStatus.BETA, enabled=True),
    "agent.genome_browser": FeatureFlag(name="agent.genome_browser", status=FeatureStatus.EXPERIMENTAL, enabled=True),
    "agent.quantum_sim": FeatureFlag(name="agent.quantum_sim", status=FeatureStatus.DISABLED, enabled=False),

    # Mission Types / Scenarios
    "scenario.mitosis_phase": FeatureFlag(name="scenario.mitosis_phase", status=FeatureStatus.STABLE, enabled=True),
    "scenario.sequence_target": FeatureFlag(name="scenario.sequence_target", status=FeatureStatus.EXPERIMENTAL, enabled=True),
    "scenario.proposal_draft": FeatureFlag(name="scenario.proposal_draft", status=FeatureStatus.BETA, enabled=True),

    # Core Features
    "brain.drift_detection": FeatureFlag(name="brain.drift_detection", status=FeatureStatus.STABLE, enabled=True),
    "brain.resource_scheduling": FeatureFlag(name="brain.resource_scheduling", status=FeatureStatus.STABLE, enabled=True),
}

class FeatureFlagController:
    """
    H4: Controlled Feature Flags Engine.
    
    Provides a centralized mechanism to toggle system capabilities,
    experimental agents, and specialized mission workflows.
    """

    def __init__(self, initial_flags: Dict[str, FeatureFlag] = DEFAULT_FLAGS):
        self._flags = initial_flags.copy()

    def is_enabled(self, feature_name: str) -> bool:
        """Checks if a feature is currently active."""
        flag = self._flags.get(feature_name)
        if not flag:
            # Default to enabled for unknown core features to prevent breakage,
            # but log a warning.
            logger.debug(f"[FeatureFlag] Unknown feature request: '{feature_name}'. Assuming enabled.")
            return True
        return flag.enabled

    def get_status(self, feature_name: str) -> FeatureStatus:
        """Returns the release status of a feature."""
        flag = self._flags.get(feature_name)
        return flag.status if flag else FeatureStatus.STABLE

    def toggle(self, feature_name: str, enabled: bool):
        """Manually toggle a feature flag at runtime."""
        if feature_name in self._flags:
            self._flags[feature_name].enabled = enabled
            logger.info(f"[FeatureFlag] Feature '{feature_name}' set to enabled={enabled}")
        else:
            logger.warning(f"[FeatureFlag] Cannot toggle unknown feature '{feature_name}'")

    def get_all_active_features(self) -> List[str]:
        """Returns a list of all currently enabled feature names."""
        return [name for name, flag in self._flags.items() if flag.enabled]

# Singleton instance for system-wide use
FEATURE_CONTROLLER = FeatureFlagController()
