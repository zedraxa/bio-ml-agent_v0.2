# Developer Notes: Agent Registry & Tiering

## Registry Discovery Mechanism
The `AgentRegistry` uses a two-phase discovery process:
1. **Canonical Registry (`agent_registry.yaml`)**: Explicitly defined agents with full metadata. These are considered the "Source of Truth".
2. **Dynamic Discovery**: The registry scans `src/bio_ml_agent/agents/` for any `.py` files inheriting from `BaseSubAgent`. Anything found here that isn't in the YAML is automatically registered as `EXPERIMENTAL`.

## Agent Maturity Tiers

| Tier | Description |
| --- | --- |
| **STABLE** | Production-ready, fully tested, and documented. Available for all mission packs. |
| **ACTIVE** | Currently in use in primary research workflows but may undergo minor interface changes. |
| **BETA** | Feature-complete but needs wider validation. |
| **EXPERIMENTAL** | Newly discovered or prototype agents. May have unstable dependencies or incomplete logic. |

## Daily Development Flow
1. **New Agent**: Create a file in `src/bio_ml_agent/agents/[domain]/my_agent.py`.
2. **Auto-Discovery**: Run the API or tests; the registry will find your agent and assign it the `EXPERIMENTAL` tier.
3. **Promotion**: Once stable, add an entry to `src/bio_ml_agent/resources/agent_registry.yaml` to promote it to `ACTIVE` or `STABLE`.

## Active vs Experimental Registry
- **Active Registry**: The set of agents currently "Vetted" for mission packs.
- **Experimental Registry**: The "Playground" where new capabilities are discovered automatically.

---
*Note: Always use `MissionOrchestrator` for multi-agent coordination. Direct agent instantiation is discouraged outside of unit tests.*
 Riverside
