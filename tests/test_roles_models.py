import unittest
from models.roles import (
    AgentContract, HandoffPayload,
    ModelStrength, BudgetPolicy,
    ModelRoutingConfig
)

class TestRolesModels(unittest.TestCase):
    def test_agent_contract(self):
        contract = AgentContract(
            role_id="bio-analyst-1",
            name="Bioinformatics Analyst",
            do_list=["analyze DNA", "visualize results"],
            dont_list=["manage budget", "write frontend code"],
            tool_allowlist=["blast_tool", "seq_viz"],
            output_schema={"type": "object", "properties": {"analysis": {"type": "string"}}}
        )
        self.assertEqual(contract.role_id, "bio-analyst-1")
        self.assertIn("analyze DNA", contract.do_list)
        self.assertIn("blast_tool", contract.tool_allowlist)

    def test_handoff_protocol(self):
        payload = HandoffPayload(
            source_agent_id="researcher-1",
            target_agent_id="coder-1",
            partial_results={"topic": "found lib for DNA"},
            open_questions=["how to install this?"],
            attached_artifacts=["art-001"],
            priority_level=5
        )
        self.assertEqual(payload.priority_level, 5)
        self.assertEqual(payload.source_agent_id, "researcher-1")
        self.assertIn("how to install this?", payload.open_questions)

    def test_budget_and_routing(self):
        policy = BudgetPolicy(
            max_cost_usd=10.0,
            preferred_strength=ModelStrength.CLOUD_POWERFUL
        )
        self.assertEqual(policy.max_cost_usd, 10.0)
        self.assertEqual(policy.preferred_strength, ModelStrength.CLOUD_POWERFUL)

        routing = ModelRoutingConfig(
            task_complexity_score=0.9,
            forced_model="gemini-pro"
        )
        self.assertEqual(routing.task_complexity_score, 0.9)
        self.assertEqual(routing.forced_model, "gemini-pro")

if __name__ == '__main__':
    unittest.main()
