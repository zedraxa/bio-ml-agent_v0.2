import unittest
from datetime import datetime
from models.consensus import (
    DebateRole, DebateEntry, DebateSession,
    ReviewOpinion, ConsensusStatus, 
    ConsensusState, ResolutionAction,
    SynthesisReport
)

class TestConsensusModels(unittest.TestCase):
    def test_debate_session(self):
        entry1 = DebateEntry(
            agent_id="proposer-1",
            role=DebateRole.PROPOSER,
            content="I suggest using Qdrant for vector storage.",
            timestamp=datetime.utcnow().isoformat()
        )
        entry2 = DebateEntry(
            agent_id="critic-1",
            role=DebateRole.CRITIC,
            content="Qdrant might be overkill for this project size.",
            timestamp=datetime.utcnow().isoformat()
        )
        session = DebateSession(
            session_id="deb-001",
            topic="Vector DB Selection",
            entries=[entry1, entry2]
        )
        self.assertEqual(len(session.entries), 2)
        self.assertEqual(session.entries[0].role, DebateRole.PROPOSER)

    def test_review_opinion(self):
        opinion = ReviewOpinion(
            reviewer_id="reviewer-agent-1",
            syntax_score=1.0,
            risk_score=0.2,
            logic_score=0.9,
            security_issues=["leaked API key in comments"],
            comment="Code is mostly fine but remove the secrets."
        )
        self.assertEqual(len(opinion.security_issues), 1)
        self.assertEqual(opinion.risk_score, 0.2)

    def test_consensus_and_deadlock(self):
        state = ConsensusState(
            task_id="task-001",
            status=ConsensusStatus.STALEMATE,
            dissenting_opinions=[{"agent_b": "wants pinecone"}]
        )
        self.assertEqual(state.status, ConsensusStatus.STALEMATE)
        
        action = ResolutionAction(
            action_type="human_review",
            reason="Agents cannot agree on DB",
            initiated_by="orchestrator"
        )
        self.assertEqual(action.action_type, "human_review")

    def test_synthesis_report(self):
        report = SynthesisReport(
            report_id="syn-001",
            contributing_agents=["bio-agent", "ml-agent"],
            summary="Combined analysis of genomic data and ML models.",
            key_findings=["Gene X correlates with Feature Y"],
            final_output={"correlation": 0.85},
            created_at=datetime.utcnow().isoformat()
        )
        self.assertIn("bio-agent", report.contributing_agents)
        self.assertEqual(report.final_output["correlation"], 0.85)

if __name__ == '__main__':
    unittest.main()
