import unittest
from bio_ml_agent.models.mcp_ops import (
    MCPResourceType, MCPResource,
    MCPTool, MCPServerConfig,
    ToolPermissionScope, UserConsent,
    ProjectAllowlist, PromptTemplateType,
    PromptTemplatePack, DomainKnowledgePack
)

class TestMCPOpsModels(unittest.TestCase):
    def test_mcp_client_server(self):
        resource = MCPResource(
            resource_id="res-001",
            name="local-db-schema",
            type=MCPResourceType.LOCAL_FILE,
            uri="file:///db/schema.sql"
        )
        self.assertEqual(resource.type, MCPResourceType.LOCAL_FILE)
        
        server = MCPServerConfig(
            server_name="postgres-mcp",
            command="npx",
            args=["@modelcontextprotocol/server-postgres", "postgres://user:pass@localhost/db"]
        )
        self.assertEqual(server.command, "npx")

    def test_permission_broker(self):
        consent = UserConsent(
            consent_id="c-999",
            tool_name="drop_database",
            granted_at="2026-03-01T10:00:00Z",
            expires_in_seconds=3600
        )
        self.assertEqual(consent.expires_in_seconds, 3600)

        allowlist = ProjectAllowlist(
            project_id="agent-v3",
            allowed_read_paths=["/src", "/data"],
            banned_tools=["rm_rf"]
        )
        self.assertIn("/src", allowlist.allowed_read_paths)
        self.assertIn("rm_rf", allowlist.banned_tools)

    def test_catalog_components(self):
        pack = PromptTemplatePack(
            pack_id="pack-bio-1",
            label="Genomics Data Analysis",
            pack_type=PromptTemplateType.WORKFLOW,
            template_content="1. Sequence data 2. Alignment",
            required_variables=["dataset_path"]
        )
        self.assertEqual(pack.pack_type, PromptTemplateType.WORKFLOW)
        
        domain_pack = DomainKnowledgePack(
            knowledge_id="dk-ml",
            domain="Machine_Learning",
            attached_resources=["res-papers", "res-code"]
        )
        self.assertIn("res-papers", domain_pack.attached_resources)

if __name__ == '__main__':
    unittest.main()
