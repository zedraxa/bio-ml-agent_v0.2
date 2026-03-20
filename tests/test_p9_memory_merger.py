import unittest
from unittest.mock import MagicMock, patch
import json

from bio_ml_agent.ultra_agent.memory.maintenance import MemoryMerger

class TestMemoryMerger(unittest.TestCase):

    def setUp(self):
        self.mock_store = MagicMock()
        self.mock_store.enabled = True
        self.mock_store.collection_name = "test_memory_collection"
        
        # 3 mock memory returns for listing
        self.mock_memories = [
            {"id": "mem1", "content": "Kullanıcı React kullanmayı seviyor.", "project": "test_project"},
            {"id": "mem2", "content": "Kullanıcı ReactJS tercih ediyor.", "project": "test_project"},
            {"id": "mem3", "content": "Django ve Python backend için kullanıldı.", "project": "test_project"}
        ]
        self.mock_store.list_memories_for_project.return_value = self.mock_memories

    @patch("bio_ml_agent.ultra_agent.memory.maintenance.synthesize_memories_llm")
    @patch("bio_ml_agent.ultra_agent.memory.qdrant_store._encode_text")
    def test_merge_clustering(self, mock_encode, mock_synthesize):
        # Arama yapıldığında, React ile ilgili olanlar birbirini bulsun
        def mock_search(query, **kwargs):
            if "React" in query:
                return [self.mock_memories[0], self.mock_memories[1]]
            return [self.mock_memories[2]]
            
        self.mock_store.search_memory.side_effect = mock_search
        
        # Sentez sonucu Fake payload
        mock_synthesize.return_value = {
            "content": "Kullanıcı frontend için kesinlikle React(JS) tercih etmektedir.",
            "summary": "ReactJS tercihi",
            "memory_type": "preference",
            "metadata": {"merged_from_ids": ["mem1", "mem2"]},
            "project": "test_project"
        }
        
        # Fake vector
        mock_encode.return_value = [0.1] * 384
        
        merger = MemoryMerger(store=self.mock_store)
        merged_count = merger.merge_project_memories("test_project", similarity_threshold=0.85)
        
        # En az 1 birleştirme olmalı (mem1 ve mem2)
        self.assertEqual(merged_count, 1)
        
        # upsert bir kez çağrılmalı (yeni anı ekleme)
        self.mock_store.client.upsert.assert_called_once()
        
        # delete 2 kez çağrılmalı (mem1 ve mem2 silinmeli)
        self.assertEqual(self.mock_store.delete_memory.call_count, 2)
        self.mock_store.delete_memory.assert_any_call("mem1")
        self.mock_store.delete_memory.assert_any_call("mem2")

if __name__ == "__main__":
    unittest.main()
