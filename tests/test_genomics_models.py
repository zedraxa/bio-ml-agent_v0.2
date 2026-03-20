import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.genomics import (
    TokenizerStrategy, SequenceTokenizerConfig, GenomicEmbeddingMeta,
    GenomicTaskType, GenomicTaskConfig, ChunkingStrategy, LongSequenceStrategy
)

class TestGenomicsModels(unittest.TestCase):
    def test_tokenizer_config(self):
        config = SequenceTokenizerConfig(
            tokenizer_id="tok-dnabert2",
            name="DNABERT-2 BPE Tokenizer",
            strategy=TokenizerStrategy.BPE,
            vocab_size=32000,
            version="1.0.0"
        )
        self.assertEqual(config.strategy, TokenizerStrategy.BPE)
        self.assertEqual(config.vocab_size, 32000)

    def test_genomic_embedding_meta(self):
        meta = GenomicEmbeddingMeta(
            embedding_id="emb-001",
            artifact_id="seq-123",
            model_id="dnabert2-base",
            tokenizer_id="tok-dnabert2",
            dimension=768,
            pooling_strategy="mean",
            species="Homo sapiens",
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            preprocessing_signature="hash_xyz",
            model_version="2.0"
        )
        self.assertEqual(meta.dimension, 768)
        self.assertEqual(meta.species, "Homo sapiens")

    def test_genomic_task_config(self):
        config = GenomicTaskConfig(
            task_id="task-promoter-01",
            name="Promoter Prediction",
            task_type=GenomicTaskType.PROMOTER_PREDICTION,
            required_context_length=512,
            labels=["promoter", "non-promoter"]
        )
        self.assertEqual(config.task_type, GenomicTaskType.PROMOTER_PREDICTION)
        self.assertIn("accuracy", config.metrics)

    def test_long_sequence_strategy(self):
        strategy = LongSequenceStrategy(
            strategy_id="strat-long-01",
            chunk_size=512,
            overlap_size=64,
            chunking_type=ChunkingStrategy.OVERLAPPING
        )
        self.assertEqual(strategy.chunk_size, 512)
        self.assertEqual(strategy.chunking_type, ChunkingStrategy.OVERLAPPING)

if __name__ == '__main__':
    unittest.main()
