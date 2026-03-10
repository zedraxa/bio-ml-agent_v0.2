import unittest
from models.drug_discovery import (
    MoleculeFormat, MoleculeStructure, GNNLayerType, GNNModelConfig,
    MolecularDatasetCard, MolecularExplanation, DrugCandidate
)

class TestDrugDiscoveryModels(unittest.TestCase):
    def test_molecule_structure(self):
        mol = MoleculeStructure(
            molecule_id="mol-aspirin",
            name="Aspirin",
            format=MoleculeFormat.SMILES,
            content="CC(=O)Oc1ccccc1C(=O)O",
            num_atoms=13,
            num_bonds=13
        )
        self.assertEqual(mol.molecule_id, "mol-aspirin")
        self.assertEqual(mol.format, MoleculeFormat.SMILES)

    def test_gnn_model_config(self):
        config = GNNModelConfig(
            model_id="gnn-01",
            name="Toxicity Predictor",
            layers=[GNNLayerType.GCN, GNNLayerType.GAT],
            hidden_channels=64,
            num_layers=2,
            dropout=0.1
        )
        self.assertEqual(config.num_layers, 2)
        self.assertIn(GNNLayerType.GAT, config.layers)

    def test_dataset_card(self):
        card = MolecularDatasetCard(
            dataset_id="mol-net-hiv",
            source="MoleculeNet",
            task_type="classification",
            targets=["HIV_active"],
            size=41127
        )
        self.assertEqual(card.dataset_id, "mol-net-hiv")
        self.assertEqual(card.split_strategy, "scaffold")

    def test_drug_candidate_ranking(self):
        candidate = DrugCandidate(
            candidate_id="cand-001",
            molecule_id="mol-xyz",
            prediction_scores={"solubility": 0.85, "toxicity": 0.02},
            uncertainty=0.05,
            novelty_score=0.9,
            rank=1
        )
        self.assertEqual(candidate.rank, 1)
        self.assertGreater(candidate.prediction_scores["solubility"], 0.8)

if __name__ == '__main__':
    unittest.main()
