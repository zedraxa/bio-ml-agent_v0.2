import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

class ProteinAnalyzer:
    """
    Protein dizilerinin temel fizikokimyasal özelliklerini analiz eder.
    Büyük ölçüde biopython kütüphanesine dayanmaktadır.
    """
    def __init__(self, protein_sequence: str):
        """
        Args:
            protein_sequence (str): Analiz edilecek amino asit dizisi.
        """
        valid_sequence = "".join(filter(lambda char: char not in "U*O", protein_sequence.upper()))
        self.sequence = valid_sequence
        if not self.sequence:
            raise ValueError("Sağlanan dizide geçerli amino asit bulunamadı.")
        self.analysis = ProteinAnalysis(self.sequence)

    def summary(self, verbose: bool = True):
        aa_percent = self.analysis.get_amino_acids_percent()
        summary_data = {
            "sequence": self.sequence,
            "length": len(self.sequence),
            "molecular_weight": self.analysis.molecular_weight(),
            "isoelectric_point": self.analysis.isoelectric_point(),
            "aromaticity": self.analysis.aromaticity(),
            "instability_index": self.analysis.instability_index(),
            "gravy": self.analysis.gravy(),
            "amino_acid_counts": {aa: self.sequence.count(aa) for aa in sorted(aa_percent.keys())},
            "amino_acid_percent": aa_percent
        }
        return summary_data

class DrugDiscoveryHelper:
    """
    SMILES formatındaki kimyasal bileşiklerin ilaç olabilirlik özelliklerini analiz eder.
    RDKit kütüphanesine dayanmaktadır.
    """
    def __init__(self, smiles_string: str):
        self.mol = Chem.MolFromSmiles(smiles_string)
        if self.mol is None:
            raise ValueError(f"Geçersiz SMILES string: {smiles_string}")
        self.smiles = smiles_string

    def check_lipinski(self, verbose: bool = True):
        mol_weight = Descriptors.MolWt(self.mol)
        logp = Descriptors.MolLogP(self.mol)
        h_donors = Lipinski.NumHDonors(self.mol)
        h_acceptors = Lipinski.NumHAcceptors(self.mol)
        violations = 0
        if mol_weight > 500: violations += 1
        if logp > 5: violations += 1
        if h_donors > 5: violations += 1
        if h_acceptors > 10: violations += 1
        passes = violations < 2
        results = {
            "mol_weight": round(mol_weight, 3),
            "logp": round(logp, 4),
            "h_donors": h_donors,
            "h_acceptors": h_acceptors,
            "passes_lipinski": passes,
            "violations": violations
        }
        return results