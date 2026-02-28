# bioeng_toolkit.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Biyomühendislik Araç Seti
#  Protein, Genomik, Medikal Görüntü, İlaç Keşfi,
#  Atık Su Analizi, EEG/EMG Sinyal İşleme
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("bio_ml_agent")

# Opsiyonel bağımlılıklar
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═════════════════════════════════════════════
#  1. Protein Analizi
# ═════════════════════════════════════════════

# Standart amino asit özellikleri
_AA_PROPERTIES = {
    "A": {"name": "Alanine",       "mw": 89.09,  "hydro": 1.8,  "charge": 0, "polar": False},
    "R": {"name": "Arginine",      "mw": 174.20, "hydro": -4.5, "charge": 1, "polar": True},
    "N": {"name": "Asparagine",    "mw": 132.12, "hydro": -3.5, "charge": 0, "polar": True},
    "D": {"name": "Aspartate",     "mw": 133.10, "hydro": -3.5, "charge": -1,"polar": True},
    "C": {"name": "Cysteine",      "mw": 121.16, "hydro": 2.5,  "charge": 0, "polar": False},
    "E": {"name": "Glutamate",     "mw": 147.13, "hydro": -3.5, "charge": -1,"polar": True},
    "Q": {"name": "Glutamine",     "mw": 146.15, "hydro": -3.5, "charge": 0, "polar": True},
    "G": {"name": "Glycine",       "mw": 75.03,  "hydro": -0.4, "charge": 0, "polar": False},
    "H": {"name": "Histidine",     "mw": 155.16, "hydro": -3.2, "charge": 0, "polar": True},
    "I": {"name": "Isoleucine",    "mw": 131.17, "hydro": 4.5,  "charge": 0, "polar": False},
    "L": {"name": "Leucine",       "mw": 131.17, "hydro": 3.8,  "charge": 0, "polar": False},
    "K": {"name": "Lysine",        "mw": 146.19, "hydro": -3.9, "charge": 1, "polar": True},
    "M": {"name": "Methionine",    "mw": 149.21, "hydro": 1.9,  "charge": 0, "polar": False},
    "F": {"name": "Phenylalanine", "mw": 165.19, "hydro": 2.8,  "charge": 0, "polar": False},
    "P": {"name": "Proline",       "mw": 115.13, "hydro": -1.6, "charge": 0, "polar": False},
    "S": {"name": "Serine",        "mw": 105.09, "hydro": -0.8, "charge": 0, "polar": True},
    "T": {"name": "Threonine",     "mw": 119.12, "hydro": -0.7, "charge": 0, "polar": True},
    "W": {"name": "Tryptophan",    "mw": 204.23, "hydro": -0.9, "charge": 0, "polar": False},
    "Y": {"name": "Tyrosine",      "mw": 181.19, "hydro": -1.3, "charge": 0, "polar": True},
    "V": {"name": "Valine",        "mw": 117.15, "hydro": 4.2,  "charge": 0, "polar": False},
}


class ProteinAnalyzer:
    """Protein sekans analizi araçları.

    Kullanım:
        pa = ProteinAnalyzer("MKWVTFISLLLLFSSAYS")
        print(pa.summary())
        print(pa.amino_acid_composition())
        print(pa.hydropathy_profile(window=7))
    """

    def __init__(self, sequence: str):
        self.sequence = sequence.upper().replace(" ", "").replace("\n", "")
        self._validate()

    def _validate(self) -> None:
        invalid = set(self.sequence) - set(_AA_PROPERTIES.keys())
        if invalid:
            raise ValueError(f"Geçersiz amino asit(ler): {', '.join(sorted(invalid))}")

    @property
    def length(self) -> int:
        return len(self.sequence)

    def molecular_weight(self) -> float:
        """Yaklaşık moleküler ağırlık (Da)."""
        water = 18.015
        mw = sum(_AA_PROPERTIES[aa]["mw"] for aa in self.sequence)
        mw -= water * (self.length - 1)  # Peptit bağı su kaybı
        return round(mw, 2)

    def amino_acid_composition(self) -> Dict[str, float]:
        """Amino asit yüzde kompozisyonu."""
        counts = Counter(self.sequence)
        return {aa: round(count / self.length * 100, 2) for aa, count in sorted(counts.items())}

    def hydropathy_profile(self, window: int = 7) -> List[float]:
        """Kyte-Doolittle hidrofobisite profili (sliding window).

        Args:
            window: Pencere boyutu (varsayılan 7).

        Returns:
            Her pozisyon için ortalama hidrofobisite değeri.
        """
        if self.length < window:
            return [self.grand_average_hydropathy()]

        profile = []
        half = window // 2
        for i in range(half, self.length - half):
            segment = self.sequence[i - half:i + half + 1]
            avg = sum(_AA_PROPERTIES[aa]["hydro"] for aa in segment) / window
            profile.append(round(avg, 3))
        return profile

    def grand_average_hydropathy(self) -> float:
        """GRAVY (Grand Average of Hydropathy) değeri."""
        total = sum(_AA_PROPERTIES[aa]["hydro"] for aa in self.sequence)
        return round(total / self.length, 3)

    def isoelectric_point(self) -> float:
        """Yaklaşık izoelektrik nokta (pI) tahmini."""
        pos_residues = sum(1 for aa in self.sequence if _AA_PROPERTIES[aa]["charge"] > 0)
        neg_residues = sum(1 for aa in self.sequence if _AA_PROPERTIES[aa]["charge"] < 0)
        # Basit Henderson-Hasselbalch yaklaşımı
        if pos_residues + neg_residues == 0:
            return 7.0
        ratio = pos_residues / max(neg_residues, 1)
        pi = 7.0 + math.log10(ratio) if ratio > 0 else 7.0
        return round(min(max(pi, 1.0), 14.0), 2)

    def secondary_structure_tendency(self) -> Dict[str, float]:
        """İkincil yapı eğilimi (Chou-Fasman basitleştirilmiş)."""
        helix_formers = set("AELM")
        sheet_formers = set("VIY")
        turn_formers = set("GNPS")

        h = sum(1 for aa in self.sequence if aa in helix_formers)
        s = sum(1 for aa in self.sequence if aa in sheet_formers)
        t = sum(1 for aa in self.sequence if aa in turn_formers)

        return {
            "helix_tendency": round(h / self.length * 100, 1),
            "sheet_tendency": round(s / self.length * 100, 1),
            "turn_tendency": round(t / self.length * 100, 1),
        }

    def summary(self) -> Dict[str, Any]:
        """Kapsamlı protein özeti."""
        polar = sum(1 for aa in self.sequence if _AA_PROPERTIES[aa]["polar"])
        return {
            "length": self.length,
            "molecular_weight_da": self.molecular_weight(),
            "gravy": self.grand_average_hydropathy(),
            "isoelectric_point": self.isoelectric_point(),
            "polar_residues_pct": round(polar / self.length * 100, 1),
            "nonpolar_residues_pct": round((self.length - polar) / self.length * 100, 1),
            "secondary_structure": self.secondary_structure_tendency(),
        }


# ═════════════════════════════════════════════
#  2. Genomik Analiz
# ═════════════════════════════════════════════

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class GenomicAnalyzer:
    """DNA/RNA sekans analizi araçları.

    Kullanım:
        ga = GenomicAnalyzer("ATGCGATCGATCG")
        print(ga.gc_content())
        print(ga.find_orfs())
        print(ga.translate())
    """

    def __init__(self, sequence: str):
        self.sequence = sequence.upper().replace(" ", "").replace("\n", "")
        self._is_rna = "U" in self.sequence
        if self._is_rna:
            self.sequence = self.sequence.replace("U", "T")  # Dahili olarak DNA kullan
        self._validate()

    def _validate(self) -> None:
        valid = set("ATGCN")
        invalid = set(self.sequence) - valid
        if invalid:
            raise ValueError(f"Geçersiz nükleotid(ler): {', '.join(sorted(invalid))}")

    @property
    def length(self) -> int:
        return len(self.sequence)

    def gc_content(self) -> float:
        """GC içeriği yüzdesi."""
        gc = sum(1 for nt in self.sequence if nt in "GC")
        return round(gc / self.length * 100, 2) if self.length > 0 else 0.0

    def nucleotide_frequency(self) -> Dict[str, int]:
        """Nükleotid frekansları."""
        return dict(Counter(self.sequence))

    def complement(self) -> str:
        """Tamamlayıcı zincir."""
        comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        return "".join(comp.get(nt, "N") for nt in self.sequence)

    def reverse_complement(self) -> str:
        """Ters tamamlayıcı zincir."""
        return self.complement()[::-1]

    def transcribe(self) -> str:
        """DNA → mRNA transkripsiyon."""
        return self.sequence.replace("T", "U")

    def translate(self, frame: int = 0) -> str:
        """DNA → Protein translasyon.

        Args:
            frame: Okuma çerçevesi (0, 1, veya 2).

        Returns:
            Amino asit dizisi (* = stop kodon).
        """
        seq = self.sequence[frame:]
        codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
        protein = []
        for codon in codons:
            aa = _CODON_TABLE.get(codon, "X")
            protein.append(aa)
        return "".join(protein)

    def codon_usage(self) -> Dict[str, int]:
        """Kodon kullanım tablosu."""
        codons = [self.sequence[i:i + 3] for i in range(0, len(self.sequence) - 2, 3)]
        return dict(Counter(c for c in codons if len(c) == 3))

    def find_orfs(self, min_length: int = 30) -> List[Dict[str, Any]]:
        """Açık okuma çerçevelerini (ORF) bul.

        Args:
            min_length: Minimum ORF uzunluğu (nükleotid).

        Returns:
            ORF bilgileri listesi.
        """
        orfs = []
        for frame in range(3):
            seq = self.sequence[frame:]
            i = 0
            while i < len(seq) - 2:
                codon = seq[i:i + 3]
                if codon == "ATG":  # Start kodon
                    start = i + frame
                    j = i + 3
                    while j < len(seq) - 2:
                        stop_codon = seq[j:j + 3]
                        if stop_codon in ("TAA", "TAG", "TGA"):
                            end = j + frame + 3
                            orf_len = end - start
                            if orf_len >= min_length:
                                protein = self.translate(frame=start)[:(orf_len // 3)]
                                orfs.append({
                                    "start": start,
                                    "end": end,
                                    "length_nt": orf_len,
                                    "length_aa": orf_len // 3,
                                    "frame": frame,
                                    "protein": protein.rstrip("*"),
                                })
                            i = j + 3
                            break
                        j += 3
                    else:
                        i += 3
                        continue
                else:
                    i += 3
        return sorted(orfs, key=lambda x: x["length_nt"], reverse=True)

    def melting_temperature(self) -> float:
        """Yaklaşık erime sıcaklığı (Tm) — kısa oligonükleotidler için.

        Wallace kuralı: Tm = 2(A+T) + 4(G+C)
        """
        a_t = sum(1 for nt in self.sequence if nt in "AT")
        g_c = sum(1 for nt in self.sequence if nt in "GC")
        return float(2 * a_t + 4 * g_c)

    def summary(self) -> Dict[str, Any]:
        """Kapsamlı sekans özeti."""
        return {
            "length": self.length,
            "gc_content_pct": self.gc_content(),
            "nucleotide_freq": self.nucleotide_frequency(),
            "melting_temp_c": self.melting_temperature(),
            "orf_count": len(self.find_orfs()),
            "is_rna_origin": self._is_rna,
        }


# ═════════════════════════════════════════════
#  3. Medikal Görüntü Analizi
# ═════════════════════════════════════════════

class MedicalImageHelper:
    """Medikal görüntü analizi yardımcıları.

    Sklearn + numpy tabanlı ön-işleme ve model pipeline.
    Opsiyonel: PIL/Pillow, scikit-image.
    """

    # Yaygın medikal görüntü veri setleri ve yapılandırmaları
    PRESETS = {
        "chest_xray": {
            "input_size": (224, 224),
            "channels": 1,
            "classes": ["NORMAL", "PNEUMONIA"],
            "normalize_range": (0, 255),
            "description": "Göğüs X-ray pnömoni sınıflandırma",
        },
        "brain_mri": {
            "input_size": (256, 256),
            "channels": 1,
            "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
            "normalize_range": (0, 255),
            "description": "Beyin MRI tümör sınıflandırma",
        },
        "skin_lesion": {
            "input_size": (224, 224),
            "channels": 3,
            "classes": ["benign", "malignant"],
            "normalize_range": (0, 255),
            "description": "Deri lezyonu melanom tespiti",
        },
        "retinal_oct": {
            "input_size": (224, 224),
            "channels": 1,
            "classes": ["CNV", "DME", "DRUSEN", "NORMAL"],
            "normalize_range": (0, 255),
            "description": "Retinal OCT hastalık sınıflandırma",
        },
    }

    @staticmethod
    def get_preprocessing_pipeline(preset_name: str) -> str:
        """Ön-işleme pipeline Python kodu üret.

        Args:
            preset_name: Preset adı (chest_xray, brain_mri, vb.)

        Returns:
            Python kodu (str).
        """
        preset = MedicalImageHelper.PRESETS.get(preset_name)
        if not preset:
            available = ", ".join(MedicalImageHelper.PRESETS.keys())
            raise ValueError(f"Bilinmeyen preset: {preset_name}. Mevcut: {available}")

        w, h = preset["input_size"]
        ch = preset["channels"]
        classes = preset["classes"]
        color_mode = "grayscale" if ch == 1 else "rgb"

        return f'''# {preset["description"]} — Ön-işleme Pipeline
import numpy as np
from pathlib import Path

def load_and_preprocess(image_path, target_size=({w}, {h})):
    """Görüntüyü yükle ve ön-işle."""
    from PIL import Image

    img = Image.open(image_path)
    img = img.convert("{"L" if ch == 1 else "RGB"}")
    img = img.resize(target_size)

    arr = np.array(img, dtype=np.float32) / 255.0
    {"arr = arr[..., np.newaxis]  # (H, W, 1)" if ch == 1 else "# (H, W, 3)"}
    return arr

def create_dataset(data_dir, target_size=({w}, {h})):
    """Dizinden veri seti oluştur."""
    X, y = [], []
    classes = {classes}
    for i, cls in enumerate(classes):
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.*"):
            try:
                arr = load_and_preprocess(img_path, target_size)
                X.append(arr)
                y.append(i)
            except Exception:
                continue
    return np.array(X), np.array(y)
'''

    @staticmethod
    def get_augmentation_code() -> str:
        """Veri augmentation kodu üret."""
        return '''# Medikal Görüntü Augmentation
import numpy as np

def augment_image(image, seed=None):
    """Basit augmentation (numpy tabanlı)."""
    if seed is not None:
        np.random.seed(seed)

    augmented = image.copy()

    # Rastgele yatay çevirme
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)

    # Rastgele döndürme (+/- 15 derece benzeri affine dönüşüm)
    if np.random.random() > 0.5:
        augmented = np.rot90(augmented, k=np.random.choice([0, 1, 2, 3]))

    # Rastgele parlaklık ayarı
    brightness = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented * brightness, 0, 1)

    # Rastgele gürültü ekleme
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.02, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)

    return augmented.astype(np.float32)
'''

    @staticmethod
    def list_presets() -> List[Dict[str, Any]]:
        """Mevcut preset'leri listele."""
        return [
            {"name": k, **v}
            for k, v in MedicalImageHelper.PRESETS.items()
        ]


# ═════════════════════════════════════════════
#  4. İlaç Keşfi (SMILES tabanlı)
# ═════════════════════════════════════════════

# Yaygın element ağırlıkları
_ELEMENT_WEIGHTS = {
    "C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007,
    "S": 32.065, "P": 30.974, "F": 18.998, "Cl": 35.453,
    "Br": 79.904, "I": 126.904,
}


class DrugDiscoveryHelper:
    """SMILES tabanlı basit moleküler analiz.

    Ağır bağımlılık (RDKit) olmadan temel özellik çıkarma.
    RDKit varsa gelişmiş özellikler kullanılabilir.

    Kullanım:
        ddh = DrugDiscoveryHelper("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
        print(ddh.molecular_formula())
        print(ddh.lipinski_rule_of_five())
    """

    def __init__(self, smiles: str):
        self.smiles = smiles.strip()
        self._has_rdkit = False
        self._mol = None
        try:
            from rdkit import Chem
            self._mol = Chem.MolFromSmiles(self.smiles)
            if self._mol is not None:
                self._has_rdkit = True
        except ImportError:
            pass

    def atom_counts(self) -> Dict[str, int]:
        """SMILES'daki atom sayıları (basit ayrıştırma)."""
        if self._has_rdkit and self._mol:
            from rdkit import Chem
            mol = Chem.AddHs(self._mol)
            counts: Dict[str, int] = {}
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                counts[sym] = counts.get(sym, 0) + 1
            return counts

        # Basit SMILES ayrıştırma (RDKit olmadan)
        counts = {}
        # Büyük harfli elementleri say
        cleaned = re.sub(r'\[.*?\]', '', self.smiles)
        cleaned = re.sub(r'[^A-Za-z]', '', cleaned)

        i = 0
        while i < len(cleaned):
            if i + 1 < len(cleaned) and cleaned[i + 1].islower():
                elem = cleaned[i:i + 2]
                i += 2
            else:
                elem = cleaned[i]
                i += 1
            counts[elem] = counts.get(elem, 0) + 1
        return counts

    def molecular_formula(self) -> str:
        """Moleküler formül."""
        counts = self.atom_counts()
        # Hill sistemi: C, H önce, sonra alfabetik
        formula = ""
        for elem in ["C", "H"]:
            if elem in counts:
                formula += elem + (str(counts[elem]) if counts[elem] > 1 else "")
        for elem in sorted(counts.keys()):
            if elem not in ("C", "H"):
                formula += elem + (str(counts[elem]) if counts[elem] > 1 else "")
        return formula

    def estimated_molecular_weight(self) -> float:
        """Tahmini moleküler ağırlık."""
        if self._has_rdkit and self._mol:
            from rdkit.Chem import Descriptors
            return round(Descriptors.MolWt(self._mol), 2)

        counts = self.atom_counts()
        mw = sum(_ELEMENT_WEIGHTS.get(elem, 12.0) * count for elem, count in counts.items())
        return round(mw, 2)

    def h_bond_donors(self) -> int:
        """Tahmini H-bağı donör sayısı."""
        if self._has_rdkit and self._mol:
            from rdkit.Chem import Descriptors
            return Descriptors.NumHDonors(self._mol)
        # Basit tahmin: OH, NH, NH2 sayısı
        return self.smiles.count("O") + self.smiles.count("N")

    def h_bond_acceptors(self) -> int:
        """Tahmini H-bağı akseptör sayısı."""
        if self._has_rdkit and self._mol:
            from rdkit.Chem import Descriptors
            return Descriptors.NumHAcceptors(self._mol)
        return self.smiles.count("O") + self.smiles.count("N")

    def rotatable_bonds(self) -> int:
        """Tahmini dönebilen bağ sayısı."""
        if self._has_rdkit and self._mol:
            from rdkit.Chem import Descriptors
            return Descriptors.NumRotatableBonds(self._mol)
        # Basit tahmin: tek bağ sayısı (halka dışı)
        single_bonds = self.smiles.count("-") + len(re.findall(r'[A-Z][a-z]?[A-Z]', self.smiles))
        return max(0, single_bonds)

    def lipinski_rule_of_five(self) -> Dict[str, Any]:
        """Lipinski'nin Beşli Kuralı kontrolü.

        İyi oral biyoyararlanım için bir ilacın:
        - MW ≤ 500 Da
        - LogP ≤ 5
        - H-bağı donörleri ≤ 5
        - H-bağı akseptörleri ≤ 10
        """
        mw = self.estimated_molecular_weight()
        hbd = self.h_bond_donors()
        hba = self.h_bond_acceptors()

        # LogP tahmini (basit — atom sayısı tabanlı)
        if self._has_rdkit and self._mol:
            from rdkit.Chem import Descriptors
            logp = round(Descriptors.MolLogP(self._mol), 2)
        else:
            counts = self.atom_counts()
            c_count = counts.get("C", 0)
            o_count = counts.get("O", 0)
            n_count = counts.get("N", 0)
            logp = round(c_count * 0.5 - o_count * 1.0 - n_count * 1.0, 2)

        violations = 0
        checks = {
            "molecular_weight": {"value": mw, "limit": 500, "pass": mw <= 500},
            "logP": {"value": logp, "limit": 5, "pass": logp <= 5},
            "h_bond_donors": {"value": hbd, "limit": 5, "pass": hbd <= 5},
            "h_bond_acceptors": {"value": hba, "limit": 10, "pass": hba <= 10},
        }
        violations = sum(1 for c in checks.values() if not c["pass"])

        return {
            "checks": checks,
            "violations": violations,
            "drug_like": violations <= 1,
            "note": "RDKit kullanılıyor" if self._has_rdkit else "Basit tahmin (RDKit olmadan)",
        }

    def summary(self) -> Dict[str, Any]:
        """Molekül özeti."""
        return {
            "smiles": self.smiles,
            "formula": self.molecular_formula(),
            "molecular_weight": self.estimated_molecular_weight(),
            "atom_counts": self.atom_counts(),
            "lipinski": self.lipinski_rule_of_five(),
        }


# ═════════════════════════════════════════════
#  5. Atık Su Analizi
# ═════════════════════════════════════════════

# WHO içme suyu kalitesi limitleri
_WHO_LIMITS = {
    "pH":             {"min": 6.5, "max": 8.5, "unit": "—",    "description": "pH değeri"},
    "turbidity":      {"min": 0,   "max": 5,   "unit": "NTU",  "description": "Bulanıklık"},
    "tds":            {"min": 0,   "max": 500, "unit": "mg/L", "description": "Toplam Çözünmüş Katı"},
    "conductivity":   {"min": 0,   "max": 400, "unit": "µS/cm","description": "İletkenlik"},
    "bod":            {"min": 0,   "max": 6,   "unit": "mg/L", "description": "Biyolojik Oksijen İhtiyacı"},
    "cod":            {"min": 0,   "max": 10,  "unit": "mg/L", "description": "Kimyasal Oksijen İhtiyacı"},
    "dissolved_oxygen":{"min": 5,  "max": 14,  "unit": "mg/L", "description": "Çözünmüş Oksijen"},
    "nitrate":        {"min": 0,   "max": 50,  "unit": "mg/L", "description": "Nitrat"},
    "phosphate":      {"min": 0,   "max": 5,   "unit": "mg/L", "description": "Fosfat"},
    "ammonia":        {"min": 0,   "max": 1.5, "unit": "mg/L", "description": "Amonyak"},
    "chloride":       {"min": 0,   "max": 250, "unit": "mg/L", "description": "Klorür"},
    "sulfate":        {"min": 0,   "max": 250, "unit": "mg/L", "description": "Sülfat"},
    "iron":           {"min": 0,   "max": 0.3, "unit": "mg/L", "description": "Demir"},
    "manganese":      {"min": 0,   "max": 0.1, "unit": "mg/L", "description": "Mangan"},
    "fluoride":       {"min": 0,   "max": 1.5, "unit": "mg/L", "description": "Florür"},
    "arsenic":        {"min": 0,   "max": 0.01,"unit": "mg/L", "description": "Arsenik"},
    "lead":           {"min": 0,   "max": 0.01,"unit": "mg/L", "description": "Kurşun"},
    "mercury":        {"min": 0,   "max": 0.006,"unit":"mg/L", "description": "Cıva"},
    "e_coli":         {"min": 0,   "max": 0,   "unit": "CFU/100mL", "description": "E. coli"},
}


class WastewaterAnalyzer:
    """Atık su kalite analizi.

    Kullanım:
        wa = WastewaterAnalyzer({
            "pH": 7.2, "turbidity": 3.5, "tds": 320,
            "bod": 4.5, "cod": 8.0, "dissolved_oxygen": 6.5,
        })
        print(wa.check_compliance())
        print(wa.water_quality_index())
    """

    def __init__(self, parameters: Dict[str, float]):
        self.parameters = {k.lower(): v for k, v in parameters.items()}

    def check_compliance(self) -> Dict[str, Dict[str, Any]]:
        """WHO standartlarına uygunluk kontrolü.

        Returns:
            Her parametre için uygunluk durumu.
        """
        results = {}
        for param, value in self.parameters.items():
            limit = _WHO_LIMITS.get(param)
            if limit is None:
                results[param] = {"value": value, "status": "unknown", "message": "Limit tanımsız"}
                continue

            if limit["min"] <= value <= limit["max"]:
                status = "compliant"
                msg = "✅ WHO standardına uygun"
            else:
                status = "non_compliant"
                if value < limit["min"]:
                    msg = f"⚠️ Minimum limitin altında (min: {limit['min']} {limit['unit']})"
                else:
                    msg = f"❌ Maksimum limiti aşıyor (max: {limit['max']} {limit['unit']})"

            results[param] = {
                "value": value,
                "unit": limit["unit"],
                "limit_min": limit["min"],
                "limit_max": limit["max"],
                "status": status,
                "message": msg,
                "description": limit["description"],
            }
        return results

    def water_quality_index(self) -> Dict[str, Any]:
        """Basitleştirilmiş Su Kalitesi İndeksi (WQI) hesapla.

        0-100 arası puan:
            90-100: Mükemmel
            70-89:  İyi
            50-69:  Orta
            25-49:  Kötü
            0-24:   Çok kötü
        """
        compliance = self.check_compliance()
        total = 0
        count = 0

        for param, result in compliance.items():
            if result["status"] == "unknown":
                continue
            count += 1
            if result["status"] == "compliant":
                # Limite ne kadar yakın
                limit = _WHO_LIMITS[param]
                limit_range = limit["max"] - limit["min"]
                if limit_range > 0:
                    # Limitlerin ortasına yakınlık
                    mid = (limit["min"] + limit["max"]) / 2
                    distance = abs(result["value"] - mid) / (limit_range / 2)
                    score = max(0, 100 * (1 - distance * 0.3))
                else:
                    score = 100 if result["value"] == 0 else 0
                total += score
            else:
                # Limit dışı — ne kadar aştığına göre puan
                total += 20  # Baseline kötü puan

        wqi = round(total / max(count, 1), 1)

        if wqi >= 90:
            quality = "Mükemmel 🟢"
        elif wqi >= 70:
            quality = "İyi 🟡"
        elif wqi >= 50:
            quality = "Orta 🟠"
        elif wqi >= 25:
            quality = "Kötü 🔴"
        else:
            quality = "Çok Kötü ⛔"

        return {
            "wqi_score": wqi,
            "quality": quality,
            "parameters_checked": count,
            "compliant_count": sum(1 for r in compliance.values() if r["status"] == "compliant"),
            "non_compliant_count": sum(1 for r in compliance.values() if r["status"] == "non_compliant"),
        }

    def treatment_suggestions(self) -> List[str]:
        """Uygun olmayan parametreler için arıtma önerileri."""
        suggestions = []
        compliance = self.check_compliance()

        treatment_map = {
            "pH": "pH ayarlama (asit/baz dozajlama)",
            "turbidity": "Koagülasyon + flokülasyon + sedimantasyon",
            "tds": "Ters ozmoz (RO) veya nanofiltrasyon",
            "bod": "Biyolojik arıtma (aktif çamur, damlatmalı filtre)",
            "cod": "İleri oksidasyon prosesleri (AOP), ozonlama",
            "dissolved_oxygen": "Havalandırma sistemi (difüzör, yüzey havalandırıcı)",
            "nitrate": "Biyolojik denitrifikasyon, iyon değişimi",
            "phosphate": "Kimyasal çöktürme (Al/Fe tuzları), biyolojik P giderimi",
            "ammonia": "Nitrifikasyon (biyolojik), air stripping",
            "iron": "Oksidasyon + filtrasyon, iyon değişimi",
            "arsenic": "Koagülasyon, adsorpsiyon (aktif karbon)",
            "lead": "Kimyasal çöktürme, iyon değişimi, RO",
            "mercury": "Aktif karbon adsorpsiyonu, kimyasal çöktürme",
        }

        for param, result in compliance.items():
            if result["status"] == "non_compliant":
                treatment = treatment_map.get(param, "Uzman değerlendirmesi gerekli")
                suggestions.append(
                    f"⚠️ {result.get('description', param)} "
                    f"(değer: {result['value']} {result.get('unit', '')}) → {treatment}"
                )
        return suggestions

    @staticmethod
    def get_who_limits() -> Dict[str, Dict[str, Any]]:
        """WHO su kalitesi limitlerini döndür."""
        return dict(_WHO_LIMITS)

    def summary(self) -> Dict[str, Any]:
        """Kapsamlı su kalitesi özeti."""
        wqi = self.water_quality_index()
        suggestions = self.treatment_suggestions()
        return {
            "water_quality_index": wqi,
            "compliance": self.check_compliance(),
            "treatment_suggestions": suggestions,
            "parameter_count": len(self.parameters),
        }


# ═════════════════════════════════════════════
#  6. EEG/EMG Sinyal İşleme
# ═════════════════════════════════════════════

class BioSignalProcessor:
    """EEG/EMG sinyal işleme araçları.

    Numpy tabanlı temel sinyal işleme.

    Kullanım:
        import numpy as np
        signal = np.random.randn(1000)  # 1s, 1000 Hz
        bsp = BioSignalProcessor(signal, sampling_rate=1000)
        print(bsp.basic_stats())
        print(bsp.frequency_bands())
    """

    # EEG frekans bantları
    EEG_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 100),
    }

    def __init__(self, signal: Any, sampling_rate: float = 256.0, channel_name: str = "CH1"):
        if HAS_NUMPY:
            self.signal = np.asarray(signal, dtype=np.float64)
        else:
            self.signal = list(signal)
        self.fs = sampling_rate
        self.channel_name = channel_name

    @property
    def duration(self) -> float:
        """Sinyal süresi (saniye)."""
        n = len(self.signal) if not HAS_NUMPY else self.signal.shape[0]
        return n / self.fs

    @property
    def n_samples(self) -> int:
        return len(self.signal) if not HAS_NUMPY else int(self.signal.shape[0])

    def basic_stats(self) -> Dict[str, float]:
        """Temel istatistikler."""
        if HAS_NUMPY:
            arr = self.signal
            return {
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "min": round(float(np.min(arr)), 6),
                "max": round(float(np.max(arr)), 6),
                "rms": round(float(np.sqrt(np.mean(arr ** 2))), 6),
                "duration_s": round(self.duration, 3),
                "n_samples": self.n_samples,
                "sampling_rate": self.fs,
            }
        else:
            vals = self.signal
            n = len(vals)
            mean = sum(vals) / n
            std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
            rms = (sum(v ** 2 for v in vals) / n) ** 0.5
            return {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "rms": round(rms, 6),
                "duration_s": round(self.duration, 3),
                "n_samples": self.n_samples,
                "sampling_rate": self.fs,
            }

    def bandpass_filter(self, low: float, high: float, order: int = 4) -> Any:
        """Basit bant geçiren filtre (numpy FFT tabanlı).

        Args:
            low: Alt kesim frekansı (Hz).
            high: Üst kesim frekansı (Hz).
            order: Filtre derecesi (kullanılmıyor — FFT tabanlı).

        Returns:
            Filtrelenmiş sinyal.
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy gerekli: pip install numpy")

        freqs = np.fft.rfftfreq(self.n_samples, d=1.0 / self.fs)
        fft_vals = np.fft.rfft(self.signal)

        # Frekans maskesi
        mask = (freqs >= low) & (freqs <= high)
        fft_filtered = fft_vals * mask

        return np.fft.irfft(fft_filtered, n=self.n_samples)

    def power_spectrum(self) -> Tuple[Any, Any]:
        """Güç spektrumu (FFT tabanlı).

        Returns:
            (frekanslar, güç_değerleri) tuple.
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy gerekli: pip install numpy")

        freqs = np.fft.rfftfreq(self.n_samples, d=1.0 / self.fs)
        fft_vals = np.fft.rfft(self.signal)
        power = np.abs(fft_vals) ** 2 / self.n_samples

        return freqs, power

    def frequency_bands(self) -> Dict[str, Dict[str, float]]:
        """EEG frekans bantlarındaki güç dağılımı.

        Returns:
            Her bant için mutlak ve yüzde güç.
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy gerekli: pip install numpy")

        freqs, power = self.power_spectrum()
        total_power = float(np.sum(power))
        if total_power == 0:
            total_power = 1.0

        bands = {}
        for band_name, (low, high) in self.EEG_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            band_power = float(np.sum(power[mask]))
            bands[band_name] = {
                "range_hz": f"{low}-{high}",
                "absolute_power": round(band_power, 4),
                "relative_power_pct": round(band_power / total_power * 100, 2),
            }
        return bands

    def time_domain_features(self) -> Dict[str, float]:
        """Zaman alanı özellikleri (EMG/EEG feature extraction)."""
        if not HAS_NUMPY:
            raise RuntimeError("numpy gerekli: pip install numpy")

        arr = self.signal
        n = self.n_samples

        # MAV (Mean Absolute Value)
        mav = float(np.mean(np.abs(arr)))

        # Waveform Length
        wl = float(np.sum(np.abs(np.diff(arr))))

        # Zero Crossing Rate
        signs = np.sign(arr)
        zc = int(np.sum(np.abs(np.diff(signs)) > 0))
        zcr = zc / (n - 1) if n > 1 else 0

        # Slope Sign Changes
        diff_arr = np.diff(arr)
        ssc = int(np.sum(np.abs(np.diff(np.sign(diff_arr))) > 0)) if len(diff_arr) > 1 else 0

        # Hjorth Parameters
        var0 = float(np.var(arr))
        diff1 = np.diff(arr)
        var1 = float(np.var(diff1)) if len(diff1) > 0 else 0
        diff2 = np.diff(diff1)
        var2 = float(np.var(diff2)) if len(diff2) > 0 else 0

        activity = var0
        mobility = math.sqrt(var1 / var0) if var0 > 0 else 0
        complexity = (math.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0

        return {
            "mav": round(mav, 6),
            "waveform_length": round(wl, 6),
            "zero_crossing_rate": round(zcr, 6),
            "slope_sign_changes": ssc,
            "hjorth_activity": round(activity, 6),
            "hjorth_mobility": round(mobility, 6),
            "hjorth_complexity": round(complexity, 6),
        }

    def summary(self) -> Dict[str, Any]:
        """Kapsamlı sinyal özeti."""
        result: Dict[str, Any] = {
            "channel": self.channel_name,
            "basic_stats": self.basic_stats(),
        }
        if HAS_NUMPY:
            result["frequency_bands"] = self.frequency_bands()
            result["time_domain_features"] = self.time_domain_features()
        return result


# ═════════════════════════════════════════════
#  Module-level Convenience
# ═════════════════════════════════════════════

def get_available_tools() -> List[str]:
    """Mevcut biyomühendislik araçlarını listele."""
    return [
        "ProteinAnalyzer — Protein sekans analizi",
        "GenomicAnalyzer — DNA/RNA analizi",
        "MedicalImageHelper — Medikal görüntü ön-işleme",
        "DrugDiscoveryHelper — SMILES tabanlı moleküler analiz",
        "WastewaterAnalyzer — Su kalitesi analizi",
        "BioSignalProcessor — EEG/EMG sinyal işleme",
        "ProteinStructureHelper — PDB yapısal analizi",
    ]

# ═════════════════════════════════════════════
#  7. Protein Yapı (PDB) Entegrasyonu
# ═════════════════════════════════════════════

class ProteinStructureHelper:
    """Protein Data Bank (PDB) yapısal analiz ve indirme araçları.

    Kullanım:
        psh = ProteinStructureHelper("1CRN", workspace_dir="/path/to/workspace")
        path = psh.download_pdb()
        print(psh.parse_header())
    """

    def __init__(self, pdb_id: str, workspace_dir: str = "."):
        self.pdb_id = str(pdb_id).strip().upper()
        self.workspace_dir = Path(workspace_dir)
        self.pdb_path = self.workspace_dir / f"{self.pdb_id}.pdb"

    def download_pdb(self) -> str:
        """PDB dosyasını RCSB API'den indirip workspace klasörüne kaydeder."""
        if self.pdb_path.exists():
            log.info(f"{self.pdb_id} zaten mevcut: {self.pdb_path}")
            return str(self.pdb_path)

        url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
        try:
            import urllib.request
            import urllib.error
            log.info(f"{self.pdb_id} RCSB'den indiriliyor...")
            req = urllib.request.Request(url, headers={'User-Agent': 'BioMLAgent/1.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
            
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            with open(self.pdb_path, "w", encoding="utf-8") as f:
                f.write(content)
            log.info(f"✅ {self.pdb_id} başarıyla indirildi: {self.pdb_path}")
            return str(self.pdb_path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"PDB ID {self.pdb_id} bulunamadı.")
            raise RuntimeError(f"İndirme hatası: HTTP {e.code}")
        except Exception as e:
            raise RuntimeError(f"PDB indirme başarısız: {e}")

    def parse_header(self) -> Dict[str, Any]:
        """PDB başlığındaki temel bilgileri (Resolution, Date, Title) çıkarır."""
        if not self.pdb_path.exists():
            self.download_pdb()

        info = {
            "pdb_id": self.pdb_id,
            "title": "",
            "deposition_date": "",
            "resolution": None,
            "chains": set(),
            "method": ""
        }

        try:
            with open(self.pdb_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("HEADER"):
                        info["deposition_date"] = line[50:59].strip()
                    elif line.startswith("TITLE"):
                        info["title"] += line[10:].strip() + " "
                    elif line.startswith("EXPDTA"):
                        info["method"] = line[10:].strip()
                    elif line.startswith("REMARK   2 RESOLUTION."):
                        try:
                            # Örn: REMARK   2 RESOLUTION. 1.20 ANGSTROMS.
                            res_parts = line[22:].strip().split()
                            if res_parts and res_parts[0] != "NOT":
                                info["resolution"] = float(res_parts[0])
                        except ValueError:
                            pass
                    elif line.startswith("COMPND   3 CHAIN:"):
                        # Örn: COMPND   3 CHAIN: A, B;
                        chains = line[18:].strip().rstrip(";").split(",")
                        for c in chains:
                            info["chains"].add(c.strip())
                    elif line.startswith("ATOM"):
                        break
        except Exception as e:
            log.warning(f"PDB header parse hatası: {e}")

        info["title"] = info["title"].strip()
        info["chains"] = sorted(list(info["chains"]))
        return info

    def summary(self) -> Dict[str, Any]:
        """PDB dosyasının genel özeti."""
        return self.parse_header()
