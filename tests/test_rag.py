import pytest
import os
from pathlib import Path
import sys

# agent.py'nin bulunduğu dizini sys.path'e ekleyelim ki modülü içeri aktarabilelim
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine import RAGEngine

@pytest.fixture
def temp_workspace(tmp_path):
    """Testler için geçici bir workspace ve içinde bazı dosyalar oluşturur."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    
    # 1. proje dosyası
    proj1 = ws / "proj1"
    proj1.mkdir()
    (proj1 / "report.md").write_text("Diyabet tahmini: RandomForest kullanılarak %85 doğruluk elde edildi.", encoding="utf-8")
    
    # Python dosyası
    (proj1 / "train.py").write_text("from sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()", encoding="utf-8")
    
    # 2. proje dosyası
    proj2 = ws / "proj2"
    proj2.mkdir()
    (proj2 / "notes.txt").write_text("Kanser veri seti üzerinde PCA uyguladık, ardından SVM çalıştırdık.", encoding="utf-8")
    
    # RAG Engine oluştur
    engine = RAGEngine(workspace_dir=ws, db_dir_name=".test_rag_db")
    return ws, engine


def test_chunk_text():
    """RAGEngine._chunk_text mantığını sına."""
    engine = RAGEngine(workspace_dir="/tmp/dummy")
    
    text = "Bu çok kısa bir metin."
    chunks = engine._chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text
    
    long_text = "A " * 100  # 200 karakter
    chunks2 = engine._chunk_text(long_text, chunk_size=50, overlap=10)
    assert len(chunks2) > 1
    assert "A" in chunks2[0]


def test_index_and_search(temp_workspace):
    """Dosyaların indekslenip, arama işleminin çalışmasını sına."""
    ws, engine = temp_workspace
    
    # İndeksleme
    indexed_files = engine.index_workspace()
    assert indexed_files == 3  # report.md, train.py, notes.txt
    
    # Arama - Kanser Projesi
    results_cancer = engine.search("pca ve svm kullanılan veri seti hangisiydi?")
    assert len(results_cancer) > 0
    assert "Kanser" in results_cancer[0]["document"]
    assert "proj2" in results_cancer[0]["source"]
    
    # Arama - Diyabet Projesi
    results_diabetes = engine.search("diyabet için hangi modeli kullandık?")
    assert len(results_diabetes) > 0
    assert "RandomForest" in results_diabetes[0]["document"]
    assert "proj1" in results_diabetes[0]["source"]
    
    # Boş arama
    empty_results = engine.search("   ")
    assert len(empty_results) == 0
