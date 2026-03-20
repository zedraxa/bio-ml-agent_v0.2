# Bio-ML Agent — Contribution Guide

Bio-ML Agent projesine katkıda bulunmak istediğiniz için teşekkürler!

## 🛠 Geliştirme Ortamı Kurulumu

```bash
# 1. Fork & Clone
git clone https://github.com/<your-username>/bio-ml-agent.git
cd bio-ml-agent

# 2. Sanal ortam
python3 -m venv venv
source venv/bin/activate

# 3. Tüm bağımlılıkları kur (test dahil)
pip install -e ".[all,test]"
```

## 🧪 Testleri Çalıştırma

```bash
# Tüm testleri (Lint + Unit + Integration) çalıştır:
./scripts/run_tests.sh

# Veya manuel:
pytest tests/
ruff check .
```

Yeni özellik eklediğinizde ilgili testleri de ekleyin veya güncelleyin.

## 🔌 Plugin Oluşturma

`plugins/` klasörüne yeni bir `.py` dosyası ekleyin:

```python
# plugins/my_tool.py
from plugin_manager import ToolPlugin
from pathlib import Path

class MyCustomTool(ToolPlugin):
    @property
    def name(self):
        return "MYTOOL"

    @property
    def description(self):
        return "Benim özel aracım"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"Sonuç: {payload}"
```

Plugin otomatik keşfedilir ve `<MYTOOL>...</MYTOOL>` tag'i ile kullanılır.

> ⚠️ **Güvenlik:** Plugin'ler ana process içinde Python kodu çalıştırır.
> Güvenilmeyen kaynaklardan plugin yüklemeyin.

- Python 3.10+ özelliklerini kullanın
- Tip ipuçları (type hints) zorunludur
- Yeni fonksiyonlar için docstring ekleyin
- `ruff` standartlarına uyun (Lint hataları CI'da engelleyicidir)
- Modüler yapıyı bozmayın (Kodlar `src/bio_ml_agent/` altında olmalı)

## 🚀 Pull Request Süreci

1. Yeni dal açın: `git checkout -b feature/yeni-ozellik`
2. Değişikliklerinizi yapın ve anlamlı commit mesajları yazın
3. Testlerin geçtiğinden emin olun: `pytest tests/ -x -q`
4. Dalınızı pushlayın: `git push origin feature/yeni-ozellik`
5. Pull Request açın

## 🐳 Docker ile Geliştirme

```bash
docker-compose up -d          # Tüm servisleri başlat
docker-compose logs -f worker # Worker loglarını izle
docker-compose down            # Servisleri durdur
```

Sorularınız için Issue açmaktan çekinmeyin!
