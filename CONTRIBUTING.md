# Bio-ML Agent — Contribution Guide

Thank you for your interest in contributing to Bio-ML Agent!

## 🛠 Development Environment Setup

```bash
# 1. Fork & Clone
git clone https://github.com/<your-username>/bio-ml-agent.git
cd bio-ml-agent

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies (including test extras)
pip install -e ".[all,test]"
```

## 🧪 Running Tests

```bash
# Run all checks (lint + unit + integration):
./scripts/run_tests.sh

# Or manually:
pytest tests/
ruff check .
```

When adding a new feature, please add or update the relevant tests.

## 🔌 Creating a Plugin

Add a new `.py` file to the `plugins/` directory:

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
        return "My custom tool"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"Result: {payload}"
```

The plugin is auto-discovered and invoked via the `<MYTOOL>...</MYTOOL>` tag.

> ⚠️ **Security:** Plugins execute Python code inside the main process.
> Never load plugins from untrusted sources.

## 📐 Code Standards
- Use Python 3.10+ features.
- Type hints are required on all public functions and methods.
- Add docstrings to new functions.
- Follow `ruff` formatting rules (lint errors are blocking in CI).
- Keep the modular structure intact — all new code must live under `src/bio_ml_agent/`.

## 🚀 Pull Request Process

1. Create a new branch: `git checkout -b feature/my-new-feature`
2. Make your changes and write meaningful commit messages.
3. Ensure all tests pass: `pytest tests/ -x -q`
4. Push your branch: `git push origin feature/my-new-feature`
5. Open a Pull Request on GitHub.

## 🐳 Development with Docker

```bash
docker-compose up -d           # Start all services
docker-compose logs -f worker  # Tail worker logs
docker-compose down            # Stop all services
```

Have questions? Feel free to open an Issue!

---

## 🇹🇷 Türkçe Katkı Rehberi (Turkish)

Bio-ML Agent projesine katkıda bulunmak istediğiniz için teşekkürler!

**Geliştirme Ortamı:**
```bash
git clone https://github.com/<kullanici-adiniz>/bio-ml-agent.git
cd bio-ml-agent
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all,test]"
```

**Testleri Çalıştırma:**
```bash
./scripts/run_tests.sh   # Lint + Unit + Integration
pytest tests/            # Manuel test
```

**Kod Standartları:**
- Python 3.10+ özellikleri kullanın.
- Tüm public fonksiyonlara tip ipuçları (type hints) ekleyin.
- `ruff` standartlarına uyun (lint hataları CI'da engelleyicidir).
- Yeni kodlar `src/bio_ml_agent/` altında olmalıdır.

**Pull Request:**
1. Yeni dal: `git checkout -b feature/yeni-ozellik`
2. Değişikliklerinizi yapın.
3. Testleri çalıştırın: `pytest tests/ -x -q`
4. Dalı pushlayın ve PR açın.
