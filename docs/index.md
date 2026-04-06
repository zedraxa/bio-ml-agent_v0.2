# Bio-ML Agent — Documentation Portal

Bio-ML Agent is a modular, secure AI assistant that automates biology and machine learning research workflows.

## 🌟 Core Capabilities
- **🧬 Bioinformatics Expertise**: Protein analysis, genomic data processing, microscopy interpretation, and academic reporting.
- **🎯 Mission-Pack Orchestration**: High-level research blueprints (`repo_review_pack`, `lab_report_pack`, `microscopy_pack`) executed by coordinated specialist agents.
- **🧪 Explainable AI (XAI)**: Scientific justification of model decisions using SHAP/LIME.
- **♻️ Auto-Recovery**: Interrupted missions resume automatically via `CheckpointStore` + `RecoveryManager`.
- **🎭 MockBackend**: Simulation support for development and testing without consuming API keys.

## 🚀 Quick Links
- **[Architecture](ARCHITECTURE.md)**: Technical layers and modules of the system.
- **[User Guide](user_guide.md)**: How to install and use the application.
- **[API Reference](api.md)**: REST API endpoints.
- **[Project Status](STATUS.md)**: Current version and branch structure.
- **[Deployment](deployment.md)**: How to deploy in different environments.

---

## ⚡ Quick Start

### 1. Demo Scenario (Recommended)
Follow the guide in `examples/golden_path_demo/` to explore the system quickly.

### 2. Installation
```bash
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git
cd bio-ml-agent_v0.2
pip install -e ".[all]"
python run_ui.py          # Web UI → http://localhost:7860
python run_api.py         # REST API → http://localhost:8001
```

CLI alternative:
```bash
bio-ml-agent ui           # Gradio Web UI
bio-ml-agent api          # FastAPI REST server
bio-ml-agent chat         # Terminal interactive chat
bio-ml-agent check        # Verify installation
```

---

*Bio-ML Agent — v0.1.0-clean (API v6.0.0)*

---

## 🇹🇷 Türkçe (Turkish)

Bio-ML Agent, biyoloji ve makine öğrenimi araştırmalarını otonomlaştıran Mission-Pack tabanlı bir AI platformudur.

**Hızlı Başlangıç:**
```bash
pip install -e ".[all]"
python run_api.py   # API → http://localhost:8001
python run_ui.py    # Web UI → http://localhost:7860
bio-ml-agent chat   # Terminal sohbet
```

Hızlı bağlantılar: [Mimari](ARCHITECTURE.md) | [Kullanım Kılavuzu](user_guide.md) | [Durum Raporu](STATUS.md) | [API Referansı](api.md)
