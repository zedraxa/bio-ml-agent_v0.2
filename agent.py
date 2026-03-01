import argparse
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import ollama

from llm_backend import (
    LLMBackend, OllamaBackend, auto_create_backend, detect_backend_name,
)

from utils.config import load_config, get_config, AppConfig
from exceptions import (
    ToolExecutionError,
    FileOperationError,
    SecurityViolationError,
    AgentError,
    ToolTimeoutError,
    ValidationError,
    LLMConnectionError,
)
from progress import Spinner
from llm_backend import LLMBackend, OllamaBackend, auto_create_backend
from plugin_manager import PluginManager
from dataset_catalog import format_catalog_for_prompt
from rag_engine import RAGEngine

# ── Yapılandırma üzerinden okunan sabitler ──
# Bu değerler config.yaml / env / CLI'dan yüklenir.
# İlk erişimde varsayılanlar kullanılır, main() içinde güncellenir.
_app_cfg = None  # load_config() sonrası set edilir
_swarm_orchestrator = None  # Swarm modunda lazy-init edilir

def _cfg() -> "AppConfig":
    """Mevcut config'i döndürür (lazy init)."""
    global _app_cfg
    if _app_cfg is None:
        _app_cfg = get_config()
    return _app_cfg

# Geriye dönük uyumluluk sabitleri (testler ve diğer modüller için)
DEFAULT_PROJECT = "scratch_project"
HISTORY_DIR_NAME = "conversation_history"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "agent.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3


# ─────────────────────────────────────────────
#  Loglama Sistemi
# ─────────────────────────────────────────────

def setup_logger(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Dosya + konsol loglaması yapan logger kur.

    Log dosyası: <log_dir>/agent.log (RotatingFileHandler — 5MB × 3 yedek)
    Konsol: sadece WARNING ve üstü (terminali kirletmemek için)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    logger = logging.getLogger("bio_ml_agent")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Eğer handler zaten eklenmişse tekrar ekleme
    if logger.handlers:
        return logger

    # ── Dosya handler (her şeyi logla) ──
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # ── Konsol handler (sadece WARNING+) ──
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("═" * 60)
    logger.info("Logger başlatıldı | seviye=%s | dosya=%s", log_level, log_file)
    logger.info("═" * 60)

    return logger


# Global logger — main() içinde setup_logger() ile yapılandırılacak
log = logging.getLogger("bio_ml_agent")

SYSTEM_PROMPT = """
You are a local Bioengineering ML Project Agent running on Linux.

GOAL
Turn the user's request into a reproducible ML project.

HARD RULES
- Use ONLY the tool protocol when performing actions (create files, run commands, web search/open).
- WRITE_FILE paths must be PROJECT-RELATIVE (e.g. src/train.py, data/raw/file.csv).
  DO NOT include "workspace/" or the project name in the path.
  The system automatically places files under workspace/<project>/.
  Example:
    CORRECT:   path: src/train.py
    CORRECT:   path: data/raw/diabetes.csv
    WRONG:     path: workspace/myproject/src/train.py
    WRONG:     path: myproject/src/train.py
- WRITE_FILE MUST use:
  path: relative/path.ext
  ---
  file content...
- BASH commands run from workspace/<project>/ directory.
  So use relative paths: python src/train.py (NOT python workspace/.../train.py)
- WEB_SEARCH is disabled unless user message includes: ALLOW_WEB_SEARCH

WORKFLOW
1) Clarify I/O + metrics (brief).
2) Find 2-5 candidate datasets (name+link+license/terms).
3) Pick dataset; download into data/raw/.
4) Create project structure and requirements.txt.
5) Implement baseline model in src/train.py.
6) **MULTI-MODEL COMPARISON** — Compare AT LEAST 3 models:
   - LogisticRegression, RandomForest, GradientBoosting, SVM, KNN
   - (For regression: LinearRegression, Ridge, RandomForest, GradientBoosting, SVR, KNN)
   - Use StandardScaler + Pipeline for each model.
   - Compute: accuracy, precision, recall, f1, roc_auc (classification)
              or r2, mae, rmse (regression).
   - Run 5-fold cross-validation for each model.
   - Print a comparison table and identify the BEST model.
   - You can use: `from utils.model_compare import compare_models`
     Example:
       comparator, results = compare_models(
           X_train, X_test, y_train, y_test,
           task_type="classification",
           output_dir="results/"
       )
   6.5) **HYPERPARAMETER OPTIMIZATION** (optional, if user requests or dataset is large):
   - Use: `from utils.hyperparameter_optimizer import optimize_model`
     Example:
       best_model, best_params, results = optimize_model(
           X_train, y_train,
           model_name="RandomForest",
           task_type="classification",
           method="random", n_iter=20
       )
7) Save results/comparison_results.json, results/comparison_report.md,
   and results/best_model.pkl (model is automatically saved by compare_models).
   The saved model can be loaded later:
     from utils.model_loader import load_and_predict
     predictions = load_and_predict("results/best_model.pkl", X_new)
8) **VISUALIZATION** — Generate plots and save as PNG to results/plots/:
   - Confusion Matrix (normal + normalized)
   - ROC Curve (binary or multi-class OvR)
   - Feature Importance (Gini or |coef|)
   - Correlation Matrix (heatmap)
   - Learning Curve (train vs validation)
   - Class Distribution (bar + donut)
   - You can use: `from utils.visualize import MLVisualizer`
     Example:
       viz = MLVisualizer(output_dir="results/plots")
       viz.plot_all(best_model, X_train, X_test, y_train, y_test,
                    feature_names=feature_cols, df=df)
   8.7) **EXPLAINABLE AI (XAI)** (If user wants model transparency/SHAP/LIME):
   - Use: `from xai_engine import XAIEngine`
     Example:
       xai = XAIEngine(best_model, X_train, feature_names=feature_cols, task_type="classification")
       xai.generate_shap_summary(X_test, output_dir="results/plots", max_display=10)
       xai.explain_instance_lime(X_test.iloc[0], output_dir="results/plots")
   8.8) **DATA PREPROCESSING** (before training, if data quality is low):
   - Use: `from utils.preprocessor import DataPreprocessor, quick_preprocess, analyze_data_quality`
   - Quick quality check:
       report = analyze_data_quality(X, feature_names=feature_cols)
       print(report)
   - Full pipeline:
       pp = DataPreprocessor(
           impute_strategy="median",
           scale_method="standard",
           detect_outliers="iqr",
           remove_outliers=True,
           pca_components=10,  # optional dimensionality reduction
       )
       X_train_clean, y_train_clean = pp.fit_transform(X_train, y_train)
       X_test_clean = pp.transform(X_test)
       print(pp.summary_text())
   - Quick one-liner: X_clean = quick_preprocess(X, scale=True, pca=5)
9) Write report.md (include comparison table + plot references + model usage instructions) and README.md.

10) **DEEP LEARNING** — For medical image classification tasks (MRI, X-Ray, etc.):
    - Use: `from deep_learning import MedicalCNN, quick_train_cnn, compare_architectures`
    - Quick start (single line):
        from deep_learning import quick_train_cnn
        metrics = quick_train_cnn("data/raw/", preset="brain_mri", 
                                   architecture="resnet18", epochs=25)
    - Available presets: chest_xray, brain_mri, skin_lesion, retinal_oct
    - Available architectures: resnet18, resnet50, efficientnet_b0, densenet121, mobilenet_v2
    - Compare multiple architectures:
        from deep_learning import compare_architectures
        results = compare_architectures("data/raw/", preset="brain_mri",
                                         architectures=["resnet18", "densenet121", "efficientnet_b0"])
    - Data directory should have class subdirectories (e.g. data/raw/glioma/, data/raw/no_tumor/).
    - The system uses PyTorch Transfer Learning (pretrained ImageNet weights).

11) **AutoML** — For automatic Neural Architecture Search:
    - Use: `from deep_learning import AutoMLSearch`
        searcher = AutoMLSearch(preset="brain_mri", max_trials=10)
        searcher.search("data/raw/", epochs_per_trial=10)
        searcher.export_summary("results/")
    - Requires: pip install autokeras tensorflow (optional, heavy dependency).

12) **RAG KNOWLEDGE SEARCH**:
    - Use the <RAG_SEARCH> tool to search past projects and reports in the workspace.
    - Example usage:
      <RAG_SEARCH>
      diabetes model comparison report
      </RAG_SEARCH>

11) **MULTI-AGENT COLLABORATION** (For complex tasks):
    - You are the Orchestrator. You can delegate specialized work to Sub-Agents.
    - Sub-Agents run in the same workspace but with focused prompts.
    - Use `from multi_agent import ask_data_engineer, ask_ml_engineer, ask_report_writer`
    - Example usage in a <PYTHON> block:
      ```python
      from multi_agent import ask_data_engineer, ask_ml_engineer
      
      # 1. Ask Data Engineer to clean data
      de_result = ask_data_engineer("Load data/raw/data.csv, handle NaN values, and save to data/processed/clean.csv")
      print("Data Engineer:", de_result)
      
      # 2. Ask ML Engineer to train models
      ml_result = ask_ml_engineer("Train RandomForest and SVM on data/processed/clean.csv. Save models to results/.")
      print("ML Engineer:", ml_result)
      ```
    - NOTE: Do not overuse sub-agents for simple tasks. Use them when tasks are logically separated.
11) **BIOENGINEERING TOOLKIT** (Specialized Biological / Medical Analysis):
    - You have a comprehensive suite of bio-focused analyzers. Import them from `bioeng_toolkit`:
      ```python
      from bioeng_toolkit import (
          ProteinAnalyzer, GenomicAnalyzer, WastewaterAnalyzer, 
          DrugDiscoveryHelper, MedicalImageHelper, BioSignalProcessor
      )
      ```
    - Use `ProteinAnalyzer("SEQ")` for amino acid stats, pI,, and GRAVY.
    - Use `GenomicAnalyzer("SEQ")` for DNA/RNA translation, ORFs, GC content.
    - Use `WastewaterAnalyzer({"pH": 7.2, "bod": 4.5, ...})` for water quality indexes and treatment rules.
    - Use `DrugDiscoveryHelper("SMILES_STRING")` for Lipinski's Rule of Five checks.
    - Use `BioSignalProcessor(np.random.randn(1000))` for EEG/EMG fast Fourier transforms and feature extractions.
    - Always output the `.summary()` or requested metrics from these classes into your text response.

Output language: Turkish (unless user asks otherwise).

TOOL PROTOCOL (ONE BLOCK ONLY):
<PYTHON>...</PYTHON>
<BASH>...</BASH>
<WEB_SEARCH>...</WEB_SEARCH>
<WEB_OPEN>...</WEB_OPEN>
<READ_FILE>...</READ_FILE>
<WRITE_FILE>...</WRITE_FILE>
<TODO>...</TODO>
"""

TOOL_TAGS = ["PYTHON", "BASH", "WEB_SEARCH", "WEB_OPEN", "READ_FILE", "WRITE_FILE", "TODO"]
TOOL_RE = re.compile(
    r"<(" + "|".join(TOOL_TAGS) + r")>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Accept fenced code in ANY case, with optional language labels
FENCED_BASH_RE = re.compile(r"```(?:bash)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
FENCED_PY_RE = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Güvenlik desenleri — config.yaml'dan yüklenir, yoksa varsayılanlar kullanılır
DENY_PATTERNS = [
    r"\brm\b.*-rf\s+/",
    r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
    r"\bdd\b\s+if=/dev/zero\b",
    r"\bmkfs\.",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\b\s+-9\s+1\b",
]

def _get_deny_patterns() -> list:
    """Config'den veya varsayılan DENY_PATTERNS'ı döndürür."""
    try:
        return _cfg().security.deny_patterns
    except Exception:
        return DENY_PATTERNS


def is_dangerous_bash(cmd: str) -> Optional[str]:
    patterns = _get_deny_patterns()
    for pat in patterns:
        if re.search(pat, cmd.strip()):
            log.warning("🚫 GÜVENLİK: Tehlikeli komut engellendi | pattern=%s | cmd=%s", pat, cmd.strip()[:100])
            return f"Blocked by denylist pattern: {pat}"
    return None


def safe_relpath(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        log.warning("🚫 GÜVENLİK: Absolute path engellendi | path=%s", path)
        raise SecurityViolationError(
            f"Absolute path kullanılamaz: {path}",
            violation_type="absolute_path",
            suggestion="Workspace içinde relative path kullanın (ör: data/raw/file.csv).",
        )
    norm = Path(os.path.normpath(str(p)))
    if str(norm).startswith(".."):
        log.warning("🚫 GÜVENLİK: Path traversal engellendi | path=%s", path)
        raise SecurityViolationError(
            f"Path traversal engellendi: {path}",
            violation_type="path_traversal",
            suggestion="Üst dizinlere erişim yasaktır. Workspace içindeki dosyaları kullanın.",
        )
    return str(norm)


def current_project() -> str:
    try:
        return os.getenv("AGENT_PROJECT", _cfg().workspace.default_project)
    except Exception:
        return os.getenv("AGENT_PROJECT", DEFAULT_PROJECT)


def run_python(code: str, workspace: Path, timeout_s: int = 180) -> str:
    log.info("🐍 PYTHON çalıştırılıyor | timeout=%ds | kod_uzunluk=%d karakter", timeout_s, len(code))
    log.debug("🐍 PYTHON kod:\n%s", code[:500])
    code = textwrap.dedent(code).strip() + "\n"
    
    # Kök dizini PYTHONPATH'e ekle
    root_dir = Path(__file__).resolve().parent
    sys_path_injection = f"import sys\nsys.path.insert(0, r'{root_dir}')\n"
    code = sys_path_injection + code
    
    tmp = workspace / "_tmp_run.py"
    tmp.write_text(code, encoding="utf-8")
    try:
        start_time = time.time()
        res = subprocess.run(
            [sys.executable, str(tmp)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - start_time
        out = (res.stdout or "") + (res.stderr or "")
        result = out.strip() if out.strip() else f"[python exit code: {res.returncode}] (no output)"
        log.info("🐍 PYTHON tamamlandı | süre=%.2fs | exit_code=%d | çıktı_uzunluk=%d", elapsed, res.returncode, len(result))
        if res.returncode != 0:
            log.warning("🐍 PYTHON hata ile bitti | exit_code=%d | stderr=%s", res.returncode, (res.stderr or "")[:300])
        return result
    except subprocess.TimeoutExpired:
        log.error("🐍 PYTHON TIMEOUT | %ds aşıldı", timeout_s)
        raise ToolTimeoutError("PYTHON", timeout_s)
    except ToolTimeoutError:
        raise
    except Exception as e:
        log.error("🐍 PYTHON beklenmeyen hata | %s", e, exc_info=True)
        raise ToolExecutionError("PYTHON", str(e), details=f"Kod uzunluğu: {len(code)} karakter")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def run_bash(cmd: str, workspace: Path, timeout_s: int = 180) -> str:
    log.info("💻 BASH çalıştırılıyor | cmd=%s | timeout=%ds", cmd.strip()[:120], timeout_s)
    reason = is_dangerous_bash(cmd)
    if reason:
        log.warning("💻 BASH ENGELLENDİ | sebep=%s | cmd=%s", reason, cmd.strip()[:100])
        raise SecurityViolationError(
            f"Tehlikeli komut engellendi: {cmd.strip()[:80]}",
            violation_type="dangerous_command",
            details=reason,
            suggestion="Bu komut güvenlik politikası tarafından engellendi.",
        )
    try:
        start_time = time.time()
        res = subprocess.run(
            cmd,
            cwd=str(workspace),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            executable="/bin/bash",
        )
        elapsed = time.time() - start_time
        out = (res.stdout or "") + (res.stderr or "")
        result = out.strip() if out.strip() else f"[bash exit code: {res.returncode}] (no output)"
        log.info("💻 BASH tamamlandı | süre=%.2fs | exit_code=%d | çıktı_uzunluk=%d", elapsed, res.returncode, len(result))
        if res.returncode != 0:
            log.warning("💻 BASH hata ile bitti | exit_code=%d | cmd=%s", res.returncode, cmd.strip()[:100])
        return result
    except subprocess.TimeoutExpired:
        log.error("💻 BASH TIMEOUT | %ds aşıldı | cmd=%s", timeout_s, cmd.strip()[:100])
        raise ToolTimeoutError("BASH", timeout_s)
    except (ToolTimeoutError, SecurityViolationError):
        raise
    except Exception as e:
        log.error("💻 BASH beklenmeyen hata | %s", e, exc_info=True)
        raise ToolExecutionError("BASH", str(e), details=f"Komut: {cmd.strip()[:100]}")


def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        log.warning("🌐 WEB_SEARCH: Boş sorgu gönderildi")
        raise ValidationError("query", "Web araması için sorgu boş olamaz.")
    log.info("🌐 WEB_SEARCH başlatıldı | sorgu=%s", query[:100])
    try:
        from ddgs import DDGS
        start_time = time.time()
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append(
                    {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
                )
        elapsed = time.time() - start_time
        log.info("🌐 WEB_SEARCH tamamlandı | süre=%.2fs | sonuç_sayısı=%d", elapsed, len(results))
        return json.dumps(results, ensure_ascii=False, indent=2)
    except (AgentError,):
        raise
    except Exception as e:
        log.error("🌐 WEB_SEARCH HATA | sorgu=%s | hata=%s", query[:80], e, exc_info=True)
        raise ToolExecutionError(
            "WEB_SEARCH", str(e),
            suggestion="ddgs paketini kurun: python -m pip install -U ddgs",
        )


def web_open(url: str) -> str:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        log.warning("📖 WEB_OPEN: Geçersiz URL | url=%s", url[:100])
        raise ValidationError(
            "url", f"Geçersiz URL: {url[:80]}",
            suggestion="URL http:// veya https:// ile başlamalıdır.",
        )
    log.info("📖 WEB_OPEN başlatıldı | url=%s", url[:150])
    try:
        import requests
        from bs4 import BeautifulSoup

        start_time = time.time()
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        elapsed = time.time() - start_time
        truncated = len(text) > 12000
        log.info("📖 WEB_OPEN tamamlandı | süre=%.2fs | status=%d | metin_uzunluk=%d | kırpıldı=%s",
                 elapsed, r.status_code, len(text), truncated)
        return (text[:12000] + "\n\n[TRUNCATED]") if truncated else text
    except (AgentError,):
        raise
    except Exception as e:
        log.error("📖 WEB_OPEN HATA | url=%s | hata=%s", url[:100], e, exc_info=True)
        raise ToolExecutionError(
            "WEB_OPEN", str(e),
            details=f"URL: {url[:100]}",
            suggestion="URL'nin erişilebilir olduğundan emin olun.",
        )


def read_file(payload: str, workspace: Path) -> str:
    rel = safe_relpath(payload.strip())
    p = workspace / rel
    if not p.exists():
        log.warning("📄 READ_FILE: Dosya bulunamadı | path=%s", rel)
        raise FileOperationError(
            "okuma", rel, "Dosya bulunamadı.",
            suggestion=f"Dosyanın var olduğundan emin olun: {rel}",
        )
    if p.is_dir():
        log.warning("📄 READ_FILE: Klasör verildi | path=%s", rel)
        raise FileOperationError(
            "okuma", rel, "Verilen yol bir klasör, dosya değil.",
            suggestion="Dosya yolunu belirtin, klasör yolunu değil.",
        )
    data = p.read_text(encoding="utf-8", errors="replace")
    log.info("📄 READ_FILE | path=%s | boyut=%d bytes", rel, len(data))
    return (data[:20000] + "\n\n[TRUNCATED]") if len(data) > 20000 else data


def sanitize_content(content: str) -> str:
    content_clean = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*$", "", content, flags=re.MULTILINE)
    content_clean = re.sub(r"^\s*```\s*$", "", content_clean, flags=re.MULTILINE)
    return content_clean.lstrip("\n")


def _strip_redundant_prefixes(rel: str, proj: str) -> str:
    """LLM'in yanlışlıkla eklediği workspace/, proje adı ve benzeri prefix'leri agresif olarak temizle.

    Örnek dönüşümler:
      workspace/myproj/src/train.py                           →  src/train.py
      scratch_project/workspace/diabetes/src/train.py         →  src/train.py
      workspace/workspace/myproj/data/raw/file.csv            →  data/raw/file.csv
      myproj/workspace/myproj/results/plots/fig.png           →  results/plots/fig.png
      src/train.py                                            →  src/train.py  (değişmez)
      report.md                                               →  report.md     (değişmez)
    """
    _KNOWN_ROOTS = {"src", "data", "results", "docs", "models", "notebooks", "tests", "config", proj}
    _KNOWN_FILES = {"report.md", "README.md", "readme.md", "pyproject.toml", "setup.py",
                    "todo.md", "report.txt", ".gitignore", "Makefile"}

    parts = list(Path(rel).parts)
    original = rel

    # Döngüsel olarak öndeki workspace ve proje_adi takılarını temizle
    while parts and parts[0] in ("workspace", proj, "scratch_project", "Kanser_Hücresi_Analiz"):
        parts = parts[1:]

    # Kalan kısmın içinde bilinen bir kök varsa, ondan öncesini de kes (Örn: breast_cancer_classification/src/ -> src/)
    for i, part in enumerate(parts):
        if part in _KNOWN_ROOTS or part in _KNOWN_FILES:
            parts = parts[i:]
            break
            
    # Eğer sonuç boş kalırsa ve orijinali salt bir dosyaysa (örn. train.py) onu geri ver
    cleaned = str(Path(*parts)) if parts else rel
    
    if cleaned != original:
        log.info("✍️ WRITE_FILE yol düzeltildi: %s → %s", original, cleaned)
        
    return cleaned


def write_file(payload: str, workspace: Path) -> str:
    raw = payload.strip()
    if "---" not in raw:
        log.warning("✍️ WRITE_FILE: Format hatası — '---' ayırıcı bulunamadı")
        raise ValidationError(
            "WRITE_FILE format",
            "'---' ayırıcı bulunamadı.",
            suggestion="Doğru format: path: dosya.py\n---\niçerik...",
        )
    head, content = raw.split("---", 1)
    m = re.search(r"^\s*path:\s*(.+)\s*$", head.strip(), re.MULTILINE)
    if not m:
        log.warning("✍️ WRITE_FILE: Format hatası — 'path:' satırı bulunamadı")
        raise ValidationError(
            "WRITE_FILE format",
            "'path: ...' satırı eksik.",
            suggestion="Blok başında 'path: dosya_adı.py' satırı olmalı.",
        )

    rel = safe_relpath(m.group(1).strip())
    proj = current_project()
    # LLM'in yanlışlıkla eklediği workspace/ veya proje adı prefix'lerini temizle
    rel = _strip_redundant_prefixes(rel, proj)
    if not rel.startswith(proj + "/") and rel != proj:
        rel = f"{proj}/{rel}"

    p = workspace / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sanitize_content(content), encoding="utf-8")
    log.info("✍️ WRITE_FILE | path=%s | boyut=%d bytes", rel, p.stat().st_size)
    return f"[OK] Wrote {rel} ({p.stat().st_size} bytes)"


def append_todo(payload: str, workspace: Path) -> str:
    todo = workspace / f"{current_project()}/todo.md"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = payload.strip()
    if not entry:
        log.warning("📝 TODO: Boş içerik gönderildi")
        raise ValidationError("TODO", "TODO bloğu boş olamaz.")
    todo.parent.mkdir(parents=True, exist_ok=True)
    with todo.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {ts}\n{entry}\n")
    log.info("📝 TODO eklendi | dosya=%s | uzunluk=%d", todo.relative_to(workspace), len(entry))
    return f"[OK] Appended to {todo.relative_to(workspace)}"


from pydantic import BaseModel, Field
from typing import Optional

class AgentConfig(BaseModel):
    model: str = Field(default="qwen2.5:7b-instruct")
    workspace: Path = Field(default=Path("workspace"))
    timeout: int = Field(default=180)
    max_steps: int = Field(default=50)
    history_dir: Path = Field(default=Path("conversation_history"))
    load_session: Optional[str] = None
    log_level: str = "INFO"
    log_dir: Path = Field(default=Path("logs"))
    config_file: str = "config.yaml"
    backend_mode: str = "auto"
    swarm: bool = Field(default=False)

# ─────────────────────────────────────────────
#  Konuşma Geçmişi Yönetimi
# ─────────────────────────────────────────────

def _ensure_history_dir(history_dir: Path) -> None:
    """Geçmiş klasörünü oluştur."""
    history_dir.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    """Benzersiz oturum kimliği üret."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def save_conversation(history_dir: Path, session_id: str, messages: List[Dict[str, str]],
                      metadata: Optional[Dict] = None) -> Path:
    """Konuşma geçmişini JSON dosyasına kaydet."""
    _ensure_history_dir(history_dir)
    filepath = history_dir / f"{session_id}.json"

    # İlk kullanıcı mesajından özet çıkar
    first_user_msg = ""
    for msg in messages:
        if msg["role"] == "user":
            first_user_msg = msg["content"][:120].replace("\n", " ")
            break

    data = {
        "session_id": session_id,
        "created_at": metadata.get("created_at", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "summary": first_user_msg,
        "message_count": len(messages),
        "messages": messages,
    }
    if metadata:
        data["metadata"] = metadata

    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.debug("💾 Oturum kaydedildi | session=%s | mesaj_sayısı=%d", session_id, len(messages))
    return filepath


def load_conversation(history_dir: Path, session_id: str) -> Tuple[List[Dict[str, str]], Dict]:
    """Konuşma geçmişini dosyadan yükle. (messages, metadata) döndürür."""
    filepath = history_dir / f"{session_id}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Oturum bulunamadı: {session_id}")

    data = json.loads(filepath.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    metadata = {
        "created_at": data.get("created_at", ""),
        "session_id": data.get("session_id", session_id),
    }
    return messages, metadata


def list_conversations(history_dir: Path, limit: int = 20) -> List[Dict]:
    """Kayıtlı konuşma oturumlarını listele (en yeniden en eskiye)."""
    _ensure_history_dir(history_dir)
    sessions = []
    for f in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", f.stem),
                "created_at": data.get("created_at", "?"),
                "updated_at": data.get("updated_at", "?"),
                "summary": data.get("summary", "")[:80],
                "message_count": data.get("message_count", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
        if len(sessions) >= limit:
            break
    return sessions


def delete_conversation(history_dir: Path, session_id: str) -> bool:
    """Bir konuşma oturumunu sil."""
    filepath = history_dir / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def print_history_help():
    """Geçmiş yönetimi komutlarının yardımını göster."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║               📜 Konuşma Geçmişi Komutları                  ║
╠══════════════════════════════════════════════════════════════╣
║  /history           → Kayıtlı oturumları listele            ║
║  /load <session_id> → Eski bir oturumu yükle                ║
║  /delete <session_id> → Bir oturumu sil                     ║
║  /new               → Yeni oturum başlat (mevcut kaydedilir)║
║  /save              → Mevcut oturumu şimdi kaydet           ║
║  /info              → Mevcut oturum bilgilerini göster       ║
╚══════════════════════════════════════════════════════════════╝
""")


# Global LLM backend — main() içinde oluşturulur, varsayılan Ollama
_llm_backend: Optional[LLMBackend] = None

# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_llm_backend(model: str = "") -> LLMBackend:
    """Aktif LLM backend'i döndür (yoksa Ollama oluştur)."""
    global _llm_backend
    if _llm_backend is None:
        _llm_backend = OllamaBackend(model=model or "qwen2.5:latest")
    return _llm_backend


def set_llm_backend(backend: LLMBackend) -> None:
    """LLM backend'i değiştir."""
    global _llm_backend
    _llm_backend = backend
    log.info("🧠 LLM backend değiştirildi: %s", backend)


def get_plugin_manager() -> PluginManager:
    """Plugin manager'ı döndür (yoksa oluştur)."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def llm_chat(model: str, messages: List[Dict[str, str]]) -> str:
    log.info("🧠 LLM isteği gönderiliyor | model=%s | mesaj_sayısı=%d", model, len(messages))
    start_time = time.time()
    try:
        backend = get_llm_backend(model)
        content = backend.chat(messages).strip()
        elapsed = time.time() - start_time
        log.info("🧠 LLM yanıt alındı | süre=%.2fs | yanıt_uzunluk=%d karakter", elapsed, len(content))
        log.debug("🧠 LLM yanıt (ilk 300 karakter): %s", content[:300])
        return content
    except LLMConnectionError:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        log.error("🧠 LLM HATA | süre=%.2fs | model=%s | hata=%s", elapsed, model, e, exc_info=True)
        raise LLMConnectionError(model, str(e))


def extract_tools(text: str) -> Tuple[List[Tuple[str, str]], str]:
    tools = []
    for m in TOOL_RE.finditer(text or ""):
        tools.append((m.group(1).upper(), m.group(2)))
    outside = TOOL_RE.sub("", text or "").strip()
    return tools, outside

def extract_tool(text: str) -> Tuple[Optional[str], Optional[str], str]:
    tools, outside = extract_tools(text)
    if not tools:
        return None, None, outside
    return tools[0][0], tools[0][1], outside


def normalize_user_message(s: str) -> str:
    s = s.replace("\r\n", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def autosave_web_outputs(cfg: AgentConfig, tool: str, out: str) -> None:
    proj = current_project()
    log_dir = cfg.workspace / proj / "datasets"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{tool.lower()}_{stamp}.json" if tool == "WEB_SEARCH" else f"{tool.lower()}_{stamp}.txt"
    (log_dir / fname).write_text(out, encoding="utf-8")

# Helper functions for V5 monolithic loop (assuming these are new)
def _run_tool(tool: str, payload: str, cfg: AgentConfig) -> str:
    # This is a placeholder. In a real scenario, this would dispatch to the actual tool function.
    # For the purpose of this edit, we'll just return a mock output.
    return f"[MOCK_TOOL_OUTPUT] {tool} executed with payload: {payload[:50]}..."

def _format_tool_output(tool: str, payload: str, output: str) -> str:
    # This is a placeholder. In a real scenario, this would format the output nicely.
    return f"\n🛠️ {tool} output:\n{output}\n"

def main():
    # ── 1. config.yaml'ı yükle (varsayılanlar + yaml + env) ──
    global _app_cfg
    _app_cfg = load_config()
    app = _app_cfg

    # ── 2. CLI argümanları (en yüksek öncelik) ──
    parser = argparse.ArgumentParser(
        description="Bio-ML Agent — Yerel LLM destekli ML proje asistanı",
        epilog="Yapılandırma: config.yaml > ortam değişkenleri > CLI argümanları",
    )
    parser.add_argument("--model", default=app.agent.model,
                        help=f"Ollama model adı (varsayılan: {app.agent.model})")
    parser.add_argument("--workspace", default=app.workspace.base_dir,
                        help=f"Çalışma alanı (varsayılan: {app.workspace.base_dir})")
    parser.add_argument("--timeout", type=int, default=app.agent.timeout,
                        help=f"Komut zaman aşımı saniye (varsayılan: {app.agent.timeout})")
    parser.add_argument("--max-steps", type=int, default=app.agent.max_steps,
                        help=f"Maks. adım sayısı (varsayılan: {app.agent.max_steps})")
    parser.add_argument("--history-dir", default=app.history.directory,
                        help=f"Konuşma geçmişi klasörü (varsayılan: {app.history.directory})")
    parser.add_argument("--load-session", default=None,
                        help="Başlangıçta yüklenecek oturum ID'si")
    parser.add_argument("--log-level", default=app.logging.level,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help=f"Log seviyesi (varsayılan: {app.logging.level})")
    parser.add_argument("--log-dir", default=app.logging.directory,
                        help=f"Log dosyaları klasörü (varsayılan: {app.logging.directory})")
    parser.add_argument("--config", dest="config_file", default="config.yaml",
                        help="Yapılandırma dosyası yolu (varsayılan: config.yaml)")
    parser.add_argument("--backend", dest="backend_mode", choices=["auto", "local", "remote"], default="auto",
                        help="LLM backend modu: local (Ollama), remote (model adına göre OpenAI/Anthropic/Gemini), auto (otomatik algıla) (varsayılan: auto)")
    parser.add_argument("--swarm", dest="swarm", action="store_true",
                        help="V6 Multi-Agent Swarm mimarisini taklit eden modu etkinleştir")
    args = parser.parse_args()

    # ── 3. CLI ile config farklıysa config'i güncelle ──
    if args.config_file: # Changed from args.config to args.config_file
        _app_cfg = load_config(config_path=args.config_file)
        app = _app_cfg

    # ── 4. Logger'ı kur ──
    log_dir = Path(args.log_dir).expanduser().resolve()
    global log
    log = setup_logger(log_dir, args.log_level)

    log.info("📋 Yapılandırma yüklendi:\n%s", app.summary())

    cfg = AgentConfig(
        model=args.model,
        workspace=Path(args.workspace).expanduser().resolve(),
        timeout=args.timeout,
        max_steps=args.max_steps,
        history_dir=Path(args.history_dir).expanduser().resolve(),
        load_session=args.load_session,
        log_level=args.log_level,
        log_dir=log_dir,
        config_file=args.config_file,
        backend_mode=args.backend_mode,
        swarm=args.swarm,
    )
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    _ensure_history_dir(cfg.history_dir)

    log.info("Agent başlatıldı | model=%s | workspace=%s | timeout=%d | max_steps=%d | swarm=%s",
             cfg.model, cfg.workspace, cfg.timeout, cfg.max_steps, cfg.swarm)

    # ── LLM backend ve plugin sistemi ──
    backend_mode = cfg.backend_mode # Changed from args.backend to cfg.backend_mode
    backend = auto_create_backend(cfg.model, mode=backend_mode)
    set_llm_backend(backend)
    log.info("🧠 LLM backend oluşturuldu | backend=%s | model=%s | mod=%s",
             type(backend).__name__, cfg.model, backend_mode)
    pm = get_plugin_manager()
    plugins_dir = Path(__file__).resolve().parent / "plugins"
    loaded = pm.discover(plugins_dir)
    if loaded:
        log.info("🔌 %d plugin yüklendi", loaded)
        print(f"🔌 {loaded} plugin yüklendi: {', '.join(pm.tool_names)}")

    # ── RAG Motoru başlat ──
    global rag
    rag = RAGEngine(workspace_dir=cfg.workspace)
    log.info("🔍 RAG Motoru başlatıldı | db_dir=%s", rag.db_dir)

    # ── Oturum başlat veya yükle ──
    session_id = generate_session_id()
    session_metadata = {"created_at": datetime.now().isoformat()}
    system_prompt = SYSTEM_PROMPT + format_catalog_for_prompt() + pm.get_prompt_additions()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if cfg.load_session: # Changed from args.load_session to cfg.load_session
        try:
            messages, session_metadata = load_conversation(cfg.history_dir, cfg.load_session)
            session_id = session_metadata.get("session_id", session_id)
            log.info("Oturum yüklendi | session=%s | mesaj_sayısı=%d", session_id, len(messages))
            print(f"📂 Oturum yüklendi: {session_id}")
            print(f"   Mesaj sayısı: {len(messages)}")
        except FileNotFoundError as e:
            log.warning("Oturum yüklenemedi | session=%s | hata=%s", cfg.load_session, e)
            print(f"❌ {e}")
            print("   Yeni oturum başlatılıyor...\n")

    log.info("Yeni oturum başlatıldı | session=%s", session_id)

    backend_label = type(backend).__name__.replace("Backend", "")
    print(f"🧠 Bio-ML Agent ready | model={cfg.model} | backend={backend_label} | workspace={cfg.workspace}")
    print(f"📜 Oturum ID: {session_id}")
    print(f"💾 Geçmiş klasörü: {cfg.history_dir}")
    print(f"📋 Log klasörü: {log_dir}")
    print(f"🔌 Backend modu: {backend_mode} | Aktif: {backend_label}")
    print("Çıkmak için: exit / quit | Komutlar: /history /load /new /save /delete /info /logs /rag /ragindex\n")

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            log.info("Kullanıcı Ctrl+C/EOF ile çıkış yaptı | session=%s", session_id)
            print("\n💾 Oturum kaydediliyor...")
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"✅ Kaydedildi: {session_id}")
            print("Çıkılıyor.")
            break

        if not user:
            continue

        # ── Çıkış komutları ──
        if user.lower() in {"exit", "quit"}:
            log.info("Kullanıcı çıkış yaptı | session=%s | komut=%s", session_id, user.lower())
            print("💾 Oturum kaydediliyor...")
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"✅ Kaydedildi: {session_id}")
            break

        # ── Geçmiş yönetimi komutları ──
        if user.lower() == "/history":
            sessions = list_conversations(cfg.history_dir)
            if not sessions:
                print("\n📭 Kayıtlı oturum bulunamadı.\n")
            else:
                print(f"\n📜 Kayıtlı Oturumlar ({len(sessions)} adet):")
                print("─" * 90)
                for i, s in enumerate(sessions, 1):
                    marker = " 👈 (aktif)" if s["session_id"] == session_id else ""
                    print(f"  {i:2}. 🆔 {s['session_id']}{marker}")
                    print(f"      📅 {s['created_at'][:19]}  |  💬 {s['message_count']} mesaj")
                    print(f"      📝 {s['summary'][:70]}")
                    print()
                print("─" * 90)
                print("  Yüklemek için: /load <session_id>\n")
            continue

        if user.lower().startswith("/load "):
            target_id = user.split(" ", 1)[1].strip()
            try:
                # Mevcut oturumu önce kaydet
                save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                print(f"💾 Mevcut oturum kaydedildi: {session_id}")

                messages, session_metadata = load_conversation(cfg.history_dir, target_id)
                session_id = session_metadata.get("session_id", target_id)
                print(f"✅ Oturum yüklendi: {session_id}")
                print(f"   Mesaj sayısı: {len(messages)}")

                # Son birkaç mesajı göster
                user_msgs = [m for m in messages if m["role"] == "user"
                             and not m["content"].startswith("TOOL_OUTPUT")]
                if user_msgs:
                    print(f"\n   📝 Son kullanıcı mesajı:")
                    print(f"      \"{user_msgs[-1]['content'][:100]}...\"\n")
            except FileNotFoundError as e:
                print(f"❌ {e}\n")
            continue

        if user.lower().startswith("/delete "):
            target_id = user.split(" ", 1)[1].strip()
            if target_id == session_id:
                print("❌ Aktif oturumu silemezsiniz! Önce /new ile yeni oturum başlatın.\n")
            elif delete_conversation(cfg.history_dir, target_id):
                print(f"🗑️  Oturum silindi: {target_id}\n")
            else:
                print(f"❌ Oturum bulunamadı: {target_id}\n")
            continue

        if user.lower() == "/new":
            # Mevcut oturumu kaydet, yenisini başlat
            log.info("Yeni oturum başlatılıyor | eski_session=%s", session_id)
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"💾 Mevcut oturum kaydedildi: {session_id}")

            session_id = generate_session_id()
            session_metadata = {"created_at": datetime.now().isoformat()}
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            log.info("Yeni oturum oluşturuldu | yeni_session=%s", session_id)
            print(f"🆕 Yeni oturum başlatıldı: {session_id}\n")
            continue

        if user.lower() == "/save":
            path = save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"💾 Oturum kaydedildi: {path}\n")
            continue

        if user.lower() == "/info":
            user_msg_count = sum(1 for m in messages if m["role"] == "user"
                                and not m["content"].startswith("TOOL_OUTPUT"))
            asst_msg_count = sum(1 for m in messages if m["role"] == "assistant")
            print(f"\n📊 Oturum Bilgileri:")
            print(f"   🆔 Oturum ID  : {session_id}")
            print(f"   📅 Oluşturulma: {session_metadata.get('created_at', '?')[:19]}")
            print(f"   💬 Toplam mesaj: {len(messages)}")
            print(f"   👤 Kullanıcı   : {user_msg_count} mesaj")
            print(f"   🤖 Asistan     : {asst_msg_count} mesaj")
            print(f"   💾 Geçmiş yolu : {cfg.history_dir / f'{session_id}.json'}")
            print()
            continue

        if user.lower() in {"/help", "/h"}:
            print_history_help()
            continue

        if user.lower().startswith("/logs"):
            # Son logları göster
            parts = user.split()
            tail_lines = 30
            if len(parts) > 1:
                try:
                    tail_lines = int(parts[1])
                except ValueError:
                    pass
            log_file = log_dir / LOG_FILE_NAME
            if log_file.exists():
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                show = lines[-tail_lines:] if len(lines) > tail_lines else lines
                print(f"\n📋 Son {len(show)} log satırı ({log_file}):")
                print("─" * 90)
                for line in show:
                    print(f"  {line}")
                print("─" * 90)
                print(f"  Toplam: {len(lines)} satır | Gösterilen: son {len(show)} satır")
                print(f"  Daha fazla görmek için: /logs <satır_sayısı>\n")
            else:
                print(f"\n📭 Log dosyası henüz oluşturulmamış: {log_file}\n")
            continue

        if user.lower() == "/ragindex":
            print("🔍 Workspace indeksleniyor. Lütfen bekleyin...")
            count = rag.index_workspace()
            print(f"✅ İndeksleme tamamlandı. {count} dosya işlendi.\n")
            continue

        if user.lower().startswith("/rag "):
            query = user.split(" ", 1)[1].strip()
            print(f"🔍 RAG araması yapılıyor: '{query}'")
            results = rag.search(query)
            if not results:
                print("📭 Eşleşen sonuç bulunamadı.\n")
            else:
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] 📄 {r['source']} (Mesafe: {r['distance']:.4f})")
                    print("─" * 40)
                    print(r['document'])
                    print("─" * 40)
                print()
            continue

        # ── Normal agent akışı ──
        log.info("👤 Kullanıcı mesajı alındı | uzunluk=%d | session=%s", len(user), session_id)
        log.debug("👤 Kullanıcı mesajı: %s", user[:300])
        user = normalize_user_message(user)
        allow_web = ("ALLOW_WEB_SEARCH" in user.upper())
        if allow_web:
            log.info("🌐 Web araması etkinleştirildi (ALLOW_WEB_SEARCH)")

        mproj = re.search(r"(?i)\bPROJECT\s*:\s*([a-z0-9_\-]+)", user)
        project = mproj.group(1) if mproj else DEFAULT_PROJECT
        os.environ["AGENT_PROJECT"] = project
        (cfg.workspace / project).mkdir(parents=True, exist_ok=True)
        log.info("📁 Aktif proje: %s", project)

        try:
            from memory_manager import memory
            mem_context = memory.get_context_string(user, n_results=2)
            if mem_context:
                enriched_user = f"{mem_context}\n\n[Mevcut Görev/Soru]:\n{user}"
                messages.append({"role": "user", "content": enriched_user})
                log.info("🧠 RAG Hafızası (%d sonuç) mesaja eklendi", 2)
            else:
                messages.append({"role": "user", "content": user})
        except Exception as e:
            log.warning("Hafıza yöneticisi hatası: %s", e)
            messages.append({"role": "user", "content": user})

        # Her kullanıcı mesajından sonra otomatik kaydet
        save_conversation(cfg.history_dir, session_id, messages, session_metadata)

        # Swarm Orchestrator Entegrasyonu (V6 Model) veya Monolitik (V5 Model)
        if cfg.swarm:
            from swarm.orchestrator import SwarmOrchestrator
            try:
                global _swarm_orchestrator
                if _swarm_orchestrator is None:
                    _swarm_orchestrator = SwarmOrchestrator(cfg)
                
                with Spinner("🧠 Swarm Orchestrator Devrede (Görev Dağıtılıyor)"):
                    assistant = _swarm_orchestrator.process(messages)
                
                # Sonucu ekrana ve hafızaya ekle
                print("\n🤖 Swarm Sonucu:\n", assistant)
                messages.append({"role": "assistant", "content": assistant})
                
                # Yeni etkileşimi RAG DB'ye kaydet
                try:
                    from memory_manager import memory
                    memory.store_interaction(session_id, user, assistant)
                    log.info("🧠 Etkileşim kalıcı hafızaya (RAG) kaydedildi")
                except Exception as e:
                    log.warning("Hafıza kaydetme hatası: %s", e)
                    
                # Asistan cevabından sonra otomatik kaydet
                save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                
            except Exception as e:
                log.error(f"[Swarm Error] {str(e)}", exc_info=True)
                print(f"\n❌ Swarm yöneticisinde kritik hata: {str(e)}")
        else:
            # Geleneksel Monolitik V5 Döngüsü
            for step in range(cfg.max_steps):
                log.info("🔄 Adım %d/%d başlıyor", step + 1, cfg.max_steps)
                
                try:
                    from llm_backend import summarize_memory
                    backend_for_mem = auto_create_backend(cfg.model)
                    messages = summarize_memory(messages, backend_for_mem, threshold=20)
                except Exception as e:
                    log.warning("Bellek özetleme adımı atlatıldı: %s", e)
                
                with Spinner(f"🧠 LLM düşünüyor (adım {step + 1}/{cfg.max_steps})"):
                    assistant = llm_chat(cfg.model, messages)

                tools_to_run, outside = extract_tools(assistant)

                if not tools_to_run:
                    py_m = FENCED_PY_RE.search(assistant)
                    bash_m = FENCED_BASH_RE.search(assistant)
                    if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                        tools_to_run = [("PYTHON", py_m.group(1))]
                        outside = FENCED_PY_RE.sub("", assistant).strip()
                        log.info("🔧 Fenced code block'tan PYTHON tool algılandı")
                    elif bash_m:
                        tools_to_run = [("BASH", bash_m.group(1))]
                        outside = FENCED_BASH_RE.sub("", assistant).strip()
                        log.info("🔧 Fenced code block'tan BASH tool algılandı")
                    else:
                        log.info("💬 Agent düz metin yanıtı verdi (tool yok) | adım=%d", step + 1)
                        print("\n🤖 Agent:\n", assistant)
                        messages.append({"role": "assistant", "content": assistant})
                        
                        try:
                            from memory_manager import memory
                            memory.store_interaction(session_id, user, assistant)
                            log.info("🧠 Etkileşim kalıcı hafızaya (RAG) kaydedildi")
                        except Exception as e:
                            log.warning("Hafıza kaydetme hatası: %s", e)
                            
                        # Asistan cevabından sonra otomatik kaydet
                        save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        break

                if outside:
                    log.warning("⚠️ Tool bloğu dışında metin vardı | dış_metin_uzunluk=%d", len(outside))
                    print("\n⚠️ Uyarı: Tool bloğu dışında metin vardı; yine de tool çalıştırılıyor.\n")

                messages.append({"role": "assistant", "content": assistant})
                
                all_outputs = []
                break_loop = False

                for tool, payload in tools_to_run:
                    log.info("🔧 Tool algılandı: %s | payload_uzunluk=%d", tool, len(payload or ""))
                    try:
                        if tool == "PYTHON":
                            # PYTHON kodlarını projenin kendi klasöründe çalıştır
                            py_cwd = cfg.workspace / project
                            py_cwd.mkdir(parents=True, exist_ok=True)
                            with Spinner("🐍 Python çalıştırılıyor"):
                                out = run_python(payload, py_cwd, timeout_s=cfg.timeout)
                        elif tool == "BASH":
                            # BASH komutlarını projenin kendi klasöründe çalıştır
                            bash_cwd = cfg.workspace / project
                            bash_cwd.mkdir(parents=True, exist_ok=True)
                            with Spinner("💻 Bash çalıştırılıyor"):
                                out = run_bash(payload, bash_cwd, timeout_s=cfg.timeout)
                        elif tool == "WEB_SEARCH":
                            if not allow_web and not _cfg().security.allow_web_search:
                                out = "[BLOCKED] WEB_SEARCH is disabled. To enable for this request, include: ALLOW_WEB_SEARCH"
                            else:
                                with Spinner("🌐 Web'de aranıyor"):
                                    out = web_search(payload)
                        elif tool == "WEB_OPEN":
                            with Spinner("📖 Sayfa okunuyor"):
                                out = web_open(payload)
                        elif tool == "READ_FILE":
                            out = read_file(payload, cfg.workspace)
                        elif tool == "WRITE_FILE":
                            out = write_file(payload, cfg.workspace)
                        elif tool == "TODO":
                            out = append_todo(payload, cfg.workspace)
                        elif tool == "RAG_SEARCH":
                            with Spinner("🔍 RAG'da aranıyor"):
                                results = rag.search(payload)
                                if not results:
                                    out = "[RAG_SEARCH] Sonuç bulunamadı."
                                else:
                                    out = "[RAG_SEARCH] Bulunan metinler:\n\n"
                                    for i, r in enumerate(results, 1):
                                        out += f"--- Kaynak: {r['source']} (Mesafe: {r['distance']:.4f}) ---\n"
                                        out += f"{r['document']}\n\n"
                        elif pm.get(tool):
                            with Spinner(f"🔌 Plugin çalıştırılıyor: {tool}"):
                                try:
                                    out = pm.execute(tool, payload, cfg.workspace) # Changed to pm.execute
                                except ToolExecutionError as e:
                                    out = f"[ERROR] {str(e)}"
                        else:
                            out = f"[ERROR] Bilinmeyen tool: {tool}"

                    except LLMConnectionError as e:
                        log.error("🧠 LLM bağlantı hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        print("\n⏳ 5 saniye sonra tekrar denenecek...\n")
                        time.sleep(5)
                        try:
                            with Spinner("🧠 LLM tekrar deneniyor"):
                                assistant = llm_chat(cfg.model, messages)
                            messages.append({"role": "assistant", "content": assistant})
                            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        except LLMConnectionError as e2:
                            log.error("🧠 LLM tekrar deneme başarısız | %s", e2)
                            print(f"\n{e2.user_message()}")
                            print("\n⚠️ LLM'e bağlanılamıyor. Lütfen Ollama servisini kontrol edin.\n")
                            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        break_loop = True
                        break

                    except SecurityViolationError as e:
                        log.warning("🔒 Güvenlik ihlali | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except ToolTimeoutError as e:
                        log.error("⏰ Zaman aşımı | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except (ToolExecutionError, FileOperationError, ValidationError) as e:
                        log.error("🛠️ Tool hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except AgentError as e:
                        log.error("❌ Agent hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except Exception as e:
                        log.error("💥 Beklenmeyen hata | tool=%s | %s", tool, e, exc_info=True)
                        print(f"\n❌ Beklenmeyen hata: {e}")
                        print(f"   💡 Öneri: Bu hatayı /logs komutuyla inceleyebilirsiniz.\n")
                        out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"

                if tool in {"WEB_SEARCH", "WEB_OPEN"} and not out.startswith("["):
                    autosave_web_outputs(cfg, tool, out)

                log.info("🛠️ Tool tamamlandı | tool=%s | çıktı_uzunluk=%d", tool, len(out))
                print(f"\n🛠️ {tool} output:\n{out}\n")
                all_outputs.append((tool, out))

            if break_loop:
                break
            
            user_msg = ""
            for t, o in all_outputs:
                user_msg += f"TOOL_OUTPUT ({t}):\n{o}\n\n"
            user_msg += "Continue. If done, answer normally (no tool)."
            
            messages.append({
                "role": "user",
                "content": user_msg
            })

            # Her tool adımından sonra otomatik kaydet
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            
        log.warning("⚠️ Maksimum adım sayısına ulaşıldı (%d) | session=%s", cfg.max_steps, session_id)
        print("\n⚠️ Max steps reached. Task may be incomplete.\n")
        save_conversation(cfg.history_dir, session_id, messages, session_metadata)


if __name__ == "__main__":
    main()
