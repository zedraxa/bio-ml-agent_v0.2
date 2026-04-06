# Bio-ML Agent Release Strategy & Guide

This document describes the versioning, packaging, and release processes for the Bio-ML Agent project.

## 🏷️ Semantic Versioning (SemVer)

The project follows `MAJOR.MINOR.PATCH` format:
- **MAJOR**: Breaking, backward-incompatible architectural changes (e.g., v1.0.0).
- **MINOR**: Backward-compatible new features (sprint completions).
- **PATCH**: Bug fixes and minor improvements.

## 🌿 Branch Structure

- `main`: Stable, always production-ready code.
- `develop`: Active development branch. Features are merged here first.
- `feature/*`: Short-lived branches for new features.

## 🚀 Release Process

1. **Version Update**: Update the `version` field in `pyproject.toml`.
2. **Changelog**: Add new version information to `walkthrough.md` (Artifact summary).
3. **CI Verification**: Ensure all GitHub Actions checks (Lint, Unit, Smoke) pass.
4. **Build**: Create packages with `python3 -m build`.
5. **Tag**: `git tag -a v0.1.0-clean -m "Phase 4: Productization Release"`

## 🧪 Reproducibility Checklist

Before each release, verify:
- [ ] `VERSION_DATASET` works for all sample datasets.
- [ ] MLflow Run IDs are visible in reports.
- [ ] All optional dependencies (`all` extra) install cleanly in a fresh venv.
- [ ] `eval_bench.py` scores are within acceptable range.

## ⚙️ Automation Status
- [x] Automatic `pytest` and `ruff` lint checks via GitHub Actions.
- [ ] Automatic PyPI publishing (planned for future release).

---

## 🇹🇷 Türkçe Sürüm Rehberi (Turkish)

**SemVer:** `MAJOR.MINOR.PATCH` — Büyük kırılmalar / Yeni özellikler / Hata düzeltmeleri.

**Dal Yapısı:** `main` (kararlı), `develop` (aktif geliştirme), `feature/*` (geçici özellik dalları).

**Yayınlama Adımları:**
1. `pyproject.toml` içinde `version` güncelle.
2. Changelog güncelle.
3. CI testlerinin geçtiğini doğrula.
4. `python3 -m build` ile paketle.
5. `git tag -a vX.Y.Z -m "..."` ile etiketle.

**Otomasyon:** GitHub Actions üzerinden `pytest` + `ruff` otomatik çalışır. PyPI yayını gelecek planında.
