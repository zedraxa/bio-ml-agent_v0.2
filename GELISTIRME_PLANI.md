# 🗺️ Bio-ML Agent — Master Super Roadmap (V3)

Bu doküman, Bio-ML Agent'ı "Scientific Operating System" (Bilimsel İşletim Sistemi) vizyonuna taşımak için hazırlanan ana stratejik plan ve gelişim yol haritasıdır.

---

## 🏗️ Mimari Katmanlar (Architecture Layers)
1. **Mission OS**: Niyet çevirisi, görev ayrıştırma ve ajan seçimi.
2. **Specialized Agents**: Browser, Microscopy, Coding, Document, Dataset ajanları.
3. **Artifact Graph**: Her adımın kanıta (evidence) dayalı takibi.
4. **Project Truth**: Proje seviyesindeki bellek ve karar geçmişi.
5. **Experience Surfaces**: Masaüstü (UI) ve Mobil (WhatsApp) operasyon yüzeyleri.
6. **Governance**: Senaryo testleri, kalite skorları ve audit.

---

## 🛤️ Yayın Trenleri (Release Trains)
- **Train A — Temel Mimari**: MissionBrain, ajan kontratları, artifact graft.
- **Train B — Yüksek Değerli Yetenekler**: Browser 2.0, Microscopy core, Lab rapor üretimi.
- **Train C — Keşif ve Yazım Derinliği**: AlphaFold, Literature Matrix, Reviewer simülasyonu.
- **Train D — Mobil ve Operasyon**: WhatsApp 2.0, Mobil onay merkezi, state sync.
- **Train E — Güvenilirlik**: Scenario replay, drift checks, recovery manager.

---

## 🚀 Faz 1: Core Agent Evolution (Part I) — Detaylı Plan

### [x] Faz E1 — Agent Runtime Standardization
Tüm alt ajanları aynı sözleşmeye bağlayarak orkestrasyon karmaşasını önlemek.
- [x] **Base Interface**: `perceive()`, `plan()`, `act()`, `verify()`, `summarize()`.
- [x] **Çıktılar**: Standart `AgentResult` (güven skoru ve kanıt linkleri ile).

### [x] Faz E2 — Browser Agent 2.0 (Profesyonel Katman)
Browser'ı tek bir döngüden çıkarıp uzmanlaşmış bir aileye dönüştürmek:
- [x] **Browser Scout**: Sayfa haritalama ve risk analizi (captcha, login).
- [x] **Browser Verifier**: "Güven ama doğrula" (Form submit oldu mu? Dosya indi mi?).
- [x] **Browser Extractor**: Kanıta dayalı yapılandırılmış veri üretimi (JSON/CSV).
- [x] **Browser Critic**: Döngü (loop) tespiti ve strateji hataları analizi.
- [x] **İleri Seviye**: Anti-loop governor ve site memory (öğrenen sistem).

### [x] Faz E3 — Document & Dataset Intelligence
- [x] **Document Agent**: Derin PDF ayrıştırma, atıf doğrulaması ve sentez.
- [x] **Dataset Agent**: Veri seti profil çıkarma (missingness, bias) ve keşif.

### [x] Faz E4 — Critic Layer (Öz-Denetim)
- [x] `CriticAgent`'ın zorunlu bir kalite kapısı (quality gate) olarak sisteme eklenmesi.
- [x] **Artifacts**: Eleştiri incelemeleri, revizyon önerileri ve güven uyarıları.

### [x] Faz E5 — Memory Evolution (Kaynaklı Bellek)
- [x] **Memory Curator**: Neyi belleğe yazacağını seçen akıllı katman.
- [x] **Site/Domain Memory**: Siteye özgü davranışların ve çalışan yöntemlerin öğrenilmesi.

### Faz E6 — Dynamic Model Orchestration 2.0
- **Capability Registry**: Görev tipi bazlı (JSON, Tool, Reasoning) model seçimi.
- **Escalation Logic**: Düşük güven durumunda daha güçlü modele veya insana devir.

### Faz E7 — Bioengineering Mission Packs
- **Literature Review Template**: DOI -> Kağıt -> Sentez otomasyonu.
- **Dataset Discovery Pack**: Omics verilerini bulma ve profilleme.

### Faz E8 — Autonomous Research Mode
- **Evidence Graph**: İddiaları kaynak belgelere/deneylere bağlama.
- **Contradiction Engine**: Farklı kaynaklardaki çelişkileri tespit etme.

---

## 📂 Önerilen Klasör Yapısı (Agent OS)
```text
src/bio_ml_agent/
  agents/         # Uzman roller (planner, critic, browser, document, dataset)
  orchestration/  # Mission graph (DAG) ve Task Router
  memory/         # Curator ve Provenance tracker
  evidence/       # Artifact Graph ve Şemalar
```

---

*Detaylı teknik plan ve görev takibi için [Active Implementation Plan](file:///home/yusuf/.gemini/antigravity/brain/a4335e25-aa48-490f-87da-e21aad0d858a/implementation_plan.md) ve [Task List](file:///home/yusuf/.gemini/antigravity/brain/a4335e25-aa48-490f-87da-e21aad0d858a/task.md) kullanılmaktadır.*
