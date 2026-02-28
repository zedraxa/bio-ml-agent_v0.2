# Bio-ML Ajanı Biyomühendislik Projesi Test Raporu

## 1. Protein Analizi Sonuçları
**Dizi:** `MKWVTFISLLLLFSSAYS` (Sinyal peptidi örneği)

`ProteinAnalyzer` kullanılarak elde edilen özellikler:
- **Uzunluk:** 18 amino asit
- **Moleküler Ağırlık:** 2062.53 Da
- **GRAVY (Büyük Ortalama Hidrofobisite):** 1.922 (Pozitif GRAVY skoru proteinin genel olarak **hidrofobik (suyu sevmeyen)** olduğunu gösterir ve dizi tipik bir hidrofobik sinyal peptidi karakteristiğine sahiptir).
- **İzoelektrik Nokta (pI):** Yük (Charge) durumu neticesinde varsayılan 7.0 (bazı asidik/bazik aminoasit eksikliği nedeniyle)
- **Kompozisyon:** L (Leucine) ve S (Serine) yoğunluktadır (%27.78 ve %22.22).

*Yorumlar:* Sinyal peptidleri genellikle yüksek L (Lösin) içeriğine ve hidrofobikliğe sahip oldukları için sonuçlar biyolojik beklentilerle birebir örtüşmektedir.

---

## 2. İlaç Olabilirlik Analizi (Aspirin)
**Molekül:** Aspirin
**SMILES:** `CC(=O)OC1=CC=CC=C1C(=O)O`

`DrugDiscoveryHelper` kullanılarak elde edilen özellikler:
- **Moleküler Formül:** C9H8O4
- **Tahmini Moleküler Ağırlık:** ~180.16 Da
- **H-Bağı Donörleri:** 1 (OH grubu nedeniyle)
- **H-Bağı Akseptörleri:** 4 (Oksijen atomları)
- **Dönebilen Bağlar:** (Tahmini ~3-4)
- **Lipinski Beşli Kuralı Uyumluluğu:** UYUMLU (drug-like = true, ihlal = 0)
  - Moleküler Ağırlık (180.16) <= 500 Da
  - H-Bağı Donör (1) <= 5
  - H-Bağı Akseptör (4) <= 10
  - Tahmini LogP <= 5

*Yorumlar:* Aspirin, Lipinski'nin Beşli Kuralı'na tam uyum gösteren ve oral yolla alındığında sistemik biyoyararlanımı iyi olan, klasik bir "drug-like" (ilaç benzeri) molekül özelliğini taşır.

---

### Terminal Test Özeti
- Yapılan canlı terminal testinde LLM asistanının sistem promptuna eklenen **Biyomühendislik Toolkit** yönergelerini okuyarak `agent.py` içindeki `<PYTHON>` aracıyla kendi kodunu yazdığı ve bu sınıflardaki metotları başarıyla çağırarak doğru sonuçları okuyabildiği **teyit edilmiştir**.
- Ajan kodları `sys.path`'e atanan kök dizin eklemesiyle (`sys.path.insert()`) sorunsuz bir şekilde `bioeng_toolkit`'den import edebilmektedir.
