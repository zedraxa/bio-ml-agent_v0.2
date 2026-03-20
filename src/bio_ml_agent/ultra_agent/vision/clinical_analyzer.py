# ultra_agent/vision/clinical_analyzer.py
# ═══════════════════════════════════════════════════════════
#  Pillar 3-1: Gemini Vision — Klinik Tıbbi Görüntü Analizi
# ═══════════════════════════════════════════════════════════
"""
MRI, BT ve Patoloji görüntülerini Gemini Vision Pro ile
analiz eden özelleştirilmiş klinik promptlar.

Kullanım:
    analyzer = ClinicalImageAnalyzer()
    result = analyzer.analyze("xray.png", modality="chest_xray")
    print(result)
"""

import logging
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List

log = logging.getLogger("bio_ml_agent")

# ── Klinik Prompt Şablonları ──────────────────────────────

CLINICAL_PROMPTS: Dict[str, str] = {
    "chest_xray": """Sen deneyimli bir radyolog asistanısın. Bu göğüs röntgeni görüntüsünü analiz et.

Lütfen şu başlıklar altında rapor hazırla:
1. **Teknik Kalite:** Görüntü kalitesi, pozisyon ve penetrasyon değerlendirmesi
2. **Kardiyak Silüet:** Kalp boyutu ve konturu (KTO tahmini)
3. **Akciğer Alanları:** Her iki akciğer alanının değerlendirmesi
   - Infiltrasyon, konsolidasyon, atelektazi var mı?
   - Plevral efüzyon belirtisi var mı?
4. **Mediastinum:** Mediastinal genişleme veya kitle
5. **Kemik Yapılar:** Kostalar ve vertebra değerlendirmesi
6. **Sonuç ve Öneri:** Klinik korelasyon önerileri

⚠️ NOT: Bu bir yapay zeka değerlendirmesidir, kesin tanı koyma amacı taşımaz.""",

    "brain_mri": """Sen deneyimli bir nöroradyolog asistanısın. Bu beyin MRI görüntüsünü analiz et.

Rapor şablonu:
1. **Sekansinformasyon:** Görülen MRI sekansı (T1, T2, FLAIR, DWI tahmini)
2. **Supratentoryal:** Serebral hemisferler, sulkus ve ventrikül değerlendirmesi
3. **İnfratentoryal:** Serebellum ve beyin sapı
4. **Lezyon Tespiti:** Varsa sinyal anormallikleri, kitle lezyonları
   - Boyut, lokalizasyon, sinyal karakteristikleri
5. **Vasküler Yapılar:** Büyük vasküler yapıların değerlendirmesi
6. **Orta Hat:** Orta hat kayması var mı?
7. **Sonuç ve Öneri:** Klinik korelasyon ve ileri tetkik önerileri

⚠️ NOT: Bu bir yapay zeka değerlendirmesidir, kesin tanı koyma amacı taşımaz.""",

    "pathology": """Sen deneyimli bir patolog asistanısın. Bu patoloji (histopatoloji) görüntüsünü analiz et.

Rapor şablonu:
1. **Genel Değerlendirme:** Doku tipi ve boyama yöntemi tahmini (H&E, IHC vb.)
2. **Mimari Pattern:** Doku mimarisi (tübüler, papiller, solid, kribriform)
3. **Hücresel Özellikler:**
   - Nükleer boyut ve pleomorfizm
   - Mitotik aktivite
   - Sitoplazma özellikleri
4. **Stroma:** Stromal reaksiyon, desmoplazi, inflamasyon
5. **Özel Bulgular:** Nekroz, vasküler invazyon, perinöral invazyon
6. **Ön Değerlendirme:** Olası tanısal kategoriler (benign/malign/atipik)
7. **Öneriler:** İleri immünohistokimya veya moleküler testler

⚠️ NOT: Bu bir yapay zeka değerlendirmesidir. Kesin patolojik tanı, uzman patolog tarafından konulmalıdır.""",

    "ct_abdomen": """Sen deneyimli bir radyolog asistanısın. Bu abdomen BT görüntüsünü analiz et.

Rapor şablonu:
1. **Teknik:** Kontrastlı/kontrastsız, pencere ayarı değerlendirmesi
2. **Karaciğer:** Boyut, kontür, parankimal homojenite, fokal lezyonlar
3. **Safra Yolları:** İntra/ekstrahepatik safra kanalları, safra kesesi
4. **Pankreas:** Boyut, morfoloji, duktus pankreatikus
5. **Dalak:** Boyut ve homojenite
6. **Böbrekler:** Bilateral böbrek değerlendirmesi, taş, kist, kitle
7. **Adrenal Bezler:** Boyut ve morfoloji
8. **Barsak ve Mezenter:** İnce/kalın barsak, LAP
9. **Vasküler:** Aort ve dalları
10. **Sonuç ve Öneri:** Klinik korelasyon önerileri

⚠️ NOT: Bu bir yapay zeka değerlendirmesidir, kesin tanı koyma amacı taşımaz.""",

    "general": """Sen deneyimli bir tıbbi görüntüleme uzmanı asistanısın. Bu tıbbi görüntüyü analiz et.

Lütfen şu başlıklar altında değerlendir:
1. **Görüntü Türü:** Modalite tahmini (X-ray, MRI, BT, Ultrason, vb.)
2. **Anatomik Bölge:** Hangi anatomik bölge görüntülenmiş
3. **Normal Bulgular:** Saptanan normal anatomik yapılar
4. **Anormal Bulgular:** Varsa patolojik görünümler
5. **Sonuç ve Öneri:** Klinik değerlendirme önerileri

⚠️ NOT: Bu bir yapay zeka değerlendirmesidir, kesin tanı koyma amacı taşımaz.""",
}


class ClinicalImageAnalyzer:
    """Gemini Vision Pro ile klinik tıbbi görüntü analizi.

    Kullanım:
        analyzer = ClinicalImageAnalyzer()
        result = analyzer.analyze("chest.png", modality="chest_xray")
    """

    SUPPORTED_MODALITIES = list(CLINICAL_PROMPTS.keys())

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """Görüntüyü base64'e çevirir."""
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """Dosya uzantısına göre MIME type döndürür."""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".dcm": "application/dicom",
        }
        return mime_map.get(ext, "image/png")

    def analyze(self, image_path: str, modality: str = "general",
                extra_context: str = "") -> Dict[str, Any]:
        """Tıbbi görüntüyü Gemini Vision ile analiz eder.

        Args:
            image_path: Görüntü dosya yolu.
            modality: Modalite türü (chest_xray, brain_mri, pathology, ct_abdomen, general).
            extra_context: Ek klinik bilgi (ör. "65 yaş erkek, öksürük şikayeti").

        Returns:
            Analiz sonucu: {"modality", "analysis", "model", "disclaimer"}
        """
        if not Path(image_path).exists():
            return {"error": f"Görüntü dosyası bulunamadı: {image_path}"}

        clinical_prompt = CLINICAL_PROMPTS.get(modality, CLINICAL_PROMPTS["general"])

        if extra_context:
            clinical_prompt += f"\n\n**Ek Klinik Bilgi:** {extra_context}"

        try:
            import google.generativeai as genai
            import os

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return {"error": "GEMINI_API_KEY veya GOOGLE_API_KEY ortam değişkeni ayarlanmamış."}

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model)

            # Görüntüyü yükle
            image_data = Path(image_path).read_bytes()
            mime_type = self._get_mime_type(image_path)

            response = model.generate_content([
                clinical_prompt,
                {"mime_type": mime_type, "data": image_data},
            ])

            analysis_text = response.text if hasattr(response, "text") else str(response)

            return {
                "modality": modality,
                "image": image_path,
                "analysis": analysis_text,
                "model": self.model,
                "disclaimer": "Bu analiz yapay zeka tarafından üretilmiştir. Klinik karar desteği amacıyla kullanılmalı, kesin tanı yerine geçmez.",
            }

        except ImportError:
            return {
                "error": "google-generativeai kütüphanesi yüklü değil. "
                         "'pip install google-generativeai' ile yükleyin.",
            }
        except Exception as e:
            log.error(f"Gemini Vision analiz hatası: {e}")
            return {"error": str(e)}

    def list_modalities(self) -> List[str]:
        """Desteklenen modalite listesini döndürür."""
        return self.SUPPORTED_MODALITIES
