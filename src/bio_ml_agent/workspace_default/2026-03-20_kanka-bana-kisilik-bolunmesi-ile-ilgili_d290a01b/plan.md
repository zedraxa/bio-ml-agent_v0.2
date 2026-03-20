# Kişilik Bölünmesi Araştırma Raporu Projesi Planı

1.  **Araştırma:** `WEB_SEARCH` aracını kullanarak Dissosiyatif Kimlik Bozukluğu (DKB), yani kişilik bölünmesi hakkında kapsamlı bilgi toplanacak. Araştırılacak ana başlıklar:
    *   Tanımı ve Tarihçesi (Çoğul Kişilik Bozukluğu)
    *   Belirtileri ve Tanı Kriterleri (DSM-5'e göre)
    *   Olası Nedenleri ve Risk Faktörleri (Özellikle travma ile ilişkisi)
    *   Eşlik Eden Diğer Psikiyatrik Durumlar (Komorbidite)
    *   Tedavi Yöntemleri ve Terapötik Yaklaşımlar
    *   Hastalıkla İlgili Yaygın Yanlış Anlaşılmalar ve Stigma
2.  **İçerik Sentezi:** Toplanan bilgiler, `rapor_icerik.md` adlı bir Markdown dosyasında mantıksal bir sıra ve akış içinde birleştirilecek. Bu dosya, PDF'in ham metnini oluşturacak.
3.  **PDF Üretimi:** Gerekli Python kütüphaneleri (`fpdf2`) kurulacak. Ardından, `rapor_icerik.md` dosyasını okuyup bunu `Kişilik_Bolunmesi_Raporu.pdf` isimli bir PDF dosyasına dönüştürecek bir Python betiği (`src/pdf_generator.py`) oluşturulacak ve çalıştırılacak.
4.  **Teslimat:** Oluşturulan PDF dosyasının hazır olduğu ve konum bilgisi kullanıcıya bildirilecek.
---