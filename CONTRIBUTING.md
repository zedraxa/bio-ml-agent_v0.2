# Bio-ML Agent Katılım Rehberi (Contribution Guide)

Bio-ML Agent projesine katkıda bulunmak istediğiniz için teşekkürler! Bu proje, biyoloji ve makine öğrenimi dünyasını otonom ajanlarla birleştirmeyi hedefler.

## 🛠 Geliştirme Ortamı Kurulumu

1.  **Depoyu Çatallayın (Fork) ve Klonlayın:**
    ```bash
    git clone https://github.com/kullanici_adi/ai-agent.git
    cd ai-agent
    ```

2.  **Sanal Ortam Oluşturun:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\\Scripts\\activate  # Windows
    ```

3.  **Bağımlılıkları Geliştirme Modunda Kurun:**
    ```bash
    pip install -e ".[all,test,dev]"
    ```

## 🧪 Testleri Çalıştırma
Yeni bir özellik eklediğinizde veya bir hata düzelttiğinizde lütfen testleri çalıştırın:
```bash
pytest tests/
```

## 📝 Kod Standartları
- Python 3.10+ özelliklerini kullanın.
- Tip ipuçları (type hints) zorunludur.
- Yeni fonksiyonlar için docstring ekleyin.
- Pydantic modelleri ile veri doğrulama yapın.

## 🚀 Pull Request Süreci
1.  Yeni bir dal (branch) açın: `git checkout -b feature/yeni-ozellik`.
2.  Değişikliklerinizi yapın ve commit mesajlarını anlamlı tutun.
3.  Dalınızı pushlayın: `git push origin feature/yeni-ozellik`.
4.  Depo üzerinden bir Pull Request açın.

Sorularınız için Issue açmaktan çekinmeyin!
