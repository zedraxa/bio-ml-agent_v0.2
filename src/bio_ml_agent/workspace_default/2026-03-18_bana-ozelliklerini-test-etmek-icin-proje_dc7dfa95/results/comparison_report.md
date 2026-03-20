# Model Karşılaştırma Raporu

Aşağıda, eğitim ve test setlerinde çeşitli modellerin performansını özetleyen bir tablo bulunmaktadır:

| Model              |   Test Accuracy |   Test Precision |   Test Recall |   Test F1-Score |   Test ROC AUC |   cv_accuracy_mean |   cv_accuracy_std |   cv_precision_mean |   cv_precision_std |   cv_recall_mean |   cv_recall_std |   cv_f1_mean |   cv_f1_std |   cv_roc_auc_mean |   cv_roc_auc_std |
|:-------------------|----------------:|-----------------:|--------------:|----------------:|---------------:|-------------------:|------------------:|--------------------:|-------------------:|-----------------:|----------------:|-------------:|------------:|------------------:|-----------------:|
| KNeighbors         |          0.7528 |           0.75   |        0.75   |          0.75   |         0.8389 |             0.6712 |            0.037  |              0.6749 |             0.04   |           0.6665 |          0.0499 |       0.6699 |      0.0396 |            0.7469 |           0.0441 |
| LogisticRegression |          0.7416 |           0.7143 |        0.7955 |          0.7527 |         0.8263 |             0.7479 |            0.0395 |              0.751  |             0.0537 |           0.7517 |          0.0241 |       0.7503 |      0.0308 |            0.8329 |           0.0368 |
| SVC                |          0.7416 |           0.7143 |        0.7955 |          0.7527 |         0.8258 |             0.7224 |            0.0333 |              0.7086 |             0.0272 |           0.7571 |          0.0458 |       0.7319 |      0.0349 |            0.8095 |           0.0333 |
| RandomForest       |          0.764  |           0.7255 |        0.8409 |          0.7789 |         0.8182 |             0.7053 |            0.057  |              0.7156 |             0.0669 |           0.689  |          0.0413 |       0.7018 |      0.0531 |            0.787  |           0.0344 |
| GradientBoosting   |          0.6966 |           0.6735 |        0.75   |          0.7097 |         0.7833 |             0.7109 |            0.057  |              0.6995 |             0.0529 |           0.7398 |          0.0717 |       0.7187 |      0.0603 |            0.7871 |           0.0416 |

**En İyi Model:** KNeighbors (Test ROC AUC: 0.8389)


## Açıklanabilir Yapay Zeka (XAI) Bulguları
SHAP ve LIME kullanarak modelin tahminlerini nasıl yaptığına dair içgörüler elde ettik.

### SHAP Özet Grafiği
SHAP özet grafiği (results/plots/shap_summary_plot.png adresinde bulunabilir) özelliklerin model çıktısı üzerindeki genel etkisini göstermektedir. Grafikteki ilk 10 özelliğin klinik olarak yorumlanması aşağıdadır:
- **BMI (Vücut Kitle İndeksi):** Diyabet tahmininde en etkili özelliklerden biridir. Yüksek BMI değerleri genellikle diyabet riskinin artmasıyla ilişkilidir.
- **Glucose (Glikoz Seviyesi):** Kan plazma glikoz konsantrasyonu, diyabet tanısında merkezi bir rol oynar. Yüksek glikoz seviyeleri diyabetin doğrudan bir göstergesidir.
- **Age (Yaş):** Yaş ilerledikçe diyabet riski artmaktadır. Modelin yaşa verdiği önem, bunun önemli bir faktör olduğunu desteklemektedir.
- **Insulin (İnsülin):** İnsülin seviyeleri, glikozun hücrelere alınmasında rol oynadığı için diyabet yönetiminde kritik öneme sahiptir. Düşük veya anormal seviyeler diyabeti işaret edebilir.
- **BloodPressure (Kan Basıncı):** Yüksek tansiyon, diyabetle sıkça görülen bir komorbiditedir ve modelin tahminlerinde etkili olmuştur.
- **SkinThickness (Cilt Kalınlığı):** Triceps cilt kıvrım kalınlığı, vücut yağ yüzdesinin bir göstergesidir ve insülin direnci ile ilişkilendirilebilir.
- **Pregnancies (Gebelik Sayısı):** Özellikle gestasyonel diyabet öyküsü olan kadınlar için diyabet riski yüksek olabilir.
- **DiabetesPedigreeFunction (Diyabet Soy Ağacı Fonksiyonu):** Aile geçmişindeki diyabet vakalarını gösteren genetik bir risk skorudur.

Bu özellikler, modelin diyabet riskini tahmin ederken klinik olarak anlamlı faktörlere odaklandığını göstermektedir. Özellikle yüksek BMI, yüksek glikoz seviyeleri ve yaş gibi faktörler, modelin pozitif diyabet tahminlerinde önemli katkı sağlamıştır.

### LIME Örnek Açıklaması
LIME, tek bir örneğin tahmini için yerel açıklamalar sağlar (results/plots/lime_explanation_instance_0.png adresinde bulunabilir). Bu, belirli bir hastanın neden diyabetli olarak sınıflandırıldığına (veya sınıflandırılmadığına) dair ayrıntılı bir bakış sunar. LIME çıktısı, o örneğin tahminine en çok katkıda bulunan özelliklerin ve bunların etkisinin anlaşılmasına yardımcı olur.
