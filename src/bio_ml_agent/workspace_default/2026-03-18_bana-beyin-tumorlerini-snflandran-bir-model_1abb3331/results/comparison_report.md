# 📊 Model Karşılaştırma Raporu

**Görev Türü:** Classification
**Karşılaştırılan Model Sayısı:** 6
**Çapraz Doğrulama:** 5-fold
**En İyi Model:** 🏆 **SVM**

## Karşılaştırma Tablosu

| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean | Süre |
|---|-------|----------|-----------|--------|-----|---------|---------|------|
| 1 | 🏆 **SVM** | 0.8550 | 0.8675 | 0.8550 | 0.8558 | 0.9668 | 0.8330 | 0.12s |
| 2 | **RandomForest** | 0.8500 | 0.8541 | 0.8500 | 0.8503 | 0.9646 | 0.7870 | 0.29s |
| 3 | **GradientBoosting** | 0.8050 | 0.8149 | 0.8050 | 0.8053 | 0.9502 | 0.7820 | 1.64s |
| 4 | **KNN** | 0.7750 | 0.7865 | 0.7750 | 0.7716 | 0.9383 | 0.7560 | 0.00s |
| 5 | **LogisticRegression** | 0.7500 | 0.7538 | 0.7500 | 0.7502 | 0.8967 | 0.7020 | 0.01s |
| 6 | **DecisionTree** | 0.7000 | 0.7088 | 0.7000 | 0.6998 | 0.8000 | 0.6480 | 0.01s |

## 🏆 En İyi Model: SVM

### Metrikler

- **accuracy:** 0.8550
- **precision:** 0.8675
- **recall:** 0.8550
- **f1:** 0.8558
- **roc_auc:** 0.9668
- **CV Mean ± Std:** 0.8330 ± 0.0196
- **Eğitim Süresi:** 0.12s

### Classification Report

```
              precision    recall  f1-score   support

           0       0.77      0.88      0.82        50
           1       1.00      0.76      0.86        50
           2       0.82      0.90      0.86        50
           3       0.88      0.88      0.88        50

    accuracy                           0.85       200
   macro avg       0.87      0.85      0.86       200
weighted avg       0.87      0.85      0.86       200

```

---
*Bu rapor Bio-ML Agent tarafından otomatik oluşturulmuştur.*
