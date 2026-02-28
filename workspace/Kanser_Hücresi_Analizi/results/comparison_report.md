# Model Karşılaştırma Raporu

Bu rapor, farklı makine öğrenimi modellerinin performansını özetler.

## Karşılaştırma Tablosu

| Model              |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |   CV_Accuracy_Mean |   CV_Accuracy_Std |
|:-------------------|-----------:|------------:|---------:|-----------:|----------:|-------------------:|------------------:|
| LogisticRegression |   0.973684 |    0.973711 | 0.973684 |   0.973616 |  0.993056 |           0.96875  |         0.0171163 |
| RandomForest       |   0.973684 |    0.974737 | 0.973684 |   0.973465 |  0.994048 |           0.946875 |         0.0233854 |
| GradientBoosting   |   0.95614  |    0.956869 | 0.95614  |   0.955776 |  0.988095 |           0.95625  |         0.030298  |
| SVM                |   0.938596 |    0.944037 | 0.938596 |   0.937229 |  0.992725 |           0.9625   |         0         |
| KNN                |   0.938596 |    0.944037 | 0.938596 |   0.937229 |  0.993717 |           0.9625   |         0.0390312 |

En iyi model: **RandomForest**

Modeller, bir StandardScaler ve ardından modelin kendisinden oluşan bir pipeline kullanılarak eğitildi.