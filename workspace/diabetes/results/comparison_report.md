# Model Karşılaştırma Raporu

Aşağıdaki tabloda farklı modellerin çapraz doğrulama ve test seti üzerindeki performans metrikleri gösterilmektedir.

| Model                  |   CV_Accuracy_Mean |   CV_Precision_Mean |   CV_Recall_Mean |   CV_F1_Mean |   CV_ROC_AUC_Mean |   Test_Accuracy |   Test_Precision |   Test_Recall |   Test_F1 |   Test_ROC_AUC |
|:-----------------------|-------------------:|--------------------:|-----------------:|-------------:|------------------:|----------------:|-----------------:|--------------:|----------:|---------------:|
| Logistic Regression    |           0.98022  |            0.982692 |         0.985965 |     0.984224 |          0.995975 |        0.982456 |         0.986111 |      0.986111 |  0.986111 |       0.995701 |
| Random Forest          |           0.962637 |            0.975519 |         0.964912 |     0.969935 |          0.989577 |        0.95614  |         0.958904 |      0.972222 |  0.965517 |       0.993882 |
| Gradient Boosting      |           0.951648 |            0.961725 |         0.961404 |     0.961363 |          0.991847 |        0.95614  |         0.946667 |      0.986111 |  0.965986 |       0.990741 |
| Support Vector Machine |           0.969231 |            0.97275  |         0.978947 |     0.975615 |          0.995562 |        0.982456 |         0.986111 |      0.986111 |  0.986111 |       0.99504  |
| K-Nearest Neighbors    |           0.962637 |            0.959179 |         0.982456 |     0.970489 |          0.988235 |        0.95614  |         0.958904 |      0.972222 |  0.965517 |       0.978836 |

**En İyi Model (Test ROC AUC'ye Göre):** Logistic Regression (ROC AUC: 0.9957)
