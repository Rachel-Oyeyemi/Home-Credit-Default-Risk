# Model Evaluation

This document provides detailed performance metrics for the models trained on the synthetic Home Credit dataset.  The evaluation uses a 20 % held‑out test set.  Given the binary classification nature of the problem and the moderate class imbalance, metrics beyond accuracy are critical for assessing model suitability.

## Evaluation metrics

| Metric            | Description |
|------------------:|-------------|
| **Accuracy**      | Overall proportion of correct predictions. |
| **Precision**     | Proportion of positive predictions that are correct (important when the cost of a false positive is high). |
| **Recall**        | Proportion of actual positives correctly identified (critical for minimising missed defaulters). |
| **F1‑score**      | Harmonic mean of precision and recall, balancing both types of errors. |
| **ROC‑AUC**       | Area under the receiver operating characteristic curve; measures the model’s ability to rank positive examples higher than negatives across thresholds. |
| **Confusion matrix** | Breakdown of true negatives (TN), false positives (FP), false negatives (FN) and true positives (TP). |

## Results by model

### Logistic Regression (baseline)

- **Accuracy:** 0.71
- **Precision:** 0.835
- **Recall:** 0.677
- **F1‑score:** 0.748
- **ROC‑AUC:** 0.783
- **Confusion matrix:**

|      | Pred 0 | Pred 1 |
|-----:|-------:|-------:|
| **True 0** | 56 | 17 |
| **True 1** | 41 | 86 |

**Interpretation:** Logistic regression predicts defaults with high precision but lower recall, leading to more missed defaulters.  As a linear model it offers transparency but may underfit complex relationships.

### Random Forest (advanced)

- **Accuracy:** 0.73
- **Precision:** 0.759
- **Recall:** 0.843
- **F1‑score:** 0.799
- **ROC‑AUC:** 0.765
- **Confusion matrix:**

|      | Pred 0 | Pred 1 |
|-----:|-------:|-------:|
| **True 0** | 39 | 34 |
| **True 1** | 20 | 107 |

**Interpretation:** The random forest yields the highest recall and F1‑score, capturing more defaulters at the cost of increased false positives.  It models non‑linearities and interactions well and is less sensitive to outliers.

### XGBoost (advanced)

- **Accuracy:** 0.71
- **Precision:** 0.763
- **Recall:** 0.787
- **F1‑score:** 0.775
- **ROC‑AUC:** 0.744
- **Confusion matrix:**

|      | Pred 0 | Pred 1 |
|-----:|-------:|-------:|
| **True 0** | 42 | 31 |
| **True 1** | 27 | 100 |

**Interpretation:** XGBoost provides a balanced trade‑off between precision and recall but does not outperform the random forest on this sample.  With larger data and more extensive tuning, boosting methods often excel.

## Metric comparison

| Model                  | Accuracy | Precision | Recall | F1‑score | ROC‑AUC |
|-----------------------:|---------:|----------:|-------:|---------:|--------:|
| **Logistic Regression**| 0.71     | **0.835** | 0.677 | 0.748   | **0.783**|
| **Random Forest**      | **0.73** | 0.759    | **0.843** | **0.799** | 0.765   |
| **XGBoost**            | 0.71     | 0.763    | 0.787 | 0.775   | 0.744   |

## Conclusions

The **Random Forest** model achieves the highest recall and F1‑score, making it the most effective at capturing defaulters with an acceptable false‑positive rate.  Logistic regression remains valuable as an interpretable baseline, while XGBoost provides a competitive alternative that may perform better with additional tuning or more complex features.  For deployment, the random forest model is recommended, but ongoing monitoring and retraining are advised as new data becomes available.
