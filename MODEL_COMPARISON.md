# Model Comparison

After preprocessing and splitting the synthetic Home Credit sample dataset, three models were trained and evaluated on the held‑out test set.  The table below summarises their performance across key metrics.

| Model                  | Accuracy | Precision | Recall | F1‑score | ROC‑AUC | Comments |
|-----------------------:|---------:|----------:|-------:|---------:|--------:|---------|
| **Logistic Regression**| **0.71** | **0.835** | 0.677 | 0.748   | **0.783**| Interpretable baseline; high precision but lower recall (more missed defaulters). |
| **Random Forest**      | **0.73** | 0.759    | **0.843** | **0.799** | 0.765   | Best overall F1 and recall; captures non‑linear relationships with minimal tuning. |
| **XGBoost**            | **0.71** | 0.763    | 0.787 | 0.775   | 0.744   | Gradient‑boosting approach; competitive but slightly lower performance on this sample. |

## Discussion

- **Baseline:** Logistic regression provides an interpretable starting point and achieves the highest precision (0.835), meaning it makes fewer false positive predictions.  However, its recall is comparatively lower (0.677), indicating it misses a significant number of actual defaulters.
- **Random Forest:** The ensemble of decision trees improves recall (0.843) and F1‑score (0.799) while maintaining reasonable precision.  It can model non‑linear relationships and interactions without extensive feature engineering, making it a strong candidate for deployment.
- **XGBoost:** The gradient‑boosted trees achieve balanced precision and recall but do not outperform the random forest on this synthetic dataset.  With additional hyperparameter tuning and a larger dataset, XGBoost may close the gap.

Given the project's goal of identifying high‑risk borrowers while minimising false negatives, **Random Forest** is selected as the recommended model.  It delivers the best balance of recall and precision and generalises well without extensive parameter tuning.
