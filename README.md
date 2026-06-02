# Customer Transaction Prediction
 
**Binary Classification using Machine Learning — Rubixe AI Solutions**
 
Predicting whether a customer will make a transaction based on
historical behavioral data.
 
---
 
## Project Overview
 
Built an end-to-end binary classification pipeline to predict
customer transactions on a dataset of 200,000+ records.
Key challenge was severe class imbalance (1:10 ratio) which
was handled using SMOTE oversampling.
 
---
 
## Dataset
 
- Size: 200,000+ customer transaction records
- Target: is_transaction (1 = transaction, 0 = no transaction)
- Class Imbalance: Severe 1:10 ratio (positive:negative)
 
---
 
## Tools & Libraries
 
- Python, Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
 
---
 
## Approach
 
1. EDA — univariate, bivariate analysis, correlation heatmap
2. Feature Engineering — created meaningful predictors
3. SMOTE — applied ONLY on training data to handle 1:10 imbalance
4. Trained and compared 4 ML models
5. Feature importance ranking to select top predictors
6. Evaluated using ROC-AUC (chosen over accuracy due to imbalance)
 
---
 
## Models Compared
 
| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.85 | 78% |
| Random Forest | 0.59 | 86% |
| LightGBM | 0.68 | 84% |
| **XGBoost** | **0.91** | **82%** |
 
---
 
## Final Model Performance
 
- **Selected Model**: XGBoost
- **ROC-AUC**: 0.91 (best among all models)
- **Improvement**: ~19% accuracy improvement over baseline
- **Why ROC-AUC**: Accuracy is misleading for imbalanced datasets.
  ROC-AUC measures discrimination ability across all thresholds.
 
---
 
## Key Results
 
- XGBoost achieved highest ROC-AUC of 0.91 on test set
- SMOTE oversampling improved minority class recall significantly
- Feature importance analysis identified top predictors
- Model performance improved by ~19% over baseline
 
---
 
## Files
 
- `Customer_Transaction_Prediction_Final.ipynb` — Full notebook with
  EDA, SMOTE, model comparison, feature importance, evaluation
 
---
 
## Summary
 
XGBoost was selected as the final model with ROC-AUC of 0.91.
SMOTE was applied only on training data to avoid data leakage.
Feature importance ranking reduced model complexity and improved
performance by 19% over the baseline.
