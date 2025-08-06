# Customer Transaction Classification using Machine Learning

## 🧾 Project Overview
This project aims to predict whether a customer will make a transaction based on historical behavioral data using machine learning techniques. The task is framed as a **binary classification problem**.

## 📊 Dataset
- Contains customer activity and profile-related features.
- Target variable: `is_transaction` (1 = transaction, 0 = no transaction)

## ⚙️ Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (for EDA & visualization)

## 🚀 Models Used
- Logistic Regression ✅ *(final model)*
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- XGBoost  

## ✅ Final Model
- **Selected Model**: Logistic Regression  
- **Accuracy**: 78%  
- **AUC Score**: 85%  
- Chosen for its balance of simplicity, performance, and generalizability.

## 📁 Files
- `Customer_Transaction_Prediction_Final.ipynb`: Main analysis and modeling notebook

## 📌 Summary
The model successfully predicts customer transactions with strong AUC performance. With further tuning and feature engineering, accuracy can potentially be improved. ROC curve and confusion matrix were also used for evaluation.


