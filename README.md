# Hotel Booking Status Prediction

This repository contains the solution notebook for **Kaggle Assignment 2** as part of the **Machine Learning Practice (MLP)** course in the **IIT Madras BS Degree in Data Science and Applications**.

> 🔗 **Kaggle Competition Link:** [MLP Term 3 2025 Kaggle Assignment 2](https://www.kaggle.com/competitions/mlp-term-3-2025-kaggle-assignment-2)  
> *Note: The competition has been made private by the authorities following the completion of the assignment evaluation, so the original work is archived and shared here.*

## 📌 Project Overview
The objective of this project is to build an end-to-end Machine Learning pipeline to predict hotel booking cancellations (`booking_status`) based on historical customer reservation data. 

## 🛠️ Workflow Breakdown

1. **Exploratory Data Analysis (EDA):**
   * Identified data types, checked shapes, and calculated statistics for training (`29,500` rows) and testing (`7,000` rows) splits.
   * Visualized price distribution, lead time vs. price relationship, and reservation segment frequencies.

2. **Data Cleaning & Engineering:**
   * Handled missing elements in columns like `room_type`, `lead_time`, and `price` via median/mode imputation.
   * Engineered temporal features (`arrival_year`, `arrival_month`, `arrival_day`) from the raw timestamp string.
   * Dropped over `7,100` duplicate rows in the training dataset to avoid evaluation bias.
   * Cleared outliers using Interquartile Range (IQR) capping method.

3. **Preprocessing Pipeline:**
   * Numerical scaling applied using `StandardScaler`.
   * Categorical feature transformation handled with robust `OneHotEncoder(handle_unknown='ignore')`.

4. **Model Development & Benchmarking:**
   * Trained and compared 7 baseline classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, and XGBoost.
   * **Hyperparameter Tuning:** Conducted `GridSearchCV` on Random Forest, SVM, and XGBoost to fine-tune regularization parameters, tree depth, and estimator limits.

## 🏆 Results Comparison

| Model | Validation Accuracy |
| :--- | :--- |
| **Random Forest (Baseline)** | **85.12% (Best Model)** |
| Random Forest (Tuned) | 84.41% |
| XGBoost (Baseline) | 84.23% |
| XGBoost (Tuned) | 84.07% |
| Gradient Boosting | 83.76% |
| SVM (Tuned) | 83.00% |
| SVM (Baseline) | 82.51% |
| KNN | 80.22% |
| Logistic Regression | 80.16% |
| Decision Tree | 79.64% |

The untuned **Random Forest Classifier** achieved the highest validation accuracy of **85.12%** and was used to make the final predictions exported to `submission.csv`.
