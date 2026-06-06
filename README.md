# Netflix Customer Churn Prediction

Predicting customer churn is vital for subscription services like Netflix to identify at-risk users, optimize retention strategies, and maximize customer lifetime value. This project implements an end-to-end Machine Learning pipeline to classify whether a subscriber will churn based on their demographic, usage, and subscription profile.

---

## Overview

Customer churn occurs when a user cancels their subscription. Retaining an existing customer is significantly more cost-effective than acquiring a new one. By leveraging customer activity metrics (like watch hours and login frequency), subscription details, and demographics, this project builds a predictive classifier to identify potential churners before they cancel, enabling proactive engagement.

---

## Dataset

The dataset used in this project is sourced from Kaggle:
👉 **[Kaggle Dataset: Netflix Customer Churn Dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)**

---

## Technical Approach

The model pipeline is built using `scikit-learn` inside a structured, reproducible `Pipeline` object:

### 1. Preprocessing Pipeline (`ColumnTransformer`)
* **Numerical Features** (`age`, `watch_hours`, `last_login_days`, `monthly_fee`, `number_of_profiles`, `avg_watch_time_per_day`):
  * **Imputation**: Median imputer for missing numeric entries.
  * **Scaling**: Standardized features to zero mean and unit variance.
* **Categorical Features** (`gender`, `subscription_type`, `region`, `device`, `payment_method`, `favorite_genre`):
  * **Imputation**: Mode imputer (most frequent category) for missing values.
  * **Encoding**: One-Hot Encoding with the first dummy column dropped to avoid multicollinearity.

### 2. Model Selection
* **Algorithm**: Random Forest Classifier (`RandomForestClassifier` with `random_state=42`).
* **Workflow**: The preprocessing steps and model are bundled in a unified `Pipeline` to prevent data leakage during training and validation.

### 3. File Structure
* `Data/`: Contains training/predictive CSV files.
* `Model/`: Directory containing the serialized model pickle file (`netflix_churn_model.pkl`).
* `Visualizations/`: Outputs from exploratory data analysis and model evaluation plots.
* `modeling.py`: Pipeline setup, dataset training, validation, evaluation metrics, and saving the model.
* `predict.py`: Inference pipeline to generate predictions on unlabelled datasets.
* `EDA.ipynb`: Jupyter notebook for Exploratory Data Analysis.

---

## Results

### Model Performance (Test Set Evaluation)

The Random Forest model demonstrated outstanding predictive capability on the test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97%** |
| **Precision** | **98%** |
| **Recall** | **97%** |
| **F1-Score** | **0.97** |
| **ROC AUC** | **1.00** |

#### Model Evaluation Graphics
The visual results are saved in the `Visualizations/` directory:
* **Confusion Matrix** (`Visualizations/confusion_matrix.png`): Demonstrates high accuracy with minimal false positives/false negatives.
* **ROC Curve** (`Visualizations/roc_curve.png`): Confirms a perfect/near-perfect area under the curve value of 1.00.

### Prediction Output

Inference was run on the unlabelled dataset `Data/data_predict.csv` (containing 500 new customer records) using `predict.py`. The resulting classification count is:
* **No Churn (no)**: 315 customers
* **Churn Predicted (yes)**: 185 customers

Output is saved to `Data/data_predict_with_churn.csv` with a visual count distribution at `Visualizations/churn_prediction_distribution.png`.

---

## Key Takeaways

1. **High Churn Propensity Metrics**: Last login days and engagement indicators like average watch hours play key roles in signaling churn.
2. **Proactive Intervention**: With a recall rate of 97%, Netflix can confidently target at-risk users with custom promotional offerings, specialized emails, or UI prompts without missing many true churners.
3. **Optimized Retention Marketing**: Using the prediction outputs (`data_predict_with_churn.csv`), marketing teams can directly focus their spend on the 185 high-probability churners rather than spreading the budget across all 500 users.
