# Task 2: End-to-End ML Pipeline with Scikit-learn

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset.

## Methodology / Approach
1. **Data Preprocessing**: We handle missing values (like empty strings in `TotalCharges`), separate the features into numerical and categorical types.
2. **Pipeline Construction**: We use `scikit-learn`'s `Pipeline` and `ColumnTransformer`. 
 - Numerical data is processed using `SimpleImputer` (median) and `StandardScaler`.
 - Categorical data is processed using `SimpleImputer` (most frequent) and `OneHotEncoder`.
3. **Model Training & Hyperparameter Tuning**: A `RandomForestClassifier` is placed at the end of the pipeline. `GridSearchCV` is used to find the optimal hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
4. **Model Export**: The final optimized pipeline is saved using `joblib` into `churn_pipeline_model.pkl` for easy loading in production environments without the need to recreate the preprocessing steps.

## Key Results
- The pipeline efficiently processes raw test data without manual transformation.
- Evaluated on a 20% holdout test set using accuracy and a classification report.
- Achieved robust performance with proper cross-validation (CV=3) during hyperparameter tuning.

## How to Run
Ensure you have the required dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```
Then execute the script:
```bash
python task2_churn_pipeline.py
```
