"""
Problem Statement & Objective:
The objective is to build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset.
The pipeline should handle data preprocessing (scaling, encoding), model training, hyperparameter tuning, and export the complete pipeline for deployment.

Methodology / Approach:
1. Dataset Loading & Preprocessing: Load data, handle missing values (TotalCharges), define numerical and categorical features.
2. Model Development & Training: Create a scikit-learn Pipeline with SimpleImputer, StandardScaler, OneHotEncoder, and a classifier (LogisticRegression / RandomForestClassifier).
3. Hyperparameter Tuning: Use GridSearchCV to find the best hyperparameters.
4. Evaluation: Evaluate using classification report, confusion matrix, accuracy.
5. Export: Save the pipeline using joblib.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def main():
 print("--- Task 2: End-to-End ML Pipeline ---")
 
 # 1. Dataset Loading
 data_path = 'Telco-Customer-Churn.csv'
 if not os.path.exists(data_path):
 print(f"Error: Dataset {data_path} not found.")
 return
 
 print("Loading dataset...")
 df = pd.read_csv(data_path)
 
 # Preprocessing: Handle empty strings in TotalCharges
 df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
 
 # Define features and target
 target = 'Churn'
 X = df.drop(columns=['customerID', target])
 y = df[target].map({'Yes': 1, 'No': 0})
 
 # Identify numerical and categorical columns
 numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
 categorical_features = [col for col in X.columns if col not in numeric_features]
 
 # Train-test split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
 # 2. Pipeline Construction
 print("Building Scikit-learn Pipeline...")
 
 numeric_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='median')),
 ('scaler', StandardScaler())
 ])
 
 categorical_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('onehot', OneHotEncoder(handle_unknown='ignore'))
 ])
 
 preprocessor = ColumnTransformer(
 transformers=[
 ('num', numeric_transformer, numeric_features),
 ('cat', categorical_transformer, categorical_features)
 ])
 
 # We will tune a Random Forest classifier
 pipeline = Pipeline(steps=[
 ('preprocessor', preprocessor),
 ('classifier', RandomForestClassifier(random_state=42))
 ])
 
 # 3. Hyperparameter Tuning with GridSearchCV
 print("Starting GridSearch for Hyperparameter Tuning...")
 param_grid = {
 'classifier__n_estimators': [50, 100],
 'classifier__max_depth': [5, 10, None],
 'classifier__min_samples_split': [2, 5]
 }
 
 grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
 grid_search.fit(X_train, y_train)
 
 print(f"Best Parameters: {grid_search.best_params_}")
 
 # 4. Evaluation
 print("Evaluating Model...")
 best_model = grid_search.best_estimator_
 y_pred = best_model.predict(X_test)
 
 acc = accuracy_score(y_test, y_pred)
 print(f"Accuracy: {acc:.4f}")
 print("\nClassification Report:")
 print(classification_report(y_test, y_pred))
 
 # 5. Export Pipeline
 model_filename = 'churn_pipeline_model.pkl'
 print(f"Exporting complete pipeline to {model_filename}...")
 joblib.dump(best_model, model_filename)
 print("Pipeline successfully exported!")
 
 print("\nFinal Summary / Insights:")
 print("- A complete, reusable machine learning pipeline was constructed handling both numeric and categorical data.")
 print("- Hyperparameter tuning found the optimal tree depth and estimator count.")
 print("- The final model is saved as a joblib file, ready for direct production deployment without repeating preprocessing logic.")

if __name__ == "__main__":
 main()
