# Author: Troy Krupinski
# Original Date: 11/15/2023
# REFACTORED 11/28/2023
# Project: Non-Profit Organization Donor Prediction and Analysis
# Course: ADTA 5230, Fall 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, confusion_matrix
from easygui import ynbox, msgbox

# Constants
AVERAGE_DONATION = 14.50
MAILING_COST = 2.00

# Load Data
data = pd.read_excel('nonprofit.xlsx')
data.drop('ID', axis=1, inplace=True) # ID is not a predictor

# Exploratory Data Analysis
if ynbox("Would you like to show the EDA?", "EDA Confirmation"):
    print("Descriptive Statistics:\n", data.describe(include='all'))
    for col in data.columns:
        plt.figure(figsize=(6, 4))
        if data[col].dtype == 'object':
            sns.countplot(x=col, data=data)
        else:
            sns.histplot(data[col], kde=True)
        plt.title(f'Distribution/Count of {col}')
        plt.show()

# Data Preparation
X = data.drop(['ID', 'donr', 'damt'], axis=1) 
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
y_classification = data['donr']  # Target for classification
y_regression = data['damt']  # Target for regression

# Handling missing values in numeric columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

# Handling missing values in non-numeric columns
# Example: Fill with 'missing' or use the most frequent value
non_numerical_cols = X.select_dtypes(include=['object']).columns
for col in non_numerical_cols:
    X[col] = X[col].fillna('missing')

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
    ('cat', OneHotEncoder(drop='first'), X.select_dtypes(include=['object', 'category']).columns)
])
X_processed = preprocessor.fit_transform(X)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_processed, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_processed, y_regression, test_size=0.2, random_state=42)

# Model Definitions
classification_models = {
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5]}
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=200),  # Increase max_iter
        'params': {'C': [0.1, 1, 10]}
    },
    'MLPClassifier': {
        'model': MLPClassifier(random_state=42),
        'params': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7]}
        # Add more parameters here...neighbors 
    },
    'SVC': {
        'model': SVC(random_state=42),
        'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [10, 20, None]}
    },
    # Add more classifiers here...
}

# Define regression models, might remove
regression_models = {
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5]}
    },
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}
    },
    # Add more regression models here...
}

# Model Training and Evaluation
best_models = {}
for model_name, model_info in {**classification_models, **regression_models}.items():
    y_target = y_train_class if 'Classifier' in model_name else y_train_reg
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy' if 'Classifier' in model_name else 'r2', n_jobs=-1)
    grid_search.fit(X_train_class if 'Classifier' in model_name else X_train_reg, y_target)
    best_model = grid_search.best_estimator_
    best_models[model_name] = {'model': best_model, 'score': grid_search.best_score_}

# Profit Calculation Function
def calculate_profit(predictions, average_donation, mailing_cost, precision=None, is_classification=True):
    if is_classification:
        if precision is None:
            return None
        true_donors = sum(predictions) * precision
        profit = true_donors * average_donation - len(predictions) * mailing_cost
    else:
        total_donations = sum(predictions)
        profit = total_donations - len(predictions) * mailing_cost
    return profit

# Evaluate and Calculate Profits for Each Model
for model_name, model_info in best_models.items():
    is_classification = 'Classifier' in model_name
    X_test = X_test_class if is_classification else X_test_reg
    y_test = y_test_class if is_classification else y_test_reg
    predictions = model_info['model'].predict(X_test)

    if is_classification:
        precision = precision_score(y_test, predictions)
        profit = calculate_profit(predictions, AVERAGE_DONATION, MAILING_COST, precision, True)
    else:
        profit = calculate_profit(predictions, AVERAGE_DONATION, MAILING_COST, is_classification=False)

    print(f"Expected profit from {model_name}: ${profit}")

# Deployment
score_data = pd.read_excel('nonprofit_score.xlsx')
score_data_processed = preprocessor.transform(score_data.drop(['id', 'donr', 'damt'], axis=1))
best_classification_model = max((model for model in best_models.items() if 'Classifier' in model[0]), key=lambda x: x[1]['score'])[1]['model']
best_regression_model = max((model for model in best_models.items() if 'Regressor' in model[0]), key=lambda x: x[1]['score'])[1]['model']
score_data['DONR'] = best_classification_model.predict(score_data_processed)
score_data['DAMT'] = best_regression_model.predict(score_data_processed)
score_data.to_csv('model_predictions.csv', index=False)

print("Model development and evaluation completed. Predictions exported to 'model_predictions.csv'.")

# Conclusion
# Summarize findings, discuss limitations, and future work
