import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# Constants
average_donation = 14.50
mailing_cost = 2.00
response_rate = 0.10

# Load the data
data = pd.read_excel('nonprofit.xlsx')

# Data Preprocessing
X = data.drop(['ID', 'donr', 'damt'], axis=1)
y_class = data['donr']
y_reg = data['damt']

# Handling missing values and standardizing numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Encoding categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Model Pipelines
# Classification model pipeline
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())])

# Regression model pipeline
reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', GradientBoostingRegressor())])

# Splitting data for classification and regression models
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Fitting the classification model
clf_pipeline.fit(X_train_class, y_train_class)

# Fitting the regression model
reg_pipeline.fit(X_train_reg, y_train_reg)

# Function to calculate expected profit
def calculate_expected_profit(model, X, y_true, is_classification):
    y_pred = model.predict(X)
    if is_classification:
        precision = precision_score(y_true, y_pred)
        profit = (precision * average_donation - mailing_cost) * len(y_pred)
    else:
        predicted_donation = np.sum(y_pred)
        profit = predicted_donation - mailing_cost * len(y_pred)
    return profit

# Evaluating Classification Model
y_pred_class = clf_pipeline.predict(X_test_class)
print(f"Classification Model Accuracy: {accuracy_score(y_test_class, y_pred_class)}")
print(f"Expected Profit (Classification): {calculate_expected_profit(clf_pipeline, X_test_class, y_test_class, True)}")

# Evaluating Regression Model
y_pred_reg = reg_pipeline.predict(X_test_reg)
print(f"Regression Model R2 Score: {r2_score(y_test_reg, y_pred_reg)}")
print(f"Expected Profit (Regression): {calculate_expected_profit(reg_pipeline, X_test_reg, y_test_reg, False)}")

# Save predictions for deployment
# Assuming 'nonprofit_score.xlsx' contains the scoring dataset
score_data = pd.read_excel('nonprofit_score.xlsx')
score_data_processed = preprocessor.transform(score_data)
score_data['Predicted_Donor'] = clf_pipeline.predict(score_data_processed)
score_data['Predicted_Donation_Amount'] = reg_pipeline.predict(score_data_processed)
score_data.to_excel('model_predictions.xlsx', index=False)
