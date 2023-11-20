# Author: Troy Krupinski
# Original Date: 11/15/2022
# HOW TO USE:
# Install Python 3.8.5 or any version of Python 3
# Install the following libraries:
#   pandas
#   sklearn
#   matplotlib
#   seaborn
#   numpy
#   xlsxwriter will be used to export the data to an excel file, and will be needed later. Not implemented.

# To go through file progression, exit out of each window / graph window to continue to next step
# For performance, comment out lines 65-73, as it's just EDA



#TODO: ADD CLUSTERING
#TODO: FIX LOGISTIC REGRESSION TO CLASSIFIER
#TODO ADD BETTER MODEL COMPARISONS
#TODO ADD CLUSTERING!!!
#TODO TOUCH UP VISUALS
#TODO WRITE TO EXCEL DOCUMENT


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


is_classification = True
# --- Introduction and Business Understanding ---
print("Author: <Troy Krupinski>")

# --- Data Understanding and EDA ---
# Load the data
file_path = 'nonprofit.xlsx'
data = pd.read_excel(file_path)

# Basic descriptive statistics
print("Descriptive Statistics:")
for column in data.columns:
    print(f"Column: {column}")
    print(data[column].describe())
    print()

import seaborn as sns
import matplotlib.pyplot as plt

# Renaming columns for better readability
data_renamed = data.rename(columns={
    'ownd': 'Homeowner (1/0)',
    'kids': 'Number of Children',
    'inc': 'Household Income Category',
    'sex': 'Gender (0: Male, 1: Female)',
    'wlth': 'Wealth Rating (0-9)',
    'hv': 'Avg Home Value ($K)',
    'incmed': 'Median Family Income ($K)',
    'incavg': 'Average Family Income ($K)',
    'low': '% Low Income',
    'npro': 'Lifetime Promotions',
    'gifdol': 'Lifetime Gift Amount ($)',
    'gifl': 'Largest Gift Amount ($)',
    'gifr': 'Most Recent Gift Amount ($)',
    'mdon': 'Months Since Last Donation',
    'lag': 'Months Between First & Second Gift',
    'gifa': 'Average Gift Amount ($)',
    'donr': 'Donor (1/0)',
    'damt': 'Donation Amount ($)'
})

# Selecting a subset of variables for better visualization
selected_columns = ['Homeowner (1/0)', 'Number of Children', 'Household Income Category', 
                    'Gender (0: Male, 1: Female)', 'Wealth Rating (0-9)', 'Avg Home Value ($K)', 
                    'Median Family Income ($K)', 'Lifetime Promotions', 'Lifetime Gift Amount ($)', 
                    'Donor (1/0)', 'Donation Amount ($)']

# Creating the pairplot
plt.figure(figsize=(12, 12))
sns.pairplot(data_renamed[selected_columns], hue='Donor (1/0)')
plt.show()


# ... [rest of the script continues without duplication] ...

# Function to evaluate and visualize model performance

# --- Data Preparation ---
# Separating features and target variables
X = data.drop(['ID', 'donr', 'damt'], axis=1)  # Excluding ID, donr, and damt
y_classification = data['donr']  # Target for classification model
y_regression = data['damt']  # Target for regression model

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Creating transformers for categorical and numerical features
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

# Combining transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Applying the transformations
X_processed = preprocessor.fit_transform(X)

# Splitting the data for classification and regression models
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_processed, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_processed, y_regression, test_size=0.2, random_state=42)

# --- Modeling ---
# Define the models and their grid search parameters
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}  # Add parameters as needed, maybe normalize in future - TK
    },    
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {'C': [0.1, 1, 10], 'solver': ['lbfgs']}
    },
    'MLPClassifier': {
        'model': MLPClassifier(random_state=42),
        'params': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'max_iter': [200, 300]}
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance']}
    }
}


# Placeholder for best models and scores
best_models = {}

# Model fitting and tuning (to be executed in your environment)
for model_name, model_info in models.items():
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_class, y_train_class)
    best_models[model_name] = {'model': grid_search.best_estimator_, 'score': grid_search.best_score_}

# --- Modeling and Evaluation ---
# Function to evaluate and visualize classification model performance
# Function to evaluate and visualize model performance
def evaluate_model(model, X_test, y_test, model_name, is_classification=True):
    y_pred = model.predict(X_test)

    if is_classification:
        # Classification metrics
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # ROC curve and AUC score
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC Curve for {model_name}")

            # Add accuracy, precision, recall, and F1 score to the plot
            plt.text(0.5, 0.2, f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}", 
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

            plt.legend(loc="lower right")
            plt.show()

    else:
        # Regression metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f"Scatter Plot for {model_name}")
        plt.show()

        print(f"R^2 Score for {model_name}: {r2:.2f}")
        print(f"Mean Squared Error for {model_name}: {mse:.2f}")

# ... [rest of the script remains the same] ...

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    if 'Classifier' in model_name:
        evaluate_model(model_info['model'], X_test_class, y_test_class, model_name, is_classification=True)
    else:
        evaluate_model(model_info['model'], X_test_reg, y_test_reg, model_name, is_classification=False)
