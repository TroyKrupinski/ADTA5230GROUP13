#Author: Troy Krupinski
#Original Date: 11/15/2022
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import unittest
from unittest.mock import MagicMock
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

# --- Introduction and Business Understanding ---
print("Author:  <Troy Krupinski>")

# --- Data Understanding and EDA ---
# Load the data
file_path = 'nonprofit.xlsx'
data = pd.read_excel(file_path)

# Basic descriptive statistics
print(data.describe())

# Visualizing distributions of key variables
# Pairplot with appropriate plots for each variable
sns.pairplot(data[['ownd', 'kids', 'inc', 'sex', 'wlth', 'hv', 'incmed', 'incavg', 'low', 'npro', 'gifdol', 'gifl', 'gifr', 'mdon', 'lag', 'gifa', 'donr', 'damt']],
             diag_kind='hist',  # Histogram for numerical variables
             hue='donr',  # Color points based on the 'donr' variable
             plot_kws={'alpha': 0.6, 's': 80},  # Adjust transparency and point size
             diag_kws={'alpha': 0.8},  # Adjust transparency of histograms
             markers=['o', 's'],  # Use different markers for 'donr' classes
             palette={0: 'blue', 1: 'red'})  # Set colors for 'donr' classes

plt.show()

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
        'params': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

# Placeholder for best models and scores
best_models = {}

# Model fitting and tuning (to be executed in your environment)
for model_name, model_info in models.items():
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_class, y_train_class)
    best_models[model_name] = {'model': grid_search.best_estimator_, 'score': grid_search.best_score_}


# Importing necessary libraries
# (The previous import statements remain the same)

# ... (Previous sections of the script remain unchanged)

# --- Modeling and Evaluation ---
# (Model definition and grid search remains the same)

# Function to evaluate and visualize classification model performance
def evaluate_classification_model(model, X_test, y_test, model_name):
    # Predictions
    y_pred = model.predict(X_test)

    # Classification report
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # going to implement roc curve and auc score
# Importing necessary libraries

# ...

# Function to evaluate and visualize classification model performance
# ...

# Function to evaluate and visualize classification and regression model performance
def evaluate_model(model, X_test, y_test, model_name, is_classification=True):
    if is_classification:
        # Predictions
        y_pred = model.predict(X_test)

        # Classification report
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # ROC curve and AUC score
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="lower right")
        plt.show()
        def evaluate_model(model, X_test, y_test, model_name, is_classification=True):
            if is_classification:
                # Predictions
                y_pred = model.predict(X_test)

                # Classification report
                print(f"Classification Report for {model_name}:")
                print(classification_report(y_test, y_pred))

                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                sns.heatmap(conf_matrix, annot=True, fmt='d')
                plt.title(f"Confusion Matrix for {model_name}")
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.show()

                # ROC curve and AUC score
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                # Plot ROC curve with statistics
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f"ROC Curve for {model_name}")
                plt.legend(loc="lower right")

                # Other statistics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Add statistics to the plot
                plt.text(0.5, 0.2, f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}", 
                         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

                plt.show()
            else:
                # Predictions
                y_pred = model.predict(X_test)

                # R^2 score
                r2 = r2_score(y_test, y_pred)

                # Other statistics
                mse = mean_squared_error(y_test, y_pred)
#                rmse = np.sqrt(mse)

                print(f"R^2 Score for {model_name}: {r2:.2f}")
                print(f"Mean Squared Error for {model_name}: {mse:.2f}")
                
                # Plot scatter plot with statistics
                plt.scatter(y_test, y_pred)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')

                # Root Mean Squared Error
#                print(f"Root Mean Squared Error for {model_name}: {rmse:.2f}")
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f"Scatter Plot for {model_name}")

                # Add statistics to the plot
                plt.text(0.5, 0.2, f"R^2 Score: {r2:.2f}\nMean Squared Error: {mse:.2f}", 
                         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

                plt.show()
      #  print(f"Root Mean Squared Error for {model_name}: {rmse:.2f}")

# ...

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_model(model_info['model'], X_test_class, y_test_class, model_name, is_classification=True)

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_model(model_info['model'], X_test_reg, y_test_reg, model_name, is_classification=False)

# ...

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_classification_model(model_info['model'], X_test_class, y_test_class, model_name)

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_classification_model(model_info['model'], X_test_class, y_test_class, model_name)

