# Author: Troy Krupinski
# Original Date: 11/15/2023
# REFACTORED 11/28/2023

#FALL 2023
#ADTA 5230


# HOW TO USE:
# Install Python 3.8.5 or any version of Python 3
# Install the following libraries using pip install:
#   pandas
#   sklearn
#   matplotlib
#   seaborn
#   numpy
#   xlsxwriter will be used to export the data to an excel file, and will be needed later. Not implemented.
#   easygui will be used to display a message box.

# To go through file progression, exit out of each window / graph window to continue to next step




#TODO FIND AND ASSOCIATE IDS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.metrics import roc_curve, auc
from easygui import *


ifEDA = False
ifoutput = False
ifDescript = False

# message / information to be displayed on the screen
message = "Would you like to show the EDA?"
 
# title of the window
title = "Troy Krupinski"
 
# creating a yes no box
output = ynbox(message, title)

if output:
     
    # message / information to be displayed on the screen
    message = "Eda will be shown"
    ifEDA = True
    # title of the window
    title = "Troy Krupinski"
  
    # creating a message box
    msg = msgbox(message, title)
 
# if user pressed No
else:
     
    # message / information to be displayed on the screen
    message = "Eda will not be shown"
  
    # title of the window
    title = "Troy Krupinski"
  
    # creating a message box
    msg = msgbox(message, title)
     

print("Author: <Troy Krupinski>")

# Load the data
data = pd.read_excel('nonprofit.xlsx')

# --- Data Understanding and EDA ---
print("Descriptive Statistics:")
print(data.describe(include='all'))

print("Descriptive Statistics per column:") #Show basic statistical details for each column/variable
for column in data.columns:
    print(f"Column: {column}")
    print(data[column].describe())
    print()
#Renaming columns for better visualization
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

#Exploring missing values in the dataset
descriptive_stats = data.describe()

# Plot heatmap of the descriptive statistics

plt.figure(figsize=(12, 8))
sns.heatmap(descriptive_stats, annot=True, fmt=".2f", cmap='viridis')
plt.title('Descriptive Statistics Heatmap')
plt.show()
print("Missing values in each column:")
print(data.isnull().sum())

# Visualization of data distribution and relationships
if ifEDA:
    for col in data_renamed.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data_renamed[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    for col in data_renamed.select_dtypes(include='object').columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=data_renamed)
        plt.title(f'Countplot of {col}')
        plt.xticks(rotation=45)
        plt.show()

print("Calculating correlation matrix...")
# --- Data Preparation ---
# Handling missing values for numerical columns, dropping ID column
X = data.drop(['ID', 'donr', 'damt'], axis=1) 
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Separating target variables
y_classification = data['donr']  # Target for classification
y_regression = data['damt']  # Target for regression

# Identifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing with transformers
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

if 'ID' in X.columns:
    X = X.drop('ID', axis=1)

# Handling missing values for numerical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())
# Combining transformers into a preprocessor
print("Preprocessing data...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Applying the transformations
X_processed = preprocessor.fit_transform(X)

# Splitting the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_processed, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_processed, y_regression, test_size=0.2, random_state=42)


# Define classification models
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
# Dropping the 'ID' column if it exists
if 'ID' in X.columns:
    X = X.drop('ID', axis=1)

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


# Function to display feature importances
def display_feature_importances(model, feature_names):
    importances = None

    if hasattr(model, 'feature_importances_'):
        # For models that have feature_importances_ attribute
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For models like LogisticRegression that use coefficients
        importances = np.abs(model.coef_[0])
    else:
        # For models like MLPClassifier, which do not provide feature importances
        print(f"Model {type(model).__name__} does not provide feature importances.")
        return

    # Create a DataFrame for visualization if importances are available
    if importances is not None:
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importances.sort_values(by='importance', ascending=False, inplace=True)

        # Display the top features
        print("Top features in the model, " + type(model).__name__ + ":")
        print(feature_importances.head(10))

# Model fitting and tuning
best_models = {}
for model_name, model_info in {**classification_models, **regression_models}.items():
    
    y_target = y_train_class if 'Classifier' in model_name else y_train_reg
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy' if 'Classifier' in model_name else 'r2', n_jobs=-1)
    grid_search.fit(X_train_class if 'Classifier' in model_name else X_train_reg, y_target)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = {'model': best_model, 'score': grid_search.best_score_}

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Display feature importances for the best model
    display_feature_importances(best_model, feature_names)


# --- Evaluation --- 

# message / information to be displayed on the screen
message = "Would you like to show the ROC graph?"
 
# title of the window
title = "Troy Krupinski"
 
# creating a yes no box
output = ynbox(message, title)

if output:
     
    # message / information to be displayed on the screen
    message = "Evaluation will be shown"
    ifoutput = True
    # title of the window
    title = "Troy Krupinski"
  
    # creating a message box
    msg = msgbox(message, title)
 
# if user pressed No
else:
     
    # message / information to be displayed on the screen
    message = "Evaluation will not be shown"
  
    # title of the window
    title = "Troy Krupinski"
  
    # creating a message box
    msg = msgbox(message, title)
     

def evaluate_model(model, X_test, y_test, model_name, is_classification=True):
    y_pred = model.predict(X_test)
    if is_classification:
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Print classification metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"F1 Score: {f1:.6f}")
        print(f"AUC Score: {auc_score:.6f}")
        print()
        if(ifoutput):
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.6f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC Curve for {model_name}")

            # Add accuracy, precision, recall, and F1 score to the plot
            plt.text(0.5, 0.2, f"Accuracy: {accuracy:.6f}\nPrecision: {precision:.6f}\nRecall: {recall:.6f}\nF1 Score: {f1:.6f}", 
                    horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

            plt.legend(loc="lower right")
            plt.show()
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_matrix, annot=True, fmt="d")
            plt.title(f"Confusion Matrix: {model_name}")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()
    else:
        # Regression metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model: {model_name}")
        print(f"R^2 Score: {r2:.6f}, Mean Squared Error: {mse:.6f}")
        if(ifoutput):
            # Scatter plot of true values vs. predictions
            plt.scatter(y_test, y_pred)
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            plt.title(f"Prediction Scatter Plot: {model_name}")
            
            # Add statistical metrics to the plot
            plt.text(0.5, 0.9, f"R^2 Score: {r2:.2f}\nMean Squared Error: {mse:.2f}", 
                    horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            
            plt.show()

# Evaluating models
for model_name, model_info in best_models.items():
    is_classification = 'Classifier' in model_name
    X_test = X_test_class if is_classification else X_test_reg
    y_test = y_test_class if is_classification else y_test_reg
    evaluate_model(model_info['model'], X_test, y_test, model_name, is_classification=is_classification)

# Finding the best models
best_classification_model = max((model for model in best_models.items() if 'Classifier' in model[0]), key=lambda x: x[1]['score'])
best_regression_model = max((model for model in best_models.items() if 'Regressor' in model[0]), key=lambda x: x[1]['score'])
best_model = max(best_models.items(), key=lambda x: x[1]['score'])

# --- Deployment ---
# Load score data
score_data = pd.read_excel('nonprofit_score.xlsx')

# Ensure that the columns to be dropped exist in the DataFrame
columns_to_drop = ['id', 'donr', 'damt']
for column in columns_to_drop:
    if column in score_data.columns:
        score_data = score_data.drop(column, axis=1)

# Process the data
score_data_processed = preprocessor.transform(score_data)

# Apply models and export predictions
classification_predictions = best_classification_model[1]['model'].predict(score_data_processed)
regression_predictions = best_regression_model[1]['model'].predict(score_data_processed)



print("Best classification model:" + str(best_classification_model[1]['model']) + "regression model:" + str(best_regression_model[1]['model']))

score_data['Predicted_Donor'] = classification_predictions
score_data['Predicted_Donation_Amount'] = regression_predictions
score_data.to_csv('model_predictions.csv', index=False)
# --- Conclusion ---
# Summarize findings, discuss limitations, and suggest future work
print("Best Classification Model: " + best_classification_model[0] + " with score: " + str(best_classification_model[1]['score']))
print("Best Regression Model: " + best_regression_model[0] + " with score: " + str(best_regression_model[1]['score']))
print("Best model overall = " + best_classification_model[0] + " with score: " + str(best_classification_model[1]['score']) + " and the best regression model being " + best_regression_model[0] + " with score: " + str(best_regression_model[1]['score']))
# Calculate expected profit


# Constants
average_donation = 14.50  # average donation amount
mailing_cost = 2.00       # cost per mailing
response_rate = 0.10      # typical overall response rate,

# Assuming best_models is a dictionary with model names as keys and dictionaries with 'score' and 'model' as values

# Finding the best models
best_classification_model = max((model for model in best_models.items() if 'Classifier' in model[0]), key=lambda x: x[1]['score'])
best_regression_model = max((model for model in best_models.items() if 'Regressor' in model[0]), key=lambda x: x[1]['score'])
best_model = max(best_models.items(), key=lambda x: x[1]['score'])
print("Best model by score: " + best_model[0] + " with score: " + str(best_model[1]['score']) + " in percent form: " + str(best_model[1]['score']*100) + "%")
# Profit calculation function
# Profit calculation function

def get_model_precision(model, X_test, y_test):
    y_pred = model.predict(X_test)
    true_positives = sum((y_pred == 1) & (y_test == 1))
    false_positives = sum((y_pred == 1) & (y_test == 0))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

def calculate_profit(classification_predictions, regression_predictions, mailing_cost, precision=None, is_classification=True):
    """
    Calculate profit considering both classification and regression predictions.
    
    :param classification_predictions: Array of classification predictions (1 for donor, 0 for non-donor)
    :param regression_predictions: Array of regression predictions (predicted donation amounts)
    :param mailing_cost: Cost of mailing per individual
    :param precision: Precision of the classification model (optional)
    :param is_classification: Flag to indicate if the model is a classification model (default: True)
    :return: Total profit
    """
    if is_classification:
        if precision is None:
            print("Precision not provided for classification model. Calculating precision.")
            true_positives = sum(classification_predictions == 1)
            false_positives = sum(classification_predictions == 0)
            precision = true_positives / (true_positives + false_positives) if len(classification_predictions) > 0 else 0
        
        # Calculate total predicted donations from those classified as donors
        total_predicted_donations = sum(regression_predictions[classification_predictions == 1])

        # Calculate costs based on the number of mails sent
        total_mailing_cost = len(classification_predictions) * mailing_cost

        # Calculate profit
        profit = total_predicted_donations - total_mailing_cost
    else:
        # For regression models, simply sum up the predictions and subtract the mailing cost
        total_predicted_donations = sum(regression_predictions)
        profit = total_predicted_donations - len(regression_predictions) * mailing_cost
    
    return profit

# Calculate and print expected profit from the best model
for model_name, model_info in best_models.items():
    is_classification = 'Classifier' in model_name
    X_test = X_test_class if is_classification else X_test_reg
    y_test = y_test_class if is_classification else y_test_reg

    model_predictions = model_info['model'].predict(X_test)

    if is_classification:
        precision = get_model_precision(model_info['model'], X_test, y_test)

        # Get regression predictions for the same test set
        regression_model = best_regression_model[1]['model']
        regression_predictions_test = regression_model.predict(X_test_reg)  # Make sure this matches your test set

        profit = calculate_profit(model_predictions, regression_predictions_test, mailing_cost, precision, True)

    else:
        zero_classifications = np.zeros_like(model_predictions)

        profit = calculate_profit(zero_classifications, model_predictions, mailing_cost, is_classification=False)

    print(f"Expected profit from {model_name}: ${profit}")



# --- Deployment ---


# predictions = [...]  # This should be your list/array of predictions
# is_classification = True  # or False, depending on the type of model
# profit = calculate_profit(predictions, is_classification)


best_classification_model_name = best_classification_model[0]
best_regression_model_name = best_regression_model[0]
best_overall_model_name = best_model[0]

best_classification_test_predictions = best_classification_model[1]['model'].predict(X_test_class)
best_regression_test_predictions = best_regression_model[1]['model'].predict(X_test_reg)
is_best_model_classification = 'Classifier' in best_model[0]
# Best classification model profit
best_classification_precision = get_model_precision(best_classification_model[1]['model'], X_test_class, y_test_class)
best_classification_profit = calculate_profit(
    best_classification_test_predictions, 
    best_regression_test_predictions,  # Use regression test predictions
    mailing_cost, 
    precision=best_classification_precision, 
    is_classification=True
)

# Best regression model profit
best_regression_profit = calculate_profit(
    np.zeros_like(best_regression_test_predictions),  # Assuming non-donor for all
    best_regression_test_predictions, 
    mailing_cost, 
    is_classification=False
)

# Best overall model profit
if is_best_model_classification:
    best_model_test_predictions = best_classification_model[1]['model'].predict(X_test_class)
    best_model_precision = get_model_precision(best_classification_model[1]['model'], X_test_class, y_test_class)
    best_profit = calculate_profit(
        best_model_test_predictions,
        best_regression_test_predictions,  # Use regression test predictions
        mailing_cost,
        precision=best_model_precision,
        is_classification=True
    )
else:
    best_model_test_predictions = best_regression_model[1]['model'].predict(X_test_reg)
    best_profit = calculate_profit(
        np.zeros_like(best_model_test_predictions),  # Assuming non-donor for all
        best_model_test_predictions,
        mailing_cost,
        is_classification=False
    )

# Corrected print statements
print(f"Expected profit from the best classification model ({best_classification_model_name}): ${best_classification_profit}")
print(f"Expected profit from the best regression model ({best_regression_model_name}): ${best_regression_profit}")
print(f"Expected profit from the best overall model ({best_overall_model_name}): ${best_profit}")

# Load score data
score_data = pd.read_excel('nonprofit_score.xlsx')

# Drop the columns that were not included in the training set
if 'ID' in score_data.columns:
    score_data = score_data.drop(['ID'], axis=1)

# Apply the same preprocessing to score_data as was applied to the training data
score_data_processed = preprocessor.transform(score_data)

# Get the best models
best_classification_model = max((model for model in best_models.items() if 'Classifier' in model[0]), key=lambda x: x[1]['score'])[1]['model']
best_regression_model = max((model for model in best_models.items() if 'Regressor' in model[0]), key=lambda x: x[1]['score'])[1]['model']

# Make predictions using the processed score data
score_data['DONR'] = best_classification_model.predict(score_data_processed)
score_data['DAMT'] = best_regression_model.predict(score_data_processed)

# Export to CSV
score_data.to_csv('nonprofit_score.csv', index=False)

print("Model development and evaluation completed. Exported to CSV file.")

#IMPLEMENT THE FOLLOWING:













