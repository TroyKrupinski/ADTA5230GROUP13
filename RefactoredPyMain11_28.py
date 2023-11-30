# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import easygui

# Flags for EDA and output
ifEDA = easygui.ynbox("Would you like to show the EDA?", "Troy Krupinski")
ifoutput = easygui.ynbox("Would you like to show the evaluation?", "Troy Krupinski")

# Load the data
data = pd.read_excel('nonprofit.xlsx')

# Data Understanding and EDA
if ifEDA:
    print("Descriptive Statistics:")
    print(data.describe(include='all'))

    for col in data.columns:
        print(f"\nColumn: {col}")
        print(data[col].describe())

    # Visualization of data distribution and relationships
    for col in data.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    for col in data.select_dtypes(include='object').columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=data)
        plt.title(f'Countplot of {col}')
        plt.xticks(rotation=45)
        plt.show()

# Data Preparation
X = data.drop(['ID', 'donr', 'damt'], axis=1)
y_classification = data['donr']
y_regression = data['damt']
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])
X_processed = preprocessor.fit_transform(X)

# Splitting the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_processed, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_processed, y_regression, test_size=0.2, random_state=42)

# Model Definitions
classification_models = {
    # Your classification models with parameters
}
regression_models = {
    # Your regression models with parameters
}

# Function to find the best model
def find_best_model(models, X_train, y_train, is_classification=True):
    best_score = -np.inf
    best_model = None
    for name, model_info in models.items():
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy' if is_classification else 'r2')
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
    return best_model

# Finding the best models
best_clf_model = find_best_model(classification_models, X_train_class, y_train_class, is_classification=True)
best_reg_model = find_best_model(regression_models, X_train_reg, y_train_reg, is_classification=False)

# Evaluation function
# ... [previous code remains the same]

# Define the profit calculation function
def calculate_profit(classification_preds, regression_preds, average_donation=14.50, mailing_cost=2.00):
    predicted_donors = classification_preds == 1
    expected_donations = regression_preds[predicted_donors].sum()
    total_mailing_cost = len(classification_preds) * mailing_cost
    profit = expected_donations - total_mailing_cost
    return profit

# Define the model evaluation function
def evaluate_model(model, X_test, y_test, model_name, is_classification=True):
    y_pred = model.predict(X_test)
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Additional classification evaluation metrics can be added here
        print(f"Classification Model: {model_name}\nAccuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        # Additional regression evaluation metrics can be added here
        print(f"Regression Model: {model_name}\nR2 Score: {r2}, Mean Squared Error: {mse}")

# Evaluate models and calculate profits
for model_name, model_info in best_models.items():
    is_classification = 'Classifier' in model_name
    X_test = X_test_class if is_classification else X_test_reg
    y_test = y_test_class if is_classification else y_test_reg

    evaluate_model(model_info['model'], X_test, y_test, model_name, is_classification)

    if is_classification:
        classification_predictions = model_info['model'].predict(X_test_class)
        regression_predictions = best_regression_model[1]['model'].predict(X_test_reg)
        profit = calculate_profit(classification_predictions, regression_predictions)
        print(f"Expected profit from {model_name}: ${profit:.2f}")

# ... [rest of your code for deployment and conclusion]


# Deployment
# Load score data, preprocess, apply models, and calculate profits
# ...

print("Model development and evaluation completed. Exported to CSV file.")
