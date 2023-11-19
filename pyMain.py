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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# --- Introduction and Business Understanding ---
print("Author:  <Troy Krupinski>")

# --- Data Understanding and EDA ---
# Load the data
file_path = 'nonprofit.xlsx'
data = pd.read_excel(file_path)

# Basic descriptive statistics
print(data.describe())

# Visualizing distributions of key variables
sns.pairplot(data[['ownd', 'kids', 'inc', 'sex', 'wlth', 'hv', 'incmed', 'incavg', 'low', 'npro', 'gifdol', 'gifl', 'gifr', 'mdon', 'lag', 'gifa', 'donr', 'damt']])
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

    # Other statistics
    print(f"AUC Score for {model_name}: {auc_score:.2f}")
    # Add any other statistics you want to display here

# ...

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_classification_model(model_info['model'], X_test_class, y_test_class, model_name)

# Evaluating and visualizing model performance
for model_name, model_info in best_models.items():
    evaluate_classification_model(model_info['model'], X_test_class, y_test_class, model_name)

