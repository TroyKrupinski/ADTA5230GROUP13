# Re-importing necessary libraries and redefining the dataset preparation steps due to code execution state reset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Reload the dataset
file_path = 'YOURDATASETADTAHERE'
nonprofit_data = pd.read_excel(file_path)

# Separating features and target variables
X = nonprofit_data.drop(['donr', 'damt'], axis=1)  # Excluding ID, donr, and damt
y_classification = nonprofit_data['donr']  # Target for classification model

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

# Splitting the data into training and testing sets for the classification model
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_processed, y_classification, test_size=0.2, random_state=42)

# Defining the models
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier()
mlp_model = MLPClassifier(random_state=42, max_iter=1000)

# Grid search parameters for tuning
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam']
}

# GridSearchCV for model tuning
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp = GridSearchCV(mlp_model, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)

# Fitting models
print("Fitting Random Forest...")
grid_rf.fit(X_train_class, y_train_class)

print("Fitting K-Nearest Neighbors...")
grid_knn.fit(X_train_class, y_train_class)

print("Fitting Multi-Layer Perceptron...")
grid_mlp.fit(X_train_class, y_train_class)

# Additional classifiers to include

logreg_model = LogisticRegression(random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)

svc_model = SVC(random_state=42)



# Grid search parameters for additional classifiers

param_grid_logreg = {

    'C': [0.01, 0.1, 1, 10],

    'solver': ['newton-cg', 'lbfgs', 'liblinear']

}



param_grid_dt = {

    'max_depth': [None, 10, 20, 30],

    'min_samples_split': [2, 5, 10]

}



param_grid_svc = {

    'C': [0.1, 1, 10],

    'kernel': ['linear', 'rbf', 'poly']

}



# GridSearchCV for additional classifiers

grid_logreg = GridSearchCV(logreg_model, param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1)

grid_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)

grid_svc = GridSearchCV(svc_model, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)



# Fitting additional models

print("Fitting Logistic Regression...")

grid_logreg.fit(X_train_class, y_train_class)



print("Fitting Decision Tree...")

grid_dt.fit(X_train_class, y_train_class)



print("Fitting Support Vector Classifier...")

grid_svc.fit(X_train_class, y_train_class)



# Retrieving best parameters and scores for each model

logreg_best_params, logreg_best_score = grid_logreg.best_params_, grid_logreg.best_score_

dt_best_params, dt_best_score = grid_dt.best_params_, grid_dt.best_score_

svc_best_params, svc_best_score = grid_svc.best_params_, grid_svc.best_score_



# TO BE IMPLEMENTED rf_best_params, rf_best_score, knn_best_params, knn_best_score, mlp_best_params, mlp_best_score,
logreg_best_params, logreg_best_score, dt_best_params, dt_best_score, svc_best_params, svc_best_score

