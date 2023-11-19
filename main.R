# Required Libraries
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(nnet)      # For Neural Network
library(class)     # For KNN
library(rpart)     # For Decision Trees
library(readxl)    # For reading Excel files

# Setting Working Directory and Checking File Existence
setwd('C:/Users/dunke/Downloads/')
train_file_path <- "nonprofit.xlsx"
score_file_path <- "nonprofit_score.xlsx"

if (!file.exists(train_file_path) || !file.exists(score_file_path)) {
    stop("File not found")
}

# Loading and Preparing Data
train_data <- read_excel(train_file_path, sheet = 1) %>% as.data.frame()
score_data <- read_excel(score_file_path, sheet = 1) %>% as.data.frame()

# Data Cleaning
train_data <- na.omit(train_data)

# Convert appropriate variables to factors
train_data$region <- as.factor(train_data$region)
train_data$ownd <- as.factor(as.character(train_data$ownd))
train_data$sex <- as.factor(as.character(train_data$sex))

# Convert target variables
train_data$donr <- as.factor(train_data$donr)
train_data$damt <- as.numeric(train_data$damt)

# Remove 'ID' column if it exists
if("ID" %in% names(train_data)) {
    train_data <- select(train_data, -ID)
}


# Distribution of Key Variables #EDA
hist(train_data$wlth, main = "Wealth Rating Distribution", xlab = "Wealth Rating")
hist(train_data$hv, main = "Home Value Distribution", xlab = "Home Value in $ Thousands")
boxplot(train_data$incmed ~ train_data$region, main = "Median Income by Region", ylab = "Income in $ Thousands")
boxplot(train_data$gifa ~ train_data$sex, main = "Average Gift Amount by Gender", ylab = "Average Gift Amount")

# Splitting Data into Training and Testing Sets
set.seed(123)
splitIndex <- createDataPartition(train_data$donr, p = .80, list = FALSE)
training_set <- train_data[splitIndex,]
testing_set <- train_data[-splitIndex,]

# Define Predictors
predictors <- c("region", "ownd", "kids", "inc", "sex", "wlth", "hv", "incmed", "incavg", "low", "npro", "gifdol", "gifl", "gifr", "mdon", "lag", "gifa")

# Handle any NA values in predictors
training_set <- training_set %>% na.omit()
testing_set <- testing_set %>% na.omit()

# Classification Models for DONR
response_class <- "donr"

# Random Forest (Classification)
rf_model <- train(training_set[, predictors], training_set[[response_class]],
                  method = "rf", trControl = trainControl(method = "cv", number = 10))

# Neural Network
nn_model <- nnet(donr ~ ., data = training_set[, c(predictors, response_class)], size = 5, maxit = 200)

# K-Nearest Neighbors
# Ensure all predictors are numeric for KNN
training_set_knn <- training_set %>% mutate_if(is.factor, as.numeric)
testing_set_knn <- testing_set %>% mutate_if(is.factor, as.numeric)

knn_model <- train(training_set_knn[, predictors], training_set_knn[[response_class]],
                   method = "knn", trControl = trainControl(method = "cv", number = 10))

# Regression Models for DAMT
response_reg <- "damt"

# Linear Regression
lm_model <- lm(damt ~ ., data = training_set[, c(predictors, response_reg)])

# Decision Tree
tree_model <- rpart(damt ~ ., data = training_set[, c(predictors, response_reg)])

training_set_knn <- training_set %>% mutate_if(is.factor, as.numeric)  # You may use one-hot encoding instead
testing_set_knn <- testing_set %>% mutate_if(is.factor, as.numeric)    # You may use one-hot encoding instead

# Check for NAs in the dataset and handle them
training_set_knn <- na.omit(training_set_knn)
testing_set_knn <- na.omit(testing_set_knn)

# K-Nearest Neighbors (Classification)
# Ensure you are using the classification method 'knn' and not 'knnreg'
knn_model <- train(x = training_set_knn[, predictors], y = training_set_knn[[response_class]],
                   method = "knn", trControl = trainControl(method = "cv", number = 10))

# Evaluation and Output Sections (Remain Unchanged)



# Evaluation for Classification Models
predictions_rf <- predict(rf_model, testing_set)
confMatrix_rf <- confusionMatrix(predictions_rf, testing_set$donr)
misclassRate_rf <- 1 - confMatrix_rf$overall['Accuracy']

predictions_nn <- predict(nn_model, testing_set[, predictors])
confMatrix_nn <- confusionMatrix(as.factor(ifelse(predictions_nn > 0.5, 1, 0)), testing_set$donr)
misclassRate_nn <- 1 - confMatrix_nn$overall['Accuracy']

predictions_knn <- predict(knn_model, testing_set)
confMatrix_knn <- confusionMatrix(predictions_knn, testing_set$donr)
misclassRate_knn <- 1 - confMatrix_knn$overall['Accuracy']

# Evaluation for Regression Models
predictions_lm <- predict(lm_model, testing_set)
rmse_lm <- RMSE(predictions_lm, testing_set$damt)

predictions_tree <- predict(tree_model, testing_set)
rmse_tree <- RMSE(predictions_tree, testing_set$damt)

predictions_knn_reg <- predict(knn_reg_model, testing_set)
rmse_knn_reg <- RMSE(predictions_knn_reg, testing_set$damt)

# Output Summary
summary_list <- list(
    Random_Forest_Accuracy = 1 - misclassRate_rf,
    Neural_Network_Accuracy = 1 - misclassRate_nn,
    KNN_Accuracy = 1 - misclassRate_knn,
    Linear_Regression_RMSE = rmse_lm,
    Decision_Tree_RMSE = rmse_tree,
    KNN_Regression_RMSE = rmse_knn_reg
)

# Summary of Model Performances
performance_summary <- data.frame(
  Model = c("Random Forest", "Neural Network", "KNN", "Linear Regression", "Decision Tree", "KNN Regression"),
  Misclassification_Rate = c(misclassRate_rf, misclassRate_nn, misclassRate_knn, NA, NA, NA),
  RMSE = c(NA, NA, NA, rmse_lm, rmse_tree, rmse_knn_reg)
)

# Business Profitability Evaluation
calculate_profit <- function(predictions, actual, cost_per_mail = 2, avg_donation = 14.50) {
  profit_per_response = avg_donation - cost_per_mail
  total_profit = sum(predictions == actual & actual == 1) * profit_per_response - sum(predictions == 1) * cost_per_mail
  return(total_profit)
}

profit_rf <- calculate_profit(predictions_rf, testing_set$donr)
profit_nn <- calculate_profit(predictions_nn, testing_set$donr)
profit_knn <- calculate_profit(predictions_knn, testing_set$donr)

# Output Evaluation Results
cat("Misclassification Rates:\n")
cat("Random Forest: ", misclassRate_rf, "\nNeural Network: ", misclassRate_nn, "\nKNN: ", misclassRate_knn, "\n\n")

cat("Root Mean Squared Errors for Regression Models:\n")
cat("Linear Regression: ", rmse_lm, "\nDecision Tree: ", rmse_tree, "\nKNN Regression: ", rmse_knn_reg, "\n\n")

cat("Business Profit Evaluation:\n")
cat("Profit with Random Forest: ", profit_rf, "\nProfit with Neural Network: ", profit_nn, "\nProfit with KNN: ", profit_knn, "\n")