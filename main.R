#Author:
#Troy Krupinski
#ADTA 5230 Group 13

# Required Libraries, you will need to install thesse.
#Install R and RStudio. Go to RStudio's website if you haven't installed it already.
#All of the EDA is taken care of.


# Required Libraries
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(nnet)      # For Neural Network
library(class)     # For KNN
library(rpart)     # For Decision Trees
library(readxl)    # For reading Excel files
library(tidyr)     # For one-hot encoding
library(pROC)      # For ROC curve analysis
library(plotly)    # For interactive plots
library(RColorBrewer) # For additional color options in plots (optional)

# These will install the packages necessary.
required_packages <- c("caret", "dplyr", "ggplo2", "randomForest", "nnet", "class", "rpart",
                       "readxl", "tidyr", "pROC", "plotly", "RColorBrewer")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the installed packages
lapply(required_packages, require, character.only = TRUE)



# Setting Working Directory and Checking File Existence
setwd('C:/Users/dunke/Downloads/') #Add your own path here that contains the data you got from Canvas
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

# EDA for all variables

# Wealth Rating Distribution
hist(train_data$wlth, main = "Wealth Rating Distribution", xlab = "Wealth Rating")

# Home Value Distribution
hist(train_data$hv, main = "Home Value Distribution", xlab = "Home Value in $ Thousands")

# Median Income by Region
boxplot(train_data$incmed ~ train_data$region, main = "Median Income by Region", ylab = "Income in $ Thousands")

# Average Gift Amount by Gender
boxplot(train_data$gifa ~ train_data$sex, main = "Average Gift Amount by Gender", ylab = "Average Gift Amount")

# Number of Children Distribution
hist(train_data$kids, main = "Number of Children Distribution", xlab = "Number of Children")

# Household Income Distribution
hist(train_data$inc, main = "Household Income Distribution", xlab = "Income Category")

# Wealth Rating Distribution
hist(train_data$wlth, main = "Wealth Rating Distribution", xlab = "Wealth Rating")

# Average Home Value Distribution
hist(train_data$hv, main = "Average Home Value Distribution", xlab = "Home Value in $ Thousands")

# Median Family Income Distribution
hist(train_data$incmed, main = "Median Family Income Distribution", xlab = "Median Income in $ Thousands")

# Average Family Income Distribution
hist(train_data$incavg, main = "Average Family Income Distribution", xlab = "Average Income in $ Thousands")

# Low Income Percent Distribution
hist(train_data$low, main = "Low Income Percent Distribution", xlab = "Low Income Percent")

# Lifetime Promotions Distribution
hist(train_data$npro, main = "Lifetime Promotions Distribution", xlab = "Number of Promotions")

# Dollar Amount of Lifetime Gifts Distribution
hist(train_data$gifdol, main = "Lifetime Gifts Amount Distribution", xlab = "Gift Amount")

# Dollar Amount of Largest Gift Distribution
hist(train_data$gifl, main = "Largest Gift Amount Distribution", xlab = "Largest Gift Amount")

# Dollar Amount of Most Recent Gift Distribution
hist(train_data$gifr, main = "Most Recent Gift Amount Distribution", xlab = "Most Recent Gift Amount")

# Number of Months Since Last Donation Distribution
hist(train_data$mdon, main = "Months Since Last Donation Distribution", xlab = "Months Since Last Donation")

# Number of Months Between First and Second Gift Distribution
hist(train_data$lag, main = "Months Between First and Second Gift", xlab = "Months Between Gifts")

# Average Gift Amount Distribution
hist(train_data$gifa, main = "Average Gift Amount Distribution", xlab = "Average Gift Amount")

# Splitting Data into Training and Testing Sets
set.seed(123)
splitIndex <- createDataPartition(train_data$donr, p = .80, list = FALSE)
training_set <- train_data[splitIndex,]
testing_set <- train_data[-splitIndex,]

# One-Hot Encoding for Categorical Variables
training_set <- training_set %>% mutate_if(is.factor, as.factor) %>%
                pivot_wider(names_from = region, values_from = region,
                            values_fn = length, values_fill = list(region = 0)) %>%
                mutate(across(everything(), as.numeric))
testing_set <- testing_set %>% mutate_if(is.factor, as.factor) %>%
              pivot_wider(names_from = region, values_from = region,
                          values_fn = length, values_fill = list(region = 0)) %>%
              mutate(across(everything(), as.numeric))

# Define Predictors
predictors <- names(training_set)[!names(training_set) %in% c("donr", "damt")]

# Classification Models for DONR
response_class <- "donr"

# Random Forest (Classification)
rf_model <- train(training_set[, predictors], training_set[[response_class]],
                  method = "rf", trControl = trainControl(method = "cv", number = 10))

# Neural Network
nn_model <- nnet(donr ~ ., data = training_set[, c(predictors, response_class)], size = 5, maxit = 200)

# K-Nearest Neighbors (Classification)
knn_model <- train(x = training_set[, predictors], y = training_set$donr,
                   method = "knn", trControl = trainControl(method = "cv", number = 10))

# Regression Models for DAMT
response_reg <- "damt"

# Linear Regression
lm_model <- lm(damt ~ ., data = training_set[, c(predictors, response_reg)])

# Decision Tree (Regression)
tree_model <- rpart(damt ~ ., data = training_set[, c(predictors, response_reg)])

# Evaluation for Classification Models
#CALCULATE PROFIT FUNCTION ----- THIS FUNCTION IS THE BASIS OF THE PROJECT
calculate_profit <- function(predictions, actual, cost_per_mail = 2, avg_donation = 14.50) {
  profit_per_response = avg_donation - cost_per_mail
  total_profit = sum(predictions == actual & actual == 1) * profit_per_response - sum(predictions == 1) * cost_per_mail
  return(total_profit)
}
# ---------------------------------------------------------------------------------------
# Ensure Factor Levels for Classification Predictions
testing_set$donr <- factor(testing_set$donr, levels = c("0", "1"))
predictions_rf <- factor(predict(rf_model, testing_set[, predictors]), levels = c("0", "1"))
predictions_nn <- predict(nn_model, testing_set[, predictors])
predictions_nn <- factor(ifelse(predictions_nn > 0.5, "1", "0"), levels = c("0", "1"))
predictions_knn <- factor(predict(knn_model, testing_set[, predictors]), levels = c("0", "1"))

# Confusion Matrix and Profit Calculation for Classification Models
confMatrix_rf <- confusionMatrix(predictions_rf, testing_set$donr)
misclassRate_rf <- 1 - confMatrix_rf$overall['Accuracy']
profit_rf <- calculate_profit(predictions_rf, testing_set$donr)

confMatrix_nn <- confusionMatrix(predictions_nn, testing_set$donr)
misclassRate_nn <- 1 - confMatrix_nn$overall['Accuracy']
profit_nn <- calculate_profit(predictions_nn, testing_set$donr)

confMatrix_knn <- confusionMatrix(predictions_knn, testing_set$donr)
misclassRate_knn <- 1 - confMatrix_knn$overall['Accuracy']
profit_knn <- calculate_profit(predictions_knn, testing_set$donr)

# Evaluation for Regression Models
predictions_lm <- predict(lm_model, testing_set)
rmse_lm <- RMSE(predictions_lm, testing_set$damt)

predictions_tree <- predict(tree_model, testing_set)
rmse_tree <- RMSE(predictions_tree, testing_set$damt)



# Output Summary to implement
#summary_list <- list(
#    Random_Forest_Accuracy = 1 - misclassRate_rf,
#    Neural_Network_Accuracy = 1 - misclassRate_nn,
#    KNN_Classification_Accuracy = 1 - misclassRate_knn,
#    Linear_Regression_RMSE = rmse_lm,
#    Decision_Tree_RMSE = rmse_tree,

#)

# Summary of Model Performances
performance_summary <- data.frame(
  Model = c("Random Forest", "Neural Network", "KNN Classification", "Linear Regression", "Decision Tree", "KNN Regression"),
  Misclassification_Rate = c(misclassRate_rf, misclassRate_nn, misclassRate_knn, NA, NA, NA),
  RMSE = c(NA, NA, NA, rmse_lm, rmse_tree, NA)
)

# Business Profitability Evaluation, duplicate code for stability.


profit_rf <- calculate_profit(predictions_rf, testing_set$donr)
profit_nn <- calculate_profit(predictions_nn, testing_set$donr)
profit_knn <- calculate_profit(predictions_knn, testing_set$donr)
profit_lm <- calculate_profit(predictions_lm, testing_set$donr)
profit_tree <- calculate_profit(predictions_tree, testing_set$donr)
#First results, reworked.):

# Prediction and Profitability Evaluation on Score Data
score_data_processed <- score_data %>%
                        mutate_if(is.factor, as.factor) %>%
                        pivot_wider(names_from = region, values_from = region,
                                    values_fn = length, values_fill = list(region = 0)) %>%
                        mutate(across(everything(), as.numeric))

# Predicting DONR for score data using Classification Models
score_predictions_rf <- predict(rf_model, score_data_processed[, predictors])
score_predictions_nn <- predict(nn_model, score_data_processed[, predictors])
score_predictions_nn <- factor(ifelse(score_predictions_nn > 0.5, "1", "0"), levels = c("0", "1"))
score_predictions_knn <- predict(knn_model, score_data_processed[, predictors])

# Predicting DAMT for score data using Regression Models
score_predictions_lm <- predict(lm_model, score_data_processed[, predictors])
score_predictions_tree <- predict(tree_model, score_data_processed[, predictors])

# Calculate Expected Profits from Classification Models
expected_profit_rf <- calculate_profit(score_predictions_rf, "1", 2, 14.50)
expected_profit_nn <- calculate_profit(score_predictions_nn, "1", 2, 14.50)
expected_profit_knn <- calculate_profit(score_predictions_knn, "1", 2, 14.50)
expected_profit_lm <- calculate_profit(score_predictions_lm, "1", 2, 14.50)
expected_profit_tree <- calculate_profit(score_predictions_tree, "1", 2, 14.50)

# Display Expected Profits
cat("Expected Profits from Classification Models:\n")
cat("Random Forest: ", expected_profit_rf, "\nNeural Network: ", expected_profit_nn, "\nKNN: ", expected_profit_knn, "\n\n")

# Display Average Predicted Donation Amounts from Regression Models
avg_donation_lm <- mean(score_predictions_lm)
avg_donation_tree <- mean(score_predictions_tree)
cat("Average Predicted Donation Amounts:\n")
cat("Linear Regression: ", avg_donation_lm, "\nDecision Tree: ", avg_donation_tree, "\n")

# Output Evaluation Results
cat("Misclassification Rates:\n")
cat("Random Forest: ", misclassRate_rf, "\nNeural Network: ", misclassRate_nn, "\nKNN Classification: ", misclassRate_knn, "\n\n")

cat("Root Mean Squared Errors for Regression Models:\n")
cat("Linear Regression: ", rmse_lm, "\nDecision Tree: ", rmse_tree, "\n\n")

cat("Business Profit Evaluation:\n")
cat("Profit with Random Forest: ", expected_profit_rf, "\nProfit with Neural Network: ", expected_profit_nn,
    "\nProfit with KNN Classification: ", expected_profit_knn, "\nProfit with Linear Regression: ", expected_profit_lm,
    "\nProfit with Decision Tree: ", expected_profit_tree, "\n")

# ... [Previous code] ...

# Install necessary additional packages
if (!require(pROC)) install.packages("pROC")
library(pROC)

# ---------------------
if (rf_model$modelType == "Classification") {
    rf_probabilities <- predict(rf_model, testing_set[, predictors], type = "prob")[, "1"]
    roc_rf <- roc(response = testing_set$donr, predictor = rf_probabilities)
    plot(roc_rf, col = "blue", main = "ROC Curves", lwd = 2)
}

if (nn_model$modelType == "Classification") {
    nn_probabilities <- predict(nn_model, testing_set[, predictors], type = "raw")
    roc_nn <- roc(response = testing_set$donr, predictor = nn_probabilities)
    plot(roc_nn, col = "red", add = TRUE, lwd = 2)
}

if (knn_model$modelType == "Classification") {
    knn_probabilities <- predict(knn_model, testing_set[, predictors], type = "prob")[, "1"]
    roc_knn <- roc(response = testing_set$donr, predictor = knn_probabilities)
    plot(roc_knn, col = "green", add = TRUE, lwd = 2)
}

if (exists("roc_rf") | exists("roc_nn") | exists("roc_knn")) {
    legend("bottomright", legend = c("Random Forest", "Neural Network", "KNN"),
           col = c("blue", "red", "green"), lwd = 2)
}

# ---------------------
# 2. Feature Importance for Random Forest
importance_rf <- varImp(rf_model)
plot(importance_rf)

# ---------------------
# 3. Decision Surface for K-NN (using first two predictors as an example)
# Note: This is a simplified representation and may not reflect the high dimensionality of the actual model
# Install necessary package
if (!require(plotly)) install.packages("plotly")
library(plotly)

feature1 <- predictors[1]
feature2 <- predictors[2]

plot_knn <- ggplot(training_set, aes_string(x = feature1, y = feature2, color = "donr")) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = c("0" = "red", "1" = "blue")) +
  labs(title = "K-NN Decision Surface", x = feature1, y = feature2)
ggplotly(plot_knn)

# ---------------------
# 4. Neural Network Performance (Misclassification Rate vs. Threshold)
thresholds <- seq(0, 1, by = 0.01)
misclass_rates <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(as.numeric(predictions_nn) > t, "1", "0"), levels = c("0", "1"))
  mean(pred != testing_set$donr)
})

plot(thresholds, misclass_rates, type = "l", col = "blue",
     xlab = "Threshold", ylab = "Misclassification Rate",
     main = "NN Performance")

# ---------------------
# 5. Histograms of Predicted Probabilities
prob_rf <- predict(rf_model, testing_set[, predictors], type = "prob")
hist(prob_rf[, "1"], main = "Predicted Probabilities - Random Forest", xlab = "Probability", col = "blue", breaks = 30)


