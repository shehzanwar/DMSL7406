# --- 1. SETUP AND LIBRARIES ---
# Purpose: Load all necessary R packages for the analysis.

# Install packages if you don't have them
# install.packages("readr")
# install.packages("caret")
# install.packages("tree")
# install.packages("randomForest")
# install.packages("gbm")
# install.packages("glmnet")
# install.packages("gam")

# Load libraries
library(readr)        # For reading the CSV file
library(caret)        # For data splitting, CV, and model tuning
library(tree)         # For the baseline decision tree
library(randomForest) # For Random Forest
library(gbm)          # For Gradient Boosting
library(glmnet)       # For Ridge and LASSO
library(gam)          # For LOESS and GAM

# Set a random seed to make our results reproducible
set.seed(7406)

# --- 2. LOAD AND PREPARE DATA ---
# Purpose: Load the Auto.csv file, handle preprocessing, and
# split into training and test sets as required by the assignment.

print("Loading and preparing data...")
# Load the dataset
auto_data <- read_csv("Auto.csv")

# Convert 'origin' to a factor, as it's a categorical variable
auto_data$origin <- as.factor(auto_data$origin)

# Split the data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(auto_data$mpg,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

train_set <- auto_data[trainIndex, ]
test_set  <- auto_data[-trainIndex, ]

print(paste("Training set size:", nrow(train_set)))
print(paste("Test set size:", nrow(test_set)))

# --- 3. CROSS-VALIDATION SETUP ---
# Purpose: Define the 10-fold cross-validation (CV) method.
# This CV will be used ONLY on the training set to tune parameters,
# as specified in the assignment[cite: 13, 19].
cv_control <- trainControl(method = "cv", number = 10)

# --- 4. BASELINE MODELS (FITTED/TUNED ON TRAINING SET) ---
# Purpose: Fit simpler models to act as baselines for comparison[cite: 9].

print("Fitting baseline models...")

# Baseline 1: Multiple Linear Regression (No tuning)
lm_fit <- lm(mpg ~ ., data = train_set)

# Baseline 2: Single Regression Tree (No tuning)
tree_fit <- tree(mpg ~ ., data = train_set)

# Baseline 3: K-Nearest Neighbors (Tuned)
# We tune 'k' using 10-fold CV.
knn_tuned_fit <- train(mpg ~ .,
                       data = train_set,
                       method = "knn",
                       trControl = cv_control,
                       preProcess = c("center", "scale"), # Important for KNN
                       tuneGrid = expand.grid(k = seq(1, 21, by = 2)))

# Baseline 4: Ridge Regression (Tuned)
# We tune 'lambda' (penalty) using 10-fold CV.
ridge_tuned_fit <- train(mpg ~ .,
                         data = train_set,
                         method = "glmnet",
                         trControl = cv_control,
                         preProcess = c("center", "scale"),
                         tuneGrid = expand.grid(
                           alpha = 0,  # alpha = 0 for Ridge
                           lambda = 10^seq(-3, 3, length = 100)
                         ))

# Baseline 5: LASSO (Tuned)
# We tune 'lambda' (penalty) using 10-fold CV.
lasso_tuned_fit <- train(mpg ~ .,
                         data = train_set,
                         method = "glmnet",
                         trControl = cv_control,
                         preProcess = c("center", "scale"),
                         tuneGrid = expand.grid(
                           alpha = 1,  # alpha = 1 for LASSO
                           lambda = 10^seq(-3, 1, length = 100)
                         ))

# Baseline 6: LOESS (Local Smoothing) (Tuned)
# We tune 'span' using 10-fold CV.
loess_tuned_fit <- train(mpg ~ .,
                         data = train_set,
                         method = "gamLoess",
                         trControl = cv_control,
                         preProcess = c("center", "scale"),
                         tuneGrid = expand.grid(
                           span = seq(0.1, 1.0, by = 0.1),
                           degree = 1 # Local linear fit
                         ))
                         
# Baseline 7: Generalized Additive Model (GAM)
# We fit using smoothing splines for continuous predictors.
gam_fit <- gam(mpg ~ s(cylinders) + s(displacement) + s(horsepower) +
                     s(weight) + s(acceleration) + s(year) + origin,
               data = train_set)


# --- 5. ENSEMBLE MODELS (TUNED ON TRAINING SET) ---
# Purpose: Fit the two required ensemble models  and tune
# their parameters using 10-fold CV on the training set[cite: 12].

print("Fitting and tuning ensemble models...")

# Ensemble 1: Random Forest
# We tune 'mtry' (number of variables per split).
rf_grid <- expand.grid(mtry = c(2, 4, 7)) # p=7 predictors

rf_tuned_fit <- train(mpg ~ .,
                      data = train_set,
                      method = "rf",
                      trControl = cv_control,
                      tuneGrid = rf_grid,
                      ntree = 500) # Use 500 trees

# Ensemble 2: Boosting (GBM)
# We tune n.trees, interaction.depth, and shrinkage.
gbm_grid <- expand.grid(
  n.trees = c(100, 500, 1000),
  interaction.depth = c(1, 2, 4),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

gbm_tuned_fit <- train(mpg ~ .,
                       data = train_set,
                       method = "gbm",
                       trControl = cv_control,
                       tuneGrid = gbm_grid,
                       verbose = FALSE) # Suppress detailed output

# --- 6. EVALUATE ALL MODELS ON THE TEST SET ---
# Purpose: Evaluate the final models on the unseen test set 
# to get an unbiased measure of performance. We use Mean Squared
# Error (MSE) as our metric.

print("Evaluating all models on the test set...")

# Baseline Predictions
lm_pred    <- predict(lm_fit, newdata = test_set)
tree_pred  <- predict(tree_fit, newdata = test_set)
knn_pred   <- predict(knn_tuned_fit, newdata = test_set)
ridge_pred <- predict(ridge_tuned_fit, newdata = test_set)
lasso_pred <- predict(lasso_tuned_fit, newdata = test_set)
loess_pred <- predict(loess_tuned_fit, newdata = test_set)
gam_pred   <- predict(gam_fit, newdata = test_set)

# Ensemble Predictions
rf_pred    <- predict(rf_tuned_fit, newdata = test_set)
gbm_pred   <- predict(gbm_tuned_fit, newdata = test_set)

# Calculate Test MSE for all 9 models
lm_mse    <- mean((lm_pred - test_set$mpg)^2)
tree_mse  <- mean((tree_pred - test_set$mpg)^2)
knn_mse   <- mean((knn_pred - test_set$mpg)^2)
ridge_mse <- mean((ridge_pred - test_set$mpg)^2)
lasso_mse <- mean((lasso_pred - test_set$mpg)^2)
loess_mse <- mean((loess_pred - test_set$mpg)^2)
gam_mse   <- mean((gam_pred - test_set$mpg)^2)
rf_mse    <- mean((rf_pred - test_set$mpg)^2)
gbm_mse   <- mean((gbm_pred - test_set$mpg)^2)

# --- 7. FINAL RESULTS ---
# Purpose: Consolidate all results into a single table
# to find the best-performing model[cite: 11].

results_df <- data.frame(
  Model = c("Linear Regression (Baseline)",
            "Single Tree (Baseline)",
            "Tuned KNN (Baseline)",
            "Ridge Regression (Tuned)",
            "LASSO (Tuned)",
            "LOESS (Tuned)",
            "GAM (Baseline)",
            "Random Forest (Tuned)",
            "Boosting (GBM) (Tuned)"),
  Test_MSE = c(lm_mse, tree_mse, knn_mse, ridge_mse, lasso_mse,
               loess_mse, gam_mse, rf_mse, gbm_mse)
)

# Sort by MSE (lower is better) to find the best model
results_df <- results_df[order(results_df$Test_MSE), ]

print("--- Final Model Comparison on Test Set (Lower MSE is Better) ---")
print(results_df)

print(paste(
  "The best performing model is:",
  results_df$Model[1],
  "with a Test MSE of:",
  round(results_df$Test_MSE[1], 4)
))

print("--- Tuned Parameters ---")
print("Best k for KNN:"); print(knn_tuned_fit$bestTune)
print("Best params for Ridge:"); print(ridge_tuned_fit$bestTune)
print("Best params for LASSO:"); print(lasso_tuned_fit$bestTune)
print("Best params for LOESS:"); print(loess_tuned_fit$bestTune)
print("Best mtry for Random Forest:"); print(rf_tuned_fit$bestTune)
print("Best params for Boosting (GBM):"); print(gbm_tuned_fit$bestTune)