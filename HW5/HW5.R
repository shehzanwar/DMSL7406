library(readr)
library(caret)
library(tree) # For the baseline decision tree
library(randomForest) # For Random Forest
library(gbm) # For Gradient Boosting
library(glmnet) # For Ridge and LASSO
library(gam) # For LOESS and GAM

# Setting the seed for reproducibility to 7406
set.seed(7406)

print("Loading and preparing data...")
auto <- read_csv("Auto.csv")

# Convert categorical var 'origin' to a factor
auto$origin <- as.factor(auto$origin)

# Split the data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(auto$mpg, p = .7, list = FALSE, times = 1)

train_set <- auto[trainIndex, ]
test_set  <- auto[-trainIndex, ]

print(paste("Training set size:", nrow(train_set)))
print(paste("Test set size:", nrow(test_set)))

# Setting up 10-fold Cross-Validation control
print("Setting up 10-fold Cross-Validation...")
cv <- trainControl(method = "cv", number = 10)

# Setting up the Baseline Models, tuned as required

print("Fitting baseline models...")
# Baseline 1: Multiple Linear Regression (No tuning)
lm_fit <- lm(mpg ~ ., data = train_set)

# Baseline 2: Single Regression Tree (No tuning)
tree_fit <- tree(mpg ~ ., data = train_set)

# Baseline 3: K-Nearest Neighbors (k Tuned using 10-fold CV)
knn_tuned_fit <- train(mpg ~ ., data = train_set,method = "knn", trControl = cv, preProcess = c("center", "scale"), tuneGrid = expand.grid(k = seq(1, 21, by = 2))) 

# Baseline 4: Ridge Regression (Tuned using 10-fold CV)
ridge_tuned_fit <- train(mpg ~ ., data = train_set, method = "glmnet", trControl = cv, preProcess = c("center", "scale"), tuneGrid = expand.grid(alpha = 0,lambda = 10^seq(-3, 3, length = 100))) # alpha = 0 for Ridge

# Baseline 5: LASSO (Tuned using 10-fold CV)
lasso_tuned_fit <- train(mpg ~ .,data = train_set, method = "glmnet", trControl = cv, preProcess = c("center", "scale"), tuneGrid = expand.grid(alpha = 1,lambda = 10^seq(-3, 1, length = 100)))  # alpha = 1 for LASSO

# Baseline 6: LOESS (span Tuned using 10-fold CV)
loess_tuned_fit <- train(mpg ~ .,data = train_set, method = "gamLoess", trControl = cv, preProcess = c("center", "scale"), tuneGrid = expand.grid(span = seq(0.1, 1.0, by = 0.1),degree = 1)) # Local linear fit
                         
# Baseline 7: Generalized Additive Model (GAM) (Smooth splines, No tuning)
gam_fit <- gam(mpg ~ s(cylinders) + s(displacement) + s(horsepower) + s(weight) + s(acceleration) + s(year) + origin, data = train_set)


# Ensemble Models, tuned as required, using 10-fold CV

print("Fitting and tuning ensemble models...")
# Ensemble 1: Random Forest (mtry Tuned)
rf_grid <- expand.grid(mtry = c(2, 4, 7)) # p=7 predictors

rf_tuned_fit <- train(mpg ~ ., data = train_set, method = "rf", trControl = cv, tuneGrid = rf_grid, ntree = 500) # Use 500 trees

# Ensemble 2: Boosting (GBM) (n.trees, interaction.depth, shrinkage Tuned)
gbm_grid <- expand.grid(n.trees = c(100, 500, 1000), interaction.depth = c(1, 2, 4), shrinkage = c(0.01, 0.1), n.minobsinnode = 10)

gbm_tuned_fit <- train(mpg ~ ., data = train_set, method = "gbm", trControl = cv, tuneGrid = gbm_grid, verbose = FALSE)


# Evaluation of the models on the test set

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

# Totals for test MSE calculations
lm_mse    <- mean((lm_pred - test_set$mpg)^2)
tree_mse  <- mean((tree_pred - test_set$mpg)^2)
knn_mse   <- mean((knn_pred - test_set$mpg)^2)
ridge_mse <- mean((ridge_pred - test_set$mpg)^2)
lasso_mse <- mean((lasso_pred - test_set$mpg)^2)
loess_mse <- mean((loess_pred - test_set$mpg)^2)
gam_mse   <- mean((gam_pred - test_set$mpg)^2)
rf_mse    <- mean((rf_pred - test_set$mpg)^2)
gbm_mse   <- mean((gbm_pred - test_set$mpg)^2)

# Print out the Test MSE results for all models

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

print("Final Model Comparison on Test Set (Lower MSE means better performance):")
print(results_df)

print(paste(
  "The best performing model is:",
  results_df$Model[1],
  "with a Test MSE of:",
  round(results_df$Test_MSE[1], 4)
))

print("Tuned Parameters for Tuned Models:")
print("Best k for KNN:"); print(knn_tuned_fit$bestTune)
print("Best params for Ridge:"); print(ridge_tuned_fit$bestTune)
print("Best params for LASSO:"); print(lasso_tuned_fit$bestTune)
print("Best params for LOESS:"); print(loess_tuned_fit$bestTune)
print("Best mtry for Random Forest:"); print(rf_tuned_fit$bestTune)
print("Best params for Boosting (GBM):"); print(gbm_tuned_fit$bestTune)