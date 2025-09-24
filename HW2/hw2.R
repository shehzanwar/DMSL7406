set.seed(7406)
source("hw2_func.R")

#### Reading the Data ####

fat <- read_csv("fat.csv", show_col_types=FALSE)
n <- dim(fat)[1]
flag = c(1, 21, 22, 57, 70, 88, 91, 94, 121, 127, 149, 151, 159, 162, ## Using the flags as per instructionsS
         164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243)
fat1train <- fat[-flag,]
fat1test <- fat[flag,]

#### Exploratory Data Analysis ####
cor_heatmap(fat)
train <- fat1train %>% subset(select=-c(siri, density, free))
test <- fat1test %>% subset(select=-c(siri, density, free)) 

#### Single Models ####
full <- full_model_mse(train,test)
kbest <- select_best_mse(train,test)
stepwise <- stepwise_mse(train,test)
ridge <- ridge_mse(train,test)
lasso <- lasso_mse(train,test,plot=TRUE)
pcar <- pca_mse(train,test,plot=TRUE)
plsr <- pls_mse(train,test,plot=TRUE)

mses <- c(full$mse, kbest$pscore$mse, kbest$search$mse, stepwise$mse, ridge$mse, 
         lasso$mse, pcar$mse, plsr$mse)
models <- list(full$model, kbest$pscore$model, kbest$search$model, stepwise$model, ridge$model, 
            lasso$model, pcar$model, plsr$model)
model_names <- c("Full", "5-Best (p-score)", "5-Best (AIC)", "Stepwise",
                 "Ridge", "LASSO", "PCA", "PLS")
p1_results <- data.frame(MSE=mses, row.names=model_names)
p1_results %>% arrange(MSE)

#### MonteCarlo CV ####
fat_red <- fat %>% subset(select=-c(siri, density, free))
trials <- 500 
cv_raw <- data.frame(matrix(nrow=0, ncol=length(model_names)))
set.seed(7406)
for (i in seq(1,trials)) {
  flag <- sample(1:nrow(fat_red), replace=FALSE, size=round(0.1*nrow(fat_red)))
  train_cv <- fat_red[-flag,]
  test_cv <- fat_red[flag,]
  
  full_cv <- full_model_mse(train_cv,test_cv)
  kbest_cv <- select_best_mse(train_cv,test_cv)
  stepwise_cv <- stepwise_mse(train_cv,test_cv)
  ridge_cv <- ridge_mse(train_cv,test_cv)
  lasso_cv <- lasso_mse(train_cv,test_cv)
  pcar_cv <- pca_mse(train_cv,test_cv)
  plsr_cv <- pls_mse(train_cv,test_cv)
  
  mses <- c(full_cv$mse, kbest_cv$pscore$mse, kbest_cv$search$mse, 
            stepwise_cv$mse, ridge_cv$mse, lasso_cv$mse, pcar_cv$mse, 
            plsr_cv$mse)
  cv_raw <- rbind(cv_raw, mses)
}
colnames(cv_raw) <- model_names
cv_results <- data.frame(Mean=sapply(cv_raw,mean), Median=sapply(cv_raw,median))