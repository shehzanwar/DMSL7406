set.seed(7406)
pacman::p_load(tidyverse, reshape2, ggplot2, leaps, glmnet, pls)
calc_mse <- function(model, test){
  df <- data.frame(pred=predict(model, test), actual=test$brozek)
  mse <- mean((df$actual - df$pred)^2)
  return(mse)
}

cor_heatmap <- function(data) {
  cormat <- round(cor(fat),2)
  cormat <- melt(cormat)
  cormat %>% ggplot() + geom_tile(aes(x=Var1,y=Var2,fill=value)) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         limit = c(-1,1), name="Correlation") +
    xlab("") + ylab("")
}                  

full_model_mse <- function(train,test) {
  full_model <- lm(brozek ~ ., data = train)
  mse <- calc_mse(full_model, test)
  return(list(mse=mse, model=full_model))
}

select_best_mse <- function(train, test) {
  full_model <- lm(brozek ~ ., data = train)
  coefs <- summary(full_model)$coefficients
  coefs <- coefs[row.names(coefs) != "(Intercept)",]
  coefs <- coefs[order(coefs[,4]),]
  sel_coef1 <- rownames(coefs[1:5,])
  sel_train1 <- train %>% subset(select=c("brozek", sel_coef1))
  pscore_model <- lm(brozek ~ ., data = sel_train1)
  mse1 <- calc_mse(pscore_model, test)
  
  full_search <- leaps(subset(train, select = -brozek), train$brozek, method="Cp", nbest=1)
  sel_coef2 <- train %>% subset(select=-brozek) %>% colnames %>% .[full_search$which[5,]]
  sel_train2 <- train %>% subset(select=c("brozek", sel_coef2))
  search_model <- lm(brozek ~ ., data = sel_train2)
  mse2 <- calc_mse(search_model, test)
  return(list(pscore=list(mse=mse1, model=pscore_model),
              search=list(mse=mse2, model=search_model)))
}

stepwise_mse <- function(train, test) {
  min_model <- lm(brozek ~ 1, data=train)
  full_model <- lm(brozek ~ ., data=train)
  step_model <- step(min_model, scope=list(lower=min_model, upper=full_model), 
                     direction="both", trace=FALSE)
  mse <- calc_mse(step_model, test)
  return(list(mse=mse, model=step_model))
}
  
ridge_mse <- function(train, test) {
  ridge_cv <- train %>% subset(select=-brozek) %>% data.matrix %>% 
    cv.glmnet(train$brozek, alpha = 0, nfolds = 5)
  ridge_model <- train %>% subset(select = -brozek) %>% data.matrix %>% 
    glmnet(train$brozek, alpha = 0, lambda = ridge_cv$lambda.min)
  ridge_pred <- predict(ridge_model, subset(test, select=-brozek) %>% as.matrix)
  mse <- mean((test$brozek - ridge_pred)^2)
  return(list(mse=mse, model=ridge_model))
}

lasso_mse <- function(train, test, plot=FALSE) {
  lasso_cv <- train %>% subset(select=-brozek) %>% data.matrix %>% 
    cv.glmnet(train$brozek, alpha = 1, nfolds = 5)
  lasso_select <- train %>% subset(select = -brozek) %>% data.matrix %>% 
    glmnet(train$brozek, alpha = 1, nlambda=100)
  if(plot) {
    plot(lasso_select, xvar="lambda", label=TRUE)
    abline(v = log(lasso_cv$lambda.min), col="black", lty=2)
  }
  sel_coef <- coefficients(lasso_select, s=lasso_cv$lambda.min) %>% as.matrix %>% 
    as.data.frame %>% rownames_to_column("feature") %>% 
    filter(s1 != 0) %>% filter(feature != "(Intercept)")
  lasso_train <- train %>% subset(select=c("brozek", sel_coef$feature))
  lasso_model <- lm(brozek ~ ., data=lasso_train)
  mse <- calc_mse(lasso_model, test)
  return(list(mse=mse, model=lasso_model, lambda=lasso_cv$lambda.min))
}

pca_mse <- function(train, test, plot=FALSE) {
  pcr_model <- pcr(brozek ~ ., data=train, scale=TRUE)
  if(plot) {
    var_expl <- cumsum(pcr_model$Xvar/pcr_model$Xtotvar)
    df <- data.frame(PCs = seq(1,length(var_expl)), expl=var_expl)
    p <- df %>% ggplot(aes(x=PCs, y=expl)) + geom_point() + geom_line() +
      geom_hline(aes(yintercept=0.95), color="red", lty=2) +
      xlab("# of PCs Included") + ylab("% Total Variance explained")
    print(p)
  }
  n_pcs <- min(which(cumsum(pcr_model$Xvar/pcr_model$Xtotvar) > 0.95))
  pca_pred <- predict(pcr_model, test, ncomp=n_pcs)
  mse <- mean((test$brozek - pca_pred)^2)
  return(list(mse=mse, model=pcr_model, n_pcs=n_pcs))
}

pls_mse <- function(train, test, plot=FALSE) {
  pls_model <- plsr(brozek ~ ., data=train, scale=TRUE, validation="CV")
  if(plot) {
    validationplot(pls_model)
  }
  n_comps <- which.min(RMSEP(pls_model)$val[1,1,]) %>% as.numeric() - 1
  pls_pred <- predict(pls_model, test, ncomp=n_comps)
  mse <- mean((test$brozek - pls_pred)^2)
  return(list(mse=mse, model=pls_model, n_comps=n_comps))
}