
##### Example for Logsitic Regression: low birth wight data 
##
## %<http://www.umass.edu/statdata/statdata/data/lowbwt.txt>
## 
## read data in
data1 <- read.table("C:/temp/lowbwt.csv", head=T, sep=",")

# take a look at first 3 rows of dataset
data1[1:3,]
## or look at the first several rows 
head(data1)

# fit a logit model with LOW as the dep. var. and AGE, LWT, and SMOKE
# as the covariates
#
logit.out <- glm(LOW~AGE+LWT+SMOKE, family=binomial(link=logit),
                 data=data1)

# take a look at the logit results
summary(logit.out)

## A. Point estimation/prediction at xnew
##
# plot low on AGE adding some jitter (noise) to LOW
plot(data1$AGE, jitter(data1$LOW, .1) )
##
## Suppose we want to investigate the impact of smoking on LOW at different ages.
##  To do so, we create two groups of subjects with different ages
##    but with the same weight (say 120 lbs).  
## One is a 120 lb. woman with different ages who are smokers. 
## the other is 120 lbs woman with different ages who are not smokers
## Below we construct two new matrices for Xnews that
# corresponds to our hypothetical women.
X1 <- cbind(1, seq(from=14, to=45, by=1), 120, 1)
X0 <- cbind(1, seq(from=14, to=45, by=1), 120, 0)



## A1. Manually computation 
##
###  A1(a) extract just the coefficients from the logit output object
coefficients(logit.out)
# put the logit coefficients in a new object called beta.logit
beta.logit <- coefficients(logit.out)

### A1(b) Next, we use the logistic model to predict 
### probabilities  of a 120 lb. woman who is a smoker giving birth 
### to a low birthweight child at different ages. 
#
# multiply the xnew data matrix by our logit coefficients 
### to get the value of the linear predictor.
Xb1 <- X1 %*% beta.logit

###  transform the linear predictor into the predicted probabilities
prob1 <- exp(Xb1)/(1+exp(Xb1))

# now plot these probabilities as a function of age on the pre-existing
# graph of low on age
lines(seq(from=14, to=45, by=1), prob1, col="red")


### A1(c) For comparison, we also use the logistic model to predict 
### probabilities  of a 120 lb. woman who is "NOT" a smoker giving birth 
### to a low birthweight child at different ages. 
# multiply this matrix by out logit coefficients to get the value of the
# linear predictor.
Xb0 <- X0 %*% beta.logit

# Now use the logistic cdf to transform the linear predictor into
# probabilities
prob0 <- exp(Xb0)/(1+exp(Xb0))

# now plot these probabilities as a function of age on the pre-existing
# graph of low on age
lines(seq(from=14, to=45, by=1), prob0, col="blue")

### A1(d) The confidence interval of all predicted probabilities  
###  This allows us to plot Confidence Interval Bands for probablities
### it is important to extract the covariance matrix
V <- vcov(logit.out)

# Confidence interval For woman who is a smoker
Var.logit1 <- diag(X1 %*% V %*% t(X1))
Xb1.upper <- Xb1 + 1.96*sqrt(Var.logit1)
Xb1.low <- Xb1 - 1.96*sqrt(Var.logit1)
# Use the logistic cdf to transform these CI into probabilities
prob1.upper <- exp(Xb1.upper)/(1+exp(Xb1.upper))
prob1.low <- exp(Xb1.low)/(1+exp(Xb1.low))
## Add these confidence bounds for smoking group 
lines(seq(from=14, to=45, by=1), prob1.upper, col="red", lty="dashed")
lines(seq(from=14, to=45, by=1), prob1.low, col="red", lty="dashed")

# Similarly, Confidence interval for woman who is NOT a smoker
Var.logit0 <- diag(X0 %*% V %*% t(X0))
Xb0.upper <- Xb0 + 1.96*sqrt(Var.logit0)
Xb0.low <- Xb0 - 1.96*sqrt(Var.logit0)
prob0.upper <- exp(Xb0.upper)/(1+exp(Xb0.upper))
prob0.low <- exp(Xb0.low)/(1+exp(Xb0.low))
lines(seq(from=14, to=45, by=1), prob0.upper, col="blue", lty="dashed")
lines(seq(from=14, to=45, by=1), prob0.low, col="blue", lty="dashed")

# A1(e) If you want, you can create a 3-d plot
weight <- seq(from=80, to=250, length=100)
age <- seq(from=14, to=45, length=100)
logit.prob.fun <- function(weight, age){
  exp(1.368225 -0.038995*age -0.012139*weight + 0.670764) /
    (1 + exp(1.368225 -0.038995*age -0.012139*weight + 0.670764))
}
prob <- outer(weight, age, logit.prob.fun)
persp(age, weight, prob, theta=30, phi=30, expand=0.5, col="lightblue")

#### A2. The above computation can also be done automatically in R
####      through the "predict" function 
logit.out <- glm(LOW~AGE+LWT+SMOKE, family=binomial(link=logit),
                 data=data1);
## A2(a) here it is important to define xnew appropriately by using 
##   the same variable names in the logisitc regression model
X1new <- data.frame(AGE=seq(from=14, to=45, by=1), LWT=120, SMOKE=1);
## Note that there are two kinds of predictions:
##  one is the X\beta value and the other is the probabilities 
Xb1b <- predict(logit.out, newdata=X1new); 
prob1b <- predict(logit.out, newdata=X1new, type="response");

## We can show that two approach lead to the same answers 
## 
cbind(Xb1, Xb1b, prob1, prob1b)

## A2(b) we can also use the point prediction function for the non-smoking group
## 
X0new <- data.frame(AGE=seq(from=14, to=45, by=1), LWT=120, SMOKE=0);
## Note that there are two kinds of predictions:
##  one is the X\beta value and the other is the probabilities 
Xb0b <- predict(logit.out, newdata=X0new); 
prob0b <- predict(logit.out, newdata=X0new, type="response");
## We can show that two approach lead to the same answers 
cbind(Xb0, Xb0b, prob0, prob0b)


### It is tricky to derive the interval estimation 
##    for the prediction in logisitc regression 
##  below is one possibility to get the interval estiamtion of X\beta 
## Suppose we want to find (1-\alpha) interval estimation on the predicted probe for xnew
alpha = 0.05;
criticalvalue = - qnorm(alpha/2);
Xb1c <- predict(logit.out, newdata=X1new, se.fit=TRUE);
Xb1c.upper <- Xb1c$fit + criticalvalue * Xb1c$se.fit;
Xb1c.low <- Xb1c$fit - criticalvalue * Xb1c$se.fit;

## These values are slightly different from (Xb1.upper and Xb1.low),
##  as criticalvalue = 1.959964 here instead of using 1.96
## if you change to 1.96, you will get the same answer!
cbind(Xb1.upper, Xb1c$fit + 1.96 * Xb1c$se.fit, Xb1.low, Xb1c$fit - 1.96 * Xb1c$se.fit);  

# 
###
### B. Hypothesis Testing in Logistic Regression
###  B1.   To start with, let us fit a baseline logit model
logit1.out <- glm(LOW~AGE+ LWT+SMOKE+HT+UI, family=binomial, data=data1)
summary(logit1.out)


###  B2, now fit another logit model including race
data1$AfrAm <- data1$RACE==2
data1$othrace <- data1$RACE==3
logit2.out <- glm(LOW~AGE+ LWT+AfrAm+othrace+SMOKE+HT+UI,
                  family=binomial, data=data1)

summary(logit2.out)

# B3. let's conduct a likelihood ratio test of model 1 vs. model 2
# Here the constrained model is model 1 and the unconstrained model is
# model 2. Since 2 constraints are applied, the test statistic under
# the null follows a chi-square distribution with 2 degrees of freedom

lr <- deviance(logit1.out)  - deviance(logit2.out)
lr
# The p-value
1 - pchisq(lr, 2)

# The p-value of 0.01994 indicates that there is reason to believe
# (at 5% level) that the constraints implied by model 1 do not hold

## B4.  Wald's Test 
# We can also use a Wald test to decide whether the
# coefficients on AfrAm and othrace are zero in the second model

R <- matrix(c(0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0), 2, 8)
beta <- coef(logit2.out)
r <- 0
V <- vcov(logit2.out)
W <- t(R %*% beta - r) %*% solve(R %*% V %*% t(R)) %*% (R %*% beta - r)
W
# The p-value from Wald test
1 - pchisq(W, 2)

## The small p-value from Wald test implies that 
##  at least one of AfrAm and othrace variable are signficantly 
##  associated with the response LOW variable
##
# We got the same conclusion as in Likelihood test
# Why is the Wald statistic "only" 7.42, while the likelihood ratio
# statistic is 7.83 and both have the same df?
# ---- likelihood ratio test is more powerful

## B5. Another use of Wald's test
##   Now suppose we want to test whether the coefficients on smoking
# and # hypertension are equal to each other in the second model.
# How to conduct a Wald test?

R <- matrix(c(0,0,0,0,0,1,-1,0), 1, 8)
beta <- coef(logit2.out)
r <- 0
V <- vcov(logit2.out)
W <- t(R %*% beta - r) %*% solve(R %*% V %*% t(R)) %*% (R %*% beta - r)
1 - pchisq(W, 1)
# the p-value of 0.293 suggests that there is no reason to believe
# the null hypothesis (that the coefficients are equal) is not true.

### C. Model or variable  Selection in logisitc regression
### 
#  C1. We could also look at BIC to pick models. The AIC() function in R 
# will return BIC values if the argument k is set to log(n)

nrow(data1)
bic1 <- AIC(logit1.out, k=log(189))
bic2 <- AIC(logit2.out, k=log(189))
bic2 - bic1
# This indicates moderate support for model 1 over model 2. Nonetheless,
# given that we have strong reason to believe that race should be in the
# model we may well want to stick with model 2.

## C2. In general, we can use the "step" function for model selection
##
logit2.out <- glm(LOW~AGE+ LWT+AfrAm+othrace+SMOKE+HT+UI,
                  family=binomial, data=data1);
logit3.out <- step(logit2.out);
summary(logit3.out)
## we can then use logit3.out as our final model for logisitc regression 
##     after variable selection 