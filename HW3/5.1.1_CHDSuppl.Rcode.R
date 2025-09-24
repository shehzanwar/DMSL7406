## R code for Logistic regression (II) and Case study

## Example A: CHD data 

##suppose that we save the data file in the folder "C:/temp"
data0 <- read.table("C:/temp/chdage.csv", head=T, sep=",")

## A1. Empirical Data Analysis
## It is better to summarize the results (instead of copying all output in the report)
## Look at the first several rows
data0[1:5,]
dim(data0)
## we have 100 observations, and each observation has three variables (ID, Age, CHD)
## We are only interested in studying the relationship between Age and CHD

## The "attach" function allows us to use the local variable of data0 directly 
attach(data0);
## plot x =age against Y= CHD status (0/1)
plot(Age, CHD);

## The function "summary" is a nice function to get your overall idea of data
##  but it is often a bad idea to put all output in the main body of the report
##  In general we should put this R output in the appendix of your report! 
summary(data0)
## If we compare the median and mean values
##  we found out that they are similar to Age, but different for CHD
sum(CHD)
## Among n=100 subjects, 43 of them have CHD=1 disease and 
##    the remaining 100-43 = 57 do not have CHD disease, e.g., CHD=0. 

## more plots to understand the data better 
## You might include these plots in the appendix of your report 
hist(Age)
hist(CHD)

## A2: logisitic regression model 
glm1 <- glm(CHD ~ Age, family=binomial(link="logit"), data=data0)
## the output of logisitc regression model
summary(glm1)
## the p-value of "Age" coefficient is very small (4.02e-6),
##  whcih implies that Age is signficantly associated with CHD disease. 

## It is useful to plot the fitted "probibility of getting CHD=1" as a function of Age:
plot(Age, CHD)
lines(Age,fitted.values(glm1), col="red")

## How to find the ???????\alpha Confidence Interval on \beta0, \beta1 in R?  
confint(glm1, level=0.95)
## this function is better than the Central Limit Theorem based code
##   confint.default(glm1, level=0.95)

### A3: The Simplest Logistic Regression Model
## In the CHD dataset, define a new variable Flag= I(Age???50).
flag <- I(Age >=50); 
glm2 <- glm(CHD ~ flag, family = binomial(link="logit"), data = data0);
## The output of this simplest logisitc regression 
summary(glm2)
## The key output is as follow
# Coefficients:
#              Estimate      Std. Error  z value   Pr(>|z|)    
#(Intercept)    -1.0380     0.2822      -3.678      0.000235 ***
#  flagTRUE      2.0989     0.4788       4.384       1.17e-05 ***
## 
## we can check that the output is consistent with 
## those we computed by hand 