## Example: Golf Putting

## A. Read the data
Distance = 2:20;
Tried = c(1443, 694, 455, 353, 272,256, 240, 217, 200, 237, 202, 192, 174, 167, 201, 195, 191, 147, 152);
Success = c(1346, 577, 337, 208, 149, 136, 111, 69, 67, 75, 52, 46, 54, 28, 27, 31, 33, 20, 24);

## B. Plot the data 
Rate = 100* Success / Tried; 
plot(Distance, Rate)


## C Building Models
# C1: Linear regression model
lm1 <- lm( Rate ~ Distance)
summary(lm1)

# C2: quadratic regression model
lm2 <- lm( Rate ~ Distance + I(Distance^2))
summary(lm2)

# Plot the fitted models 
plot(Distance, Rate)
abline(lm1)
lines(Distance, fitted(lm2), col="red")

### It looks nice, right? 
### However, let us look at larger ranges of Distance
xx <- seq(0.1, 40, 0.01);
xnew <- data.frame(Distance=xx);
pred1 <- predict(lm1, xnew);  pred2 <- predict(lm2, xnew); 

plot(xx, pred1, col="black", "l",  xlim=c(0,40), ylim=c(-50,150), xlab="Distance", ylab="Rate")
points(Distance, Rate)
lines(xx, pred2, col="red")
abline(0,0, lty=3); abline(100,0, lty=3)


### C3; Logistic Regression
### Two equivalent ways to fit logistic regression in R
glm1 <- glm( cbind(Success, Tried-Success) ~ Distance, family = binomial)
glm1b <- glm(Success / Tried ~ Distance,  weights=Tried, family = binomial)

summary(glm1)
summary(glm1b)

## Plot the logisitc regression fit over a larger ranges of Distance
xx <- seq(0.1, 40, 0.01);
xnew <- data.frame(Distance=xx);
pred3 <- 100*predict(glm1, xnew, type="response"); 
lines(xx, pred3, col="blue")

plot(Distance, Rate, xlim=c(0,40), ylim=c(-50,150), xlab="Distance", ylab="Rate")
lines(xx, pred3, col="blue")
abline(0,0, lty=3)
abline(100,0, lty=3)


## C4: domain-knowledge based new model
## 
## Two methods for NLS for new model
## C4(a) The first is to use the "nls" package (you might need to install it first)
nls1 <- nls(Rate ~ 100*(2*pnorm((1/sigma0) * asin((4.25-1.68) / (24*Distance))) -1),
            start = list(sigma0=5)); 
xnew <- data.frame(Distance=seq(0.1, 40, 0.01));
pred4 <- predict(nls1, xnew); 
## There are some warnigng messages 
##  In asin((4.25 - 1.68)/(24 * Distance)) : NaNs produced
## this is because asin is defined in the range of [-1,1]
## Those NaNs values will become 0 in our context, which is okay.
##  IN general you need to pay attension if there are warning messages. 

##
## C4(b) The second is to compute by ourselves for this 1-dim problem
sigma = seq(0.00001, 1, 0.00001);
PP <- NULL;
n = length(Distance);
for (i in 1:n)  PP <- rbind(PP, 100*(2* pnorm( (1/sigma) * asin( (4.25-1.68) / (24*Distance[i]) )  ) -1) );
RSS <- apply((PP- Rate)^2, 2, sum);
plot(sigma, RSS, "l")
sigmahat <- sigma[which.min(RSS)] ##[1] 0.02505

## A comparision of logisitc regression vs new model fit
## The lwd=3 to increase the line width relative to the default (lwd)
plot(Distance, Rate, xlim=c(0,40), ylim=c(-50,150), xlab="Distance", ylab="Rate")
lines(xx, pred3, col="blue", lwd=3)
lines(xx, pred4, col="red", lwd=3)
abline(0,0, lty=3)
abline(100,0, lty=3)
