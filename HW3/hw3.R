library (ggplot2)
library (dplyr)
library (tidyr)
library (corrplot)
library (MASS)
library (e1071)
library (class)

# Part A: Let's load the data
auto <- read.csv("Auto.csv", header = TRUE)

# Part B: Let's create the binary variable 'mpg01'
median_mpg <- median(auto$mpg)
print(paste("The value of the median mpg is: ",median_mpg))

# We're going to create a new column for mpg01 and set a condition: If mpg > median_mpg, return 1, else return 0
# and since the mpg would be redundant, let's drop it.
auto$mpg01 <- ifelse(auto$mpg > median_mpg, 1, 0)
auto <- auto %>% dplyr::select(-mpg)

# Part C: Exploratory Data Analysis
# Since mpg01's value is either a 1 or a 0, let's not consider it as an integer value, 
# but rather a factor.
auto$mpg01 <- as.factor(auto$mpg01)

# Convert the data from wide to long format so that we can plot it on the same graph.
auto_long <- auto %>% pivot_longer(cols = -mpg01, # Grab all columns except for mpg01
    names_to = "characteristic", values_to = "value")

eda_plot <- ggplot(auto_long, aes(x = mpg01, y = value, fill = mpg01)) + geom_boxplot() +
    facet_wrap(~characteristic, scales = "free_y") + labs(title = "How Car Characteristics Relate to MPG", 
    x = "MPG (0 = Less than Median, 1 = Greater than Median)", y = "Value") + theme_minimal()

# Now, let's do a correlation plot to compare the variables.
auto$mpg01 <- as.numeric(as.character(auto$mpg01)) # Need to convert mpg01 back to a numeric to include it in the calculation
num_val <- auto %>% select_if(is.numeric) # Let's only take the numeric vals.
corr_mat <- cor(num_val)
corrplot(corr_mat, method = "number", type = "upper", order = "hclust", tl.col = "black", tl.srt = 45, diag = FALSE, title = "Correlation Matrix", mar=c(0,0,1,0))

# Part D: Let's split the data into training and testing
set.seed(7406)

# Implement a 100 trial Monte Carlo Cross Validation
trials <- 100
error_num <- c()

for (i in 1:trials) { # loop to run 100 times
train_idx <- sample(1:nrow(auto), 2/3 * nrow(auto)) # using sample code from class, using 2/3 because it gives a decent amount of data to either set
train_set <- auto[train_idx,] 
test_set <- auto[-train_idx,]

train_set$mpg01 <- as.factor(train_set$mpg01)
test_set$mpg01 <- as.factor(test_set$mpg01)

# Part E: Let's perform classification methods on the data
# 4: Logistic Regression
auto_log <- glm(mpg01 ~ cylinders + weight + displacement + horsepower + year, data = train_set, family = binomial)

# Test set predictions
test_prob <- predict(auto_log, test_set, type = "response")
test_class <- ifelse(test_prob > 0.5, 1, 0) # if prob > 0.5, return 1, else 0

# Build a confusion matrix
confusion <- table(test_class, test_set$mpg01)

# Test error for Logistic Regression
log_err <- mean(test_class != test_set$mpg01)
error_num <- c(error_num, log_err) # Run CV
}

avg_log_err <- mean(error_num)
std_log_err <- sd(error_num)
print(paste("The average test error for Logistic Regression over", trials, "trials:",round(avg_log_err, 4)))
print(paste("The standard deviation of the test error for Logistic Regression: ", round(std_log_err, 4)))

# 1: LDA
for (i in 1:trials) { # loop to run 100 times
lda_model <- lda(mpg01 ~ cylinders + weight + displacement + horsepower + year, data = train_set)
lda_pred <- predict(lda_model, test_set)
lda_err <- lda_pred$class # we're separating the predicted classes from the rest of the model

# LDA Confusion matrix
lda_confusion <- table(lda_err, test_set$mpg01)

# LDA Test error
lda_test_err <- mean(lda_err != test_set$mpg01)
error_num <- c(error_num, lda_test_err) # Run CV
}

avg_lda_err <- mean(error_num)
std_lda_err <- sd(error_num)
print(paste("The average test error for LDA over", trials, "trials:",round(avg_lda_err, 4)))
print(paste("The standard deviation of the test error for LDA: ", round(std_lda_err, 4)))

# 2: QDA
for (i in 1:trials) { # loop to run 100 times
qda_model <- qda(mpg01 ~ cylinders + weight + displacement + horsepower + year, data = train_set)
qda_pred <- predict(qda_model, test_set)
qda_err <- qda_pred$class

# QDA Confusion matrix
qda_confusion <- table(qda_err, test_set$mpg01)

# QDA Test error
qda_test_err <- mean(qda_err != test_set$mpg01)
error_num <- c(error_num, qda_test_err) # Run CV
}

avg_qda_err <- mean(error_num)
std_qda_err <- sd(error_num)
print(paste("The average test error for QDA over", trials, "trials:",round(avg_qda_err, 4)))
print(paste("The standard deviation of the test error for QDA: ", round(std_qda_err, 4)))

# 3: Naive Bayes
for (i in 1:trials) { # loop to run 100 times
nb_model <- naiveBayes(mpg01 ~ cylinders + weight + displacement + horsepower + year, data = train_set)
nb_pred <- predict(nb_model, test_set)

# Naive Bayes Confusion matrix
nb_confusion <- table(nb_pred, test_set$mpg01)

# Naive Bayes Test error
nb_test_err <- mean(nb_pred != test_set$mpg01)
error_num <- c(error_num, nb_test_err) # Run CV
}

avg_nb_err <- mean(error_num)
std_nb_err <- sd(error_num)
print(paste("The average test error for Naive Bayes over", trials, "trials:",round(avg_nb_err, 4)))
print(paste("The standard deviation of the test error for Naive Bayes: ", round(std_nb_err, 4)))

# 5: KNN
k_values <- seq(1, 30, 2)
k_error_rate <- matrix(nrow = trials, ncol = length(k_values))
colnames(k_error_rate) <- paste("k = ", k_values)

for (i in 1:trials) { # This is the outer loop which will do the cross validation trials
    train_idx <- sample(1:nrow(auto), 2/3 * nrow(auto))
    train_set <- auto[train_idx,]
    test_set <- auto[-train_idx,]

    train_pred <- train_set %>% dplyr::select(cylinders, weight, displacement, horsepower, year)
    test_pred <- test_set %>% dplyr::select(cylinders, weight, displacement, horsepower, year)
    train_labels <- train_set$mpg01
    test_labels <- test_set$mpg01

    for (k in 1:length(k_values)) { # This is the inner loop that handles the k values
        k_val = k_values[k]
        knn_mod <- knn(train = scale(train_pred), test = scale(test_pred), cl = train_labels, k = k_val)
        k_error_rate[i, k] <- mean(knn_mod != test_labels)
    }
}

avg_knn_err <- colMeans(k_error_rate)
std_knn_err <- apply(k_error_rate, 2, sd)
print(paste("The average test error for KNN over", trials, "trials:",round(avg_knn_err, 4)))
print(paste("The standard deviation of the test error for KNN: ", round(std_knn_err, 4)))
