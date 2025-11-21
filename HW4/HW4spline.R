pacman::p_load(tidyverse, ggplot2, matrixStats, magrittr)

# 2. Define the true "Mexican Hat" function
mexican_hat <- function(x) {
  y <- (1 - (x^2)) * exp(-0.5 * (x^2))
  return(y)
}

tryCatch({
  x_data <- read.csv("HW04part2-1.x.csv")
  x_i <- x_data$x # Extract the 'x' column into a vector
}, error = function(e) {
  stop("Error: File not found.")
})

# This is the required spar value from the homework
spar_value <- 0.7163

# 4. Generate a single noisy dataset for visualization
set.seed(7406)
fx_i <- mexican_hat(x_i)
Y_i <- fx_i + rnorm(length(x_i), mean = 0, sd = 0.2)

# 5. Fit the smoothing spline using the specified 'spar' parameter
# This is the line of code that cannot be directly translated to Python.
splines_model <- smooth.spline(Y_i ~ x_i, spar = spar_value)

# 6. Create a dataframe for plotting
plot_data <- data.frame(
  x = x_i,
  Noisy_Y = Y_i,
  True_Function = fx_i,
  Fitted_Spline = splines_model$y
)

# 7. Visualize the result
print(ggplot(plot_data, aes(x = x)) +
  geom_point(aes(y = Noisy_Y), color = "grey", alpha = 0.6) +
  geom_line(aes(y = True_Function, color = "True Function"), linewidth = 1) +
  geom_line(aes(y = Fitted_Spline, color = "Fitted Spline"), linewidth = 1.2) +
  scale_color_manual(values = c("True Function" = "black", "Fitted Spline" = "darkgreen")) +
  ggtitle(paste("Smoothing Spline with spar =", spar_value)) +
  theme_minimal() +
  labs(y = "Value", color = "Legend")
)