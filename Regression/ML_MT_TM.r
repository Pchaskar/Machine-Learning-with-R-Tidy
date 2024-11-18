# Cubist Model Tree

# Path to the scripts
library("rstudioapi")

# Set working directory
setwd(dirname(getActiveDocumentContext()$path))  # Set the working directory to the parent directory of the active document

# Load required libraries
library(tidymodels)

# Load required libraries
library(Cubist)

# Load the wine dataset
wine <- read.csv("whitewines.csv")
glimpse(wine)

# Split the data into training and testing sets
set.seed(123)
wine_split <- initial_split(wine, prop = 0.75)
wine_train <- training(wine_split)
wine_test <- testing(wine_split)

# Train a Cubist model tree
cubist_model <- cubist(x = wine_train %>% dplyr::select(-quality), 
                       y = wine_train$quality)

# Generate predictions for the test set
cubist_predictions <- predict(cubist_model, wine_test %>% dplyr::select(-quality))

# Evaluate the Cubist model
cubist_metrics <- data.frame(
  truth = wine_test$quality,
  prediction = cubist_predictions
) %>%
  metrics(truth = truth, estimate = prediction)

print(cubist_metrics)

# correlation between the predicted and true values
cor(cubist_predictions, wine_test$quality)

# function to calculate the mean absolute error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

# mean absolute error between predicted and actual values
MAE(cubist_predictions, wine_test$quality)

