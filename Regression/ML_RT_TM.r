# Regression Trees with Tidymodels

# Path to the scripts
library("rstudioapi")

# Set working directory
setwd(dirname(getActiveDocumentContext()$path))  # Set the working directory to the parent directory of the active document

# Load required libraries
library(tidymodels)
library(rpart.plot)

# Load the wine dataset
wine <- read.csv("whitewines.csv")
glimpse(wine)

# Split the data into training and testing sets
set.seed(123)
wine_split <- initial_split(wine, prop = 0.75)
wine_train <- training(wine_split)
wine_test <- testing(wine_split)

# Create a recipe
wine_recipe <- recipe(quality ~ ., data = wine_train)

# Specify the regression tree model
tree_spec <- decision_tree(
  mode = "regression",
  cost_complexity = tune(), # To optimize the complexity parameter
  tree_depth = tune()       # To optimize the depth
) %>%
  set_engine("rpart", model = TRUE)

# Create a workflow
tree_workflow <- workflow() %>%
  add_recipe(wine_recipe) %>%
  add_model(tree_spec)

# Perform cross-validation to tune hyperparameters
set.seed(123)
tree_resamples <- vfold_cv(wine_train, v = 5)

tree_grid <- grid_regular(
  cost_complexity(range = c(-3, -1)), # log-scale for better tuning
  tree_depth(range = c(1, 10)),
  levels = 10
)

tree_tune_results <- tree_workflow %>%
  tune_grid(
    resamples = tree_resamples,
    grid = tree_grid,
    metrics = metric_set(rmse, rsq)
  )

# Select the best hyperparameters
best_tree <- tree_tune_results %>%
  tune::select_best(metric = "rmse")

# Finalize the workflow with the best parameters
final_tree_workflow <- tree_workflow %>%
  finalize_workflow(best_tree)

# Train the final model
final_tree_fit <- final_tree_workflow %>%
  fit(data = wine_train)

# Extract the fitted rpart model
rpart_model <- extract_fit_engine(final_tree_fit)

# Basic Decision Tree with Increased Text Size
rpart.plot(rpart_model, 
           digits = 3, 
           main = "Basic Decision Tree", 
           cex = 0.7)  # Increase text size

# Enhanced Decision Tree with Larger Text and Adjustments
rpart.plot(rpart_model, 
           digits = 4, 
           fallen.leaves = TRUE, 
           type = 3, 
           extra = 101, 
           main = "Enhanced Decision Tree", 
           cex = 0.7)  # Increase text size

# Evaluate the model on the test data
tree_predictions <- predict(final_tree_fit, wine_test) %>%
  bind_cols(wine_test)

# Calculate performance metrics
tree_metrics <- tree_predictions %>%
  metrics(truth = quality, estimate = .pred)

# compare the correlation
cor(tree_predictions$.pred, tree_predictions$quality)

# function to calculate the mean absolute error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

# mean absolute error between predicted and actual values
MAE(tree_predictions$.pred, tree_predictions$quality)

print(tree_metrics)