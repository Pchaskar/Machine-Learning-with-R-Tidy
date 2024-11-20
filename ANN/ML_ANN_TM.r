# Neural Networks

library(rstudioapi)  # For setting working directory

# Set working directory to the current script's location
setwd(dirname(getActiveDocumentContext()$path))

# Load necessary libraries
library(tidymodels)  # For modeling and workflows
library(dplyr)       # For data manipulation
library(ggplot2)     # For visualization
library(NeuralNetTools)  # For neural network visualization

# Set seed for reproducibility
set.seed(12345)

# Step 1: Data Preparation -----
# Load the data
concrete <- read.csv("concrete.csv")

# Inspect the structure of the dataset
str(concrete)

# Split the data into training, validation, and test sets
set.seed(12345)
data_split <- initial_split(concrete, prop = 0.75)  # 75% train + validation, 25% test
train_validation <- training(data_split)
concrete_test <- testing(data_split)

# Further split the training data into training and validation sets
validation_split <- initial_split(train_validation, prop = 0.8)  # 80% train, 20% validation
concrete_train <- training(validation_split)
concrete_validation <- testing(validation_split)

# Step 2: Define the Model -----
# Specify a neural network model with tuning parameters
nn_model <- mlp(hidden_units = tune(), penalty = tune()) %>%
  set_engine("nnet") %>%
  set_mode("regression")

# Step 3: Create a Recipe -----
# Define a recipe for preprocessing
# Normalize predictors within the recipe
concrete_recipe <- recipe(strength ~ ., data = concrete_train) %>%
  step_normalize(all_predictors())  # Normalize predictors only

# Step 4: Combine Model and Recipe into a Workflow -----
# Combine the recipe and model into a workflow
concrete_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(concrete_recipe)

# Step 5: Hyperparameter Tuning -----
# Perform 10-fold cross-validation
folds <- vfold_cv(concrete_train, v = 10)

# Define a grid of hyperparameters for tuning
grid <- grid_regular(
  hidden_units(range = c(1, 10)),  # Number of hidden units
  penalty(range = c(0.001, 0.1)),  # Regularization parameter
  levels = 5
)

# Tune the model using grid search
tuned_results <- tune_grid(
  concrete_workflow,
  resamples = folds,
  grid = grid,
  metrics = metric_set(rmse, rsq)
)

# Display tuning results
tuned_results %>%
  collect_metrics()

# Select the best hyperparameters based on RMSE
best_params <- tuned_results %>%
  tune::select_best(metric = "rmse")

# Step 6: Finalize the Model -----
# Finalize the workflow with the best hyperparameters
final_workflow <- concrete_workflow %>%
  finalize_workflow(best_params)

# Train the finalized model on the training dataset
final_fit <- final_workflow %>%
  fit(data = concrete_train)

# Step 7: Evaluate the Model -----
# Evaluate on the validation dataset
validation_predictions <- predict(final_fit, new_data = concrete_validation) %>%
  bind_cols(concrete_validation %>% select(strength))

# Calculate validation metrics
validation_metrics <- validation_predictions %>%
  metrics(truth = strength, estimate = .pred)

# Display validation metrics
print(validation_metrics)

# Evaluate on the test dataset
test_predictions <- predict(final_fit, new_data = concrete_test) %>%
  bind_cols(concrete_test %>% select(strength))

# Calculate test metrics
test_metrics <- test_predictions %>%
  metrics(truth = strength, estimate = .pred)

# Display test metrics
print(test_metrics)

# Step 8: Visualize Results -----
# Plot actual vs predicted values for the test set
ggplot(test_predictions, aes(x = strength, y = .pred)) +
  geom_point(alpha = 0.7) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(
    title = "Actual vs Predicted Strength (Test Set)",
    x = "Actual Strength",
    y = "Predicted Strength"
  ) +
  theme_minimal()

cor(test_predictions$strength, test_predictions$.pred)

# Step 9: Visualize the Neural Network -----
# Extract the trained model
nn_final_model <- extract_fit_parsnip(final_fit)

# Visualize the neural network structure
if ("nnet" %in% class(nn_final_model$fit)) {
  plotnet(nn_final_model$fit, circle_col = "lightgreen", bord_col = "black")
}

# End of Script -----
