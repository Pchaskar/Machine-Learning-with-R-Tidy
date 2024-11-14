# Load required libraries
library(tidymodels)
library(dplyr)
library(ggplot2)
library(tune)
library(themis)


# Step 1: Load and Prepare Data
credit <- read.csv("credit.csv", stringsAsFactors = TRUE)

# Explore data
glimpse(credit)

# Check class distribution
credit %>% 
  dplyr::count(default)

# Step 2: Split Data into Training, Validation, and Test Sets
set.seed(123)

# Initial train-test split
credit_split <- initial_split(credit, prop = 0.8, strata = default)
credit_train_val <- training(credit_split)
credit_test <- testing(credit_split)

# Further split train-val data
credit_train_val_split <- initial_split(credit_train_val, prop = 0.8, strata = default)
credit_train <- training(credit_train_val_split)
credit_validation <- testing(credit_train_val_split)

# Step 3: Preprocessing
# Create a recipe for preprocessing
# Adjust the recipe to handle class imbalance
# Updated recipe with proper handling of numeric and categorical variables
tree_recipe <- recipe(default ~ ., data = credit_train) %>%
  step_dummy(all_nominal_predictors()) %>%     # Convert categorical variables to dummy variables
  step_normalize(all_numeric_predictors()) %>% # Normalize only numeric variables
  step_upsample(default)                       # Upsample the minority class (requires themis)

# Step 4: Model Specification
# Define a tree model with hyperparameters to tune
tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Create a workflow
tree_workflow <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_spec)

# Set up cross-validation folds
set.seed(123)
cv_folds <- vfold_cv(credit_train, v = 10, strata = default)

# Define a grid of hyperparameters to search
tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# Tune the model
tree_tune_results <- tune_grid(
  tree_workflow,
  resamples = cv_folds,
  grid = tree_grid,
  metrics = metric_set(accuracy, roc_auc)
)

# Select the best hyperparameters
best_tree_params <- tree_tune_results %>%
  tune::select_best(metric = "accuracy")

# Finalize the workflow with the best parameters
final_tree_workflow <- tree_workflow %>%
  finalize_workflow(best_tree_params)

# Train on training data and evaluate on validation data
final_fit <- final_tree_workflow %>%
  last_fit(split = credit_train_val_split)

# Collect metrics
final_metrics <- final_fit %>%
  collect_metrics()
print(final_metrics)

# Extract the workflow with the fitted model
final_workflow <- extract_workflow(final_fit)

# Make predictions on the test dataset
test_predictions <- predict(final_workflow, new_data = credit_test) %>%
  bind_cols(credit_test)

# View the predictions
head(test_predictions)

# Evaluate predictions
conf_mat(test_predictions, truth = default, estimate = .pred_class)

# load the "gmodels" library
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(test_predictions$.pred_class, test_predictions$default, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual')
)