# Load required libraries
library(tidymodels)
library(dplyr)

# Step 1: Load and Explore Data
mushrooms <- read.csv("mushrooms.csv", stringsAsFactors = TRUE)

# Drop unnecessary column
mushrooms <- mushrooms %>%
  dplyr::select(-veil_type)

# Examine class distribution
mushrooms %>%
  dplyr::count(type)

# Step 2: Split Data into Training and Testing Sets
set.seed(123)

# Initial split into training and testing sets
mushroom_split <- initial_split(mushrooms, prop = 0.8, strata = type)
mushroom_train <- training(mushroom_split)
mushroom_test <- testing(mushroom_split)

# Step 3: Preprocessing
mushroom_recipe <- recipe(type ~ ., data = mushroom_train) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Step 4: Model Specification
# Specify a rule-based model using C5.0 (no tuning)
rule_spec <- boost_tree(
  trees = 50,  # Set a fixed number of trees
  min_n = 5    # Set a fixed minimum number of data points in a terminal node
) %>%
  set_engine("C5.0") %>%
  set_mode("classification")

# Step 5: Workflow
rule_workflow <- workflow() %>%
  add_recipe(mushroom_recipe) %>%
  add_model(rule_spec)

# Step 6: Fit Final Model on Training Data
final_rule_fit <- rule_workflow %>%
  fit(data = mushroom_train)

# Step 7: Make Predictions on the Test Set
rule_test_predictions <- final_rule_fit %>%
  predict(new_data = mushroom_test) %>%
  dplyr::bind_cols(mushroom_test %>% dplyr::select(type))

# Confusion matrix
rule_test_predictions %>%
  yardstick::conf_mat(truth = type, estimate = .pred_class)

# Evaluate the model on the test set
rule_test_metrics <- rule_test_predictions %>%
  yardstick::metrics(truth = type, estimate = .pred_class)
rule_test_metrics
