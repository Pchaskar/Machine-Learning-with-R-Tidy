# Path to the scripts
library("rstudioapi")

# Set working directory
setwd(dirname(getActiveDocumentContext()$path))  # Set the working directory to the parent directory of the active document

# Load required libraries
library(tidymodels)
library(naivebayes)
library(pROC)  # For AUC calculation and ROC plotting
library(ggplot2) # For visualization
library(MLmetrics) # For MCC
library(gmodels)  # For cross-tabulation
library(dplyr) # For dplyr functions

# Step 1: Load and Prepare Data
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)

# Remove unnecessary columns (e.g., ID)
wbcd <- wbcd %>%
  dplyr::select(-id)

# Recode the 'diagnosis' column as a factor
wbcd <- wbcd %>%
  dplyr::mutate(diagnosis = factor(
    diagnosis,
    levels = c("B", "M"),
    labels = c("Benign", "Malignant")
  ))

# Display counts of each diagnosis type
table(wbcd$diagnosis)

# Step 2: Split the Data
set.seed(123) # For reproducibility

# Split into Train+Validation (80%) and Test (20%)
train_val_split <- initial_split(wbcd, prop = 0.8, strata = diagnosis)
train_val_data <- training(train_val_split)
test_data <- testing(train_val_split)

# Now split the Train+Validation data into Train (50%) and Validation (30%)
train_split <- initial_split(train_val_data, prop = 0.8, strata = diagnosis)  # 50% for training
train_data <- training(train_split)
validation_data <- testing(train_split)  # 30% for validation (because it's 50% of 80%)

# Step 3: Preprocessing with recipe
# Use a recipe for consistent preprocessing steps
rec <- recipe(diagnosis ~ ., data = train_data) %>%
  step_normalize(all_predictors())  # Normalize all predictors (numeric features)

# Step 4: Define the Naive Bayes Model Specification
# Define Naive Bayes model with a tuning parameter for Laplace smoothing
nb_spec <- naive_Bayes(smoothness = tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

# Step 5: Define a Workflow
# Create a workflow that combines the recipe and model specification
nb_workflow <- workflow() %>%
  add_recipe(rec) %>%  # Add preprocessing recipe
  add_model(nb_spec)  # Add Naive Bayes model

# Step 6: Cross-Validation Setup (using validation set)
set.seed(123)
cv_folds <- vfold_cv(validation_data, v = 10, strata = diagnosis)

# Step 7: Hyperparameter Tuning for Laplace smoothing
# Define a grid for the smoothness (Laplace) parameter
laplace_grid <- grid_regular(smoothness(range = c(0.5, 3)), # Tune Laplace smoothing factor between 0.5 and 3
                             levels = 10)

# Tune the model using the grid and cross-validation
nb_tune_results <- tune_grid(
  nb_workflow,
  resamples = cv_folds,
  grid = laplace_grid,
  metrics = metric_set(accuracy)
)

# Step 8: Get the Best Laplace Parameter
best_nb_tune <- tune::select_best(nb_tune_results, metric = "accuracy")

cat("Best Laplace smoothing value: ", best_nb_tune$smoothness, "\n")

# Step 9: Finalize the Workflow with the Best Laplace Parameter
final_nb_workflow <- finalize_workflow(nb_workflow, best_nb_tune)

# Step 10: Fit the Final Model
final_nb_fit <- final_nb_workflow %>%
  last_fit(train_val_split)  # Fit the model on the train_val_split

# Step 11: Evaluate the Model
# Collect and view final performance metrics
final_metrics <- collect_metrics(final_nb_fit)
print(final_metrics)

# Step 12: Make Predictions with Final Model
# ---------------------------
# Extract the fitted model and the preprocessed test data from the last_fit() result
final_model <- extract_workflow(final_nb_fit)  # Extract the final workflow object

# Make predictions on the test data using the final fitted model
test_predictions <- predict(final_model, new_data = test_data)

# Optionally, view the predictions along with the actual labels
predictions_with_labels <- test_predictions %>%
  dplyr::bind_cols(test_data %>% dplyr::select(diagnosis))  # Assuming 'diagnosis' is the true label

# View the first few predictions
head(predictions_with_labels)

# Step 13: Confusion Matrix
# ---------------------------
# Confusion Matrix to assess the classification performance
conf_matrix_yardstick <- predictions_with_labels %>%
  conf_mat(truth = diagnosis, estimate = .pred_class)

# Print the confusion matrix
print(conf_matrix_yardstick)

# Step 14: MCC (Matthews Correlation Coefficient)
# ---------------------------
# Calculate MCC for the predictions
manual_conf_matrix <- table(predictions_with_labels$.pred_class,
                            predictions_with_labels$diagnosis)

TP <- manual_conf_matrix[2, 2]  # True Positive
TN <- manual_conf_matrix[1, 1]  # True Negative
FP <- manual_conf_matrix[1, 2]  # False Positive
FN <- manual_conf_matrix[2, 1]  # False Negative

# Compute Matthews Correlation Coefficient (MCC)
mcc_value <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
cat("Matthews Correlation Coefficient (MCC): ", mcc_value, "\n")

# Step 15: ROC Curve
# ---------------------------
# Use pROC for ROC curve and AUC calculation
roc_curve <- roc(
  predictions_with_labels$diagnosis,
  as.numeric(predictions_with_labels$.pred_class) - 1
)

# Plot ROC curve
plot(roc_curve,
     main = "ROC Curve",
     col = "blue",
     lwd = 2)

# AUC Score
cat("AUC: ", auc(roc_curve), "\n")

# Step 16: Precision-Recall Curve
# ---------------------------
# For Precision-Recall curve, we use pROC
pr_curve <- roc(
  predictions_with_labels$diagnosis,
  as.numeric(predictions_with_labels$.pred_class) - 1,
  plot = TRUE,
  print.auc = TRUE,
  col = "green",
  main = "Precision-Recall Curve"
)

# Create the cross tabulation of predicted vs. actual
CrossTable(
  x = predictions_with_labels$diagnosis,
  y = predictions_with_labels$.pred_class,
  prop.chisq = FALSE
)

# End of script
