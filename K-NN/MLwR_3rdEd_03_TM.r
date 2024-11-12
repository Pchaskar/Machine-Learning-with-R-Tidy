##### Classification using Nearest Neighbors --------------------
# Prasad Chaskar

# Path to the scripts
library("rstudioapi")

# Set working directory
setwd(dirname(getActiveDocumentContext()$path))  # Set the working directory to the parent directory of the active document

# Load required libraries
library(tidymodels)
library(kknn)
library(pROC)  # For AUC calculation and ROC plotting
library(ggplot2) # For visualization
library(MLmetrics) # For MCC

# Step 1: Load and Prepare Data
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)

# Remove unnecessary columns (e.g., ID)
wbcd <- wbcd %>%
  select(-id)

# Recode the 'diagnosis' column as a factor
wbcd <- wbcd %>%
  mutate(diagnosis = factor(
    diagnosis,
    levels = c("B", "M"),
    labels = c("Benign", "Malignant")
  ))

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

# Step 3: Preprocessing
# Use a recipe for consistent preprocessing steps
rec <- recipe(diagnosis ~ ., data = train_data) %>%
  step_normalize(all_predictors())  # Normalize all predictors (numeric features)

# Step 4: Define the Model
# parsnip_addin() for automatic parameter generation
knn_spec <- nearest_neighbor(neighbors = tune()) %>%  # Tune neighbors (k)
  set_engine("kknn") %>%
  set_mode("classification")

# Step 5: Define a Workflow
knn_workflow <- workflow() %>%
  add_recipe(rec) %>%  # Add the preprocessing recipe
  add_model(knn_spec)  # Add the model to the workflow

# Step 6: Cross-Validation Setup (using validation set)
set.seed(123)
cv_folds <- vfold_cv(validation_data, v = 10, strata = diagnosis)

# Step 7: Hyperparameter Tuning
# Define grid for neighbors (k)
knn_grid <- grid_regular(neighbors(range = c(1, 31)), levels = 15)

# Tune the model
knn_results <- tune_grid(
  knn_workflow,
  resamples = cv_folds,
  grid = knn_grid,
  metrics = metric_set(accuracy)
)

# Step 8: Get the Best K (neighbors)
best_k <- knn_results %>%
  select_best(metric = "accuracy")

cat("Best value of k: ", best_k$neighbors, "\n")

# Step 9: Finalize the Workflow with the Best k
final_knn <- finalize_workflow(knn_workflow, best_k)

# Step 10: Fit the Final Model
final_fit <- final_knn %>%
  last_fit(train_val_split)  # Fit using the original train_val_split which includes 50% training and 30% validation

# Step 11: Evaluate the Model
# Collect metrics from the final model
final_metrics <- collect_metrics(final_fit)
print(final_metrics)

# Step 12: Make Predictions with Final Model
# ---------------------------
# Extract the fitted model and the preprocessed test data from the last_fit() result
final_model <- extract_workflow(final_fit)  # Extract the final workflow object

# Make predictions on the test data using the final fitted model
test_predictions <- predict(final_model, new_data = test_data)

# Optionally, view the predictions along with the actual labels
predictions_with_labels <- test_predictions %>%
  bind_cols(test_data %>% select(diagnosis))  # Assuming 'diagnosis' is the true label

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

# load the "gmodels" library
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(
  x = predictions_with_labels$diagnosis,
  y = predictions_with_labels$.pred_class,
  prop.chisq = FALSE
)

# End of script
