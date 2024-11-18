# Linear Regression with Tidymodels 

# Path to the scripts
library("rstudioapi")

# Set working directory
setwd(dirname(getActiveDocumentContext()$path))  # Set the working directory to the parent directory of the active document

# Load required libraries
library(tidymodels)
library(dplyr) # For explicit dplyr calls
library(psych) # For visualizations
library(ggplot2) # For plotting

# Load the dataset
insurance <- read.csv("insurance.csv", stringsAsFactors = TRUE)
glimpse(insurance)

# Split the data into training and testing sets
set.seed(123)
insurance_split <- initial_split(insurance, prop = 0.75)
insurance_train <- training(insurance_split)
insurance_test <- testing(insurance_split)

# Explore and summarize the data
dplyr::glimpse(insurance_train)  # Summary of the training data
summary(insurance_train$expenses)  # Summary of the target variable
ggplot(insurance_train, aes(x = expenses)) + 
  geom_histogram(binwidth = 1000, fill = "blue", color = "black") + 
  labs(title = "Histogram of Medical Expenses", x = "Expenses", y = "Frequency")

# Analyze relationships using correlations
correlations <- insurance_train %>%
  dplyr::select(age, bmi, children, expenses) %>%
  cor()
print(correlations)

# Pair plots
pairs.panels(insurance_train %>% dplyr::select(age, bmi, children, expenses), 
             method = "pearson", hist.col = "blue", main = "Scatterplot Matrix")

# Prepare the model specification
lm_spec <- linear_reg() %>%
  set_engine("lm")

# Create a recipe for preprocessing
insurance_recipe <- recipe(expenses ~ ., data = insurance_train)

# Create a workflow and fit the model
lm_workflow <- workflow() %>%
  add_recipe(insurance_recipe) %>%
  add_model(lm_spec)

lm_fit <- lm_workflow %>% fit(data = insurance_train)
summary(lm_fit)

# Evaluate on test data
lm_predictions <- predict(lm_fit, insurance_test) %>%
  bind_cols(insurance_test)

# Calculate performance metrics
lm_metrics <- lm_predictions %>%
  metrics(truth = expenses, estimate = .pred)

cor(lm_predictions$expenses, lm_predictions$.pred)

plot(lm_predictions$.pred, lm_predictions$expenses)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)

print(lm_metrics)

# Improving model performance ----
# Add a higher-order "age" term to capture non-linear relationships
insurance_train <- insurance_train %>%
  mutate(age2 = age^2)

insurance_test <- insurance_test %>%
  mutate(age2 = age^2)

# Add an indicator variable for BMI >= 30
insurance_train <- insurance_train %>%
  mutate(bmi30 = ifelse(bmi >= 30, 1, 0))

insurance_test <- insurance_test %>%
  mutate(bmi30 = ifelse(bmi >= 30, 1, 0))

# Update the recipe
improved_recipe <- recipe(expenses ~ ., data = insurance_train) %>%
  step_mutate(
    age2 = age^2,                    # Higher-order term for age
    bmi30 = ifelse(bmi >= 30, 1, 0)  # Indicator for BMI >= 30
  ) %>%
  step_interact(
    terms = ~ bmi30:smoker           # Interaction term between bmi30 and smoker
  )

# Create a new workflow
improved_workflow <- workflow() %>%
  add_recipe(improved_recipe) %>%
  add_model(lm_spec)

# Train the improved model
improved_fit <- improved_workflow %>% fit(data = insurance_train)

# Make predictions with the improved model
improved_predictions <- predict(improved_fit, insurance_test) %>%
  bind_cols(insurance_test)

# Evaluate the improved model
improved_metrics <- improved_predictions %>%
  metrics(truth = expenses, estimate = .pred)

# Print improved metrics
print(improved_metrics)

cor(improved_predictions$expenses, improved_predictions$.pred)

plot(improved_predictions$.pred, improved_predictions$expenses)
abline(a = 0, b = 1, col = "red", lwd = 3, lty = 2)
