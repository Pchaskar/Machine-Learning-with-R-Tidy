# Load necessary libraries
library(tidymodels)  # For ML workflow
library(discrim)     # For Naive Bayes implementation
library(textrecipes) # For text preprocessing
library(tidyverse)   # For data manipulation and visualization
library(wordcloud)   # For word cloud visualization
library(stopwords)
library(klaR)
library(tidytext)


# Set a seed for reproducibility
set.seed(123)

# Step 1: Load and Explore Data
sms_raw <- read_csv("sms_spam.csv")
sms_raw <- sms_raw %>%
  mutate(type = as_factor(type))

# Split the data into training and testing sets
sms_split <- initial_split(sms_raw, prop = 0.75, strata = type)
sms_train <- training(sms_split)
sms_test <- testing(sms_split)

# Step 2: Data Preprocessing
sms_recipe <- recipe(type ~ text, data = sms_train) %>%
  step_tokenize(text) %>%                           # Tokenize the text column
  step_stopwords(text) %>%                          # Remove stopwords
  step_stem(text) %>%                               # Apply stemming
  step_tokenfilter(text, max_tokens = 500) %>%      # Keep only the top 500 tokens
  step_tf(text)                                     # Convert to term frequency

# Step 3: Define the Model
nb_model <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("klaR")  # Supported by discrim

# Step 4: Create a Workflow
sms_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(nb_model)

# Step 5: Train the Model
sms_fit <- sms_workflow %>%
  fit(data = sms_train)

# Step 6: Evaluate Model Performance
sms_predictions <- sms_fit %>%
  predict(new_data = sms_test) %>%
  bind_cols(sms_test)

# load the "gmodels" library
library(gmodels)

# Create the cross tabulation of predicted vs. actual
CrossTable(sms_predictions$.pred_class, sms_predictions$type, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual')
)

sms_metrics <- sms_predictions %>%
  metrics(truth = type, estimate = .pred_class)

print(sms_metrics)

# Step 7: Visualize Word Frequencies
sms_train %>%
  filter(type == "spam") %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  with(wordcloud(word, n, max.words = 40, scale = c(3, 0.5)))

sms_train %>%
  filter(type == "ham") %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  with(wordcloud(word, n, max.words = 40, scale = c(3, 0.5)))

# Step 8: Improve Model Performance
nb_model_laplace <- naive_Bayes(smoothness = 1) %>%
  set_mode("classification") %>%
  set_engine("klaR")

sms_workflow_laplace <- sms_workflow %>%
  update_model(nb_model_laplace)

sms_fit_laplace <- sms_workflow_laplace %>%
  fit(data = sms_train)

sms_predictions_laplace <- sms_fit_laplace %>%
  predict(new_data = sms_test) %>%
  bind_cols(sms_test)

sms_metrics_laplace <- sms_predictions_laplace %>%
  metrics(truth = type, estimate = .pred_class)

print(sms_metrics_laplace)
