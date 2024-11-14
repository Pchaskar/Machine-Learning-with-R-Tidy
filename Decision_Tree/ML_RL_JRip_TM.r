# Load required libraries
library(dplyr)
library(RWeka)   # For JRip model
library(caret)    # For confusion matrix

# Step 1: Load and Explore Data
mushrooms <- read.csv("mushrooms.csv", stringsAsFactors = TRUE)

# Examine the structure of the data frame
str(mushrooms)

# Drop the veil_type feature
mushrooms$veil_type <- NULL

# Examine the class distribution
table(mushrooms$type)

# Step 2: Train-test split (80% training, 20% testing)
set.seed(123)
train_index <- createDataPartition(mushrooms$type, p = 0.8, list = FALSE)
mushroom_train <- mushrooms[train_index, ]
mushroom_test <- mushrooms[-train_index, ]

# Step 3: Train the JRip model using RWeka
mushroom_JRip <- JRip(type ~ ., data = mushroom_train)

# Step 4: Model Performance Evaluation
# Print model summary
summary(mushroom_JRip)

# Step 5: Make Predictions on the Test Set
mushroom_JRip_pred <- predict(mushroom_JRip, mushroom_test)

# Step 6: Confusion Matrix
conf_matrix <- confusionMatrix(mushroom_JRip_pred, mushroom_test$type)
print(conf_matrix)

# Step 7: Accuracy Evaluation
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Accuracy: ", round(accuracy, 4)))
