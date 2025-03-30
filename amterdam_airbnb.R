rm(list=ls())
# Airbnb Price Prediction Analysis -  Part 1
# =================================
# This R script performs data preparation, model training, evaluation, and prediction 
# for Airbnb listing price prediction. We use raw price (not log-transformed) as the target.

# Setup and Data Loading ----
# Load required libraries
library(tidyverse)   # for data manipulation (dplyr, tidyr, ggplot2, etc.)
library(stringr)     # for string operations
library(readr)       # for fast CSV reading
library(caret)       # for utility functions (e.g. createDataPartition, RMSE)
library(glmnet)      # for LASSO/Ridge regression
library(randomForest)# for Random Forest
library(xgboost)     # for XGBoost
# (Install any missing packages if necessary, e.g.: install.packages("xgboost"))

# Read the core dataset (Amsterdam 2024 Q4)
core <- read_csv("Amsterdam_core(2024Q4).csv")
# 1. Data Cleaning and Feature Engineering (Amsterdam Core) ----
# Clean the price column: remove currency symbols, spaces, commas, then convert to numeric
core <- core %>%
  mutate(price = str_remove_all(price, "[$€\\s,]"),
         price = as.numeric(price)) %>%
  filter(!is.na(price) & price > 0)  # filter out missing or zero prices

# Extract numeric bathroom count from bathrooms_text (e.g. "1 bath", "1.5 baths")
core <- core %>%
  mutate(bathrooms = str_extract(bathrooms_text, "[0-9\\.]+"),
         bathrooms = as.numeric(bathrooms))

# Clean and expand amenities:
# Remove brackets, braces, and quotes from the amenities text
core <- core %>%
  mutate(amenities = str_replace_all(amenities, "[\\[\\]{}\"]", ""))

# Split amenities into multiple rows (long format), trim whitespace, count frequencies
top_amenities <- core %>%
  separate_rows(amenities, sep = ",") %>%
  mutate(amenities = str_trim(amenities)) %>%
  filter(amenities != "") %>%                      # drop empty entries if any
  count(amenities, sort = TRUE) %>%
  slice_max(order_by = n, n = 15) %>%              # select top 15 most frequent amenities
  pull(amenities)

# Create binary indicator columns for each of the top 15 amenities
for (amenity in top_amenities) {
  # Create a safe column name by prefixing with "amenity_" and making it syntactically valid
  clean_name <- make.names(paste0("amenity_", amenity))
  core[[clean_name]] <- str_detect(core$amenities, fixed(amenity))
}

# Drop the original amenities column (no longer needed after encoding top amenities)
core <- core %>% select(-amenities)

# Select relevant features:
# Define the base feature set (target + key predictors before dummy encoding)
features <- c("price", "accommodates", "bedrooms", "beds", "bathrooms",
              "room_type", "property_type", "review_scores_rating",
              "number_of_reviews", "availability_365", "neighbourhood_cleansed")

# Subset the core data to keep only the selected features and the new amenity indicators
core <- core %>% select(any_of(features), starts_with("amenity_"))

# Handle missing values:
# Impute numeric NAs with median, and drop rows with missing categorical values (if any)
core <- core %>%
  mutate(
    bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms),
    beds = ifelse(is.na(beds), median(beds, na.rm = TRUE), beds),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    review_scores_rating = ifelse(is.na(review_scores_rating), 
                                  median(review_scores_rating, na.rm = TRUE), review_scores_rating)
  ) %>%
  drop_na(room_type, property_type, neighbourhood_cleansed)

# Create dummy variables for categorical features (room_type, property_type, neighbourhood)
core_dummies <- model.matrix(~ room_type + property_type + neighbourhood_cleansed - 1, data = core) %>%
  as.data.frame()

# Combine numeric + amenity + dummy variables into one modeling dataset
core_model <- bind_cols(
  core %>% select(-room_type, -property_type, -neighbourhood_cleansed),
  core_dummies
)
# Ensure all column names are syntactically valid (no spaces or special chars)
colnames(core_model) <- make.names(colnames(core_model))

# Drop one dummy from each category group to avoid multicollinearity in models that include an intercept
# (We drop the "baseline" categories: e.g., "Shared room" for room_type, "Private room in home" for property_type,
#  and the most common neighbourhood as baseline which was "Centrum West" in training data.)
core_model_clean <- core_model %>% select(
  -any_of(c("room_typeShared.room", 
            "property_typePrivate.room.in.home", 
            "neighbourhood_cleansedCentrum.West"))
)
# The dataset core_model_clean is now ready for modeling.
# Split into training and test sets for model evaluation
set.seed(123)  # for reproducibility
train_index <- createDataPartition(core_model_clean$price, p = 0.8, list = FALSE)
train_data <- core_model_clean[train_index, ]
test_data  <- core_model_clean[-train_index, ]

# For convenience, define matrices and vectors for modeling
x_train <- train_data %>% select(-price)
y_train <- train_data$price
x_test  <- test_data %>% select(-price)
y_test  <- test_data$price

# Save the predictor names for later use (to ensure new data has same features)
predictor_names <- colnames(x_train)

# 2. Model Training ----
# We will train five models: OLS, LASSO, Ridge, Random Forest, XGBoost.

## Model 1: OLS (Linear Regression)
ols_model <- lm(price ~ ., data = train_data)  # using all predictors in train_data
# Predict on test set and calculate RMSE
pred_ols <- predict(ols_model, newdata = test_data)
rmse_ols <- RMSE(pred_ols, y_test)
print(paste("OLS RMSE (cleaned):", round(rmse_ols, 2)))

## Model 2: LASSO Regression
# We use cross-validation to find the best lambda for LASSO (alpha = 1)
x_train <- model.matrix(price ~ ., data = train_data)[, -1]  # remove intercept
y_train <- train_data$price
x_test <- model.matrix(price ~ ., data = test_data)[, -1]
y_test <- test_data$price
# Step 2: Cross-validation to find best lambda
set.seed(123)
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1)  # alpha = 1 for LASSO
# Get best lambda
best_lambda <- lasso_cv$lambda.min
# Step 3: Fit LASSO model using best lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)
# Step 4: Predict + Evaluate
pred_lasso <- predict(lasso_model, s = best_lambda, newx = x_test)
rmse_lasso <- RMSE(pred_lasso, y_test)
print(paste("LASSO RMSE:", round(rmse_lasso, 2)))
# Set output to PNG file (adjust file name or path as needed)
png("lasso_cv_plot.png", width = 800, height = 600)
# Plot the cross-validation curve
plot(lasso_cv)
# Close the PNG device
dev.off()


## Model 3: Random Forest
# Use the same train/test split from earlier
x_train <- train_data %>% select(-price)
y_train <- train_data$price

x_test <- test_data %>% select(-price)
y_test <- test_data$price
# Train Random Forest Model
set.seed(123)
rf_model <- randomForest(x = x_train, y = y_train, ntree = 100, importance = TRUE)
#  Step 3: Predict + RMSE
pred_rf <- predict(rf_model, newdata = x_test)
rmse_rf <- RMSE(pred_rf, y_test)
print(paste("Random Forest RMSE:", round(rmse_rf, 2)))
#  Step 4: Feature Importance Plot
# Save the Random Forest importance plot to a PNG
png("rf_feature_importance.png", width = 1000, height = 800)
varImpPlot(rf_model, main = "Random Forest Feature Importance")
dev.off()


## Model 4: XGBoost
# Prepare data in matrix/DMatrix form for XGBoost
x_train_matrix <- as.matrix(x_train)
x_test_matrix  <- as.matrix(x_test)
y_train_vector <- y_train
y_test_vector <- y_test
dtrain <- xgb.DMatrix(data = x_train_matrix, label = y_train_vector)
dtest  <- xgb.DMatrix(data = x_test_matrix, label = y_test_vector)
set.seed(123)
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror",
                     eval_metric = "rmse", verbose = 0)
# Predict and evaluate
pred_xgb <- predict(xgb_model, dtest)
rmse_xgb <- RMSE(pred_xgb, y_test_vector)
print(paste("XGBoost RMSE:", round(rmse_xgb, 2)))
#  Feature Importance (Optional)
importance_matrix <- xgb.importance(model = xgb_model)
# Save a high-resolution version (recommended fix)
png("xgb_feature_importance_hd.png", width = 1200, height = 1000, res = 150)
xgb.plot.importance(importance_matrix, top_n = 15, main = "XGBoost Feature Importance")
dev.off()


## Model 5: Ridge Regression
# Make sure x_train and x_test are numeric matrices
x_train <- model.matrix(price ~ ., data = train_data)[, -1]  # remove intercept
y_train <- train_data$price

x_test <- model.matrix(price ~ ., data = test_data)[, -1]
y_test <- test_data$price
# Cross-validation to find best lambda for Ridge (alpha = 0)
set.seed(123)
ridge_cv <- cv.glmnet(as.matrix(x_train), y_train, alpha = 0)  # alpha = 0 for Ridge
# Optional: Plot the CV curve
png("ridge_cv_plot.png", width = 800, height = 600)
plot(ridge_cv)
dev.off()
# Best lambda value
best_lambda_ridge <- ridge_cv$lambda.min
#  Fit Ridge Model
ridge_model <- glmnet(as.matrix(x_train), y_train, alpha = 0, lambda = best_lambda_ridge)
# Predict + Evaluate RMSE
pred_ridge <- predict(ridge_model, s = best_lambda_ridge, newx = as.matrix(x_test))
rmse_ridge <- RMSE(pred_ridge, y_test)
print(paste("Ridge RMSE:", round(rmse_ridge, 2)))


###       SHAP with iml + xgboost      ____________________________________
# Install if not already installed
if (!require(iml)) install.packages("iml")
library(iml)
# Convert matrix back to data frame (required for iml)
X_train_df <- as.data.frame(x_train_matrix)
# Create Predictor object for iml
predictor_xgb <- Predictor$new(
  model = xgb_model,
  data = X_train_df,
  y = y_train_vector,
  predict.function = function(model, newdata) {
    predict(model, newdata = as.matrix(newdata))
  }
)
# Pick a random observation (e.g., row 10)
shap <- Shapley$new(predictor_xgb, x.interest = X_train_df[10, ])
png("shap_xgb_explanation.png", width = 1000, height = 800, res = 150)
plot(shap)
dev.off()
# Aggregate SHAP for all features (approx)
# Keep only the top 20 most important features
# Calculate feature importance using iml
feature_imp <- FeatureImp$new(predictor_xgb, loss = "rmse")
feature_imp$results <- head(
  feature_imp$results[order(-feature_imp$results$importance), ],
  20
)
png("shap_global_top20.png", width = 1200, height = 800, res = 150)
plot(feature_imp)
dev.off()

# Bootstrapped RMSE ----
set.seed(123)
B <- 30
n_train <- nrow(train_data)
rmse_boot <- data.frame(OLS = numeric(B), LASSO = numeric(B), Ridge = numeric(B),
                        RandomForest = numeric(B), XGBoost = numeric(B))

for (b in 1:B) {
  boot_idx <- sample(seq_len(n_train), size = n_train, replace = TRUE)
  boot_train <- train_data[boot_idx, ]
  oob_idx <- setdiff(seq_len(n_train), unique(boot_idx))
  if (length(oob_idx) == 0) next
  boot_oob <- train_data[oob_idx, ]
  
  # Ensure correct columns
  missing_cols <- setdiff(names(x_train), names(boot_train))
  for (mc in missing_cols) {
    boot_train[[mc]] <- 0
    boot_oob[[mc]] <- 0
  }
  boot_train <- dplyr::select(boot_train, all_of(c(names(x_train), "price")))
  boot_oob <- dplyr::select(boot_oob, all_of(c(names(x_train), "price")))
  
  # OLS
  try({
    boot_ols <- lm(price ~ ., data = boot_train)
    pred_ols <- predict(boot_ols, newdata = boot_oob)
    rmse_boot$OLS[b] <- RMSE(pred_ols, boot_oob$price)
  }, silent = TRUE)
  
  # LASSO & Ridge
  try({
    x_bt <- model.matrix(price ~ ., data = boot_train)[, -1]
    y_bt <- boot_train$price
    x_oob <- model.matrix(price ~ ., data = boot_oob)[, -1]
    
    boot_lasso <- glmnet(x_bt, y_bt, alpha = 1, lambda = best_lambda)
    pred_lasso <- predict(boot_lasso, s = best_lambda, newx = x_oob)
    rmse_boot$LASSO[b] <- RMSE(pred_lasso, boot_oob$price)
    
    boot_ridge <- glmnet(x_bt, y_bt, alpha = 0, lambda = best_lambda_ridge)
    pred_ridge <- predict(boot_ridge, s = best_lambda_ridge, newx = x_oob)
    rmse_boot$Ridge[b] <- RMSE(pred_ridge, boot_oob$price)
  }, silent = TRUE)
  
  # Random Forest
  try({
    boot_rf <- randomForest(x = boot_train %>% select(-price), y = boot_train$price, ntree = 100)
    pred_rf <- predict(boot_rf, newdata = boot_oob %>% select(-price))
    rmse_boot$RandomForest[b] <- RMSE(pred_rf, boot_oob$price)
  }, silent = TRUE)
  
  # XGBoost
  try({
    boot_dtrain <- xgb.DMatrix(data = as.matrix(boot_train %>% select(-price)), label = boot_train$price)
    boot_dtest  <- xgb.DMatrix(data = as.matrix(boot_oob %>% select(-price)), label = boot_oob$price)
    boot_xgb <- xgboost(data = boot_dtrain, nrounds = 100, objective = "reg:squarederror", eval_metric = "rmse", verbose = 0)
    pred_xgb <- predict(boot_xgb, boot_dtest)
    rmse_boot$XGBoost[b] <- RMSE(pred_xgb, boot_oob$price)
  }, silent = TRUE)
}

boot_stats <- sapply(rmse_boot, function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
boot_stats <- as.data.frame(t(boot_stats))
colnames(boot_stats) <- c("RMSE_boot_mean", "RMSE_boot_sd")
boot_stats$Model <- rownames(boot_stats)
boot_stats <- boot_stats %>% select(Model, RMSE_boot_mean, RMSE_boot_sd)
print(boot_stats)

rmse_long <- pivot_longer(rmse_boot, cols = everything(), names_to = "Model", values_to = "RMSE")
ggplot(rmse_long, aes(x = RMSE, fill = Model)) +
  geom_density(alpha = 0.5) +
  labs(title = "Bootstrapped RMSE Distributions", x = "RMSE", y = "Density") +
  theme_minimal()


###  3. Compare models  ___________________________________________
###  Horserace Table
model_results <- data.frame(
  Model = c("OLS", "LASSO", "Ridge", "Random Forest", "XGBoost"),
  RMSE = c(rmse_ols, rmse_lasso, rmse_ridge, rmse_rf, rmse_xgb)
)
print(model_results)
# Optional: Plot
# Create your plot and assign to a variable
plot_rmse <- ggplot(model_results, aes(x = reorder(Model, RMSE), y = RMSE)) +
  geom_col(fill = "orange") +
  coord_flip() +
  labs(title = "Model Comparison (RMSE)", x = "Model", y = "RMSE") +
  theme_minimal()
# Save the plot as a PNG
ggsave("model_comparison_rmse.png", plot = plot_rmse, width = 8, height = 6, dpi = 150)


###   Model comparison ____________________________________
model_results <- data.frame(
  Model = c("OLS", "LASSO", "Ridge", "Random Forest", "XGBoost"),
  RMSE = c(rmse_ols, rmse_lasso, rmse_ridge, rmse_rf, rmse_xgb)
)
print(model_results)

## For OLS Model
# R-squared
r2_ols <- summary(ols_model)$r.squared
# BIC
bic_ols <- BIC(ols_model)
# Training RMSE
ols_preds_train <- predict(ols_model, newdata = train_data)
rmse_train_ols <- RMSE(ols_preds_train, train_data$price)


## For LASSO Model
lasso_fit <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)
# Fit full model again to entire training data (for BIC and R2)
# Predict on training set
lasso_preds_train <- predict(lasso_fit, newx = as.matrix(x_train))
# Calculate SSE and SST
sse <- sum((lasso_preds_train - y_train)^2)
sst <- sum((y_train - mean(y_train))^2)
# R-squared
r2_lasso <- 1 - sse / sst
print(paste("LASSO R²:", round(r2_lasso, 4)))
# BIC for LASSO (approximation)
n <- length(y_train)
k <- lasso_fit$df  # number of non-zero coefficients
bic_lasso <- n * log(sse/n) + log(n) * k
print(paste("LASSO BIC:", round(bic_lasso, 2)))
# Training RMSE
rmse_train_lasso <- RMSE(lasso_preds_train, y_train)
print(paste("LASSO Training RMSE:", round(rmse_train_lasso, 2)))

##  For Random Forest
library(tidyverse)  
x_train <- train_data[, setdiff(names(train_data), "price")]
y_train <- train_data$price
# Training predictions
rf_train_preds <- predict(rf_model, newdata = x_train)
# Training RMSE
rmse_train_rf <- RMSE(rf_train_preds, y_train)
# R-squared
sst <- sum((y_train - mean(y_train))^2)
sse <- sum((rf_train_preds - y_train)^2)
r2_rf <- 1 - sse / sst
k_rf <- length(rf_model$forest$xlevels)  # approximate number of predictors used
n_rf <- length(y_train)
bic_rf <- n_rf * log(sse/n_rf) + log(n_rf) * k_rf
rmse_train_rf <- RMSE(predict(rf_model, newdata = x_train), y_train)

## For XG Boost
xgb_train_preds <- predict(xgb_model, dtrain)
# R-squared
sst <- sum((y_train_vector - mean(y_train_vector))^2)
sse <- sum((xgb_train_preds - y_train_vector)^2)
r2_xgb <- 1 - sse / sst
k_xgb <- length(xgb.importance(model = xgb_model)$Feature)  # number of important features
n_xgb <- length(y_train_vector)
bic_xgb <- n_xgb * log(sse/n_xgb) + log(n_xgb) * k_xgb
# Training RMSE
rmse_train_xgb <- RMSE(xgb_train_preds, y_train_vector)

## For Ridge Model
ridge_fit <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda_ridge)
# Training predictions
ridge_preds_train <- predict(ridge_fit, newx = as.matrix(x_train))
# Calculate SSE and SST
sse <- sum((ridge_preds_train - y_train)^2)
sst <- sum((y_train - mean(y_train))^2)
# R-squared
r2_ridge <- 1 - sse / sst
print(r2_ridge)
# BIC
n <- length(y_train)
k <- ridge_fit$df
bic_ridge <- n * log(sse/n) + log(n) * k
# Training RMSE
rmse_train_ridge <- RMSE(ridge_preds_train, y_train)

# --- Print All Metrics ---
cat("\n===== Model Metrics (Training) =====\n")
cat("OLS     - R2:", round(r2_ols, 4), ", BIC:", round(bic_ols, 2), ", Training RMSE:", round(rmse_train_ols, 2), ", RMSE:", round(rmse_ols, 2), "\n")
cat("LASSO   - R2:", round(r2_lasso, 4), ", BIC:", round(bic_lasso, 2), ", Training RMSE:", round(rmse_train_lasso, 2), ", RMSE:", round(rmse_lasso, 2), "\n")
cat("RF      - R2:", round(r2_rf, 4), ", BIC:", round(bic_rf, 2), ", Training RMSE:", round(rmse_train_rf, 2), ", RMSE:", round(rmse_rf, 2), "\n")
cat("XGBoost - R2:", round(r2_xgb, 4), ", BIC:", round(bic_xgb, 2), ", Training RMSE:", round(rmse_train_xgb, 2), ", RMSE:", round(rmse_xgb, 2), "\n")
cat("Ridge   - R2:", round(r2_ridge, 4), ", BIC:", round(bic_ridge, 2), ", Training RMSE:", round(rmse_train_ridge, 2), ", RMSE:", round(rmse_ridge, 2), "\n")

# Create model comparison table
model_metrics <- data.frame(
  Model = c("OLS", "LASSO", "Ridge", "Random Forest", "XGBoost"),
  R_squared = c(r2_ols, r2_lasso, r2_ridge, r2_rf, r2_xgb),
  BIC = c(bic_ols, bic_lasso, bic_ridge, bic_rf, bic_xgb),
  Training_RMSE = c(rmse_train_ols, rmse_train_lasso, rmse_train_ridge, rmse_train_rf, rmse_train_xgb),
  Test_RMSE = c(rmse_ols, rmse_lasso, rmse_ridge, rmse_rf, rmse_xgb)
)
# Round values for display
model_metrics <- model_metrics %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))
print(model_metrics)


## Plot Training & Test RMSE
# Load required library
library(ggplot2)
# Create a data frame with your actual results
model_perf <- data.frame(
  Model = c("OLS", "LASSO", "Random Forest", "XGBoost", "Ridge"),
  Coefficients = c(87, 55, 70, 70, 87),  # You can adjust these based on your model
  Training_RMSE = c(0.354, 0.294, 0.0364, 0.0012, 0.295),
  Test_RMSE = c(0.402, 0.336, 0.115, 0.016, 0.329)
)
# Convert to long format for ggplot
library(tidyr)
plot_data <- model_perf %>%
  pivot_longer(cols = c("Training_RMSE", "Test_RMSE"),
               names_to = "Set", values_to = "RMSE")
# Make the plot
ggplot(plot_data, aes(x = Coefficients, y = RMSE, color = Set)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(
    title = "Training and Test RMSE for Each Model",
    x = "Number of Coefficients",
    y = "RMSE"
  ) +
  scale_color_manual(values = c("Training_RMSE" = "green4", "Test_RMSE" = "steelblue")) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.title = element_blank()
  )
ggsave("model_rmse_plot.png", width = 8, height = 5)


### 4. Feature importance Random Forest Vs. XG Boost   ____________________________________________

library(randomForest)
library(xgboost)
library(tidyverse)
library(dplyr)
library(data.table)  # For xgb.importance
varImpPlot(rf_model)  # For Random Forest
xgb.plot.importance(importance_matrix)  # For XGBoost
# Random Forest: Extract and Plot Top 10 Feature Importance
# Plot and save
png("rf_feature_importance_top10.png", width = 1000, height = 800)
varImpPlot(rf_model, n.var = 10, main = "Top 10 Features - Random Forest")
dev.off()
rf_importance <- as.data.frame(importance(rf_model))
rf_importance$Feature <- rownames(rf_importance)
top10_rf <- rf_importance[order(-rf_importance$IncNodePurity), ][1:10, ]
print(top10_rf)
# XGBoost: Extract and Plot Top 10 Feature Importance
xgb_importance <- xgb.importance(model = xgb_model)
top10_xgb <- xgb_importance[order(-xgb_importance$Gain), ][1:10, ]
# Plot and save
png("xgb_feature_importance_top10.png", width = 1000, height = 800)
xgb.plot.importance(top10_xgb, main = "Top 10 Features - XGBoost")
dev.off()
print(top10_xgb)
# Combine Top 10 from Both Models
top10_combined <- tibble(
  Rank = 1:10,
  Random_Forest = top10_rf$Feature,
  XGBoost = top10_xgb$Feature
)
print(top10_combined)


# ==============================================
# Airbnb Price Prediction Analysis - Part II
# Validity: Apply models on new data
# ==============================================

# 1. Load New Datasets ----

# A. Amsterdam 2025 Q1
amsterdam_later <- read_csv("Amsterdam_later(2025Q1).csv")

# B. Brussels 2025 Q1
brussels <- read_csv("Brussels_2025Q1.csv")

# 2. Reusable Data Preparation Function ----

prepare_new_data <- function(df, top_amenities, features, train_data) {
  df <- df %>%
    mutate(price = str_remove_all(price, "[$\u20ac\\s,]"),
           price = as.numeric(price)) %>%
    filter(!is.na(price) & price > 0) %>%
    mutate(bathrooms = str_extract(bathrooms_text, "[0-9\\.]+"),
           bathrooms = as.numeric(bathrooms),
           amenities = str_replace_all(amenities, "[\\[\\]{}\"]", ""))
  
  # Top amenities
  for (amenity in top_amenities) {
    clean_name <- make.names(paste0("amenity_", amenity))
    df[[clean_name]] <- str_detect(df$amenities, fixed(amenity))
  }
  
  df <- df %>% select(-amenities)
  
  df <- df %>% select(any_of(features), starts_with("amenity_")) %>%
    mutate(
      bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms),
      beds = ifelse(is.na(beds), median(beds, na.rm = TRUE), beds),
      bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
      review_scores_rating = ifelse(is.na(review_scores_rating), 
                                    median(review_scores_rating, na.rm = TRUE), review_scores_rating)
    ) %>% drop_na(room_type, property_type, neighbourhood_cleansed)
  
  dummies <- model.matrix(~ room_type + property_type + neighbourhood_cleansed - 1, data = df) %>% as.data.frame()
  df_model <- bind_cols(df %>% select(-room_type, -property_type, -neighbourhood_cleansed), dummies)
  colnames(df_model) <- make.names(colnames(df_model))
  
  df_model <- df_model %>% select(-any_of(c("room_typeShared.room", "property_typePrivate.room.in.home", "neighbourhood_cleansedCentrum.West")))
  
  missing_cols <- setdiff(colnames(train_data), colnames(df_model))
  for (col in missing_cols) {
    df_model[[col]] <- 0
  }
  df_model <- df_model %>% select(colnames(train_data))
  return(df_model)
}

# 3. Preprocess New Data ----
later_clean <- prepare_new_data(amsterdam_later, top_amenities, features, train_data)
brussels_clean <- prepare_new_data(brussels, top_amenities, features, train_data)

# 4. Predict and Evaluate on Later Date ----
pred_ols_later   <- predict(ols_model, newdata = later_clean)
pred_lasso_later <- predict(lasso_model, s = best_lambda, newx = as.matrix(later_clean %>% select(-price)))
pred_ridge_later <- predict(ridge_model, s = best_lambda_ridge, newx = as.matrix(later_clean %>% select(-price)))
pred_rf_later    <- predict(rf_model, newdata = later_clean %>% select(-price))
pred_xgb_later   <- predict(xgb_model, xgb.DMatrix(data = as.matrix(later_clean %>% select(-price))))

rmse_later <- c(
  OLS = RMSE(pred_ols_later, later_clean$price),
  LASSO = RMSE(pred_lasso_later, later_clean$price),
  Ridge = RMSE(pred_ridge_later, later_clean$price),
  RandomForest = RMSE(pred_rf_later, later_clean$price),
  XGBoost = RMSE(pred_xgb_later, later_clean$price)
)

print("RMSE - Amsterdam Later:"); print(rmse_later)

# 5. Predict and Evaluate on Other City ----
pred_ols_brussels   <- predict(ols_model, newdata = brussels_clean)
pred_lasso_brussels <- predict(lasso_model, s = best_lambda, newx = as.matrix(brussels_clean %>% select(-price)))
pred_ridge_brussels <- predict(ridge_model, s = best_lambda_ridge, newx = as.matrix(brussels_clean %>% select(-price)))
pred_rf_brussels    <- predict(rf_model, newdata = brussels_clean %>% select(-price))
pred_xgb_brussels   <- predict(xgb_model, xgb.DMatrix(data = as.matrix(brussels_clean %>% select(-price))))

rmse_brussels <- c(
  OLS = RMSE(pred_ols_brussels, brussels_clean$price),
  LASSO = RMSE(pred_lasso_brussels, brussels_clean$price),
  Ridge = RMSE(pred_ridge_brussels, brussels_clean$price),
  RandomForest = RMSE(pred_rf_brussels, brussels_clean$price),
  XGBoost = RMSE(pred_xgb_brussels, brussels_clean$price)
)

print("RMSE - Brussels:"); print(rmse_brussels)

