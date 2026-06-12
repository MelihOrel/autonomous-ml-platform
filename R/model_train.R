# ============================================================
# Module: model_train.R
# Purpose: Train and evaluate classification models (Random
#          Forest or XGBoost) using the caret framework, and
#          return standardized performance metrics.
# ============================================================

library(caret)
library(dplyr)
library(randomForest)
library(xgboost)

#' Train and Evaluate a Classification Model
#'
#' Splits the input dataframe into an 80/20 train/test split,
#' trains the specified algorithm using \code{caret}, and
#' computes evaluation metrics (Accuracy, Precision, Recall),
#' a confusion matrix, and a feature importance table.
#'
#' @param df A cleaned data.frame. The target column must already
#'   be a factor (as produced by \code{auto_clean_data}).
#' @param target_col Character string naming the target column.
#' @param algorithm Character string, either \code{"Random Forest"}
#'   or \code{"XGBoost"}.
#'
#' @return A named list containing:
#'   \describe{
#'     \item{accuracy}{Numeric overall accuracy on the test set.}
#'     \item{precision}{Numeric macro-averaged precision.}
#'     \item{recall}{Numeric macro-averaged recall.}
#'     \item{confusion_matrix}{A \code{caret} confusionMatrix object.}
#'     \item{feature_importance}{A data.frame with columns
#'       \code{Feature} and \code{Importance}, sorted descending.}
#'     \item{model}{The trained \code{caret} train object.}
#'   }
#'
#' @examples
#' \dontrun{
#' result <- train_and_evaluate(iris, "Species", "Random Forest")
#' print(result$accuracy)
#' }
#'
#' @export
train_and_evaluate <- function(df, target_col, algorithm = "Random Forest") {

  if (!target_col %in% names(df)) {
    stop(paste0("Target column '", target_col, "' not found in dataset."))
  }

  if (!is.factor(df[[target_col]])) {
    df[[target_col]] <- as.factor(df[[target_col]])
  }

  if (nlevels(df[[target_col]]) < 2) {
    stop("Target column must have at least 2 classes for classification.")
  }

  set.seed(42)

  # ---- 1. Train/Test split (80/20), stratified by target -------
  train_index <- createDataPartition(df[[target_col]], p = 0.8, list = FALSE)
  train_data <- df[train_index, ]
  test_data  <- df[-train_index, ]

  # Formula: target ~ all other columns
  predictors <- setdiff(names(df), target_col)
  if (length(predictors) == 0) {
    stop("No predictor columns available after cleaning. Cannot train a model.")
  }

  formula_str <- paste0("`", target_col, "` ~ ",
                         paste0("`", predictors, "`", collapse = " + "))
  model_formula <- as.formula(formula_str)

  # ---- 2. Cross-validation control ------------------------------
  train_control <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = multiClassSummary,
    savePredictions = "final"
  )

  # ---- 3. Train model based on selected algorithm ----------------
  if (algorithm == "Random Forest") {

    model <- train(
      model_formula,
      data = train_data,
      method = "rf",
      trControl = train_control,
      metric = "Accuracy",
      tuneLength = 3,
      importance = TRUE,
      na.action = na.omit
    )

  } else if (algorithm == "XGBoost") {

    tune_grid <- expand.grid(
      nrounds = c(50, 100),
      max_depth = c(3, 6),
      eta = c(0.1, 0.3),
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    )

    model <- train(
      model_formula,
      data = train_data,
      method = "xgbTree",
      trControl = train_control,
      metric = "Accuracy",
      tuneGrid = tune_grid,
      na.action = na.omit,
      verbosity = 0
    )

  } else {
    stop(paste0("Unsupported algorithm: '", algorithm,
                 "'. Choose 'Random Forest' or 'XGBoost'."))
  }

  # ---- 4. Predict on the test set ---------------------------------
  predictions <- predict(model, newdata = test_data)
  actuals <- test_data[[target_col]]

  # Align factor levels between predictions and actuals
  common_levels <- levels(actuals)
  predictions <- factor(predictions, levels = common_levels)
  actuals <- factor(actuals, levels = common_levels)

  conf_matrix <- confusionMatrix(predictions, actuals)

  # ---- 5. Extract Accuracy, Precision, Recall ----------------------
  accuracy <- as.numeric(conf_matrix$overall["Accuracy"])

  if (nlevels(actuals) == 2) {
    # Binary classification: use byClass directly
    precision <- as.numeric(conf_matrix$byClass["Precision"])
    recall    <- as.numeric(conf_matrix$byClass["Recall"])
    if (is.na(precision)) precision <- as.numeric(conf_matrix$byClass["Pos Pred Value"])
    if (is.na(recall))    recall    <- as.numeric(conf_matrix$byClass["Sensitivity"])
  } else {
    # Multi-class: macro-average across classes
    by_class <- conf_matrix$byClass
    precision <- mean(by_class[, "Precision"], na.rm = TRUE)
    recall    <- mean(by_class[, "Recall"], na.rm = TRUE)
  }

  # Handle potential NaN (e.g., a class with no predictions)
  if (is.nan(precision) || is.na(precision)) precision <- 0
  if (is.nan(recall) || is.na(recall)) recall <- 0

  # ---- 6. Feature Importance ---------------------------------------
  importance_obj <- varImp(model)
  importance_df <- as.data.frame(importance_obj$importance)

  if ("Overall" %in% names(importance_df)) {
    importance_df$Feature <- rownames(importance_df)
    importance_df <- importance_df[, c("Feature", "Overall")]
    names(importance_df)[2] <- "Importance"
  } else {
    # Multi-class importance: average across class columns
    importance_df$Feature <- rownames(importance_df)
    numeric_cols <- sapply(importance_df, is.numeric)
    importance_df$Importance <- rowMeans(importance_df[, numeric_cols, drop = FALSE])
    importance_df <- importance_df[, c("Feature", "Importance")]
  }

  importance_df <- importance_df %>%
    arrange(desc(Importance)) %>%
    mutate(Importance = round(Importance, 2))

  rownames(importance_df) <- NULL

  # ---- 7. Return standardized results -------------------------------
  list(
    accuracy = round(accuracy, 4),
    precision = round(precision, 4),
    recall = round(recall, 4),
    confusion_matrix = conf_matrix,
    feature_importance = importance_df,
    model = model
  )
}
