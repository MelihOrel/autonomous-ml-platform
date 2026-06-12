# ============================================================
# Module: auto_clean.R
# Purpose: Automated data cleaning pipeline for the No-Code ML
#          platform. Handles missing values, zero-variance
#          predictors, and target-column type coercion.
# ============================================================

library(dplyr)

#' Automatically Clean a Dataset for Machine Learning
#'
#' Performs an end-to-end automated cleaning routine on a raw
#' dataframe in preparation for model training. The function:
#' \itemize{
#'   \item Imputes missing numeric values with the column median.
#'   \item Imputes missing categorical/character values with the
#'         column mode (most frequent value).
#'   \item Removes zero-variance (constant) predictor columns.
#'   \item Converts the target column to a factor (for
#'         classification tasks).
#' }
#'
#' @param df A data.frame or tibble containing the raw dataset.
#' @param target_col A character string giving the name of the
#'   target/outcome column.
#'
#' @return A named list with two elements:
#'   \describe{
#'     \item{data}{The cleaned data.frame.}
#'     \item{summary}{A character string summarizing all cleaning
#'       actions taken, suitable for display in the UI.}
#'   }
#'
#' @examples
#' \dontrun{
#' result <- auto_clean_data(iris, "Species")
#' cleaned_df <- result$data
#' cat(result$summary)
#' }
#'
#' @export
auto_clean_data <- function(df, target_col) {

  if (!is.data.frame(df)) {
    stop("Input 'df' must be a data.frame.")
  }
  if (!target_col %in% names(df)) {
    stop(paste0("Target column '", target_col, "' not found in dataset."))
  }

  actions <- character(0)
  original_dims <- dim(df)

  # ---- 1. Drop fully empty columns (all NA) -----------------
  all_na_cols <- names(df)[sapply(df, function(x) all(is.na(x)))]
  if (length(all_na_cols) > 0) {
    df <- df[, !(names(df) %in% all_na_cols), drop = FALSE]
    actions <- c(actions, paste0(
      "Dropped ", length(all_na_cols),
      " column(s) that were entirely empty: ",
      paste(all_na_cols, collapse = ", ")
    ))
  }

  # Re-check target still exists after dropping empty columns
  if (!target_col %in% names(df)) {
    stop(paste0("Target column '", target_col,
                 "' was removed because it contained only NA values."))
  }

  # ---- 2. Impute missing values ------------------------------
  numeric_imputed <- character(0)
  categorical_imputed <- character(0)

  for (col in names(df)) {
    if (col == target_col) next  # don't impute target; handled separately

    col_data <- df[[col]]
    n_missing <- sum(is.na(col_data))

    if (n_missing == 0) next

    if (is.numeric(col_data)) {
      med_val <- median(col_data, na.rm = TRUE)
      df[[col]][is.na(col_data)] <- med_val
      numeric_imputed <- c(numeric_imputed,
                            paste0(col, " (", n_missing, " values -> median = ",
                                   round(med_val, 3), ")"))
    } else {
      # Mode imputation for categorical / character / factor
      tbl <- table(col_data)
      if (length(tbl) > 0) {
        mode_val <- names(tbl)[which.max(tbl)]
        df[[col]][is.na(col_data)] <- mode_val
        categorical_imputed <- c(categorical_imputed,
                                  paste0(col, " (", n_missing, " values -> mode = '",
                                         mode_val, "')"))
      }
    }
  }

  if (length(numeric_imputed) > 0) {
    actions <- c(actions, paste0(
      "Imputed missing NUMERIC values using median for: ",
      paste(numeric_imputed, collapse = "; ")
    ))
  }

  if (length(categorical_imputed) > 0) {
    actions <- c(actions, paste0(
      "Imputed missing CATEGORICAL values using mode for: ",
      paste(categorical_imputed, collapse = "; ")
    ))
  }

  # ---- 3. Drop rows where target is missing -------------------
  target_na_count <- sum(is.na(df[[target_col]]))
  if (target_na_count > 0) {
    df <- df[!is.na(df[[target_col]]), ]
    actions <- c(actions, paste0(
      "Dropped ", target_na_count,
      " row(s) with missing target values in '", target_col, "'."
    ))
  }

  # ---- 4. Drop zero-variance predictor columns -----------------
  predictor_cols <- setdiff(names(df), target_col)
  zero_var_cols <- character(0)

  for (col in predictor_cols) {
    col_data <- df[[col]]
    unique_vals <- length(unique(col_data[!is.na(col_data)]))
    if (unique_vals <= 1) {
      zero_var_cols <- c(zero_var_cols, col)
    } else if (is.numeric(col_data) && stats::var(col_data, na.rm = TRUE) == 0) {
      zero_var_cols <- c(zero_var_cols, col)
    }
  }

  if (length(zero_var_cols) > 0) {
    df <- df[, !(names(df) %in% zero_var_cols), drop = FALSE]
    actions <- c(actions, paste0(
      "Removed ", length(zero_var_cols),
      " zero-variance predictor column(s): ",
      paste(zero_var_cols, collapse = ", ")
    ))
  }

  # ---- 5. Convert target column to factor (classification) -----
  if (!is.factor(df[[target_col]])) {
    original_class <- class(df[[target_col]])[1]
    df[[target_col]] <- as.factor(df[[target_col]])
    actions <- c(actions, paste0(
      "Converted target column '", target_col,
      "' from ", original_class, " to factor with levels: ",
      paste(levels(df[[target_col]]), collapse = ", ")
    ))
  }

  # ---- 6. Clean character columns -> factors for modeling -------
  char_cols <- names(df)[sapply(df, is.character) & names(df) != target_col]
  if (length(char_cols) > 0) {
    for (col in char_cols) {
      df[[col]] <- as.factor(df[[col]])
    }
    actions <- c(actions, paste0(
      "Converted ", length(char_cols),
      " character predictor column(s) to factors: ",
      paste(char_cols, collapse = ", ")
    ))
  }

  # ---- 7. Final summary ------------------------------------------
  final_dims <- dim(df)
  actions <- c(
    paste0("Original dataset dimensions: ", original_dims[1],
           " rows x ", original_dims[2], " columns."),
    actions,
    paste0("Final cleaned dataset dimensions: ", final_dims[1],
           " rows x ", final_dims[2], " columns.")
  )

  summary_text <- paste(
    "AUTO-CLEAN SUMMARY:\n",
    paste0("- ", actions, collapse = "\n"),
    sep = ""
  )

  list(
    data = df,
    summary = summary_text
  )
}
