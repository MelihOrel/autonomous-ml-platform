# ============================================================
# Module: ai_analyst.R
# Purpose: Generate a natural-language interpretation of model
#          performance using an LLM (OpenAI API), acting as a
#          virtual Senior Data Scientist.
# ============================================================

library(httr)
library(jsonlite)

#' Generate an AI-Driven Interpretation of Model Results
#'
#' Constructs a dynamic prompt summarizing model performance
#' metrics and top feature importances, sends it to the OpenAI
#' Chat Completions API, and returns a natural-language
#' interpretation written from the perspective of a senior data
#' scientist explaining results to a non-technical stakeholder.
#'
#' The OpenAI API key is read from the environment variable
#' \code{OPENAI_API_KEY}, which should be loaded from a \code{.env}
#' file using \code{dotenv::load_dot_env()} at application startup.
#'
#' @param metrics_list A named list containing at least
#'   \code{accuracy}, \code{precision}, and \code{recall} (numeric,
#'   on a 0-1 scale), and optionally \code{algorithm} (character).
#' @param top_features A data.frame with columns \code{Feature} and
#'   \code{Importance}, typically the top 5 rows of the feature
#'   importance table from \code{train_and_evaluate}.
#'
#' @return A character string containing the AI-generated
#'   interpretation. If the API call fails (e.g., missing key,
#'   network error, or non-200 response), a graceful fallback
#'   message is returned instead of throwing an error.
#'
#' @examples
#' \dontrun{
#' metrics <- list(accuracy = 0.91, precision = 0.89, recall = 0.87,
#'                  algorithm = "Random Forest")
#' top_feats <- data.frame(Feature = c("Age", "Income"),
#'                          Importance = c(95.2, 80.1))
#' insight <- generate_ai_insight(metrics, top_feats)
#' cat(insight)
#' }
#'
#' @export
generate_ai_insight <- function(metrics_list, top_features) {

  api_key <- Sys.getenv("OPENAI_API_KEY")

  if (is.null(api_key) || nchar(trimws(api_key)) == 0) {
    return(paste(
      "AI ANALYST UNAVAILABLE:",
      "No OpenAI API key was found. Please set OPENAI_API_KEY in your .env file",
      "to enable automated natural-language insights.",
      sep = "\n"
    ))
  }

  # ---- 1. Build a readable feature importance string -----------
  n_show <- min(5, nrow(top_features))
  feature_lines <- character(0)

  if (n_show > 0) {
    for (i in seq_len(n_show)) {
      feature_lines <- c(feature_lines, sprintf(
        "  %d. %s (importance score: %.2f)",
        i, top_features$Feature[i], top_features$Importance[i]
      ))
    }
    feature_text <- paste(feature_lines, collapse = "\n")
  } else {
    feature_text <- "  (No feature importance data available)"
  }

  algorithm_name <- if (!is.null(metrics_list$algorithm)) {
    metrics_list$algorithm
  } else {
    "the selected machine learning algorithm"
  }

  # ---- 2. Construct the dynamic prompt ---------------------------
  prompt_text <- sprintf(
    paste(
      "You are a Senior Data Scientist explaining the results of a freshly trained",
      "%s classification model to a business stakeholder who is NOT technical.",
      "",
      "Here are the model's evaluation metrics on a held-out test set:",
      "- Accuracy: %.2f%%",
      "- Precision: %.2f%%",
      "- Recall: %.2f%%",
      "",
      "Here are the top predictive features, ranked by importance:",
      "%s",
      "",
      "Please write a clear, professional interpretation that:",
      "1. Explains in plain language what these metrics mean for this model's quality.",
      "2. Highlights any risks or limitations the user should be aware of (e.g., if",
      "   precision and recall are imbalanced, or accuracy is misleadingly high due to",
      "   class imbalance).",
      "3. Explains, in simple terms, what the top features suggest about the underlying",
      "   patterns in the data.",
      "4. Gives 1-2 concrete, actionable recommendations for next steps.",
      "",
      "Keep the tone confident, concise, and business-friendly. Limit the response to",
      "around 250 words. Do not use markdown headers; use short paragraphs.",
      sep = "\n"
    ),
    algorithm_name,
    metrics_list$accuracy * 100,
    metrics_list$precision * 100,
    metrics_list$recall * 100,
    feature_text
  )

  # ---- 3. Build the request body -----------------------------------
  request_body <- list(
    model = "gpt-4o-mini",
    messages = list(
      list(
        role = "system",
        content = "You are an expert AI data analyst embedded in a no-code machine learning platform. You explain model results clearly, honestly, and without unnecessary jargon."
      ),
      list(
        role = "user",
        content = prompt_text
      )
    ),
    temperature = 0.4,
    max_tokens = 500
  )

  # ---- 4. Call the OpenAI API ----------------------------------------
  response <- tryCatch({
    httr::POST(
      url = "https://api.openai.com/v1/chat/completions",
      httr::add_headers(
        Authorization = paste("Bearer", api_key),
        `Content-Type` = "application/json"
      ),
      body = jsonlite::toJSON(request_body, auto_unbox = TRUE),
      encode = "raw",
      httr::timeout(60)
    )
  }, error = function(e) {
    return(e)
  })

  # ---- 5. Handle connection-level errors -------------------------------
  if (inherits(response, "error") || inherits(response, "simpleError")) {
    return(paste(
      "AI ANALYST ERROR:",
      "Could not connect to the OpenAI API. Please check your internet connection",
      "and try again.",
      sep = "\n"
    ))
  }

  # ---- 6. Handle non-200 HTTP responses ---------------------------------
  status <- httr::status_code(response)

  if (status != 200) {
    error_content <- tryCatch({
      httr::content(response, as = "parsed", type = "application/json")
    }, error = function(e) NULL)

    error_msg <- if (!is.null(error_content) && !is.null(error_content$error$message)) {
      error_content$error$message
    } else {
      paste("HTTP status code:", status)
    }

    return(paste(
      "AI ANALYST ERROR:",
      paste0("The OpenAI API returned an error - ", error_msg),
      "Please verify your OPENAI_API_KEY in the .env file and your account quota.",
      sep = "\n"
    ))
  }

  # ---- 7. Parse the successful response -----------------------------------
  parsed <- tryCatch({
    httr::content(response, as = "parsed", type = "application/json")
  }, error = function(e) NULL)

  if (is.null(parsed) ||
      is.null(parsed$choices) ||
      length(parsed$choices) == 0 ||
      is.null(parsed$choices[[1]]$message$content)) {
    return(paste(
      "AI ANALYST ERROR:",
      "Received an unexpected response format from the OpenAI API.",
      sep = "\n"
    ))
  }

  insight_text <- trimws(parsed$choices[[1]]$message$content)

  return(insight_text)
}
