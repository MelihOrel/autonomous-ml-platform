# ============================================================
# app.R
# Autonomous No-Code Machine Learning Platform & AI Analyst
# A Shiny dashboard for automated data cleaning, model training,
# and LLM-driven results interpretation.
# ============================================================

library(shiny)
library(shinydashboard)
library(shinycssloaders)
library(dplyr)
library(plotly)
library(DT)
library(dotenv)

# ---- Load environment variables (.env) -----------------------------
if (file.exists(".env")) {
  dotenv::load_dot_env(file = ".env")
}

# ---- Source custom backend modules ----------------------------------
source("R/auto_clean.R")
source("R/model_train.R")
source("R/ai_analyst.R")

# ============================================================
# UI
# ============================================================

ui <- dashboardPage(

  skin = "blue",

  dashboardHeader(
    title = "Autonomous No-Code ML Platform"
  ),

  dashboardSidebar(
    width = 320,
    sidebarMenu(
      menuItem("Modeling Studio", tabName = "studio", icon = icon("robot"))
    ),
    tags$div(style = "padding: 15px;",

      fileInput(
        inputId = "data_file",
        label = "1. Upload CSV Dataset",
        accept = c(".csv"),
        buttonLabel = "Browse...",
        placeholder = "No file selected"
      ),

      uiOutput("target_selector_ui"),

      radioButtons(
        inputId = "algorithm",
        label = "3. Select Algorithm",
        choices = c("Random Forest" = "Random Forest",
                     "XGBoost" = "XGBoost"),
        selected = "Random Forest"
      ),

      hr(),

      actionButton(
        inputId = "run_pipeline",
        label = "Auto-Clean & Train Model",
        icon = icon("play"),
        class = "btn-primary btn-block",
        style = "font-weight: bold;"
      ),

      hr(),

      tags$small(
        style = "color: #888;",
        "Upload a CSV file, choose a target column and algorithm, then click the button above. The platform will automatically clean your data, train a model, and generate an AI-powered analysis of the results."
      )
    )
  ),

  dashboardBody(

    tags$head(
      tags$link(rel = "stylesheet", type = "text/css", href = "custom_styles.css")
    ),

    tabItems(
      tabItem(
        tabName = "studio",

        fluidRow(
          valueBoxOutput("accuracy_box", width = 4),
          valueBoxOutput("precision_box", width = 4),
          valueBoxOutput("recall_box", width = 4)
        ),

        fluidRow(
          box(
            title = "Data Cleaning Summary",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            collapsible = TRUE,
            withSpinner(
              verbatimTextOutput("cleaning_summary"),
              type = 6
            )
          )
        ),

        fluidRow(
          box(
            title = "Feature Importance",
            status = "primary",
            solidHeader = TRUE,
            width = 6,
            withSpinner(
              plotlyOutput("importance_plot", height = "400px"),
              type = 6
            )
          ),
          box(
            title = "Confusion Matrix",
            status = "primary",
            solidHeader = TRUE,
            width = 6,
            withSpinner(
              verbatimTextOutput("confusion_matrix_output"),
              type = 6
            )
          )
        ),

        fluidRow(
          box(
            title = tagList(icon("brain"), "AI Analyst Interpretation"),
            status = "success",
            solidHeader = TRUE,
            width = 12,
            withSpinner(
              verbatimTextOutput("ai_insight_output"),
              type = 6
            )
          )
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {

  # ---- Reactive: read the uploaded CSV --------------------------------
  raw_data <- reactive({
    req(input$data_file)

    tryCatch({
      df <- read.csv(input$data_file$datapath, stringsAsFactors = FALSE,
                      na.strings = c("NA", "N/A", "", "null", "NULL"))
      if (nrow(df) == 0 || ncol(df) == 0) {
        stop("The uploaded file appears to be empty.")
      }
      df
    }, error = function(e) {
      showNotification(
        paste("Error reading CSV file:", e$message),
        type = "error", duration = 8
      )
      NULL
    })
  })

  # ---- Dynamic UI: target variable dropdown ----------------------------
  output$target_selector_ui <- renderUI({
    df <- raw_data()

    if (is.null(df)) {
      return(
        selectInput(
          inputId = "target_col",
          label = "2. Select Target Variable",
          choices = c("Upload a file first" = ""),
          selected = ""
        )
      )
    }

    # Suggest a default: prefer the last column, or a column with few unique values
    candidate_cols <- names(df)
    selectInput(
      inputId = "target_col",
      label = "2. Select Target Variable",
      choices = candidate_cols,
      selected = candidate_cols[length(candidate_cols)]
    )
  })

  # ---- Reactive Values store for pipeline results -----------------------
  pipeline_results <- reactiveValues(
    cleaning_summary = NULL,
    metrics = NULL,
    importance = NULL,
    conf_matrix = NULL,
    ai_insight = NULL,
    error = NULL
  )

  # ---- Main pipeline: triggered by action button -------------------------
  observeEvent(input$run_pipeline, {

    # --- Validation guards -------------------------------------------
    df <- raw_data()

    if (is.null(df)) {
      showNotification(
        "Please upload a valid CSV file before running the pipeline.",
        type = "error", duration = 6
      )
      return(NULL)
    }

    if (is.null(input$target_col) || input$target_col == "") {
      showNotification(
        "Please select a target variable.",
        type = "error", duration = 6
      )
      return(NULL)
    }

    if (!input$target_col %in% names(df)) {
      showNotification(
        "Selected target column is not present in the uploaded data.",
        type = "error", duration = 6
      )
      return(NULL)
    }

    # Reset previous results / errors
    pipeline_results$error <- NULL
    pipeline_results$cleaning_summary <- NULL
    pipeline_results$metrics <- NULL
    pipeline_results$importance <- NULL
    pipeline_results$conf_matrix <- NULL
    pipeline_results$ai_insight <- NULL

    withProgress(message = "Running Autonomous ML Pipeline", value = 0, {

      # ---- STEP 1: Auto-clean data -------------------------------------
      incProgress(0.1, detail = "Cleaning data...")

      clean_result <- tryCatch({
        auto_clean_data(df, input$target_col)
      }, error = function(e) {
        pipeline_results$error <- paste("Data Cleaning Error:", e$message)
        NULL
      })

      if (is.null(clean_result)) {
        showNotification(pipeline_results$error, type = "error", duration = 10)
        return(NULL)
      }

      pipeline_results$cleaning_summary <- clean_result$summary
      cleaned_df <- clean_result$data

      # ---- STEP 2: Train and evaluate model -----------------------------
      incProgress(0.4, detail = paste("Training", input$algorithm, "model..."))

      train_result <- tryCatch({
        train_and_evaluate(cleaned_df, input$target_col, input$algorithm)
      }, error = function(e) {
        pipeline_results$error <- paste("Model Training Error:", e$message)
        NULL
      })

      if (is.null(train_result)) {
        showNotification(pipeline_results$error, type = "error", duration = 10)
        return(NULL)
      }

      pipeline_results$metrics <- list(
        accuracy = train_result$accuracy,
        precision = train_result$precision,
        recall = train_result$recall,
        algorithm = input$algorithm
      )
      pipeline_results$importance <- train_result$feature_importance
      pipeline_results$conf_matrix <- train_result$confusion_matrix

      # ---- STEP 3: Generate AI Insight -----------------------------------
      incProgress(0.8, detail = "Consulting AI Analyst...")

      ai_text <- tryCatch({
        generate_ai_insight(
          metrics_list = pipeline_results$metrics,
          top_features = head(train_result$feature_importance, 5)
        )
      }, error = function(e) {
        paste("AI ANALYST ERROR: Unexpected failure -", e$message)
      })

      pipeline_results$ai_insight <- ai_text

      incProgress(1, detail = "Done!")
    })

    showNotification("Pipeline completed successfully.", type = "message", duration = 5)
  })

  # ============================================================
  # OUTPUTS
  # ============================================================

  # ---- Value Boxes ------------------------------------------------------
  output$accuracy_box <- renderValueBox({
    val <- if (!is.null(pipeline_results$metrics)) {
      paste0(round(pipeline_results$metrics$accuracy * 100, 2), "%")
    } else {
      "N/A"
    }
    valueBox(val, "Accuracy", icon = icon("bullseye"), color = "green")
  })

  output$precision_box <- renderValueBox({
    val <- if (!is.null(pipeline_results$metrics)) {
      paste0(round(pipeline_results$metrics$precision * 100, 2), "%")
    } else {
      "N/A"
    }
    valueBox(val, "Precision", icon = icon("crosshairs"), color = "blue")
  })

  output$recall_box <- renderValueBox({
    val <- if (!is.null(pipeline_results$metrics)) {
      paste0(round(pipeline_results$metrics$recall * 100, 2), "%")
    } else {
      "N/A"
    }
    valueBox(val, "Recall", icon = icon("magnet"), color = "yellow")
  })

  # ---- Cleaning Summary --------------------------------------------------
  output$cleaning_summary <- renderText({
    if (is.null(pipeline_results$cleaning_summary)) {
      return("Upload a dataset and click 'Auto-Clean & Train Model' to see the cleaning summary here.")
    }
    pipeline_results$cleaning_summary
  })

  # ---- Feature Importance Plot (plotly) ----------------------------------
  output$importance_plot <- renderPlotly({
    imp_df <- pipeline_results$importance

    if (is.null(imp_df) || nrow(imp_df) == 0) {
      return(plotly_empty(type = "bar") %>%
               layout(title = "No data yet. Run the pipeline to view feature importance."))
    }

    top_n <- head(imp_df, 10)
    top_n <- top_n[order(top_n$Importance), ]  # ascending for horizontal bar

    plot_ly(
      data = top_n,
      x = ~Importance,
      y = ~reorder(Feature, Importance),
      type = "bar",
      orientation = "h",
      marker = list(color = "rgba(58, 142, 226, 0.8)")
    ) %>%
      layout(
        title = "Top Feature Importances",
        xaxis = list(title = "Importance Score"),
        yaxis = list(title = ""),
        margin = list(l = 120)
      )
  })

  # ---- Confusion Matrix Output -------------------------------------------
  output$confusion_matrix_output <- renderText({
    cm <- pipeline_results$conf_matrix

    if (is.null(cm)) {
      return("Run the pipeline to generate the confusion matrix.")
    }

    cm_table <- cm$table
    cm_str <- capture.output(print(cm_table))

    overall_stats <- paste0(
      "\nOverall Statistics:\n",
      "Accuracy : ", round(cm$overall["Accuracy"], 4), "\n",
      "Kappa    : ", round(cm$overall["Kappa"], 4)
    )

    paste(c(cm_str, overall_stats), collapse = "\n")
  })

  # ---- AI Analyst Output ---------------------------------------------------
  output$ai_insight_output <- renderText({
    if (is.null(pipeline_results$ai_insight)) {
      return("The AI Analyst's interpretation will appear here once the model has been trained.")
    }
    pipeline_results$ai_insight
  })
}

# ============================================================
# Run the application
# ============================================================
shinyApp(ui = ui, server = server)
