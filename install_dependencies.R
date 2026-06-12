# ============================================================
# install_dependencies.R
# Installs all R packages required to run the Autonomous
# No-Code Machine Learning Platform & AI Analyst.
# ============================================================

required_packages <- c(
  "shiny",
  "shinydashboard",
  "shinycssloaders",
  "caret",
  "randomForest",
  "xgboost",
  "dplyr",
  "plotly",
  "httr",
  "jsonlite",
  "dotenv",
  "DT",
  "e1071"   # required by caret::confusionMatrix / multiClassSummary
)

installed <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    message(paste("Installing package:", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(paste("Package already installed:", pkg))
  }
}

message("\nAll dependencies are ready. You can now run the app with:")
message('  shiny::runApp("app.R")')
