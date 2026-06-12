# 🤖 Autonomous No-Code ML Platform & AI Analyst

> **Upload data. Get a trained model. Get a plain-English explanation — automatically.**

An end-to-end, no-code machine learning platform built in **R Shiny** that lets anyone — no coding required — upload a CSV dataset, automatically clean it, train a classification model (**Random Forest** or **XGBoost**), and receive a **GenAI-powered analysis** of the results from a virtual Senior Data Scientist.

This project combines a production-style **R Shiny dashboard**, a robust **`caret`-based modeling pipeline**, and **LLM API integration (OpenAI)** to deliver automated insights — bridging the gap between raw data and business decisions.

---

## ✨ Features

### 🧹 Autonomous Data Cleaning
- Automatically imputes missing **numeric values with the median** and **categorical values with the mode**.
- Detects and removes **zero-variance columns** that add no predictive value.
- Drops rows with missing target values and converts the target column to a classification-ready **factor**.
- Returns a full, human-readable **audit log** of every transformation applied.

### 🧠 Automated Model Training & Evaluation
- Choose between **Random Forest** and **XGBoost** via a simple radio button — no hyperparameter tuning required.
- Performs an **80/20 train/test split** with 5-fold cross-validation under the hood (via `caret`).
- Automatically computes **Accuracy, Precision, Recall**, a full **Confusion Matrix**, and ranked **Feature Importances**.

### 💬 GenAI-Driven Interpretation
- Sends model metrics and top features to the **OpenAI API**.
- Returns a natural-language report written from the perspective of a **Senior Data Scientist**, covering:
  - What the metrics mean in plain English
  - Risks and limitations (e.g., class imbalance, overfitting signals)
  - What the top features suggest about underlying patterns
  - Actionable next steps

### 📊 Interactive Dashboard
- Built with `shinydashboard` for a clean, professional layout.
- **Plotly** bar chart for feature importance — fully interactive.
- Loading spinners (`shinycssloaders`) during training and AI inference.
- Graceful error handling for missing files, invalid targets, and API failures.

---

## 🗂️ Project Structure

```
autonomous-ml-platform/
├── app.R                  # Main Shiny application (UI + Server)
├── install_dependencies.R # One-click dependency installer
├── .env.example           # Template for API key configuration
├── .env                    # Your local API key configuration (not committed)
├── .gitignore
├── R/
│   ├── auto_clean.R        # Data cleaning module
│   ├── model_train.R       # Model training & evaluation module
│   └── ai_analyst.R         # LLM-powered insight generator
├── data/
│   └── sample_dataset.csv  # Example dataset for testing
└── www/
    └── custom_styles.css   # UI styling
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/autonomous-ml-platform.git
cd autonomous-ml-platform
```

### 2. Install R dependencies

Open R or RStudio in the project directory and run:

```r
source("install_dependencies.R")
```

This installs `shiny`, `shinydashboard`, `shinycssloaders`, `caret`, `randomForest`, `xgboost`, `dplyr`, `plotly`, `httr`, `jsonlite`, `dotenv`, `DT`, and `e1071`.

### 3. Configure your `.env` file

Copy the provided template and add your OpenAI API key:

```bash
cp .env.example .env
```

Then open `.env` and set:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

> 🔒 **Never commit your real `.env` file.** It is already excluded via `.gitignore`.

---

## 🚀 Usage

1. Launch the app from R or RStudio:

```r
shiny::runApp("app.R")
```

2. In the sidebar:
   - **Upload a CSV file** containing your dataset (a sample dataset is provided at `data/sample_dataset.csv`).
   - Select the **target variable** (the column you want to predict) from the dropdown — it updates automatically based on your uploaded columns.
   - Choose an algorithm: **Random Forest** or **XGBoost**.

3. Click **"Auto-Clean & Train Model"**.

4. The platform will automatically:
   - Clean your dataset and display a summary of all changes made.
   - Train and evaluate the selected model (80/20 split, cross-validated).
   - Display **Accuracy, Precision, and Recall** in value boxes.
   - Render an interactive **Feature Importance** chart and **Confusion Matrix**.
   - Generate an **AI Analyst report** explaining the results in plain language.

---

## 🧪 Try It with the Sample Dataset

A ready-to-use synthetic dataset is included at `data/sample_dataset.csv` — a loan approval dataset with columns such as `age`, `income`, `credit_score`, `years_employed`, `education`, `home_owner`, `region`, and the target column `loan_status`. Upload it, select `loan_status` as the target, and click **"Auto-Clean & Train Model"** to see the full pipeline in action.

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| UI / Dashboard | `shiny`, `shinydashboard`, `shinycssloaders` |
| Data Wrangling | `dplyr` |
| Modeling | `caret`, `randomForest`, `xgboost` |
| Visualization | `plotly` |
| AI Insights | OpenAI Chat Completions API via `httr` / `jsonlite` |
| Configuration | `dotenv` |

---

## ⚠️ Notes & Limitations

- Currently supports **classification** tasks only (the target column is automatically converted to a factor).
- Requires a valid **OpenAI API key** for the AI Analyst feature; without it, the app will display a graceful fallback message instead of an LLM-generated report.
- Designed for small-to-medium tabular datasets suitable for in-memory training with `caret`.

---

## 📄 License

This project is released under the MIT License. Feel free to use, modify, and extend it for your own portfolio or production use.
