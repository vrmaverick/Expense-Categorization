# Mint Sage: Personal Finance AI manager

Personal finance management is often tedious and error-prone due to scattered expense records, multi-channel transaction sources, and the absence of intelligent decision-making support. Most individuals struggle to track spending patterns, manually categorize purchases, and forecast future expenses, which ultimately affects long-term budgeting discipline. Mint Sage addresses these challenges by developing a unified AI-driven financial assistant capable of automating end-to-end expense management.

The system integrates three major components — expense categorization, expense forecasting, and budget optimization. Transaction text descriptions are preprocessed and converted into dense semantic vectors using MiniLM (Sentence-BERT embeddings), enabling contextual understanding beyond keyword matching. These embeddings are fed into SVM and XGBoost classifiers, later combined into an ensemble model for improved robustness and generalization across overlapping categories. The categorization engine achieves an accuracy of ~90%, demonstrating strong performance even on short and noisy transaction descriptions.
For financial prediction, time-series statistical modeling is employed using feature-engineered rolling windows, seasonal patterns, and smoothed trend signals. This enables forecasting of expected future expenditure, helping users anticipate overspending with reasonable predictive stability. In the final pipeline stage, an LLM-based recommendation layer analyzes categorized spending and predicted costs to generate personalized budgeting strategies, savings insights, and potential expense optimizations.

Overall, Mint Sage functions as a scalable personal finance intelligence platform that not only automates tracking but also interprets financial behavior, forecasts upcoming spending, and guides users toward data-driven financial decisions.

# Local setup
1) Clone the repository
```
git clone https://github.com/lune07/Expense-Categorization.git
cd Expense-Categorization
```


2)  Create and activate a virtual environment
```
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3) Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

4) Check model paths: ensure the following files exist in the expected locations:

```
Categorization_model/svm_classifier.pkl

Categorization_model/xgb_classifier.pkl

Categorization_model/label_encoder.pkl

Categorization_model/scaler.pkl
```

5) Running the Streamlit demo

a) Activate the environment If not already active:

```
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```
b) Start the app
From the project root:

```
python demo.py
This runs streamlit run app.py under the hood and opens the app in your browser (default http://localhost:8501)
```

