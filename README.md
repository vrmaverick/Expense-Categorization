# FAI Project 104

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