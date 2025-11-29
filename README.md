# Expense Categorization using MiniLM, SVM, and XGBoost

This project builds an **expense categorization system** that predicts categories (e.g., `Food`, `Transport`, `Shopping`, `Rent`, etc.) from raw transaction descriptions like:

- "Uber ride to office"
- "Starbucks latte and croissant"
- "Amazon purchase – phone charger"

It is designed as a core module for a **Personal Finance AI Agent** that can analyze transactions, categorize them, and support downstream analytics such as budgeting and trend detection.

---

## Project Overview

**Goal:** Given a text description of a financial transaction, assign it to a meaningful category.

**Key ideas:**

- Use a **pre-trained language model** (MiniLM via `sentence-transformers`) to convert text into dense semantic embeddings.
- Train **classical machine learning classifiers** (SVM, XGBoost) on top of these embeddings.
- Combine models into an **ensemble** to improve robustness and accuracy.
- Provide interpretable outputs such as classification reports and confusion matrices.

---

## Model Architecture

1. **Text Input**  
   Raw expense description (e.g., `"Dominos Pizza delivery"`).

2. **Embedding Layer (MiniLM)**  
   - Model: `sentence-transformers/all-MiniLM-L6-v2`  
   - Converts each description into a dense vector that captures semantic meaning.

3. **Classical ML Classifiers**
   - **SVM (RBF Kernel)**  
     - Applied on standardized embeddings.  
     - Hyperparameters tuned using `GridSearchCV` (`C`, `gamma`).  
   - **XGBoost**  
     - Gradient boosting over embedding features.  
     - Configured with `n_estimators`, `learning_rate`, `max_depth`, etc.

4. **Ensemble Layer**
   - Average the predicted probabilities from:
     - Tuned SVM
     - XGBoost
   - Final prediction = argmax of the averaged probability vector.
