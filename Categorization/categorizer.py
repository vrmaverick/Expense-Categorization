import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# # Load Models Once (Global Load)

# print("Loading Sentence-BERT Embedding Model...")
# encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# #change paths to models
# print("Loading trained ML models...")
# svm = joblib.load("/content/svm_classifier.pkl")
# xgb = joblib.load("/content/xgb_classifier.pkl")
# label_encoder = joblib.load("/content/label_encoder.pkl") # Category Encoder
# scaler = joblib.load("/content/scaler.pkl")               # Feature scaler for SVM

# print("Categorizer Ready ✔")


# # Core Function: Predict Category for Single or Batch Input

def Categorization(text):
    # Load Models Once (Global Load)

    print("Loading Sentence-BERT Embedding Model...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    #change paths to models
    print("Loading trained ML models...")
    svm = joblib.load("./Categorization_model/svm_classifier.pkl")
    xgb = joblib.load("./Categorization_model/xgb_classifier.pkl")
    label_encoder = joblib.load("./Categorization_model/label_encoder.pkl") # Category Encoder
    scaler = joblib.load("./Categorization_model/scaler.pkl")               # Feature scaler for SVM

    print("Categorizer Ready ✔")

    return predict_category(text,encoder,scaler,svm,xgb,label_encoder)


# Core Function: Predict Category for Single or Batch Input


def predict_category(text,encoder,scaler,svm,xgb,label_encoder):
    """
    Predicts category of a single expense string.

    Parameters:
    -----------
    text : str
        Expense description like "Starbucks coffee" or "Uber ride"
    return_probs : bool
        If True, returns (category, probability_distribution)

    Returns:
    --------
    category : str
    """

    if not text or not isinstance(text, str):
        return "Invalid Input"

    emb = encoder.encode([text], convert_to_numpy=True)

    # Scale for SVM (XGBoost doesn't need scaling)
    emb_scaled = scaler.transform(emb)

    # Get model probabilities
    proba_svm = svm.predict_proba(emb_scaled)
    proba_xgb = xgb.predict_proba(emb)

    # Weighted Ensemble (tune ratio if needed)
    final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)

    pred_idx = np.argmax(final_proba)
    category = label_encoder.inverse_transform([pred_idx])[0]

    # if return_probs:
    #     return category, final_proba.tolist()

    return category

# Batch Prediction Support

def predict_bulk(list_of_expenses,encoder,scaler,svm,xgb,label_encoder):
    """
    Predict multiple descriptions at once.
    Returns list of categories.

    Example:
    --------
    predict_bulk(["Dominos pizza", "Electricity bill"]) ->
    ['Food', 'Utilities']
    """

    embeddings = encoder.encode(list_of_expenses, convert_to_numpy=True)
    embeddings_scaled = scaler.transform(embeddings)

    proba_svm = svm.predict_proba(embeddings_scaled)
    proba_xgb = xgb.predict_proba(embeddings)

    final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)

    idx = np.argmax(final_proba, axis=1)
    return label_encoder.inverse_transform(idx)
