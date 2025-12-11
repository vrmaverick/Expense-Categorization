# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer

# # # Load Models Once (Global Load)

# # print("Loading Sentence-BERT Embedding Model...")
# # encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # #change paths to models
# # print("Loading trained ML models...")
# # svm = joblib.load("/content/svm_classifier.pkl")
# # xgb = joblib.load("/content/xgb_classifier.pkl")
# # label_encoder = joblib.load("/content/label_encoder.pkl") # Category Encoder
# # scaler = joblib.load("/content/scaler.pkl")               # Feature scaler for SVM

# # print("Categorizer Ready ✔")


# # # Core Function: Predict Category for Single or Batch Input

# def Categorization(text):
#     # Load Models Once (Global Load)

#     print("Loading Sentence-BERT Embedding Model...")
#     encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#     #change paths to models
#     print("Loading trained ML models...")
#     svm = joblib.load("./Categorization_model/svm_classifier.pkl")
#     xgb = joblib.load("./Categorization_model/xgb_classifier.pkl")
#     label_encoder = joblib.load("./Categorization_model/label_encoder.pkl") # Category Encoder
#     scaler = joblib.load("./Categorization_model/scaler.pkl")               # Feature scaler for SVM

#     print("Categorizer Ready ✔")

#     return predict_category(text,encoder,scaler,svm,xgb,label_encoder)


# # Core Function: Predict Category for Single or Batch Input


# def predict_category(text,encoder,scaler,svm,xgb,label_encoder):
#     """
#     Predicts category of a single expense string.

#     Parameters:
#     -----------
#     text : str
#         Expense description like "Starbucks coffee" or "Uber ride"
#     return_probs : bool
#         If True, returns (category, probability_distribution)

#     Returns:
#     --------
#     category : str
#     """

#     if not text or not isinstance(text, str):
#         return "Invalid Input"

#     emb = encoder.encode([text], convert_to_numpy=True)

#     # Scale for SVM (XGBoost doesn't need scaling)
#     emb_scaled = scaler.transform(emb)

#     # Get model probabilities
#     proba_svm = svm.predict_proba(emb_scaled)
#     proba_xgb = xgb.predict_proba(emb)

#     # Weighted Ensemble (tune ratio if needed)
#     final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)

#     pred_idx = np.argmax(final_proba)
#     category = label_encoder.inverse_transform([pred_idx])[0]

#     # if return_probs:
#     #     return category, final_proba.tolist()

#     return category

# # Batch Prediction Support

# def predict_bulk(list_of_expenses,encoder,scaler,svm,xgb,label_encoder):
#     """
#     Predict multiple descriptions at once.
#     Returns list of categories.

#     Example:
#     --------
#     predict_bulk(["Dominos pizza", "Electricity bill"]) ->
#     ['Food', 'Utilities']
#     """

#     embeddings = encoder.encode(list_of_expenses, convert_to_numpy=True)
#     embeddings_scaled = scaler.transform(embeddings)

#     proba_svm = svm.predict_proba(embeddings_scaled)
#     proba_xgb = xgb.predict_proba(embeddings)

#     final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)

#     idx = np.argmax(final_proba, axis=1)
#     return label_encoder.inverse_transform(idx)

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ---- Load once (module import) ----
print("Loading Sentence-BERT Embedding Model...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # [web:118]

print("Loading trained ML models...")
svm = joblib.load("./Categorization_model/svm_classifier.pkl")
xgb = joblib.load("./Categorization_model/xgb_classifier.pkl")
label_encoder = joblib.load("./Categorization_model/label_encoder.pkl")  # Category Encoder
scaler = joblib.load("./Categorization_model/scaler.pkl")                # Feature scaler for SVM

print("Categorizer Ready ✔")


# def Categorization(text):
#     if not text or not isinstance(text, str):
#         return "Invalid Input", None, None

#     emb = encoder.encode([text], convert_to_numpy=True)
#     emb_scaled = scaler.transform(emb)

#     proba_svm = svm.predict_proba(emb_scaled)
#     proba_xgb = xgb.predict_proba(emb)
#     final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)  # (1, n_classes)

#     # Use classes_ from the model
#     class_labels = list(svm.classes_)
#     # OR, if xgb is the canonical one:
#     # class_labels = list(xgb.classes_)

#     # argmax index in that same order
#     pred_idx = int(np.argmax(final_proba, axis=1)[0])
#     category = class_labels[pred_idx]

#     return category, final_proba, class_labels

def Categorization(text):
    if not text or not isinstance(text, str):
        return "Invalid Input", None, None

    emb = encoder.encode([text], convert_to_numpy=True)
    emb_scaled = scaler.transform(emb)

    proba_svm = svm.predict_proba(emb_scaled)
    proba_xgb = xgb.predict_proba(emb)
    final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)  # (1, n_classes)

    # ids are the encoded labels the models were trained with
    class_ids = svm.classes_          # shape (n_classes,)

    # Convert to original string labels using label_encoder
    # This requires that label_encoder was fit on the same y used for svm/xgb
    class_labels = label_encoder.inverse_transform(class_ids)

    # argmax index in prob array → index in class_ids → index in class_labels
    pred_pos = int(np.argmax(final_proba, axis=1)[0])
    predicted_label = class_labels[pred_pos]

    return predicted_label, final_proba, class_labels





def predict_category(text):
    """Simple helper if you only need the label."""
    category, probs, _ = Categorization(text)
    return category


def predict_bulk(list_of_expenses):
    """
    Batch prediction, returns np.array of labels.
    """
    if not list_of_expenses:
        return np.array([])

    embeddings = encoder.encode(list_of_expenses, convert_to_numpy=True)
    embeddings_scaled = scaler.transform(embeddings)

    proba_svm = svm.predict_proba(embeddings_scaled)
    proba_xgb = xgb.predict_proba(embeddings)

    final_proba = (0.7 * proba_svm) + (0.3 * proba_xgb)
    idx = np.argmax(final_proba, axis=1)
    return label_encoder.inverse_transform(idx)

