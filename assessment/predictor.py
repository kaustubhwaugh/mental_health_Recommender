import joblib
import os
import pandas as pd
from django.conf import settings

BASE_DIR = settings.BASE_DIR
model_path = os.path.join(BASE_DIR, "models")

# Load models
presence_model = joblib.load(os.path.join(model_path, "presence_model_rf.pkl"))
severity_models = {
    "Stress": joblib.load(os.path.join(model_path, "severity_rf_stress.pkl")),
    "Anxiety": joblib.load(os.path.join(model_path, "severity_rf_anxiety.pkl")),
    "Depression": joblib.load(os.path.join(model_path, "severity_rf_depression.pkl")),
}

severity_mapping = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

def make_predictions(X):
    """
    Predicts the presence and severity of Stress, Anxiety, and Depression.
    Args:
        X (pd.DataFrame): Input features in correct column order.
    Returns:
        dict: Condition -> Severity Level (e.g. {"Stress": "Mild"})
    """
    presence_preds = presence_model.predict(X)[0]
    severity_results = {}

    for i, cond in enumerate(["Stress", "Anxiety", "Depression"]):
        if presence_preds[i]:
            level = int(severity_models[cond].predict(X)[0])
            severity_results[cond] = severity_mapping.get(level, "None")
        else:
            severity_results[cond] = "None"

    return severity_results
