import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "iris_model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def predict_data(X):
    """X: list of [sepal_length, sepal_width, petal_length, petal_width]"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    return [
        {
            "response": int(p),
            "species": CLASS_NAMES[int(p)],
            "confidence": round(float(prob.max()), 3),
        }
        for p, prob in zip(preds, probs)
    ]