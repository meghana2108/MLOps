import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data import load_data, split_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "iris_model.pkl")

def fit_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, random_state=12)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")