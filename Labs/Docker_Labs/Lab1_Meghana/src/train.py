import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print(f"Model trained. Test accuracy: {acc:.4f}")
    print("Saved to model/model.pkl")


if __name__ == "__main__":
    main()
