import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model/model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify(error='Send JSON like {"features": [5.1, 3.5, 1.4, 0.2]}'), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify(error="Expected exactly 4 features"), 400

    pred = int(model.predict([features])[0])
    return jsonify(prediction=pred, class_name=CLASS_NAMES[pred])


if __name__ == "__main__":
    # 0.0.0.0 is required inside Docker; 127.0.0.1 would be unreachable from your Mac
    app.run(host="0.0.0.0", port=5000)
