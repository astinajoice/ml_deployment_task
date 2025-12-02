from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)

model = load("model.joblib")
tfidf = load("tfidf_vectorizer.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Spam Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field in JSON"}), 400

    user_text = data["text"]
    text_tfidf = tfidf.transform([user_text])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0].tolist()

    return jsonify({
        "input_text": user_text,
        "prediction": prediction,
        "probabilities": {
            "ham": probabilities[0],
            "spam": probabilities[1]
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
