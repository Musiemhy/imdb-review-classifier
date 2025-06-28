from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("models/sentiment_model.joblib")

@app.route("/")
def home():
    return render_template("test_sentiment.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    proba = model.predict_proba([text])[0]
    label = model.predict([text])[0]
    sentiment = "positive" if label == 1 else "negative"
    confidence = float(proba[label])
    return jsonify({"sentiment": sentiment, "confidence": round(confidence, 2)})

if __name__ == "__main__":
    app.run(debug=True)
