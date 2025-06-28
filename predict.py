import sys
import joblib
import numpy as np

def predict(text):
    model = joblib.load("models/sentiment_model.joblib")
    proba = model.predict_proba([text])[0]
    label = model.predict([text])[0]
    
    sentiment = "positive" if label == 1 else "negative"
    confidence = proba[label]
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        sys.exit(1)
    text = sys.argv[1]
    predict(text)
