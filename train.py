import joblib
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

def load_imdb_data():
    dataset = load_files("./imdb_data", categories=['positive', 'negative'], shuffle=True, encoding="utf-8")
    return dataset.data, dataset.target

def main():
    print("Loading data...")
    X, y = load_imdb_data()

    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/sentiment_model.joblib")
    print("Model saved to models/sentiment_model.joblib")

if __name__ == "__main__":
    main()
