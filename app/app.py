import os
import re
import html
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import keras
from keras.utils import pad_sequences

app = Flask(__name__)

# Base directories setup
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODELS_DIR / "bigru.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"

# Model hyperparameters matching training specifications
MAX_LEN = 28
VOCAB_LIMIT = 50000

print("[INFO] Loading BiGRU model and Tokenizer...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer file not found at: {TOKENIZER_PATH}")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

model = keras.models.load_model(str(MODEL_PATH))
print("[INFO] Model and Tokenizer loaded successfully!")


def clean_text(text: str) -> str:
    """Preprocess text according to the training pipeline:
    1. Unescape HTML entities (&amp; -> &)
    2. Strip URLs and @mentions
    3. Space out punctuation (!, ?, #)
    4. Normalize whitespace
    """
    if not text:
        return ""
    # 1. HTML unescape
    text = html.unescape(text)
    # 2. Remove URLs and user mentions
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    # 3. Space out sentiment-carrying punctuation (!, ?, #)
    text = re.sub(r"([!?#])", r" \1 ", text)
    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str):
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "error": "The input text is empty or contains only URLs/mentions."
        }

    # Convert text to sequences using fitted tokenizer
    sequences = tokenizer.texts_to_sequences([cleaned])
    token_indices = sequences[0] if sequences else []
    
    # Map back indices to words known in vocabulary
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    matched_words = [index_to_word.get(idx, "<UNK>") for idx in token_indices]

    # Pad sequences to 28 tokens with post-padding as defined in training
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Raw probability prediction (sigmoid output)
    pred_prob = float(model.predict(padded, verbose=0)[0][0])
    
    is_positive = pred_prob >= 0.5
    sentiment = "Positive" if is_positive else "Negative"
    
    # Confidence represents confidence in the predicted class
    confidence = pred_prob if is_positive else (1.0 - pred_prob)

    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "is_positive": is_positive,
        "score": round(pred_prob, 4),
        "confidence": round(confidence * 100, 2),
        "positive_pct": round(pred_prob * 100, 1),
        "negative_pct": round((1.0 - pred_prob) * 100, 1),
        "token_count": len(token_indices),
        "tokens": matched_words
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please enter a valid tweet or sentence to analyze."}), 400

        result = predict_sentiment(text)
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Host on 127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
