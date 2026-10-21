# Tweet Sentiment Analysis: BiLSTM vs BiGRU

Binary sentiment classification is performed on the Sentiment140 dataset (1.6M tweets). A tuned bidirectional LSTM is compared against a tuned bidirectional GRU, and a small web app is provided for live predictions.

## Summary

- About **82.4–82.6% test accuracy** is reached by both models. The ranking between them was flipped across two training runs, so the difference is considered to be within run-to-run variation.
- Inference time is identical (1.2 s for 159,719 tweets).
- Over 99% of each model's parameters are contained in the embedding layer, so total model size is mostly determined by embedding width and not by the recurrent cell.
- Overfitting is started by both models after 1–2 epochs. Early stopping with best-weight restoration is therefore required.

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) is used: 1.6M tweets, balanced between negative (0) and positive (1).

The labels were assigned automatically from emoticons, and the emoticons were then removed from the text. The labels are therefore noisy by construction, and accuracy on this dataset is naturally capped well below 100%.

## Preprocessing

1. HTML entities are unescaped (`&amp;` → `&`)
2. URLs and @mentions are removed, and whitespace is normalized
3. Tweets that became empty are dropped
4. `!`, `?` and `#` are spaced out so that punctuation becomes its own token (sentiment signal is carried by it)
5. A **stratified 80/10/10 train/val/test split is made before the tokenizer is fitted**
6. The Keras `Tokenizer` is fitted on the **training set only** (lowercased, OOV token, vocabulary capped at 50,000)
7. Sequences are padded/truncated to **28 tokens** (99th percentile of tweet length), with `padding="post"`

Because the split is made before the tokenizer is fitted, test-set vocabulary is not leaked into training.

## Models and tuning

Two architectures are used, and each is tuned independently with Keras Tuner (Hyperband, objective `val_accuracy`):

- Search space: embedding dim {50, 100, 200}, units {32, 64, 128}, depth 1–3, dropout 0.2–0.5, dense units {32, 64}, learning rate {1e-2, 3e-3, 1e-3, 3e-4}
- `SpatialDropout1D` is added after the embedding in the final models, and early stopping on validation loss (`restore_best_weights=True`) is used during training
- Tuning is performed on the validation set only. The test set is used for final evaluation.

Best configurations found: **[TODO: paste the final hyperparameters for each model]**

## Results

| | Tuned BiLSTM | Tuned BiGRU |
|---|---|---|
| Test accuracy (run 1) | 82.46% | 82.41% |
| Test accuracy (run 2) | 82.36% | 82.61% |
| Embedding (dim / params) | 200 / 10.0M | 100 / 5.0M |
| Recurrent + dense params | 61,761 | 48,769 |
| Total params | 10,061,761 | 5,048,769 |
| Inference time (159,719 tweets) | 1.2 s | 1.2 s |
| Epochs until early stop | 4 | 3 |


Precision, recall and F1 are all about 0.82–0.83 for both classes, so neither label is favored by either model. Validation and test accuracy agree closely (about 82.4%), which indicates that no leakage has occurred.

![Training curves](results/curves)

## Key findings

1. **No clear winner.** BiLSTM and BiGRU are found to be statistically indistinguishable in accuracy, and their order was flipped between runs.
2. **Model size is dominated by the embedding.** It accounts for 99.4% of the LSTM's parameters and 99.0% of the GRU's. The 2x total-size difference is caused by different embedding dimensions being chosen by the tuner (200 vs 100), not by LSTM vs GRU.
3. **Overfitting is fast.** Training accuracy is observed to climb while validation accuracy peaks within 1–2 epochs. A large embedding table memorizing rare words and the noisy labels are suspected to be the causes. Training was stabilized by a reduced vocabulary, spatial dropout and early stopping, but the validation peak was not raised.
4. **Accuracy is probably label-limited.** Because validation accuracy stayed near 82.4% across many configurations, the ceiling is believed to come from the emoticon-derived labels more than from the architecture.

## Limitations

- **The comparison is not controlled.** The two models were tuned independently, so differences in embedding size, depth and dense width exist in addition to LSTM vs GRU. "Best-tuned BiLSTM" is compared with "best-tuned BiGRU".
- **The domain is narrow.** The models were trained on 2009-era tweets, so lower performance is expected on other text or on modern slang.

## Web app

An interactive web application is provided to test live inference using the trained **BiGRU** model (`models/bigru.keras`) and tokenizer (`models/tokenizer.pkl`). The BiGRU was chosen because it achieves parity with the BiLSTM in classification accuracy while maintaining half the parameter footprint.

### Key Features
- **Real-time JS Inference**: Submits text asynchronously via `fetch` API without page reloads.
- **Strict Preprocessing Pipeline**: Matches training exactly (HTML unescape, stripping URLs and `@mentions`, punctuation spacing for `!`, `?`, `#`, and post-padding to 28 tokens).
- **Confidence & Probability Distribution**: Displays predicted sentiment class, confidence percentage, raw sigmoid score, and a visual split bar.
- **Inspection Accordion**: View the cleaned text representation and vocabulary tokens recognized by the Keras tokenizer.
- **Interactive UI**: Character counter (280 max), keyboard shortcut (`Ctrl+Enter` / `Cmd+Enter`), quick sample buttons, and responsive bright modern design.

## Project structure

```
SA pro/
├── notebooks/
│   └── sentiment_analysis.ipynb
├── app/
│   ├── app.py
│   ├── templates/index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── models/
│   ├── bilstm.keras
│   ├── bigru.keras
│   └── tokenizer.pkl
├── results/
│   ├── curves.png
│   └── metrics.csv
├── requirements.txt
└── README.md
```

## How to run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask application**:
   ```bash
   cd app
   python app.py
   ```

3. **Open in browser**:
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

To retrain, `notebooks/sentiment_analysis.ipynb` should be opened in Google Colab (GPU runtime). The dataset is downloaded by the notebook through `kagglehub`.
