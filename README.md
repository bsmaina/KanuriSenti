# KanuriSenti

A lexicon- and TF–IDF + logistic-regression baseline for Kanuri sentiment analysis.

## 1. Environment Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/bsmaina/KanuriSenti.git
   cd KanuriSenti
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Directory Structure

```
KanuriSenti/
├── data/
│   ├── train.csv          # training split (texts + labels)
│   └── test.csv           # test split (texts + labels)
├── lexicons/
│   ├── polarity_lexicon.json
│   └── vad_lexicon.json
├── src/
│   ├── preprocess.py      # tokenization, cleaning routines
│   ├── train.py           # TF–IDF + logistic-regression training script
│   └── evaluate.py        # evaluation script
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── logreg_model.joblib
├── README.md
└── requirements.txt
```

## 3. Data Loading & Preprocessing

In `src/preprocess.py` you’ll find functions to:
- Read `data/*.csv`
- Clean and tokenize text
- Return `(X_train, y_train), (X_test, y_test)`

Example usage in `train.py`:
```python
from preprocess import load_data
X_train, X_test, y_train, y_test = load_data("data/train.csv", "data/test.csv")
```

## 4. Training

Run:
```bash
python src/train.py   --train-path data/train.csv   --output-dir models/   --random-seed 42
```
This will fit a TF–IDF vectorizer + logistic regression and save them under `models/`.

## 5. Hyperparameters

See **HYPERPARAMETERS.md** for the full list.

## 6. Evaluation

To reproduce our reported metrics, run:
```bash
python src/evaluate.py   --model models/logreg_model.joblib   --vectorizer models/tfidf_vectorizer.joblib   --test-path data/test.csv   --output results/metrics.json
```
This script will output accuracy, macro-F1, per-class precision/recall/F1, and a confusion matrix.
