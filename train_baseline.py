#!/usr/bin/env python3
"""
train_baseline.py

A minimal supervised baseline for KanuriSenti polarity classification.
Usage example:
  python train_baseline.py \
    --polarity-lexicon path/to/polarity_lexicon.xlsx \
    --test-size 0.2 \
    --ngram-range 1 2 \
    --class-weight balanced \
    --output metrics.json
"""

import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_polarity(path):
    df = pd.read_excel(path)
    df = df[['kanuri', 'Polarity']].dropna()
    df = df[df['Polarity'].isin(['positive', 'neutral', 'negative'])]
    return df

def train_and_evaluate(df, test_size, ngram_range, class_weight):
    X = df['kanuri']
    y = df['Polarity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['negative','neutral','positive']).tolist(),
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train a TF–IDF + LogisticRegression baseline on KanuriSenti.")
    parser.add_argument('--polarity-lexicon', required=True, help="Path to the polarity lexicon Excel file")
    parser.add_argument('--test-size', type=float, default=0.2, help="Proportion of data to reserve for testing")
    parser.add_argument('--ngram-range', type=int, nargs=2, default=[1,2], help="TF–IDF ngram range, e.g. 1 2")
    parser.add_argument('--class-weight', default=None, help="Classifier class_weight parameter (e.g., balanced)")
    parser.add_argument('--output', default='metrics.json', help="Path to save JSON metrics output")
    args = parser.parse_args()

    df_pol = load_polarity(args.polarity_lexicon)
    metrics = train_and_evaluate(df_pol, args.test_size, tuple(args.ngram_range), args.class_weight)

    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training complete. Metrics saved to {args.output}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

if __name__ == '__main__':
    main()
