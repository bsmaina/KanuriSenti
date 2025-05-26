import argparse
import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

def load_test_data(path):
    df = pd.read_csv(path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()
    return texts, labels

def main(args):
    X_test, y_test = load_test_data(args.test_path)
    vectorizer = joblib.load(args.vectorizer)
    model = joblib.load(args.model)
    X_vec = vectorizer.transform(X_test)
    preds = model.predict(X_vec)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    precision, recall, f1, support = precision_recall_fscore_support(y_test, preds, average=None, labels=model.classes_)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    results = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class': {
            cls: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i, cls in enumerate(model.classes_)
        },
        'confusion_matrix': cm.tolist(),
        'classes': model.classes_.tolist()
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KanuriSenti baseline.")
    parser.add_argument("--model", required=True, help="Path to trained logistic-regression model (.joblib)")
    parser.add_argument("--vectorizer", required=True, help="Path to TF–IDF vectorizer (.joblib)")
    parser.add_argument("--test-path", required=True, help="Path to test CSV with columns 'text' and 'label'")
    parser.add_argument("--output", required=True, help="Filepath to save metrics JSON")
    args = parser.parse_args()
    main(args)
