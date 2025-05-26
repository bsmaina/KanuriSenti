# Hyperparameter Settings

All hyperparameters below were fixed for our baseline.

## TF–IDF Vectorizer

| Parameter       | Value         | Description                                  |
|-----------------|---------------|----------------------------------------------|
| ngram_range     | (1, 2)        | unigrams + bigrams                           |
| max_df          | 0.95          | ignore tokens in >95% of documents           |
| min_df          | 5             | ignore tokens in fewer than 5 documents      |
| max_features    | 10000         | limit vocabulary size                        |
| strip_accents   | 'unicode'     | remove diacritics                            |
| token_pattern   | r"(?u)\b\w\w+\b" | tokens of 2+ alphanumeric characters |

## Logistic Regression

| Parameter      | Value        | Description                              |
|----------------|--------------|------------------------------------------|
| C              | 1.0          | inverse of regularization strength       |
| penalty        | 'l2'         | L2 regularization                        |
| solver         | 'liblinear'  | suitable for small datasets              |
| max_iter       | 1000         | maximum optimization iterations          |
| random_state   | 42           | seed for reproducibility                 |
