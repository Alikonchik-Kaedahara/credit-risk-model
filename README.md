# Credit risk model

Final project of the ML Junior course. A bank wants to know how likely a client is to default on a credit. For
3 million clients there is their credit history from the credit bureau: 26 million credits, one row per credit,
all values already cut into bins. The model predicts the default flag for a client. The metric is ROC-AUC, the
requirement is at least 0.75 on the test set.

## Result

ROC-AUC, cross-validation on 600,000 clients of the train set and the test set (20% of clients, 600,000):

| model | cv roc_auc | test roc_auc |
|---|---|---|
| LogisticRegression | 0.7365 | 0.7370 |
| RandomForest | 0.7407 | 0.7385 |
| HistGradientBoosting | 0.7512 | 0.7523 |
| LightGBM, default parameters | 0.7522 | 0.7537 |
| LightGBM, tuned, trained on the whole train set | | **0.7676** |

The models in the first four rows are trained on the 600,000 sample, the last one on all 2.4 million train
clients. The most important features are the share of credits that had an overdue, utilization of the limit and
payment statuses.

## Files

- `credit_risk_model.ipynb` - all steps: the data, features, experiments with features and models, tuning, the
  final model, predictions and the pipeline
- `pipeline.py` - sklearn Pipeline that makes the features from raw credit records and trains LightGBM, trained
  on all 3 million clients and saved to `model/credit_risk_pipeline.pkl`
- `predictions/test_predictions.csv` - probability of default and 0/1 for the 600,000 clients of the test set
- `requirements.txt` - libraries

GitHub is slow to show notebooks with plots, the same notebook in nbviewer:
[credit_risk_model.ipynb](https://nbviewer.org/github/Alikonchik-Kaedahara/credit-risk-model/blob/main/credit_risk_model.ipynb)

## Data

The data is from the course materials (based on an Open Data Science competition) and is not in the repo,
it is about 420 MB. To run the notebook or `pipeline.py` put it into `data/`:

```
data/train_data/train_data_0.pq ... train_data_11.pq
data/train_target.csv
```

## Features

Every column is cut into bins with random numbers, so the numbers have no order and I treat them as categories.
For every client:
- counts of credits in each bin of each column (419 features);
- number of credits, number and share of credits with any overdue, totals of each payment status;
- status, type, age, utilization and the last payment of the newest credit.

The files are read one by one with int8 types, all features for 3 million clients are made in about a minute.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS LightGBM also needs `brew install libomp`.

The notebook takes about 40 minutes with all experiments. `python pipeline.py` trains the pipeline on all data
(about 8 minutes and 8 GB of memory) and saves it.

Using the trained pipeline, it takes raw credit records (one row per credit):

```python
import dill
import pandas as pd

with open('model/credit_risk_pipeline.pkl', 'rb') as file:
    pipeline = dill.load(file)

records = pd.read_parquet('data/train_data/train_data_11.pq')
proba = pipeline.predict_proba(records)[:, 1]  # one value per client, clients sorted by id
pred = pipeline.predict(records)               # 0/1, threshold 0.1 (defaults are rare, 0.5 gives zeros)
```
