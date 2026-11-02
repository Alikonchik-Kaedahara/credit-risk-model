import datetime
import time

import dill
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = 'data/train_data'
TARGET_PATH = 'data/train_target.csv'
MODEL_PATH = 'model/credit_risk_pipeline.pkl'
N_FILES = 12

# flag = 0 means the credit had an overdue of this length
OVERDUE_FLAGS = ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']
PAYM_COLS = [f'enc_paym_{i}' for i in range(25)]
LAST_CREDIT_COLS = ['enc_loans_credit_status', 'enc_loans_credit_type', 'pre_since_opened', 'pre_util', 'enc_paym_0']

# parameters chosen in credit_risk_model.ipynb
PARAMS = {'learning_rate': 0.05, 'n_estimators': 754, 'num_leaves': 127, 'min_child_samples': 300,
          'colsample_bytree': 0.3, 'subsample': 0.8, 'subsample_freq': 1, 'reg_lambda': 5.0}


def read_records(path):
    # all values are small bin numbers, int8 takes 8 times less memory than int64
    records = pd.read_parquet(path)
    feature_cols = [col for col in records.columns if col not in ('id', 'rn')]
    records[feature_cols] = records[feature_cols].astype('int8')
    records['id'] = records['id'].astype('int32')
    records['rn'] = records['rn'].astype('int16')
    return records


def make_features(records):
    # one row per client from many rows (credits) per client
    feature_cols = [col for col in records.columns if col not in ('id', 'rn')]
    ids = records['id']

    # bin numbers are random, so they are categories: count credits in each bin of each column
    dummies = pd.get_dummies(records[feature_cols], columns=feature_cols, dtype='uint8')
    dummies['id'] = ids.values
    features = dummies.groupby('id').sum()

    features['n_credits'] = ids.value_counts().sort_index().astype('int16')
    has_overdue = (records[OVERDUE_FLAGS] == 0).any(axis=1)
    features['n_overdue_credits'] = has_overdue.groupby(ids).sum().astype('int16')
    features['share_overdue_credits'] = (features['n_overdue_credits'] / features['n_credits']).astype('float32')
    for status in range(5):
        status_count = (records[PAYM_COLS] == status).sum(axis=1)
        features[f'paym_status_{status}_total'] = status_count.groupby(ids).sum().astype('int16')

    # the newest credit (biggest rn) says the most about the current situation
    last = records.sort_values(['id', 'rn']).groupby('id').tail(1).set_index('id')[LAST_CREDIT_COLS]
    last = pd.get_dummies(last, columns=LAST_CREDIT_COLS, prefix=['last_' + col for col in LAST_CREDIT_COLS],
                          dtype='uint8')
    return features.join(last)


class CreditHistoryFeatures(BaseEstimator, TransformerMixin):
    """Turns credit records (one row per credit) into one row of features per client.

    Clients come out sorted by id, so the target for fit has to be sorted by id too.
    """

    def __init__(self, chunk_size=250_000):
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        self.columns_ = self._build(X).columns
        return self

    def fit_transform(self, X, y=None):
        features = self._build(X)
        self.columns_ = features.columns
        return features

    def transform(self, X):
        # bins that were not seen in fit are dropped, missing ones are zeros
        return self._build(X).reindex(columns=self.columns_, fill_value=0)

    def _build(self, X):
        # in chunks of clients, otherwise get_dummies on all 26 million rows does not fit in memory
        ids = np.sort(X['id'].unique())
        parts = []
        for start in range(0, len(ids), self.chunk_size):
            chunk_ids = ids[start:start + self.chunk_size]
            parts.append(make_features(X[X['id'].between(chunk_ids[0], chunk_ids[-1])]))
        features = pd.concat(parts)
        # a bin can be missing in one chunk, after concat it becomes NaN
        missing = features.columns[features.isna().any()]
        features[missing] = features[missing].fillna(0).astype('uint8')
        return features


def make_pipeline():
    return Pipeline([
        ('features', CreditHistoryFeatures()),
        ('model', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **PARAMS)),
    ])


def main():
    print('Credit risk model: probability of default')
    start = time.time()

    records = pd.concat([read_records(f'{DATA_PATH}/train_data_{i}.pq') for i in range(N_FILES)], ignore_index=True)
    target = pd.read_csv(TARGET_PATH).sort_values('id').set_index('id')['flag']
    print('credits:', len(records), 'clients:', len(target), f'({time.time() - start:.0f} s)')

    # quick check on clients that the model does not see (same split as in the notebook)
    train_ids, test_ids = train_test_split(target.index.values, test_size=0.2, random_state=42, stratify=target.values)
    train_ids, test_ids = np.sort(train_ids), np.sort(test_ids)
    pipe = make_pipeline()
    pipe.fit(records[records['id'].isin(train_ids)], target.loc[train_ids].values)
    test_proba = pipe.predict_proba(records[records['id'].isin(test_ids)])[:, 1]
    test_auc = roc_auc_score(target.loc[test_ids].values, test_proba)
    print(f'test roc_auc: {test_auc:.4f} ({time.time() - start:.0f} s)')

    # the final pipeline is trained on all clients
    pipe = make_pipeline()
    pipe.fit(records, target.values)

    # without recurse=True the saved pipeline did not load in a new session
    with open(MODEL_PATH, 'wb') as file:
        dill.dump(pipe, file, recurse=True)
    print('saved to', MODEL_PATH, f'({time.time() - start:.0f} s),',
          datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))


if __name__ == '__main__':
    main()
