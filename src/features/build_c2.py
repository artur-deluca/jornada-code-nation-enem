import os
import pandas as pd
from pathlib import Path
from src.features.build_features import clean_dataset, transform_dataset, one_hot_dataset

if __name__ == '__main__':
    data = os.path.join(Path(__file__).resolve().parents[2], "data")
    # input data
    train = pd.read_csv(os.path.join(data, 'raw/train.csv'), index_col=0).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(data, 'raw/test2.csv')).set_index('NU_INSCRICAO')
    columns = test.columns

    train_set = one_hot_dataset(transform_dataset(clean_dataset(train, columns)))
    test_set = one_hot_dataset(transform_dataset(clean_dataset(test, columns)))

    missing_fields = list(set(train_set.columns) - set(test_set.columns)) + list(set(test_set.columns) - set(train_set.columns))
    train_set.drop(missing_fields, axis=1, errors='ignore', inplace=True)
    test_set.drop(missing_fields, axis=1, errors='ignore', inplace=True)

    train_set.to_csv(os.path.join(data, 'interim/train2.csv'))
    test_set.to_csv(os.path.join(data, 'interim/test2.csv'))
