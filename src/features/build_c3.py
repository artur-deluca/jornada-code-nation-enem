import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.features.build_features import estimate_math, create_groups, build_quantiles

if __name__ == '__main__':
    data = os.path.join(Path(__file__).resolve().parents[2], "data")
    # input data
    train = pd.read_csv(os.path.join(data, 'raw/train.csv'), index_col=0).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(data, 'raw/test3.csv')).set_index('NU_INSCRICAO')

    # quick data clean-up
    train.loc[:, 'TX_RESPOSTAS_MT'] = train.loc[:, 'TX_RESPOSTAS_MT'].str.replace('\.', '*')
    train = train.loc[train.TX_RESPOSTAS_MT.dropna(axis=0).index]

    # predict the grades on the test set using the Quantile Transformation
    grade_prediction = estimate_math(train.drop('TX_RESPOSTAS_MT', axis=1), test.drop('TX_RESPOSTAS_MT', axis=1))
    test.loc[list(grade_prediction.index), 'NU_NOTA_MT'] = grade_prediction.loc[:, 'NU_NOTA_MT']

    # remove the 0 scores from the training set
    train = train.loc[train.NU_NOTA_MT != 0, :]

    # reposition the training set
    train = train.copy()[list(test.columns)]
    answer = train.pop('TX_RESPOSTAS_MT')
    train['TX_RESPOSTAS_MT'] = answer

    # separte the datasets in quantiles based on the math grade
    train['MT_QT'], test['MT_QT'] = build_quantiles(train, test)

    # separte the datasets in groups using k_means and the math answers
    train['group'], test['group'] = create_groups(train, test)

    train['PREDICTION'], test['PREDICTION'] = '', ''

    train, validation = train_test_split(train, train_size=0.8, random_state=42)

    train.to_csv(os.path.join(data, 'processed/train3.csv'))
    validation.to_csv(os.path.join(data, 'processed/validation3.csv'))
    test.to_csv(os.path.join(data, 'processed/test3.csv'))
