import pandas as pd
from src.data.clean import clean_dataset
from src.models.regression import TransformedLinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def transform_dataset(dataset):

    df = dataset.copy(deep=True)

    # Transform irregularities in a single label(0)
    TP_columns = list(df.columns[df.columns.str.contains('PRESENCA')])
    df.loc[:, TP_columns] = (df.loc[:, TP_columns]
                               .astype(str)
                               .replace(r'^(?!1.0).*$', value=0, regex=True)
                               .astype(float))

    # Fill the unexisting values in the continuous fields (NU) with their average
    NU_columns = list(df.columns[df.columns.str.startswith('NU')])
    df.loc[:, NU_columns] = df.loc[:, NU_columns].fillna(df.loc[:, NU_columns].mean())

    # Transform 'state' feature in 'region'
    regioes = {
        'N': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
        'NE': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
        'CE': ['GO', 'MS', 'MT'],
        'SE': ['SP', 'ES', 'RJ', 'MG'],
        'S': ['PR', 'SC', 'RS']
    }

    for reg in regioes:
        df.loc[df.SG_UF_RESIDENCIA.isin(regioes[reg]), 'SG_UF_RESIDENCIA'] = reg

    # Essay status - reduce to 3: Writing problems
    df.loc[~df['TP_STATUS_REDACAO'].isin([1, 4]), 'TP_STATUS_REDACAO'] = 3

    # Merge fields from Q001 and Q002
    to_replace = {
        'C': 'B',
        'E': 'D'
    }
    df.loc[:, ['Q001', 'Q002']].replace(to_replace, inplace=True)

    # Merge Q006 fields
    to_replace = {
        'C': 'B',
        'H': 'G',
        'I': 'G',
        'E': 'D',
        'F': 'D',
        'K': 'J',
        'L': 'J',
        'M': 'J',
        'O': 'N',
        'P': 'N'
    }
    df['Q006'].replace(to_replace, inplace=True)

    # Merge Q024 fields
    to_replace = {
        'D': 'C',
        'E': 'C'
    }
    df['Q024'].replace(to_replace, inplace=True)

    return df


def one_hot_dataset(dataset):

    # select only categorical columns
    one_hot_columns = list(dataset.columns[~dataset.columns.str.startswith('NU')])

    # apply one hot encoding to the dataset
    df = pd.get_dummies(dataset, columns=one_hot_columns)

    # move the target column to the last place
    target = df.columns.get_loc('NU_NOTA_MT')
    columns = list(df.columns.values)
    df = df.loc[:, columns[:target]+columns[target+1:]+[columns[target]]]

    return df


def estimate_math(train, test):

    test = test.copy()
    train = train.copy()
    columns = test.columns

    train_set = one_hot_dataset(transform_dataset(clean_dataset(train, columns)))
    test_set = one_hot_dataset(transform_dataset(clean_dataset(test, columns)))

    answer = test.copy().loc[:, []]
    answer['NU_NOTA_MT'] = 0

    missing_fields = list(set(train_set.columns) - set(test_set.columns)) + list(set(test_set.columns) - set(train_set.columns))

    train_set.drop(missing_fields, axis=1, errors='ignore', inplace=True)
    test_set.drop(missing_fields, axis=1, errors='ignore', inplace=True)

    train_X = train_set.iloc[:, :-1]
    train_Y = train_set.iloc[:, -1]
    test_X = test_set.iloc[:, :-1]

    # test-set
    model = TransformedLinearRegression(1500)
    model.fit_set(train_X, train_Y)
    prediction = model.predict_set(test_X)

    # send answers
    answer_qt = answer.copy()
    answer_qt.loc[test_X.index, 'NU_NOTA_MT'] = prediction

    return answer_qt

def build_quartiles(train, test, quartiles=4):
    merged_grades = pd.qcut(pd.concat([train.NU_NOTA_MT, test.NU_NOTA_MT]), quartiles, labels=False)
    return merged_grades.loc[train.index].values, merged_grades.loc[test.index].values

def create_groups(train, test):

    train = train.copy()
    test = test.copy()

    codes = list(train['CO_PROVA_MT'].unique())

    prev_answers_train = pd.DataFrame(list(map(lambda x: list(x), train.TX_RESPOSTAS_MT.str[:-5]))).set_index(train.index)
    prev_answers_test = pd.DataFrame(list(map(lambda x: list(x), test.TX_RESPOSTAS_MT))).set_index(test.index)

    prev_answers_train['code'] = train['CO_PROVA_MT']
    prev_answers_test['code'] = test['CO_PROVA_MT']

    prev_answers = pd.concat([prev_answers_train, prev_answers_test])
    prev_answers['group'] = ''

    label_encod = LabelEncoder()
    label_encod.fit(['A', 'B', 'C', 'D', 'E', '*'])
    prev_answers_enc = prev_answers.copy()
    prev_answers_enc.iloc[:, :-2] = prev_answers_enc.iloc[:, :-2].apply(label_encod.transform)

    k_clusters = 6

    for code in codes:
        X = prev_answers_enc.loc[prev_answers.code == code].iloc[:, :-2].values
        kmeanModel = KMeans(n_clusters=k_clusters, init='random')
        kmeanModel.fit(X)
        prev_answers.loc[prev_answers.code == code, 'group'] = kmeanModel.labels_

    return prev_answers.loc[train.index, 'group'], prev_answers.loc[test.index, 'group']
