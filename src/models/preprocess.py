import pandas as pd
import numpy as np


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


def clean_dataset(dataset, columns):

    df = dataset.copy(deep=True)

    # We're interessed in predicting only the ones that completed the test
    df = df[df.TP_PRESENCA_LC == 1.0]

    # if the dataset to clean is the training set then remove the unexisting columns in the test set

    key = 'NU_NOTA_MT'
    if key not in df.columns:
        df[key] = 0
    else:
        df = df.loc[df.NU_NOTA_MT != 0]
    df = df.loc[:, list(columns)+[key]]

    # the fields below were judged redundant or badly distributed
    to_remove = ['CO_UF_RESIDENCIA',
                 'TP_ANO_CONCLUIU',
                 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
                 'TP_DEPENDENCIA_ADM_ESC',
                 'TP_ENSINO',
                 'Q027']

    # remove TP_ESCOLA - very similar to Q047
    to_remove.append('TP_ESCOLA')
    # to_remove.append('Q047')

    # remove the considated essay grade, since its componenents are also in the dataset
    to_remove.extend(['NU_NOTA_REDACAO'])
    # to_remove.extend(['NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5'])

    # remove the special indicators due to very uneven distribution
    to_remove.extend(list(df.columns[df.columns.str.startswith('IN')]))

    # drop all 'to_remove' columns
    df.drop(to_remove, axis=1, inplace=True)

    # drop unexpected columns with single values
    try:
        singular_columns = df.columns[list(map(lambda x: len(df[x].unique()) == 1, list(df.columns)[:-1]))]
        df.drop(singular_columns, axis=1, inplace=True)
    except IndexError:
        pass

    # transform the non-continuous fields in category data-types
    for i in list(df.columns[~df.columns.str.startswith('NU')]):
        df[i].astype('category')

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
