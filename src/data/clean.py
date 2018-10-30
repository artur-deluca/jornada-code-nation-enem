import pandas as pd

def clean_dataset(dataset, columns):
    '''
    Remove all the unnecessary fields and values in the dataset

    Parameters:
    -----------
    dataset: pandas DataFrame
        Dataset to perform the cleaning
    columns: list
        columns to keep in the DataFrame
    '''

    df = dataset.copy(deep=True)

    # We're interessed in predicting only the ones that completed the test
    df = df.loc[df.TP_PRESENCA_LC == 1.0, :]

    # if the dataset to clean is the training set then remove the unexisting columns in the test set

    key = 'NU_NOTA_MT'
    columns = list(columns)

    try:
        columns.pop(columns.index(key))
    except ValueError:
        pass
    columns.append(key)

    if key not in df.columns:
        df[key] = 0
    else:
        df = df.loc[df.NU_NOTA_MT != 0]

    df = df.loc[:, list(columns)]

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
        singular_columns = df.columns[list(map(lambda x: len(pd.Series(df[x]).unique()) == 1, list(df.columns)[:-1]))+[False]]
        df.drop(singular_columns, axis=1, inplace=True)
    except IndexError:
        pass

    # transform the non-continuous fields in category data-types
    for i in list(df.columns[~df.columns.str.startswith('NU')]):
        df[i].astype('category')

    return df
