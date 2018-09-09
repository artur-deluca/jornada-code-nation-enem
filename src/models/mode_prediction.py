import pandas as pd

def predict_answers(train, test, streak=5):
    '''
    Predict answers for the dataset using a mode strategy for each answer

    Parameters
    ----------
        train: pandas DataFrame
            Dataset containing the math answers in the 'TX_RESPOSTAS_MT' field
        test: pandas DataFrame
            Dataset to estimate the filled answers
        streak: int, default 5
            Number of answers to predict
    '''
    train = train.copy()
    test = test.copy()
    filter_df = lambda df, code, quartile: df.loc[(df['CO_PROVA_MT'] == code) & (df['MT_QT'] == quartile)].index
    # iterate through each type of math test in the training set
    for code in train.CO_PROVA_MT.unique():

        # iterate through each quartile in this math test
        for quartile in train.loc[train.CO_PROVA_MT == code, 'MT_QT'].unique():
            enem_answer = ''
            for _ in range(streak):
                # accumulate the last n-enem_answers for each row
                enem_answer += train.loc[filter_df(train, code, quartile), 'TX_RESPOSTAS_MT'].str[-streak+len(enem_answer)].mode()[0]
            test.loc[filter_df(test, code, quartile), 'PREDICTION'] = enem_answer
    return test.PREDICTION
