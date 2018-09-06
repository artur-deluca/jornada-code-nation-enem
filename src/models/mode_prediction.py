import pandas as pd

def predict_answers(train, test, predict_n=5):

    train = train.copy()
    test = test.copy()
    filter_df = lambda df, code, quartile: df.loc[(df['CO_PROVA_MT'] == code) & (df['MT_QT'] == quartile)].index
    # iterate through each type of math test in the training set
    for code in train.CO_PROVA_MT.unique():

        # iterate through each quartile in this math test
        for quartile in train.loc[train.CO_PROVA_MT == code, 'MT_QT'].unique():
            enem_answer = ''
            for _ in range(predict_n):
                # accumulate the last n-enem_answers for each row
                enem_answer += train.loc[filter_df(train, code, quartile), 'TX_RESPOSTAS_MT'].str[-predict_n+len(enem_answer)].mode()[0]
            test.loc[filter_df(test, code, quartile), 'PREDICTION'] = enem_answer
    return test.PREDICTION
