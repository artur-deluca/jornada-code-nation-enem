import pandas as pd
import numpy as np


def score(y, y_pred, mean=True):
    scores = []

    def similar(a, b):
        score = 0
        for i in range(len((a))):
            score += 1 if a[i] == b[i] else 0
        return score/max(len(a), len(b))
    for i in range(len(y)):
        scores.append(similar(y.iloc[i], y_pred.iloc[i]))
    if mean:
        return np.mean(scores)
    else:
        return np.array(scores)


def naive_approach(dataset, predict_n=5):
    df = dataset.copy()
    df['PREDICTION'] = ""
    for i in df.index:
        df.loc[i, 'PREDICTION'] = df.loc[i, 'PREDICTION'].join(''.join(str(x) for x in np.random.choice(['A', 'B', 'C', 'D', 'E'], size=predict_n)))
    return score(df.TX_RESPOSTAS_MT.str[-predict_n:], df.PREDICTION)
