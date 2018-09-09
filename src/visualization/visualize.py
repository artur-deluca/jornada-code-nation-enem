import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns

from pathlib import Path
from src.models.markov import Markov
from src.models.score import score, naive_approach


def plot_dist(target):
    '''Plot distribution of data

    Parameters:
    -----------
        target pandas Series
    '''

    from scipy import stats

    # histogram and normal probability plot
    try:
        to_analyse = target.reset_index(drop=True)
    except AttributeError:
        to_analyse = target

    sns.distplot(to_analyse, fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(to_analyse, plot=plt)


def sweep_orders(lazy=True):
    '''Sweep the orders of a markov chain and analyze its performance with a trained set, a validation and a naive approach.

    Parameters:
    -----------
    lazy: Bool, default True
        If true, loads previously calculated sweep, otherwise calculates a new sweep
    '''

    path = Path(__file__).resolve().parents[2]

    # input data
    train = pd.read_csv(os.path.join(path, 'data/processed/train3.csv')).set_index('NU_INSCRICAO')
    validation = pd.read_csv(os.path.join(path, 'data/processed/validation3.csv')).set_index('NU_INSCRICAO')

    if lazy:
        with open(os.path.join(path, 'reports/data/order_sweep.json'), 'rb') as f:
            scores = json.load(f)
    else:
        scores = {'train': [], 'validation': [], 'naive_approach': []}
        for order in range(1, 11):
            streak = 5
            model = Markov(order, streak)

            predict_df = lambda df: model.predict(df['TX_RESPOSTAS_MT'][-(order+streak):-streak], tuple([df['CO_PROVA_MT'], df['group']]))
            model.train_chain(train)

            scores['naive_approach'].append(naive_approach(validation)*100)
            scores['train'].append(score(train.TX_RESPOSTAS_MT.str[-streak:], train.apply(predict_df, axis=1))*100)
            scores['validation'].append(score(validation.TX_RESPOSTAS_MT.str[-streak:], validation.apply(predict_df, axis=1))*100)
        with open('../../reports/data/order_sweep.json', 'w') as f:
            json.dump(scores, f)

    x = list(range(1, 11))


    plt.plot(x, scores['train'])
    plt.plot(x, scores['validation'])
    plt.plot(x, scores['naive_approach'])


    plt.legend(['Train', 'Valdation', 'Naive approach'], loc='upper left')
    plt.xlabel('Order')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of prediction with different orders in a Markov Chain ')
    plt.show()
