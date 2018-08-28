import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_dist(target):

    from scipy import stats

    # histogram and normal probability plot
    try:
        to_analyse = target.reset_index(drop=True)
    except AttributeError:
        to_analyse = target

    sns.distplot(to_analyse, fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(to_analyse, plot=plt)
