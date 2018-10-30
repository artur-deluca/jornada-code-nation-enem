import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, make_scorer as scorer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from src.visualization.visualize import plot_dist


class TransformedLinearRegression(LinearRegression):
    '''
    Linear Regression model. Methods and attributes inherited from sklearn's LinearRegression class.
    Applies a Quantile transformation to the dataset to increase the math grade prediction accuracy

    Parameters
    ----------
    n_quantiles: int, default 1500
        Number of quantiles to divide the math grades' spectrum
    '''
    def __init__(self, n_quantiles=1500):
        super().__init__(self, normalize=True)
        self.n_quantiles = n_quantiles

    def fit(self, train_X, train_Y):
        '''
        Transform data and create a model to predict its behavior.

        Parameters
        ----------
        train_X: pandas DataFrame
            data used to train the model
        train_Y: pandas DataFrame
            data used to adjust the predictions of the model
        '''
        self.qt = {
            'X': QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution='normal'),
            'Y': QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution='normal')
        }
        self.train_X = pd.DataFrame(self.qt['X'].fit_transform(train_X.values))
        self.train_Y = self.qt['Y'].fit_transform(train_Y.to_frame())
        self.shape = self.train_Y.shape
        self.__fit_set(self.train_X, self.train_Y.reshape((1, self.shape[0]))[0])

    # original fit inherited from LinearRegression class
    def __fit_set(self, train_X, train_Y):
        return super().fit(train_X, train_Y)

    # original predict inherited from LinearRegression class
    def __predict_set(self, test_X):
        return super().predict(test_X)

    def predict(self, test_X):
        '''
        Transform data and create a model to predict its behavior.

        Parameters
        ----------
        test_X: pandas DataFrame
            Data used to predict the results. It must have the same format as the train dataset

        returns:
            numpy array with the corresponding prediction
        '''
        test_X = pd.DataFrame(self.qt['X'].transform(test_X.copy().values))
        shape = test_X.shape
        prediction = self.__predict_set(test_X).reshape(shape[0], 1)
        return self.qt['Y'].inverse_transform(prediction).reshape(1, shape[0])[0]

    def plot_dist(self, train_Y):
        '''
        Plot data distribution with the quantile transformation
        Parameters
        ----------
        train_Y: pandas DataFrame
            Collection of values to be transformed an analyzed
        '''
        plot_dist(quantile_transform(train_Y.to_frame(), n_quantiles=self.n_quantiles, output_distribution='normal')[:, 0])

    def save(self):
        '''
        Store model as a pickle in 'models/trained_models'
        '''
        path = os.path.join(Path(__file__).resolve().parents[2], 'models/trained_models')
        print(path)
        pkl.dump(self, open(path+"/regression.pkl", 'wb'))

if __name__ == '__main__':

    # Logger set-up
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create log file to store messages
    handler = logging.FileHandler('models/trained_models/regression.log')
    handler.setLevel(logging.INFO)

    # format logger
    formatter = logging.Formatter("%(asctime)s - [%(name)s] - [%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # display log messages in console
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    path = Path(__file__).resolve().parents[2]
    # input data
    logging.info('Ingesting files')
    train = pd.read_csv(os.path.join(path, 'data/interim/train2.csv')).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(path, 'data/interim/test2.csv')).set_index('NU_INSCRICAO')
    train_X, validation_X, train_Y, validation_Y = train_test_split(train.iloc[:, :-1], train.iloc[:, -1], train_size=0.8, random_state=42)
    test_X = test.iloc[:, :-1]

    # tries to open the pickle file where the trained model is saved
    try:
        with open(os.path.join(path, 'models/trained_models/regression.pkl'), "rb") as f:
            model = pkl.load(f)
        logging.info('Loading model')

    # if file is not found it creates and saves a model from scratch
    except FileNotFoundError:
        model = TransformedLinearRegression(1500)
        model.fit(train_X, train_Y)
        model.predict(train_X)
        model.save()
        logging.info('Creating and saving model')

    # Mean absolute percentage error (MAPE)
    mape = lambda y_true, y_pred: abs((y_true-y_pred)/y_true).mean()

    logging.info('Evaluating model')

    predicted_train_Y = model.predict(train_X)
    predicted_validation_Y = model.predict(validation_X)

    logging.info('\n\
        +---------+-----------+----------------+\n\
        | Metrics | Train set | Validation set |\n\
        +---------+-----------+----------------+\n\
        | MAPE    | {:.4f}    | {:.4f}         |\n\
        | R²      | {:.4f}    | {:.4f}         |\n\
        +---------+-----------+----------------+\n\
    '.format(
        mape(y_true=train_Y, y_pred=predicted_train_Y),
        mape(y_true=validation_Y, y_pred=predicted_validation_Y),
        r2_score(y_true=train_Y, y_pred=predicted_train_Y),
        r2_score(y_true=validation_Y, y_pred=predicted_validation_Y)
    ))

    logging.info('Generating learning curves')

    train_sizes, train_scores, valid_scores = learning_curve(TransformedLinearRegression(1500), train.iloc[:,:-1], train.iloc[:,-1], scoring=scorer(mape), cv=5)

    # plot configuration
    fig = plt.gcf()
    fig.canvas.set_window_title('Learning curves of Linear Regression model with Quantile Transformation')
    plt.plot(train_sizes, train_scores.mean(axis=1), color='r', label='Training Score')
    plt.plot(train_sizes, valid_scores.mean(axis=1), color='g', label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()

    # predict values for the test set
    logging.info('Infering test answers to send to CodeNation')
    prediction = model.predict(test_X)
    answer = test.copy().loc[:, []]
    answer['NU_NOTA_MT'] = 0
    answer.loc[test_X.index, 'NU_NOTA_MT'] = prediction
    answer.to_csv(os.path.join(path, 'models/prediction/regression/test2.csv'))
