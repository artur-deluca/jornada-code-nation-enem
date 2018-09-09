
import pandas as pd
import pickle as pkl
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
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
        LinearRegression.__init__(self, normalize=True)
        self.n_quantiles = n_quantiles

    def fit_set(self, train_X, train_Y):
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
        self.fit(self.train_X, self.train_Y.reshape((1, self.shape[0]))[0])

    def predict_set(self, test_X):
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
        prediction = self.predict(test_X).reshape(shape[0], 1)
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
        Store model as a pickle
        '''
        path = os.path.join(Path(__file__).resolve().parents[2], 'models\\trained_models\\regression.pkl')
        pkl.dump(self, open(path, 'wb'))


if __name__ == '__main__':
    print('Linear Regression model')
    path = Path(__file__).resolve().parents[2]
    # input data
    train = pd.read_csv(os.path.join(path, 'data/processed/train2.csv')).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(path, 'data/processed/test2.csv')).set_index('NU_INSCRICAO')

    train_X = train.iloc[:, :-1]
    train_Y = train.iloc[:, -1]
    test_X = test.iloc[:, :-1]

    # predict values for the test set
    # tries to open the pickle file where the trained model is saved
    try:
        with open(os.path.join(path, 'models/trained_models/regression.pkl'), "rb") as f:
            model = pkl.load(f)

    # if file is not found it creates and saves a model from scratch
    except FileNotFoundError:
        model = TransformedLinearRegression(1500)
        model.fit_set(train_X, train_Y)
        model.save()

    # send answers
    prediction = model.predict_set(test_X)
    answer = test.copy().loc[:, []]
    answer['NU_NOTA_MT'] = 0
    answer.loc[test_X.index, 'NU_NOTA_MT'] = prediction
    export = os.path.join(path, 'models/prediction/regression/test2.csv')
    answer.to_csv(export)

    # Mean absolute percentage error (MAPE)
    mape = lambda y_true, y_pred: (abs(y_true-y_pred)/y_true).mean()
    print('MAPE: %.4f'%mape(y_true=train_Y, y_pred=model.predict_set(train_X)))
    print('Test file results stored at: {}\n'.format(export))
