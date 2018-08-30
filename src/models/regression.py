import pandas as pd
from src.data.preprocess import clean_dataset, transform_dataset, one_hot_dataset


def predict(train, test):

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import QuantileTransformer

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
    test_Y = test_set.iloc[:, -1]

    n_quantiles = 1500

    qt = {
        'X': QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal'),
        'Y': QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
    }

    qt_train_X = pd.DataFrame(qt['X'].fit_transform(train_X.values))
    qt_Y = qt['Y'].fit_transform(train_Y.to_frame())
    qt_shape = qt_Y.shape

    model_qt = LinearRegression(normalize=True)
    model_qt.fit(qt_train_X, qt_Y.reshape((1,qt_Y.shape[0]))[0])
    Y_pred = qt['Y'].inverse_transform(model_qt.predict(qt_train_X).reshape(qt_shape)).reshape(1,qt_shape[0])[0]

    # test-set

    qt_test_X = pd.DataFrame(qt['X'].transform(test_X.values))
    prediction = qt['Y'].inverse_transform(model_qt.predict(qt_test_X).reshape((test_X.shape[0],1))).reshape(1, test_X.shape[0])[0]

    # send answers
    answer_qt = answer.copy()
    answer_qt.loc[test_X.index, 'NU_NOTA_MT'] = prediction

    return answer_qt
