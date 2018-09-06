import numpy as np
import pandas as pd
import pickle as pkl
import os
from pathlib import Path
from src.models.score import score, naive_approach


class Markov:
    '''
    Markov Chain in dictionary structure to assimilate and predict the math answers in the code:nation challenge

    Parameters
    ----------
    order: int
        Represents how many steps backwards are necessary to predict the next value.
        For instance, to estimate the last char of the sequence "ABC" a 2nd order Markov chain needs "AB" to predict "C"
    streak: int, default 1
        Represents how many elements of a sequence are going to be predicted.
        For instance, to estimate the last two elements of a sequence, streak is 2.
    '''
    def __init__(self, order=3, streak=1, target='TX_RESPOSTAS_MT', id=['CO_PROVA_MT', 'group']):
        self.id = id
        self.model = {}
        self.order = order
        self.streak = streak
        self.target = target
        self.__path = Path(__file__).resolve().parents[2]

    def __store(self, elements, i=0):
        # creates the keys based on the order of the Markov Chain
        length = len(elements)
        key = tuple(elements[i-length:self.order+i-length])
        try:
            # select elements to predict based on the order, the position of the iterator and the streak
            to_add = elements[self.order+i-length:self.order+i+self.streak-length]
            # in case the variable is empty the addition is ignored
            if to_add != '':
                # check if key already exists and if not it adds to the dict
                if key not in self.model[self.__key].keys():
                    self.model[self.__key][key] = []
                self.model[self.__key][key].append(to_add)
        # when order plus position and streak overflow the index of elements
        except IndexError:
            pass

    def __train(self, elements):
        '''
        Train the markov model

        Parameters
        ----------
        elements: str
            String of elements to feed into the Markov chain
        multiple: bool, default True
            If True, it iterate through the entire sequence to train the Markov chain. If False only trains the chain using the last elements.
        '''
        if self.multiple:
            # iterate through elements
            for i in range(len(elements)):
                self.__store(elements, i)
        else:
            self.__store(elements)

    def train_chain(self, df, multiple=False, save=True):
        self.multiple = multiple
        for index, element in df[self.target].iteritems():
            self.__key = tuple(df.loc[index, self.id])
            if self.__key not in self.model.keys():
                self.model[self.__key] = {}
            self.__train(element)
        # save model in a pickle
        pkl.dump(self, open(os.path.join(self.__path, 'models\\trained_models\\markov.pkl'), 'wb'))

    def predict(self, elements):
        '''
        Predicts the next element(s) of the sequence based on the input

        Parameters
        ----------
        elements: str
            String of elements to feed into the Markov chain
        '''
        input_elements = elements[-self.order:]
        flat_list = [item for sublist in list(self.model[self.__key].values()) for item in sublist]
        if self.multiple:
            answer = ""
            for i in range(self.streak):
                try:
                    answer += np.random.choice(self.model[self.__key][tuple(input_elements)])
                except KeyError:
                    answer += np.random.choice(list(map(lambda x: x[i], flat_list)))
                input_elements = input_elements[-self.order+1:] + answer[-1]
        else:
            try:
                answer = np.random.choice(self.model[self.__key][tuple(input_elements)])
            except KeyError:
                answer = ""
                for i in range(self.streak):
                    answer += np.random.choice(list(map(lambda x: x[i], flat_list)))
        # selects on the values for the corresponding key, the ones more present in the list are more likely to be selected
        return answer

    def load_model(self):
        with open(os.path.join(self.__path, 'models/trained_models/markov.pkl'), "rb") as f:
            model = pkl.load(f)
        self.__dict__ = model.__dict__.copy()


if __name__ == '__main__':
    # path to general folder of the project
    path = Path(__file__).resolve().parents[2]

    # set the order of the markov chain
    order = 3
    # set the number of predictions for each row of the datasets
    n_predictions = 5

    shift = n_predictions + order

    # input data
    train = pd.read_csv(os.path.join(path, 'data/interim/train3.csv')).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(path, 'data/interim/test3.csv')).set_index('NU_INSCRICAO')
    validation = pd.read_csv(os.path.join(path, 'data/interim/validation3.csv')).set_index('NU_INSCRICAO')

    # open stored markov models
    try:
        model = load_model()
    # trains and saves the models if the models are not stored
    except FileNotFoundError:
        __train_whole_chain(train)
        # open stored markov models
        model = load_model()

    test_codes = train.CO_PROVA_MT.unique()
    groups = train.group.unique()

    filter_df = lambda df, code, group: df.loc[(df['CO_PROVA_MT'] == code) & (df['group'] == group)].index

    # iterate through all the math test codes
    for code in test_codes:
        # iterate through the classified groups
        for group in groups:
            # merge all subsets to be predicted as one
            train_validation_test_set = pd.concat([
                train.loc[filter_df(train, code, group), 'TX_RESPOSTAS_MT'].str[-shift:-n_predictions],
                validation.loc[filter_df(validation, code, group), 'TX_RESPOSTAS_MT'].str[-order:],
                test.loc[filter_df(test, code, group), 'TX_RESPOSTAS_MT'].str[-order:]
            ])
            # generates key to access the desired model in markov dictionary
            key = tuple([code, group])
            for index, element in train_validation_test_set.iteritems():
                # build answer from empty string
                enem_answer = ''
                try:
                    enem_answer += model[key].predict(element)
                except KeyError:
                    # In case it tries to make an unseen prediction, the result will be the mode of each element
                    for _ in range(n_predictions):
                        enem_answer += train_set.loc[:, 'TX_RESPOSTAS_MT'].str[-n_predictions+len(enem_answer)].mode()[0]
                # stores the answer
                train_validation_test_set.loc[index] = enem_answer
            # store the answers to each corresponding dataset
            train.loc[filter_df(train, code, group), 'PREDICTION'] = train_validation_test_set.loc[filter_df(train, code, group)]
            validation.loc[filter_df(validation, code, group), 'PREDICTION'] = train_validation_test_set.loc[filter_df(validation, code, group)]
            test.loc[filter_df(test, code, group), 'PREDICTION'] = train_validation_test_set.loc[filter_df(test, code, group)]

    # saves a separate set to submit
    answer = test.copy().loc[:, ['PREDICTION']]
    answer = answer.rename(index=str, columns={"PREDICTION": "TX_RESPOSTAS_MT"})
    answer.to_csv(os.path.join(path, 'models/prediction/markov/test3.csv'))

    # accuracy measures
    print('Naive approach accuracy: {:.2f}%'.format(naive_approach(train)*100))
    print('Traning set accuracy: {:.2f}%'.format(score(train.TX_RESPOSTAS_MT.str[-n_predictions:], train.PREDICTION)*100))
    print('Validation set accuracy: {:.2f}%'.format(score(validation.TX_RESPOSTAS_MT.str[-n_predictions:], validation.PREDICTION)*100))
