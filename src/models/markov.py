import numpy as np
import pandas as pd
import pickle as pkl
import os
from pathlib import Path
import warnings
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

    def __init__(self, order=3, streak=1, target='TX_RESPOSTAS_MT', id=['CO_PROVA_MT', 'group'], lazy=False):
        self.id = id
        self.model = {}
        self.order = order
        self.streak = streak
        self.target = target
        self.__path = Path(__file__).resolve().parents[2]
        if lazy:
            try:
                self.load_model()
            except FileNotFoundError:
                warnings.warn('File not found. Building model from scratch')

    def __store(self, elements, forward, i=0):
        # creates the keys based on the order of the Markov Chain
        key = tuple(elements[-self.order-self.streak+i:-self.streak+i])
        try:
            # select elements to predict based on the order, the position of the iterator and the streak
            to_add = elements[-self.streak+i:len(elements)+forward]
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
        if self.sequential:
            # iterate through elements
            for i in range(self.streak-1):
                forward = -len(elements) -self.streak +i +1
                self.__store(elements, forward, i)
            i = self.streak-1
            forward = 0
            self.__store(elements, forward, i)
        else:
            forward = 0
            self.__store(elements, forward)

    def train_chain(self, df, sequential=False, save=True):
        '''
        Train the markov model

        Parameters
        ----------
        elements: str
            String of elements to feed into the Markov chain
        sequential: bool, default False
            If True, it iterate through the elements progressevely, predicting an element one by one afterwards on 'predict'
        save: bool, default True
            Store trained model in a pickle
        '''
        self.sequential = sequential
        for index, element in df[self.target].iteritems():
            self.__key = tuple(df.loc[index, self.id])
            if self.__key not in self.model.keys():
                self.model[self.__key] = {}
            self.__train(element)
        # save model in a pickle
        pkl.dump(self, open(os.path.join(self.__path, 'models\\trained_models\\markov.pkl'), 'wb'))

    def predict(self, elements, key):
        '''
        Predicts the next element(s) of the sequence based on the input

        Parameters
        ----------
        elements: str
            String of elements to feed into the Markov chain
        key: tuple
            Elements used to reference the segment of the model used
        '''

        input_elements = elements[-self.order:]
        flat_list = [item for sublist in list(self.model[key].values()) for item in sublist]
        if self.sequential:
            answer = ""
            for i in range(self.streak):
                try:
                    flat_list = [item for sublist in list(self.model[key].values()) for item in sublist]
                    answer += np.random.choice(self.model[key][tuple(input_elements)])
                except KeyError:
                    answer += np.random.choice(list(map(lambda x: x[0], flat_list)))
                input_elements = input_elements[-self.order+1:] + answer[-1]
        else:
            try:
                answer = np.random.choice(self.model[key][tuple(input_elements)])
            except KeyError:
                answer = ""
                for i in range(self.streak):
                    answer += np.random.choice(list(map(lambda x: x[i], flat_list)))
        # selects on the values for the corresponding key, the ones more present in the list are more likely to be selected
        return answer

    def load_model(self):
        '''
        Loads model from a pickle file
        '''
        with open(os.path.join(self.__path, 'models/trained_models/markov.pkl'), "rb") as f:
            model = pkl.load(f)
        self.__dict__ = model.__dict__.copy()


if __name__ == '__main__':
    print('Markov model')
    # path to general folder of the project
    path = Path(__file__).resolve().parents[2]
    # input data
    train = pd.read_csv(os.path.join(path, 'data/processed/train3.csv')).set_index('NU_INSCRICAO')
    test = pd.read_csv(os.path.join(path, 'data/processed/test3.csv')).set_index('NU_INSCRICAO')
    validation = pd.read_csv(os.path.join(path, 'data/processed/validation3.csv')).set_index('NU_INSCRICAO')

    # set the order of the markov chain
    order = 3
    # set the number of predictions for each row of the datasets
    streak = 5

    target = 'TX_RESPOSTAS_MT'
    id = ['CO_PROVA_MT', 'group']

    # initialize markov model
    model = Markov(order, streak, target, id, lazy=True)
    # trains and saves the models if the models are not stored
    if model.model == {}:
        model.train_chain(train)

    predict = {
        'train': lambda df, id, target: model.predict(df[target][-(order+streak):-streak], tuple(df.loc[id].values)),
        'test': lambda df, id, target: model.predict(df[target][-order:], tuple(df.loc[id].values))
    }

    # saves a separate set to submit
    try:
        test['PREDICTION'] = test.apply(predict['test'], id=id, target=target, axis=1)
    except KeyError:
        warnings.warn('Markov model incompatible with the dataset. Training model to encompass data')
        model = Markov(order, streak, target, id)
        model.train_chain(train)
        test['PREDICTION'] = test.apply(predict['test'], id=id, target=target, axis=1)
    answer = test.copy().loc[:, ['PREDICTION']]
    answer = answer.rename(index=str, columns={"PREDICTION": "TX_RESPOSTAS_MT"})
    export = os.path.join(path, 'models/prediction/markov/test3.csv')
    answer.to_csv(export)

    # accuracy measures
    print('Naive approach accuracy: {:.2f}%'.format(naive_approach(train)*100))
    print('Traning set accuracy: {:.2f}%'.format(score(train.TX_RESPOSTAS_MT.str[-streak:], train.apply(predict['train'], id=id, target=target, axis=1))*100))
    print('Validation set accuracy: {:.2f}%'.format(score(validation.TX_RESPOSTAS_MT.str[-streak:], validation.apply(predict['train'], id=id, target=target, axis=1))*100))
    print('Test file results stored at: {}\n'.format(export))
