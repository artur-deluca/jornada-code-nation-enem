import numpy as np


class Markov:

    def __init__(self, order=3, streak=1):
        self.states = {}
        self.order = order
        self.streak = streak

    def train(self, elements, multilple=True):
        if multilple:
            for i in range(len(elements)):
                # create the keys based on the order of the Markov Chain
                key = tuple(elements[i:self.order+i])
                if key not in self.states.keys():
                    self.states[key] = []
                try:
                    to_add = elements[self.order+i:self.order+i+self.streak]
                    if to_add != '':
                        self.states[key].append(to_add)
                except IndexError:
                    pass
        else:
            key = tuple(elements[:self.order])
            if key not in self.states.keys():
                self.states[key] = []
            try:
                to_add = elements[self.order:self.order+self.streak]
                if to_add != '':
                    self.states[key].append(to_add)
            except IndexError:
                pass

    def predict(self, elements):
        try:
            return np.random.choice(self.states[tuple(elements[-self.order:])])
        except ValueError:
            raise KeyError
