from random import choice


class MarkovCN:
    def __init__(self, order=3):
        self.states = {}
        self.order = order

    def train(self, elements):
        for i in range(len(elements)):
            key = tuple(elements[i:self.order+i])
            if key not in self.states.keys():
                self.states[key] = []
            try:
                self.states[key].append(elements[self.order+i])
            except:
                pass

    def predict(self, elements):
        return choice(self.states[tuple(elements[-self.order:])])
