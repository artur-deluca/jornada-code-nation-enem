import numpy as np
def score(y, y_pred, mean=True):
    scores = []
    def similar(a, b):
        score = 0
        for i in range(len((a))):
            score += 1 if a[i]==b[i] else 0
        return score/max(len(a),len(b))
    for i in range(len(y)):
        scores.append(similar(y.iloc[i], y_pred.iloc[i]))
    if mean:
    	return np.mean(scores)
    else:
    	return np.array(scores)