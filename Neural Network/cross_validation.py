import numpy as np
import random

def cross_validate(model, X, y, k = 5):
    X, y = random.shuffle(X, y)
    sz = len(y)/k
    scores = []
    for K in range(k):
        xtr = np.concatenate([ X[:k*sz, :], X[ (k*sz + sz):, :]])
        ytr = np.concatenate([ y[:k*sz], y[ (k*sz + sz):]])
        xte = X[k*sz: (k*sz + sz), :]
        yte = y[k*sz: (k*sz + sz)]

        model.fit(xtr, ytr)
        score = model.score(xte, yte)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
