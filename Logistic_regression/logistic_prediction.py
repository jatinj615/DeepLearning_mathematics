import numpy as np
from process import get_binary_data

X, y = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W)+b)

P_y_given_X = forward(X, W, b)
predictions = np.round(P_y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(y, predictions))