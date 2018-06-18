import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import  shuffle
from process import get_data

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

X, y = get_data()
X, y = shuffle(X, y)
y = y.astype(np.int32)
D = X.shape[1]
K = len(set(y))

X_train = X[:-100]
y_train = y[:-100]
y_train_ind = y2indicator(y_train, K)
X_test = X[-100:]
y_test = y[-100:]
y_test_ind = y2indicator(y_test, K)

W = np.random.randn(D, K)
b = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis = 1, keepdims = True)

def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

train_cost = []
test_cost = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(X_train, W, b)
    pYtest = forward(X_test, W, b)

    c_train = cross_entropy(y_train_ind, pYtrain)
    c_test = cross_entropy(y_test_ind, pYtest)

    train_cost.append(c_train)
    test_cost.append(c_test)

    W -= learning_rate*X_train.T.dot(pYtrain - y_train_ind)
    b -= learning_rate*(pYtrain - y_train_ind).sum(axis = 0)

    if i % 1000 == 0:
        print(i, c_train, c_test)
    
print("Final train classification_rate : ", classification_rate(y_train, predict(pYtrain)))
print("Final test classification_rate : ", classification_rate(y_test, predict(pYtest)))

plt.plot(train_cost, label='train cost')
plt.plot(test_cost, label='test cost')
plt.show()