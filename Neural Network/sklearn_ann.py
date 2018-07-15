from process import get_data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# get the data
X, Y = get_data()

# split into train and test
X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

#create neural network
model = MLPClassifier(hidden_layer_sizes=(25,25), max_iter=2000)

model.fit(Xtrain, Ytrain)

# print the train and test accuracy
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)

