from data import *
from model import NNFromScratch


X, y = generate_data()
X,y = shuffle_data(X,y)
X_train, Y_train, X_test, Y_test = split_data(X,y)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

model = NNFromScratch(2,10,1)
model.fit(X_train, Y_train, X_test, Y_test)
model.train()

