import numpy as np
import matplotlib.pyplot as plt
from utils import *

class NNFromScratch:
    def __init__(self,h0 , h1 , h2 , alpha = 0.1, num_epochs = 1000):
        self.W1 = np.random.rand(h1,h0)
        self.W2 = np.random.rand(h2,h1)
        self.b1 = np.random.rand(h1,1)
        self.b2 = np.random.rand(1,1)
        self.alpha = alpha
        self.num_epochs = num_epochs

    def fit(self, X, Y,X_test, Y_test):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test

    def forward_pass(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = sigmoid(Z2)
        return A2, Z2, A1, Z1
    
    def backward_pass(self, A2, Z2, A1, Z1):
        m = self.X.shape[1]
        dZ2 = A2 - self.Y
        dW2 = (1/m) * (dZ2 @ A1.T)
        db2 = (1/m) * (np.sum(dZ2, axis =1,keepdims=True))
        dA1 = self.W2.T @ dZ2
        dZ1 = d_sigmoid(Z1) * dA1
        dW1 = (1/m) * (dZ1 @ self.X.T)
        db1 = (1/m) * (np.sum(dZ1, axis =1,keepdims=True))

        return dW1, dW2, db1, db2
    
    def predict(self, X):
        A2, Z2, A1, Z1 = self.forward_pass(X)
        return (A2>=0.5).astype(int)

    def update(self ,dW1, dW2, db1, db2):
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        return self.W1, self.W2, self.b1, self.b2
    
    def train(self):
        train_loss = []
        test_loss = []
        for i in range(self.num_epochs):
            ## forward pass
            A2, Z2, A1, Z1 = self.forward_pass(self.X)
            ## backward pass
            dW1, dW2, db1, db2 = self.backward_pass(A2, Z2, A1, Z1)
            ## update parameters
            self.update(dW1, dW2, db1, db2)
            ## save the train loss
            train_loss.append(loss(A2, self.Y))
             ## compute test loss
            A2, Z2, A1, Z1 = self.forward_pass(self.X_test)
            test_loss.append(loss(A2, self.Y_test))
            ## plot boundary
            if i %1000 == 0:
                self.plot_decision_boundary()

        ## plot train et test losses
        plt.plot(train_loss)
        plt.plot(test_loss)

        y_pred = self.predict(self.X)
        train_accuracy = accuracy(y_pred, self.Y)
        print ("train accuracy :", train_accuracy)

        y_pred = self.predict(self.X_test)
        test_accuracy = accuracy(y_pred, self.Y_test)
        print ("test accuracy :", test_accuracy)

    
    def plot_decision_boundary(self):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        A2, Z2, A1, Z1 = self.forward_pass(X_)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= A2)
        plt.show()