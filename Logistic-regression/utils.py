import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification


def train_test_split(X,y, percentage = 0.8):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

  n = int(len(X)*percentage)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test



def add_ones(X):
    ones = np.ones((X.shape[0],1))
    return np.hstack((ones, X))

def sigmoid(X, w):
    z = X @ w
    s = (1/(1+ np.exp(-z)) )
    return s

def cross_entropy(X, y_true, w):
    y_pred = sigmoid(X, w)
    loss = - np.mean( (y_true * np.log(y_pred)) + ((1- y_true) * np.log(1-y_pred)) )
    return loss

def gradient(X,y, y_pred):
    return (-1/X.shape[0])*(X.T @ (y - y_pred))


def predict_proba(X, w):  
    return sigmoid(X, w)

def accuracy(y_true, y_pred):
    acc = np.mean(y_true.reshape(-1,1) == y_pred ) * 100
    return acc


def plot_decision_boundary(X, w, b,y_train):

    # z = w1x1 + w2x2 + w0
    # one can think of the decision boundary as the line x2=mx1+c
    # Solving we find m and c
    x1 = [X[:,0].min(), X[:,0].max()]
    m = -w[1]/w[2]
    c = -b/w[2]
    x2 = m*x1 + c

    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.scatter(X[:, 0], X[:, 1],c=y_train)
    plt.scatter(X[:, 0], X[:, 1], c=y_train)
    plt.xlim([-2, 3])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x1, x2, 'y-')