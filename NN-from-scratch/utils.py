import numpy as np

def sigmoid(z):
  sgmoid = 1 / ( 1 + np.exp(-z) )
  return sgmoid

def d_sigmoid(z):

  d = sigmoid(z)*(1 - sigmoid(z))

  return d

def loss(y_pred, Y):

  l = -np.mean( (Y * np.log(y_pred)) + ((1- Y) * np.log(1-y_pred)) )

  return  l

def accuracy(y_pred, y):

  m = y.shape[1]
  correct_predictions = np.sum( y == y_pred )
  return (correct_predictions *100)/ m

