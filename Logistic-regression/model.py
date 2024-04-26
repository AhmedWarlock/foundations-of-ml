from utils import *

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''

  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []


  def predict(self,x):
    probas = predict_proba(add_ones(x), self.w)
    #convert the probalities into 0 and 1 by using a treshold=0.5
    output = (probas >= 0.5).astype(int)
    return output

  def fit(self,x,y):

    # Add ones to x
    x = add_ones(x)
    print(x.shape)

    # reshape y if needed
    y = y.reshape(-1,1)

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w = np.zeros((x.shape[1], 1))

    for epoch in range(self.n_epochs):
      # make predictions
      y_pred = predict_proba(x, self.w)

      #compute the gradient
      grad = gradient(x,y,y_pred)

      #update rule
      self.w -= self.lr * grad

      #Compute and append the training loss in a list
      loss = cross_entropy(x,y, self.w)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')
