import numpy as np

def generate_data(var = 0.2, n = 800):
    class_0_a = var * np.random.randn(n//4,2)
    class_0_b =var * np.random.randn(n//4,2) + (2,2)

    class_1_a = var* np.random.randn(n//4,2) + (0,2)
    class_1_b = var * np.random.randn(n//4,2) +  (2,0)
    X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
    Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
    return X,Y


def shuffle_data(X, Y):
    rand_perm = np.random.permutation(X.shape[0])
    X = X[rand_perm, :]
    Y = Y[rand_perm, :]
    return X,Y

def split_data(X,Y, ratio = 0.8):
    X = X.T
    Y = Y.T
    n = X.shape[1]
    X_train = X [:, :int (n*ratio)]
    Y_train = Y [:, :int (n*ratio)]
    X_test = X [:, int (n*ratio):]
    Y_test = Y [:, int (n*ratio):]
    return X_train, Y_train, X_test, Y_test




