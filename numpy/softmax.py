import numpy as np

def softmax(X):
    X = X - np.max(X)
    X_exp = np.exp(X)
    softmax_X = X_exp / np.sum(X_exp)
    return softmax_X
    

if __name__ == '__main__':
    X = [[11, 12, 13]]
    print(softmax(X))