import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # - p * log(p) - (1 - p) * log(1 - p)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss 
