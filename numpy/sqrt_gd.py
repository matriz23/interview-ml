'''
Square root function with gradient descending
'''

import numpy as np


def sqrt_gd(x, max_iter=1000, lr=0.01):
    """
    Compute the square root of each element in the input array using gradient descent.
    """
    if not np.all(x >= 0):
        raise ValueError("Input array must contain only non-negative values.")
    r = x / 2
    for i in range(max_iter):
        grad = r * (r**2 - x)
        r -= grad * lr
        if np.all((r**2 - x) ** 2 < 1e-6):
            break
    return r


if __name__ == "__main__":
    print(sqrt_gd(np.array([0, 1, 2, 9])))
