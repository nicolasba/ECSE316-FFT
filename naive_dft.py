import numpy as np


def helper_naive_dft(x, k):
    N = len(x)
    sum = 0

    for n in range(N):
        omega = 2 * np.pi / N
        exponent = -1j * omega * k * n
        sum += x[n] * np.exp(exponent)
    return sum


def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        X[k] = helper_naive_dft(x, k)
    return X


# print(naive_dft([1, 2, 3]))
