import numpy as np
import naive_dft


# Recursive helper function that computes sum of two halves to obtain X_k
def helper_fft(x, k):
    N = len(x)
    if N <= 8:                  # We can play around with this value to find best efficiency vs overhead
        return naive_dft.helper_naive_dft(x,k)

    x_odd = x[1:N:2]            # Subarray of odd indices
    x_even = x[0:N:2]           # Subarray of even indices
    omega = 2 * np.pi / N
    const = np.exp(-1j * omega * k)

    return helper_fft(x_even, k) + const * helper_fft(x_odd, k)


def fft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)      # List containing transform

    for k in range(N):
        X[k] = helper_fft(x, k)
    return X


x = [1, 2, 3, 4, 5, 6, 7, 8]
print(naive_dft.naive_dft(x))
print(fft(x))
print(np.fft.fft(x))
