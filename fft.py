import numpy as np
import naive_dft


# Recursive helper function that computes sum of two halves to obtain X_k
def helper_fft(x, k):
    N = len(x)
    if N <= 8:                  # We can play around with this value to find best efficiency vs overhead
        return naive_dft.helper_naive_dft(x, k)

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


# Recursive helper function that computes sum of two halves to obtain x_n
# The only diff between inv_fft and fft is the sign of the exponent in the constant multiplied to the odd recursive call
def helper_inv_fft(X, k):
    N = len(X)
    if N <= 8:                  # We can play around with this value to find best efficiency vs overhead
        sum = 0
        for n in range(N):      # Naive inverse dft for small N
            omega = 2 * np.pi / N
            exponent = 1j * omega * k * n
            sum += X[n] * np.exp(exponent)
        return sum

    X_odd = X[1:N:2]            # Subarray of odd indices
    X_even = X[0:N:2]           # Subarray of even indices
    omega = 2 * np.pi / N
    const = np.exp(1j * omega * k)

    return helper_inv_fft(X_even, k) + const * helper_inv_fft(X_odd, k)


def inv_fft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)      # List containing transform

    for n in range(N):
        x[n] = 1 / N * helper_inv_fft(X, n)
    return x


def fft_2d(x):
    A = np.array(x, dtype=complex)
    N = len(x)          # Number of rows
    M = len(x[0])       # Number of columns
    for n in range(N):
        A[n] = fft(A[n])
    A = A.T             # Transpose the matrix, so that it is easier to operate on columns
    for m in range(M):
        A[m] = fft(A[m])
    return A.T


def inv_fft_2d(X):
    A = np.array(X, dtype=complex)
    N = len(X)          # Number of rows
    M = len(X[0])       # Number of columns
    for n in range(N):
        A[n] = inv_fft(A[n])
    A = A.T             # Transpose the matrix, so that it is easier to operate on columns
    for m in range(M):
        A[m] = inv_fft(A[m])
    return A.T


def round_complex(c):
    return complex(round(c.real, 8), round(c.imag, 8))


def round_complex_list(l):
    size = len(l)
    new_l = np.zeros(size, dtype=complex)
    for i in range(size):
        new_l[i] = round_complex(l[i])
    return new_l


def round_complex_2d_list(l):
    rows = len(l)
    cols = len(l[0])
    new_l = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            new_l[i][j] = round_complex(l[i][j])
    return new_l


x1 = [1, 2, 3, 4, 5, 6, 7, 8]
print("x1: ")
print(str(x1) + "\n")
print("naive dft of x1: ")
print(str(round_complex_list(naive_dft.naive_dft(x1))) + "\n")
print("fast ft of x1: ")
print(str(round_complex_list(fft(x1))) + "\n")
print("numpy fast ft of x1: ")
print(str(np.fft.fft(x1)) + "\n")
print("F^-1(F(x1)): ")
print(str(round_complex_list(inv_fft(fft(x1)))) + "\n")

x_2d = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
print("numpy 2d fast ft of x_2d: ")
print(str(np.fft.fft2(x_2d)) + "\n")
print("2d fast ft of x_2d: ")
print(str(round_complex_2d_list(fft_2d(x_2d))) + "\n")
print("inverse 2d fft of 2d fft of x_2d: ")
print(str(round_complex_2d_list(inv_fft_2d(fft_2d(x_2d)))) + "\n")