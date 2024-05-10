import numpy as np
import time
import matplotlib.pyplot as plt

def matrix_multiplication_with_blocking(matrix_size, blocking_size):
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    matrix_c = np.zeros((matrix_size, matrix_size))

    start_time = time.time()
    for i in range(0, matrix_size, blocking_size):
        for j in range(0, matrix_size, blocking_size):
            for k in range(0, matrix_size, blocking_size):
                matrix_c[i:i+blocking_size, j:j+blocking_size] += np.dot(matrix_a[i:i+blocking_size, k:k+blocking_size],
                                                                         matrix_b[k:k+blocking_size, j:j+blocking_size])
    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time

def matrix_multiplication(matrix_size):
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    matrix_c = np.zeros((matrix_size, matrix_size))

    start_time = time.time()
    for i in range(matrix_size):
        for j in range(matrix_size):
            for k in range(matrix_size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time

matrix_sizes = [64,128,256,512]
blocking_sizes = [2,4,8,16,32]

results_blocking = {}
results_no_blocking = {}

for matrix_size in matrix_sizes:
    for blocking_size in blocking_sizes:
        elapsed_time_blocking = matrix_multiplication_with_blocking(matrix_size, blocking_size)
        results_blocking[(matrix_size, blocking_size)] = elapsed_time_blocking

        elapsed_time_no_blocking = matrix_multiplication(matrix_size)
        results_no_blocking[matrix_size] = elapsed_time_no_blocking


for matrix_size in matrix_sizes:
    x = blocking_sizes
    y1 = [results_blocking[(matrix_size, blocking_size)] for blocking_size in blocking_sizes]
    y2 = results_no_blocking[matrix_size]
    plt.plot(x, y1, label=f"Matrix Size: {matrix_size} - With Blocking")
    plt.axhline(y=y2, linestyle='--', color='r', label=f"Matrix Size: {matrix_size} - Without Blocking")

plt.xlabel("Blocking Size")
plt.ylabel("Time (s)")
plt.title("Relationship between Blocking Size and Time for Matrix Multiplication")
plt.legend()
plt.show()
