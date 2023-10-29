import numpy as np
from numba import cuda

@cuda.jit
def matrix_multiply_cuda(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        dot_product = 0.0
        for k in range(A.shape[1]):
            dot_product += A[i, k] * B[k, j]
        C[i, j] = dot_product

if __name__ == "__main__":
    # Define two random matrices
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C = np.zeros((100, 100))

    # Configure the CUDA grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (A.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Copy matrices to the GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(C)

    # Launch the CUDA kernel
    matrix_multiply_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    # Copy the result back to the CPU
    d_C.copy_to_host(C)

    # Print the result
    print(C)
