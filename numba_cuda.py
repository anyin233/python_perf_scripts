from numba import cuda
import numpy as np

@cuda.jit
def axpy(y, x, a):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    bw = cuda.threadIdx.x
    pos = tx + ty * bw
    
    if pos < y.size:
        y[pos] += x[pos] * a[0]

N = 1 << 30

x = np.random.rand(N)
a = np.random.rand(1)
y = np.zeros(N)

x = x.dot(a[0])
threadsperblock = 32
blockpergrid = (x.size + (threadsperblock - 1)) // threadsperblock
axpy[blockpergrid, threadsperblock](y, x, a)
axpy[blockpergrid, threadsperblock](x, y, a)
