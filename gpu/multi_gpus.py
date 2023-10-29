import cupy as cp
import numpy as np

import time

N = 1 << 29

x_cpu = np.random.rand(N)
a = np.random.rand(1)
y_cpu = np.zeros(N)

# cpu
start = time.time()
y_cpu = x_cpu.dot(a[0])
end = time.time()

print("CPU time {:.6f}".format(end - start))


# single gpu
x_gpu = cp.asarray(x_cpu)
a_gpu = cp.asarray(a)
y_gpu = cp.asarray(y_cpu)

start = time.time()
for i in range(1000):
    y_gpu = x_gpu.dot(a_gpu[0])
y_cpu = cp.asnumpy(y_gpu)
end = time.time()

print("GPU time {:.6f}".format(end - start))

# clean GPU memory
x_gpu = None
a_gpu = None
y_gpu = None

# two gpus

s = []
for i in range(2):
    cp.cuda.Device(i).use()
    s.append(cp.cuda.Stream(non_blocking=True))

x_gpu = []
y_gpu = []
a_gpu = []

y_cpu = np.zeros(N)
len = N // 2
for i in range(2):
    cp.cuda.Device(i).use()
    with s[i]:
        x_gpu.append(cp.asarray(x_cpu[len * i:len * (i + 1)]))
        y_gpu.append(cp.asarray(y_cpu[len * i:len * (i + 1)]))
        a_gpu.append(cp.asarray(a))
        
        
start = time.time()
for i in range(2):
    cp.cuda.Device(i).use()
    with s[i]:
        for _ in range(1000):
            y_gpu[i] = x_gpu[i].dot(a_gpu[i][0])

for i in range(2):
    cp.cuda.Device(i).use()
    with s[i]:
        y_cpu[len * i:len * (i + 1)] = cp.asnumpy(y_gpu[i])

end = time.time()

print("2 GPUs Time {:.6f}".format(end - start))