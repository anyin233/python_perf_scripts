import numpy as np
from argparse import ArgumentParser
import concurrent.futures
def vallina(a, b):
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    c = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c

def vallina_for_parallel(a, b):
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    c = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c

def numpy_mm(a, b):
    return np.dot(a, b)

def parallel_mm(a, b, x_threads = 2, y_threads = 2):
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    
    x_blk_len = m // x_threads
    y_blk_len = p // y_threads
    
    assert(x_blk_len * x_threads == m)
    assert(y_blk_len * y_threads == p)
    
    c = np.zeros((m, p))
    a_blks = []
    b_blks = []
    for i in range(x_threads):
        for j in range(y_threads):
            m_blk_start = i * x_blk_len
            m_blk_end = (i + 1) * x_blk_len
            p_blk_start = j * y_blk_len
            p_blk_end = (j + 1) * y_blk_len
            a_blk = a[m_blk_start:m_blk_end, :]
            b_blk = b[:, p_blk_start:p_blk_end]
            a_blks.append(a_blk)
            b_blks.append(b_blk)
            # c_blk = vallina_for_parallel(a_blk, b_blk)
            # c[m_blk_start:m_blk_end, p_blk_start:p_blk_end] = c_blk
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=x_threads * y_threads) as executor:
        c_blks = executor.map(vallina_for_parallel, a_blks, b_blks)
        for index, c_blk in enumerate(c_blks):
            x_index = index // x_threads
            y_index = index % x_threads
            m_blk_start = x_index * x_blk_len
            p_blk_start = y_index * y_blk_len
            m_blk_end = (x_index + 1) * x_blk_len
            p_blk_end = (y_index + 1) * y_blk_len
            c[m_blk_start:m_blk_end, p_blk_start:p_blk_end] = c_blk
        
    
    return c
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-x', type=int, default=100)
    parser.add_argument('-y', type=int, default=100)
    parser.add_argument('-z', type=int, default=100)
    parser.add_argument('-x_threads', type=int, default=2)
    parser.add_argument('-y_threads', type=int, default=2)
    return parser.parse_args()
    
def main():
    args = parse_args()
    x, y, z = args.x, args.y, args.z
    x_threads, y_threads = args.x_threads, args.y_threads
    a = np.random.rand(x, y)
    b = np.random.rand(y, z)
    # c = vallina(a, b)
    d = numpy_mm(a, b)
    e = parallel_mm(a, b, x_threads, y_threads)
    # print(np.allclose(c, d))
    print(np.allclose(d, e))

if __name__ == '__main__':
    main()
