import cupy as cp
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=int, default=100)
    parser.add_argument("-y", type=int, default=-1)
    parser.add_argument("-z", type=int, default=-1)
    
    return parser.parse_args()

def main():
    args = parse_args()
    x, y, z = args.x, args.y, args.z
    test_size = (x)
    if y > 0:
        test_size = (x, y)
    if z > 0:
        if y <= 0:
            raise ValueError("y must be greater than 0")
        test_size = (x, y, z)
    data = np.random.uniform(size=test_size)
    
    with cp.cuda.Device(0):
        g_data = cp.asarray(data)
    
    with cp.cuda.Device(1):
        g_data1 = cp.asarray(g_data)
    
    with cp.cuda.Device(0):
        g_data = cp.asarray(g_data1)

if __name__ == "__main__":
    main()