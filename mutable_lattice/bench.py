from . import Lattice, Vector
from .pylattice import PyLattice

import random
from time import perf_counter

VALUES =  [-3, -2, -1, 0, 1, 2, 3]
WEIGHTS = [1, 1, 1, 5, 1, 1, 1]

def bench(N, HNF_policy):
    random.seed(0)
    total_time = 0.0
    total_count = 0
    while True:
        data = [
            Vector(random.choices(VALUES, weights=WEIGHTS, k=N))
            for _ in range(N)
        ]
        t0 = perf_counter()
        L = Lattice(N, HNF_policy=HNF_policy)
        for vec in data:
            L.add_vector(vec)
            for vec2 in data:
                vec2 in L
        t1 = perf_counter()
        total_time += t1 - t0
        total_count += 1
        if total_time > 0.1 and total_count >= 100:
            break
        # L._assert_consistent()
    print(f"{N=}, {HNF_policy=} --> {total_time/total_count*10**6:.3f}us")

def bench_pylattice(N):
    random.seed(0)
    total_time = 0.0
    total_count = 0
    while True:
        data = [
            list(random.choices(VALUES, weights=WEIGHTS, k=N))
            for _ in range(N)
        ]
        t0 = perf_counter()
        L = PyLattice(N)
        for vec in data:
            L.add_vector(vec)
            for vec2 in data:
                vec2 in L
        t1 = perf_counter()
        total_time += t1 - t0
        total_count += 1
        if total_time > 0.1 and total_count >= 100:
            break
        # L._assert_consistent()
    print(f"{N=}, PyLattice --> {total_time/total_count*10**6:.3f}us")

if __name__ == "__main__":
    for N in list(range(15)) + list(range(15, 50, 5)):
        # bench_pylattice(N)
        # for HNF_policy in (0, 1, 2, 3, 4, 5, 6):
        for HNF_policy in (1,):
            bench(N, HNF_policy)
        # print()
