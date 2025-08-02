from . import Lattice, Vector
from .pylattice import PyLattice

import random
from time import perf_counter

def bench(N, hnfify):
    total_time = 0.0
    random.seed(0)
    for _ in range(5000):
        data = [
            Vector(random.choices([-3, -2, -1, 0, 1, 2, 3],
                                weights=[1, 1, 1, 20, 1, 1, 1],
                                k=N))
            for _ in range(N)
        ]
        t0 = perf_counter()
        L = Lattice(N)
        for vec in data:
            if vec not in L:
                L.add_vector(vec)
                if hnfify is True:
                    L.HNFify()
                elif hnfify == "half":
                    if L.rank * 2 > L.ambient_dimension:
                        L.HNFify()
            for vec2 in data:
                vec2 in L
        t1 = perf_counter()
        total_time += t1 - t0
    print(f"{N=}, {hnfify=: >5} --> {total_time*1000:.0f}ms")

if __name__ == "__main__":
    for N in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]:
        bench(N, False)
        bench(N, "half")
        bench(N, True)
        print()