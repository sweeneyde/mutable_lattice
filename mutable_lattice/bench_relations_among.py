from time import perf_counter
from random import Random
from itertools import product
from . import Vector, relations_among, relations_among_simple
from statistics import geometric_mean

SIZES = [
    (2, 1, [1,2]),
    (5, 2, [1,2,5,10]),
    (10, 5, [1,2,5,10,20,50]),
    (20, 10, [1,2,5,10,20,50,100,200]),
    (50, 20, [1,2,5,10,20,50,100,200,500,1000]),
    (100, 50, [1,2,5,10,20,50,100,200,500,1000]),
    (200, 100, [1,2,5,10,20,50,100,200,500,1000]),
    (500, 200,  [1,2,5,10,20,50,100,200,500,1000]),
    (1000, 500, [1,2,5,10,20,50,100,200,500,1000]),
]

def bench(function, num_vectors, N, num_nonzero, seed):
    r = Random(seed)
    data = [[0] * N for _ in range(num_vectors)]
    entries = r.sample(list(product(range(num_vectors), range(N))), k=num_nonzero)
    values = r.choices([1, -1, 2, -2, 3, -3, 4, -4, 5, -5],
                       weights=[5,5,4,4,3,3,2,2,1,1],
                       k=num_nonzero)
    for (i, j), v in zip(entries, values):
        data[i][j] = v
    total_time = 0.0
    num_iterations = 0
    while total_time < 0.1:
        vectors = list(map(Vector, data))
        t0 = perf_counter()
        result = function(vectors)
        t1 = perf_counter()
        total_time += t1 - t0
        num_iterations += 1
    h = hash(tuple(map(tuple, result.get_basis())))
    return total_time / num_iterations, h


def main():
    ratios = []
    for R, N, num_nonzero_list in SIZES:
        for num_nonzero in num_nonzero_list:
            for seed in (10, 20, 30):
                tx, hx = bench(relations_among_simple, R, N, num_nonzero, seed)
                ty, hy = bench(relations_among, R, N, num_nonzero, seed)
                assert hx == hy
                ratio = ty/tx
                print(f"{R=},{N=},{num_nonzero=},{seed=}:  {tx*1000:.2f} --> {ty*1000:.2f}   ({ratio=})")
                ratios.append(ratio)
    g = geometric_mean(ratios)
    print(f"geometric mean ratio: {g}")

if __name__ == "__main__":
    main()