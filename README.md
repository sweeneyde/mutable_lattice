`mutable_lattice` is a Python package that implements fast membership testing and mutation
for sublattices of an integral lattice Z^n, with miscellaneous other integer linear algebra features.

# Installation

Install this package with `python -m pip install mutable_lattice`.

Run the tests with `python -m mutable_lattice.tests`.

# Demo

The following constructs a sublattice of `Z^3`, adds some vectors to it,
then tests which vectors are in the integer span of the previously added vectors.

```pycon
>>> from mutable_lattice import Vector, Lattice
>>> L = Lattice(3)
>>> L.add_vector(Vector([2, 2, 2]))
>>> L.add_vector(Vector([2, 3, 3]))
>>> L.get_basis()
[Vector([2, 0, 0]), Vector([0, 1, 1])]
>>> Vector([-4, 7, 7]) in L
True
>>> Vector([3, 0, 0]) in L
False
>>> L.add_vector(Vector([3, 3, 3]))
>>> L.get_basis()
[Vector([1, 0, 0]), Vector([0, 1, 1])]
>>> Vector([1, 0, 0]) in L
True
>>> L.add_vector(Vector([1, 0, 1]))
>>> L.get_basis()
[Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])]
>>> Vector([314, -159265, 3589793238462643383279]) in L
True

```

This behavior was originally implemented to help construct small projective resolutions
for computing integral homology of finite monoids, but is useful for integer linear algebra
in general.

There are more general integer linear algebra libraries included in SageMath (including PARI, GAP, and IML),
and some of these may be preferred, especially when matrix entries include large integers,
but this `mutable_lattice` package specializes in fast `add_vector` and `__contains__` operations,
and is especially fast with machine-word-sized integers.

# Features

`mutable_lattice` exposes two types: `Vector` and `Lattice`.

## Efficient `Vector`s of integers

The `Vector` class is implemented in C, and stores a sequence of integers,
packed efficiently using [tagged pointers](https://en.wikipedia.org/wiki/Tagged_pointer)
to distinguish between machine-sized integers and pointers to Python `int` objects.
This optimizes performance for integers that fit in one machine word (up to around `2^30` or `2^62`),
while also seamlessly intermixing integers of arbitrary size.

Vector arithmetic (addition, subtraction, negation, and integer scaling)
behaves conventionally:

```pycon
>>> v = Vector([10, 20, 30])
>>> w = Vector([7, 7, 7])
>>> v + w
Vector([17, 27, 37])
>>> v - w
Vector([3, 13, 23])
>>> 10*v
Vector([100, 200, 300])
>>> (-2) * v
Vector([-20, -40, -60])
>>> -w
Vector([-7, -7, -7])
>>> v * 10**20
Vector([1000000000000000000000, 2000000000000000000000, 3000000000000000000000])

```

One can access entries of a vector with the `__getitem__`, `tolist`, and `__iter__` methods:

```pycon
>>> v = Vector([10, 20, 30])
>>> v[0]
10
>>> v[-1]
30
>>> v.tolist()
[10, 20, 30]
>>> [-x for x in v]
[-10, -20, -30]
>>> len(v)
3

```

Vector are mutable and have `__iadd__`, `__isub__`, `__imul__`, and `__setitem__` methods:

```pycon
>>> v = Vector([1, 1, 1, 1, 1])
>>> v[1] = 4
>>> v
Vector([1, 4, 1, 1, 1])
>>> v += Vector([1, 1, 1, 1, 1])
>>> v
Vector([2, 5, 2, 2, 2])
>>> v *= 2
>>> v
Vector([4, 10, 4, 4, 4])
>>> v -= Vector([4, 4, 4, 4, 4])
>>> v
Vector([0, 6, 0, 0, 0])

```

`Vector.zero(n)` is equivalent to `Vector([0]*n)`.

## Efficient `Lattice`s of integer vectors: Overview

The `Lattice` class by default stores a basis of vectors in
[Hermite normal form (HNF)](https://en.wikipedia.org/wiki/Hermite_normal_form),
i.e., the integer version of reduced row echelon form.

The `Lattice.get_basis()` method returns a list of basis `Vector`s for the `Lattice`.
The `Lattice.__str__()` method prints this basis as the rows of a matrix.
Each time a new vector is added via `Lattice.add_vector(v)`,
HNF is restored.

```pycon
>>> L = Lattice(5)
>>> print(L)
<zero Lattice in Z^5>

>>> L.add_vector(Vector([1, 1, 1, 1, 1]))
>>> L.get_basis()
[Vector([1, 1, 1, 1, 1])]
>>> print(L)
[1 1 1 1 1]

>>> L.add_vector(Vector([10, 0, 10, 0, 10]))
>>> L.get_basis()
[Vector([1, 1, 1, 1, 1]), Vector([0, 10, 0, 10, 0])]
>>> print(L)
[ 1  1  1  1  1]
[ 0 10  0 10  0]

>>> L.add_vector(Vector([1, 0, 0, 0, 0]))
>>> L.get_basis()
[Vector([1, 0, 0, 0, 0]), Vector([0, 1, 1, 1, 1]), Vector([0, 0, 10, 0, 10])]
>>> print(L)
[ 1  0  0  0  0]
[ 0  1  1  1  1]
[ 0  0 10  0 10]

```

The `Lattice.rank` attribute gives the number of vectors currently stored in a `Lattice`,
and the `Lattice.ambient_dimension` attribute gives the length of each vector.
So `L.__str__()` is displaying a matrix of height `L.rank` and width `L.ambient_dimension`:

```pycon
>>> print(L) # Continuing from above
[ 1  0  0  0  0]
[ 0  1  1  1  1]
[ 0  0 10  0 10]
>>> L.rank
3
>>> L.ambient_dimension
5

```

The `Lattice.__contains__(v)` method identifies whether the argument `v`
is the `Lattice`, i.e., whether `v` is in the integer span of the stored basis:

```pycon
>>> print(L) # Continuing from above
[ 1  0  0  0  0]
[ 0  1  1  1  1]
[ 0  0 10  0 10]
>>> Vector([777, 1, 11,  1, 11]) in L
True
>>> Vector([0, 0, 10, 10, 10]) in L
False

```

To reconstruct the linear combination of the `Lattice` basis
that produces the given vector, use the `Lattice.coefficients_of(v)` method.
For the inverse operation, use the `Lattice.linear_combination(w)` method
to evaluate a linear combination of the `Lattice` basis given a `Vector` of coefficients `w`:

```pycon
>>> print(L) # Continuing from above
[ 1  0  0  0  0]
[ 0  1  1  1  1]
[ 0  0 10  0 10]
>>> L.coefficients_of(Vector([0, 0, 10, 10, 10]))
Traceback (most recent call last):
    ...
ValueError: Vector not present in Lattice
>>> L.coefficients_of(Vector([777, 1, 11, 1, 11]))
Vector([777, 1, 1])
>>> L.linear_combination(Vector([777, 1, 1]))
Vector([777, 1, 11, 1, 11])

```

`Lattice.__add__`  and `Lattice.__iadd__` methods allow adding two lattices,
so if `v` is a vector and `L1`, `L2` are `Lattice`s of the same `ambient_dimension`,
then `v in (L1 + L2)` if and only if we can write `v = v1 + v2` where `v1 in L1` and `v2 in L2`:

```pycon
>>> L1 = Lattice(4)
>>> L1.add_vector(Vector([2, 0, 0, 0]))
>>> L1.add_vector(Vector([0, 2, 0, 0]))
>>> L1.add_vector(Vector([0, 0, 2, 0]))
>>> L2 = Lattice(4)
>>> L2.add_vector(Vector([0, 4, 0, 0]))
>>> L2.add_vector(Vector([0, 0, 4, 0]))
>>> L2.add_vector(Vector([0, 0, 0, 4]))
>>> print(L1 + L2)
[2 0 0 0]
[0 2 0 0]
[0 0 2 0]
[0 0 0 4]

```

Comparison methods such as `L1 < L2`, `L1 <= L2`, `L1 == L2` are also supported
for detecting when one lattice is a subset of another.

## `Lattice(n, data=..., /, *, maxrank=-1, HNF_policy=1)`

The `Lattice(n, data)` constructor creates a sublattice of `Z^n`
and accepts an optional second positional `data` argument
which must be a list of any length, of `Vector`s or lists of length `n`,
which are added to the lattice as if by calling
`add_vector(v)` or `add_vector(Vector(v))` for each `v` in `data`.

For memory locality and to avoid reallocations, `Lattice` objects are allocated
in one contiguous memory block with enough room for all `n*n` integer entries.
If `n` is large, this may be needlessly memory intensive. If the optional
keyword-only integer argument `maxrank` is provided then only enough
memory is allocated for `n*maxrank` integer entries, potentially saving memory.
If `maxrank > n` or if `maxrank == -1` then `maxrank` is replaced by `n`.

For more uniformly fast performance in more situations, by default
all `Lattice`s are stored as a basis in HNF form.
If a `Lattice` is constructed instead with `HNF_policy=0` then
the basis of the `Lattice` is stored in some row echelon form,
but the pivots are not necessarily positive,
and entries of the matrix above pivots are not normalized to HNF.
This lazier policy may be faster if your basis is
particularly sparse or small (say `n <= 10` but always be sure to measure for your particular application).
If you have a Lattice not stored in HNF then calling `L.HNFify()`
will perform row operations to convert to HNF.

The full lattice that contains every vector in `Z^n` can be constructed
using the classmethod `Lattice.full(n)`. `L.is_full()`
returns whether `L == Lattice.full(L.ambient_dimension)`.

## SNF invariants

Given a Lattice `L` in `Z^n`, some (integer-)invertible n-by-n integer matrix takes `L`
to a Lattice with a basis `[Vector([d0, 0, 0, ...]), Vector([0, d1, 0, ...]), Vector([0, 0, d2, ...]), ...]`
in which the `i`th `Vector` is positive in its `i`th entry and zero elsewhere.
The matrix with these rows (and an additional and divisibility constraint)
is the [Smith normal form (SNF)](https://en.wikipedia.org/wiki/Smith_normal_form)
of the matrix with the original basis of `L` for rows. The `Lattice.nonzero_invariants()` method
returns a list `[d0, d1, d2, ...]` of these diagonal entries of the Smith normal form.
The length of `L.nonzero_invariants()` is always `L.rank`. A related method
`L.invariants()` returns the same list with zeros appended to the end until
the length of the result is `L.ambient_dimension`.

```pycon
>>> L = Lattice(4)
>>> L.add_vector(Vector([10, 10, 10, 10]))
>>> L.add_vector(Vector([0, 20, 20, 20]))
>>> L.add_vector(Vector([0, 0, 30, 30]))
>>> L.nonzero_invariants()
[10, 10, 60]
>>> L.invariants()
[10, 10, 60, 0]

```

Note that the quotient group `Z^n/L` is isomorphic
to the direct sum of cyclic groups `Z/dZ` for `d` in `L.invariants()`,
where `Z/0Z = Z`.

The homology/cohomology of a chain complex of integer matrices between free abelian groups
can be computed using this method:

```pycon
>>> # Chain complex from a Klein bottle Delta-complex
>>> cell_counts = {-1: 0, 0: 1, 1: 2, 2: 1, 3: 0}
>>> matrices = {
...     0: [Vector([])],
...     1: [Vector([0]), Vector([0])],
...     2: [Vector([2, 0])],
...     3: [],
... }
>>> invariant_lists = {
...    i: Lattice(cell_counts[i-1], m).nonzero_invariants()
...    for i, m in matrices.items()
... }
>>> for i in (0, 1, 2):
...     free_rank = cell_counts[i] - len(invariant_lists[i]) - len(invariant_lists[i+1])
...     torsion = [d for d in invariant_lists[i+1] if d != 1]
...     print(f"H_{i} =", torsion + [0] * free_rank)
H_0 = [0]
H_1 = [2, 0]
H_2 = []
>>> for i in (0, 1, 2):
...     free_rank = cell_counts[i] - len(invariant_lists[i]) - len(invariant_lists[i+1])
...     torsion = [d for d in invariant_lists[i] if d != 1]
...     print(f"H^{i} =", torsion + [0] * free_rank)
H^0 = [0]
H^1 = [0]
H^2 = [2]

```

## Kernel computations

Given a list of `k` vectors, the `relations_among([v1, ..., vk])` function
constructs the Lattice of linear dependencies among these vectors.

```pycon
>>> from mutable_lattice import relations_among
>>> L = relations_among([Vector([1, 2]), Vector([-1, 0]), Vector([0, 1]), Vector([0, 0])])
>>> print(L)
[ 1  1 -2  0]
[ 0  0  0  1]

```

Thinking of the provided list `[v1, ..., vk]` as a matrix *M* of row vectors,
the result of `relations_among()` is the `Lattice` of vectors *w* such that *wM*=0,
i.e, the left-kernel of *M*.

To instead compute the classic right-kernel that solves *Mv*=0, first the transpose
the matrix with the `transpose(n, [v1, ..., vk])` function (which requires each `Vector` in the list to have length `n`):

```pycon
>>> from mutable_lattice import transpose
>>> M = [Vector([1, -1, 0, 0]), Vector([2, 0, 1, 0])]
>>> print(relations_among(M))
<zero Lattice in Z^2>
>>> transpose(4, M)
[Vector([1, 2]), Vector([-1, 0]), Vector([0, 1]), Vector([0, 0])]
>>> print(relations_among(transpose(4, M)))
[ 1  1 -2  0]
[ 0  0  0  1]

```

## `v.shuffled_by_action(a, result_size=..., /)`

It is occasionally useful to shuffle around the entries of a `Vector` by a permutation:

```pycon
>>> perm = Vector([0, 2, 4, 1, 3])
>>> Vector([0, 10, 20, 30, 40]).shuffled_by_action(perm)
Vector([0, 30, 10, 40, 20])

```

Note that each entry `perm[i]` above specifies where the `i`th entry moves to after the shuffle.

More generally, we can shuffle the entries of the `Vector`
Using some function from `range(len(v))` to `range(result_size)`,
represented as a "action" `Vector`.
If `result_size` is not provided, it defaults to `len(v)`.
If the has duplicate entries `a[j1] == a[j2] == ...`,
then `v.shuffled_by_action(a)[a[j1]]` will be their sum `a[j1] + a[j2] + ...`,
so each action `Vector` specifies a linear operation.

```pycon
>>> Vector([0, 10, 20, 30, 40]).shuffled_by_action(Vector([0, 1, 0, 1, 0]))
Vector([60, 40, 0, 0, 0])
>>> Vector([0, 10, 20, 30, 40]).shuffled_by_action(Vector([0, 1, 0, 1, 0]), 2)
Vector([60, 40])

```

This package does not provide a built-in way of applying more general linear operations
on `Vector`s via matrices, nor a way to multiply matrices, but these can be emulated
using appropriate `Vector` addition and scaling.