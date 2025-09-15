import unittest
import itertools
import random
import math
from array import array
import sys
array_typecode = {2**31-1: 'i', 2**63-1: 'q'}[sys.maxsize]

from . import (
    Vector,
    row_op,
    generalized_row_op,
    Lattice,
    xgcd,
)
from .pylattice import PyLattice

def make_some_values():
    return sorted({
        sgn * 2**e + d
        # for e in [33, 62]
        for e in [0,1,2,3,4,
                  6,7,8,9,
                  14,15,16,17,
                  30,31,32,33,
                  62,63,64,65, 100]
        # for d in [0]
        for d in [-2, -1, 0, 1, 2]
        for sgn in (-1, 1)
    }, key=abs)

class TestVector(unittest.TestCase):
    def setUp(self):
        self.VALUES = make_some_values()
    
    def tearDown(self):
        self.VALUES.clear()
        del self.VALUES

    def test_do_nothing(self):
        pass

    def test_Vector_tolist(self):
        self.assertEqual(Vector([]).tolist(), [])
        self.assertEqual(Vector([10]).tolist(), [10])
        self.assertEqual(Vector([10, 20, 30]).tolist(), [10, 20, 30])
        self.assertEqual(Vector([10**100]).tolist(), [10**100])
        self.assertEqual(Vector([-10*100, -10, -1, 0, 1, 10, 10**100]).tolist(),
                         [-10*100, -10, -1, 0, 1, 10, 10**100])
        self.assertEqual(Vector(self.VALUES).tolist(), self.VALUES)

    def test_Vector_numbigints(self):
        self.assertEqual(Vector([])._num_bigints(), 0)
        self.assertEqual(Vector([10, 20, 30])._num_bigints(), 0)
        self.assertEqual(Vector(list(range(-100, 100)))._num_bigints(), 0)
        self.assertEqual(Vector([2**65, 0, 0]*5)._num_bigints(), 5)

    def test_iadd_isub(self):
        v = Vector([1, 2, 3])
        self.assertEqual(v._num_bigints(), 0)
        w = Vector([0, 10**50, 10**100])
        v += w
        self.assertEqual(v.tolist(), [1, 10**50+2, 10**100 + 3])
        self.assertEqual(v._num_bigints(), 2)

        v = Vector([1, 2, 3])
        self.assertEqual(v._num_bigints(), 0)
        w = Vector([0, 10**50, 10**100])
        v -= w
        self.assertEqual(v.tolist(), [1, 2-10**50, 3-10**100])
        self.assertEqual(v._num_bigints(), 2)

        v = Vector([])
        v += v
        self.assertEqual(v.tolist(), [])
        v -= v
        self.assertEqual(v.tolist(), [])

        VALUES = self.VALUES
        for i in range(len(VALUES)):
            other_data = VALUES[i:] + VALUES[:i]
            vec2 = Vector(other_data)

            expected1 = [x+y for x, y in zip(VALUES, other_data)]
            result1 = Vector(VALUES)
            result1 += vec2
            self.assertEqual(result1.tolist(), expected1)
            self.assertEqual(vec2.tolist(), other_data)

            expected2 = [x-y for x, y in zip(VALUES, other_data)]
            result2 = Vector(VALUES)
            result2 -= vec2
            self.assertEqual(result2.tolist(), expected2, (VALUES, other_data))
            self.assertEqual(vec2.tolist(), other_data)

    def test_add_sub(self):
        self.assertEqual((Vector([]) + Vector([])).tolist(), [])
        self.assertEqual((Vector([]) - Vector([])).tolist(), [])
        self.assertEqual((Vector([1,2]) + Vector([3, 4])).tolist(), [4, 6])
        self.assertEqual((Vector([1,2]) - Vector([3, 5])).tolist(), [-2, -3])

        VALUES = self.VALUES
        for i in range(len(VALUES)):
            other_data = VALUES[i:] + VALUES[:i]
            vec1 = Vector(VALUES)
            vec2 = Vector(other_data)

            expected1 = [x+y for x, y in zip(VALUES, other_data)]
            result1 = vec1 + vec2
            self.assertEqual(result1.tolist(), expected1)
            self.assertEqual(vec1.tolist(), VALUES)
            self.assertEqual(vec2.tolist(), other_data)

            expected2 = [x-y for x, y in zip(VALUES, other_data)]
            result2 = vec1 - vec2
            self.assertEqual(result2.tolist(), expected2)
            self.assertEqual(vec1.tolist(), VALUES)
            self.assertEqual(vec2.tolist(), other_data)

    def test_imul(self):
        VALUES = self.VALUES
        for m in VALUES:
            expected = [m * x for x in VALUES]
            v = Vector(VALUES)
            v *= m
            self.assertEqual(v.tolist(), expected, (VALUES, m))

    def test_mul(self):
        VALUES = self.VALUES
        v = Vector(VALUES)
        for m in VALUES:
            expected = [m * x for x in VALUES]
            self.assertEqual((m * v).tolist(), expected)
            self.assertEqual((v * m).tolist(), expected)
    
    def test_copy(self):
        VALUES = self.VALUES
        v = Vector(VALUES)
        w = v.copy()
        self.assertEqual(v.tolist(), VALUES)
        self.assertEqual(v.tolist(), VALUES)
        self.assertIsNot(v, w)

    def test_negative(self):
        VALUES = self.VALUES
        v = Vector(VALUES)
        w = -v
        self.assertEqual(v.tolist(), VALUES)
        self.assertEqual(w.tolist(), [-x for x in VALUES])
    
    def test_repr(self):
        self.assertEqual(repr(Vector([])), "Vector([])")
        self.assertEqual(repr(Vector([17])), "Vector([17])")
        self.assertEqual(repr(Vector([10, 20, 30])), "Vector([10, 20, 30])")
        self.assertEqual(repr(Vector([10**100])), f"Vector([1" + "0"*100 + "])")

    def test_str(self):
        self.assertEqual(str(Vector([])), "[]")
        self.assertEqual(str(Vector([17])), "[17]")
        self.assertEqual(str(Vector([10, 20, 30])), "[10 20 30]")
        self.assertEqual(str(Vector([1, 20, 3])), "[ 1 20  3]")
        self.assertEqual(str(Vector([10**100])), f"[1" + "0"*100 + "]")
        self.assertEqual(str(Vector([10**100, 5])), f"[1" + "0"*100 + " " + " "*100 + "5]")

    def test_zero(self):
        for n in range(10):
            self.assertEqual(Vector.zero(n).tolist(), [0]*n)

    def test_equality(self):
        VALUES = self.VALUES
        vecs = [Vector([x]) for x in VALUES]
        for v in vecs:
            for w in vecs:
                self.assertEqual(v == w, v.tolist() == w.tolist())
                self.assertEqual(v != w, v.tolist() != w.tolist())
                self.assertEqual((v + w) - w, v)
                self.assertEqual((v - w) + w, v)
        for v in vecs:
            self.assertEqual(v, Vector(v.tolist()))
        for v in vecs:
            for w in vecs:
                for u in [v + w, v - w]:
                   self.assertEqual(u, Vector(u.tolist()))

    def test_errors(self):
        with self.assertRaises(TypeError): Vector()
        with self.assertRaises(TypeError): Vector([], foo="bar")
        with self.assertRaises(TypeError): Vector("hello")
        with self.assertRaises(TypeError): Vector(0)
        with self.assertRaises(TypeError): Vector([1], [2], [3])
        with self.assertRaises(TypeError): Vector([False])
        with self.assertRaises(TypeError): Vector.zero("foobar")
        with self.assertRaises(ValueError): Vector.zero(-1)
        v = Vector([1,2,3])
        w = Vector([4,5,6])
        with self.assertRaises(TypeError): v *= w
        with self.assertRaises(TypeError): v * w
        with self.assertRaises(TypeError): v + 0
        with self.assertRaises(TypeError): v += 0

class TestRowOps(unittest.TestCase):
    def setUp(self):
        self.VALUES = make_some_values()

    def tearDown(self):
        self.VALUES.clear()
        del self.VALUES

    def test_row_op_single(self):
        VALUES = [-10**50, -2, -1, 0, 1, 2, 10**50]
        for x, y, m in itertools.product(VALUES, repeat=3):
            vx = Vector([x])
            vy = Vector([y])
            row_op(vx, vy, m)
            self.assertEqual(vx.tolist(), [x])
            self.assertEqual(vy.tolist(), [y+m*x], (x,y,m))

    def test_row_op_many(self):
        VALUES = self.VALUES
        vx_data = [x for x in VALUES for _ in VALUES]
        vy_data = [y for _ in VALUES for y in VALUES]
        for m in VALUES:
            vx = Vector(vx_data)
            vy = Vector(vy_data)
            new_vy = vy+m*vx
            row_op(vx, vy, m)
            self.assertEqual(vx.tolist(), vx_data)
            self.assertEqual(vy.tolist(), new_vy.tolist())

    def test_generalized_row_op_single(self):
        VALUES = [-10**50, -2, -1, 0, 1, 2, 10**50]
        for tup in itertools.product(VALUES, repeat=6):
            a, b, c, d, x, y = tup
            vx = Vector([x])
            vy = Vector([y])
            generalized_row_op(vx, vy, a, b, c, d)
            self.assertEqual(vx.tolist(), [a*x + b*y], tup)
            self.assertEqual(vy.tolist(), [c*x + d*y], tup)
    
    def test_generalized_row_op_many(self):
        VALUES = list(self.VALUES)
        random.Random(0).shuffle(VALUES)
        vx_data = [x for x in VALUES for _ in VALUES]
        vy_data = [y for _ in VALUES for y in VALUES]
        for i in range(len(VALUES) - 4):
            a, b, c, d = VALUES[i:i+4]
            vx = Vector(vx_data)
            vy = Vector(vy_data)
            new_vx = a*vx + b*vy
            new_vy = c*vx + d*vy
            generalized_row_op(vx, vy, a, b, c, d)
            self.assertEqual(vx.tolist(), new_vx.tolist())
            self.assertEqual(vy.tolist(), new_vy.tolist())

    def test_xgcd(self):
        for a, b in itertools.product(self.VALUES, repeat=2):
            x, y, g = xgcd(a, b)
            self.assertEqual(abs(g), math.gcd(a, b))
            self.assertEqual(x*a + y*b, g)
    
    def test_shuffled_by_array(self):
        random.seed(0)
        VALUES = list(self.VALUES)
        for N in [0, 1, 2, 3, 4, 5, 10, 20]:
            for _ in range(100):
                data = random.choices(VALUES, k=N)
                a = array(array_typecode, random.choices(range(N), k=N))
                expected = [0] * N
                for i, ai in enumerate(a):
                    expected[ai] += data[i]
                self.assertEqual(Vector(data).shuffled_by_array(a).tolist(), expected)

    def test_shuffled_out_of_bounds(self):
        v = Vector([0, 10, 20, 30])
        self.assertEqual(
            v.shuffled_by_array(array(array_typecode, [1, 0, 3, 2])).tolist(),
            [10, 0, 30, 20],
        )
        a = array(array_typecode, [0, 4, 0, 0])
        with self.assertRaises(IndexError):
            v.shuffled_by_array(a)

class LatticeTests:

    # define in subclasses
    HNF_POLICY = None

    def setUp(self):
        assert self.HNF_POLICY is not None
        self.VALUES = make_some_values()

    def tearDown(self):
        self.VALUES.clear()
        del self.VALUES

    def test_random(self):
        VALUES = self.VALUES
        for N in [0, 1, 2, 3, 4, 5, 10, 20]:
            for _ in range(20):
                mod = Lattice(N, HNF_policy=self.HNF_POLICY)
                # mod._assert_consistent()
                vecs = [
                    Vector(random.choices(VALUES, k=N))
                    for _ in range(random.randrange(N + 2))
                ]
                zero = Vector.zero(N)
                vec0 = vecs[0] if vecs else zero
                for vec in vecs:
                    mod.add_vector(vec)
                    mod._assert_consistent()
                    self.assertIn(vec, mod, msg=vecs)
                    self.assertIn(zero-vec, mod, msg=vecs)
                    self.assertIn(10*vec, mod, msg=vecs)
                    self.assertIn(vec0 + vec, mod, msg=vecs)
                    self.assertIn(3*vec0-100*vec, mod, msg=vecs)

                lin_combo = zero.copy()
                for vec in vecs:
                    lin_combo += random.choice(VALUES)*vec
                self.assertIn(lin_combo, mod)

                basis = mod.get_basis()
                for to_remove in basis:
                    submod = Lattice(N, HNF_policy=self.HNF_POLICY)
                    for vec in basis:
                        if vec != to_remove:
                            submod.add_vector(vec)
                    # mod._assert_consistent()
                    self.assertNotIn(to_remove, submod)
                    also_outside = to_remove
                    for vec in submod.get_basis():
                        q = random.randint(-2, 2)
                        also_outside += q*vec
                        self.assertNotIn(also_outside, submod)

    def test_against_pylattice(self):
        VALUES = self.VALUES
        for N in [0, 1, 2, 3, 4, 5]:
            for _ in range(100):
                vecs = [
                    list(random.choices(VALUES, k=N))
                    for _ in range(random.randrange(N + 2))
                ]
                L = Lattice(N, HNF_policy=self.HNF_POLICY)
                PyL = PyLattice(N)
                for vec in vecs:
                    vvec = Vector(vec)
                    self.assertEqual(vvec in L, vec in PyL)
                    L.add_vector(vvec)
                    PyL.add_vector(vec)
                    self.assertEqual(L.rank, len(PyL.basis))

                self.assertEqual(L._get_col_to_pivot(), PyL.pivot_location_in_column)
                self.assertEqual(L._get_row_to_pivot(), PyL.pivot_location_in_row)
                self.assertEqual(L._get_zero_columns(), PyL.zero_columns)

                L_basis = L.tolist()
                for vec in L_basis:
                    self.assertIn(vec, PyL)
                for v in PyL.basis:
                    self.assertIn(Vector(v), L)
                for i, j in enumerate(L._get_row_to_pivot()):
                    self.assertEqual(abs(L_basis[i][j]), abs(PyL.basis[i][j]))

    def test_sum_reorder(self):
        VALUES = self.VALUES
        for N in [0, 1, 2, 3, 4, 5]:
            for _ in range(200):
                vecs = [
                    Vector(random.choices(VALUES, k=N))
                    for _ in range(random.randrange(N + 2))
                ]
                mod = Lattice(N, HNF_policy=self.HNF_POLICY)
                for vec in vecs:
                    mod.add_vector(vec)
                for i in range(len(vecs) + 1):
                    vecs1, vecs2 = vecs[:i], vecs[i:]
                    random.shuffle(vecs1)
                    random.shuffle(vecs2)
                    mod1 = Lattice(N, HNF_policy=self.HNF_POLICY)
                    for vec in vecs1:
                        mod1.add_vector(vec)
                    mod2 = Lattice(N, HNF_policy=self.HNF_POLICY)
                    for vec in vecs2:
                        mod2.add_vector(vec)
                    mod3 = mod1 + mod2
                    self.assertEqual(mod3, mod)

    def test_str(self):
        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        self.assertEqual(str(L), "<zero lattice in Z^4>")
        L.add_vector(Vector([0, 1, 0, 1]))
        L.add_vector(Vector([0, 0, 2, 2]))
        self.assertEqual(str(L), 
                        "[0 1 0 1]\n"
                        "[0 0 2 2]")

    def test_constructor_with_data(self):
        self.assertEqual(Lattice(0, [], HNF_policy=self.HNF_POLICY).tolist(), [])
        self.assertEqual(Lattice(1, [], HNF_policy=self.HNF_POLICY).tolist(), [])
        self.assertEqual(Lattice(1, [[0]], HNF_policy=self.HNF_POLICY).tolist(), [])
        self.assertEqual(Lattice(1, [[1]], HNF_policy=self.HNF_POLICY).tolist(), [[1]])
        self.assertEqual(Lattice(1, [[2]], HNF_policy=self.HNF_POLICY).tolist(), [[2]])
        self.assertEqual(Lattice(2, [], HNF_policy=self.HNF_POLICY).tolist(), [])
        self.assertEqual(Lattice(2, [[0, 0]], HNF_policy=self.HNF_POLICY).tolist(), [])
        self.assertEqual(Lattice(2, [[1, 5]], HNF_policy=self.HNF_POLICY).tolist(), [[1, 5]])
        self.assertEqual(Lattice(2, [[1, 0], [0, 5]], HNF_policy=self.HNF_POLICY).tolist(), [[1, 0], [0, 5]])
        self.assertEqual(Lattice(2, [[2, 0], [0, 2], [-100, -200]], HNF_policy=self.HNF_POLICY).tolist(), [[2, 0], [0, 2]])

    def test_hnf(self):
        L = Lattice(2, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1, 3]))
        L.add_vector(Vector([0, 2]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 1],
             [0, 2]]
        )

        L = Lattice(2, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([5, -10]))
        L.add_vector(Vector([0, -1]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[5, 0],
             [0, 1]]
        )

        L = Lattice(2, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1, 3]))
        L.add_vector(Vector([0, 1]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 0],
             [0, 1]]
        )

        L = Lattice(2, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 1]))
        L.add_vector(Vector([-2, 0]))
        L.add_vector(Vector([-3, -1]))
        L.add_vector(Vector([-2, -1]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 0],
             [0, 1]]
        )

        # From https://en.wikipedia.org/wiki/Hermite_normal_form
        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([2, 3, 6, 2]))
        L.add_vector(Vector([5, 6, 1, 6]))
        L.add_vector(Vector([8, 3, 1, 1]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 0, 50, -11],
             [0, 3, 28, -2],
             [0, 0, 61, -13]]
        )

        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 0, 1, 1]))
        L.add_vector(Vector([0, 1, 0, 0]))
        L.add_vector(Vector([0, 0, 19, 1]))
        L.add_vector(Vector([0, 0, 0, 3]))
        self.assertEqual(L.tolist(),
            [[3, 0,  1, 1],
             [0, 1,  0, 0],
             [0, 0, 19, 1],
             [0, 0,  0, 3]]
        )

        # From https://github.com/sagemath/sage/blob/develop/src/sage/matrix/matrix_integer_dense.pyx
        L = Lattice(5, HNF_policy=self.HNF_POLICY)
        for k in range(0, 25, 5):
            L.add_vector(Vector(list(range(k, k+5))))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[   5,   0,  -5, -10, -15],
             [   0,   1,   2,   3,   4]],
        )

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([ 1,  2,  3]))
        L.add_vector(Vector([ 4,  5,  6]))
        L.add_vector(Vector([ 7,  8,  9]))
        L.add_vector(Vector([10, 11, 12]))
        L.add_vector(Vector([13, 14, 15]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 2, 3],
            [0, 3, 6]],
        )

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 0, 0]))
        L.add_vector(Vector([0, 0, 0]))
        L.add_vector(Vector([0, 0, 0]))
        L.HNFify()
        self.assertEqual(L.tolist(), [])

        L = Lattice(1, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0]))
        L.add_vector(Vector([0]))
        L.add_vector(Vector([0]))
        L.HNFify()
        self.assertEqual(L.tolist(), [])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 0, 0]))
        L.HNFify()
        self.assertEqual(L.tolist(), [])

        L = Lattice(0, HNF_policy=self.HNF_POLICY)
        L.HNFify()
        self.assertEqual(L.tolist(), [])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1, 2, 3]))
        L.add_vector(Vector([4, 5, 6]))
        L.add_vector(Vector([7, 8, 9]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 2, 3],
             [0, 3, 6]],
        )

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 2, 3]))
        L.add_vector(Vector([4, 5, 6]))
        L.add_vector(Vector([7, 8, 9]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 3]],
        )

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 0, 3]))
        L.add_vector(Vector([0,-2, 2]))
        L.add_vector(Vector([0, 1, 2]))
        L.add_vector(Vector([0,-2, 5]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[0, 1, 2],
             [0, 0, 3]],
        )

        L = Lattice(10, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([-2, 1, 9, 2, -8, 1, -3, -1, -4, -1]))
        L.add_vector(Vector([5, -2, 0, 1, 0, 4, -1, 1, -2, 0]))
        L.add_vector(Vector([-11, 3, 1, 0, -3, -2, -1, -11, 2, -2]))
        L.add_vector(Vector([-1, 1, -1, -2, 1, -1, -1, -1, -1, 7]))
        L.add_vector(Vector([-2, -1, -1, 1, 1, -2, 1, 0, 2, -4]))
        for i in range(10):
            L.add_vector(Vector([0]*i + [200] + [0]*(9-i)))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[  1,   0,   2,   0,  13,   5,   1, 166,  72,  69],
             [  0,   1,   1,   0,  20,   4,  15, 195,  65, 190],
             [  0,   0,   4,   0,  24,   5,  23,  22,  51, 123],
             [  0,   0,   0,   1,  23,   7,  20, 105,  60, 151],
             [  0,   0,   0,   0,  40,   4,   0,  80,  36,  68],
             [  0,   0,   0,   0,   0,  10,   0, 100, 190, 170],
             [  0,   0,   0,   0,   0,   0,  25,   0, 100, 150],
             [  0,   0,   0,   0,   0,   0,   0, 200,   0,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0, 200,   0],
             [  0,   0,   0,   0,   0,   0,   0,   0,   0, 200]]
        )

        # From Arne Storjohann (1998).
        # "Computing Hermite and Smith normal forms of triangular integer matrices"
        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([5, 2342, 1843]))
        L.add_vector(Vector([0,   78, 8074]))
        L.add_vector(Vector([0,    0,   32]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[5,  2,  7],
             [0, 78, 10],
             [0,  0, 32]]
        )

        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([8, 11286,  4555,  46515]))
        L.add_vector(Vector([0,     1, 66359, 153094]))
        L.add_vector(Vector([0,     0,     9,  43651]))
        L.add_vector(Vector([0,     0,     0,     77]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[8, 0, 1, 51],
             [0, 1, 2, 20],
             [0, 0, 9, 69],
             [0, 0, 0, 77]]
        )

        # From Pernet and Stein (2009)
        # "Fast computation of Hermite normal forms of random integer matrices"
        L = Lattice(6, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([-5,  8, -3, -9,  5,  5]))
        L.add_vector(Vector([-2,  8, -2, -2,  8,  5]))
        L.add_vector(Vector([ 7, -5, -8,  4,  3, -4]))
        L.add_vector(Vector([ 1, -1,  6,  0,  8, -3]))
        L.HNFify()
        self.assertEqual(L.tolist(),
            [[1, 0, 3, 237, -299,  90],
             [0, 1, 1, 103, -130,  40],
             [0, 0, 4, 352, -450, 135],
             [0, 0, 0, 486, -627, 188]]
        )

    def test_hnf_random(self):
        VALUES = self.VALUES
        for N in [0, 1, 2, 3, 4, 5]:
            for _ in range(2000):
                vecs = [
                    Vector(random.choices(VALUES, k=N))
                    for _ in range(random.randrange(N + 2))
                ]
                L = Lattice(N, HNF_policy=self.HNF_POLICY)
                for vec in vecs:
                    L.add_vector(vec)
                    L2 = L.copy()
                    L2.HNFify()
                    for v in L.get_basis():
                        self.assertIn(v, L2)
                    for v2 in L2.get_basis():
                        self.assertIn(v2, L)
                    L2._assert_consistent()
                    basis = L2.tolist()
                    row_to_pivot = L2._get_row_to_pivot()
                    for i, j in enumerate(row_to_pivot):
                        pivot = basis[i][j]
                        self.assertGreater(pivot, 0)
                        for ii in range(i):
                            self.assertIn(basis[ii][j], range(pivot))

    def test_invariants(self):
        # From Arne Storjohann (1998).
        # "Computing Hermite and Smith normal forms of triangular integer matrices"
        L = Lattice(5, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 113344, 95472, 42884, 12302]))
        L.add_vector(Vector([0,      2,  1576, 98594, 11872]))
        L.add_vector(Vector([0,      0,     2, 99206, 94692]))
        L.add_vector(Vector([0,      0,     0,  9456,  7080]))
        L.HNFify()
        self.assertEqual(L.invariants(), [1, 2, 6, 24, 0])

        # From https://github.com/sagemath/sage/blob/develop/src/sage/matrix/matrix_integer_dense.pyx
        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 0, 1]))
        L.add_vector(Vector([0, 1, 0]))
        self.assertEqual(L.invariants(), [1, 1, 0])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 1, 2]))
        L.add_vector(Vector([3, 4, 5]))
        L.add_vector(Vector([6, 7, 8]))
        self.assertEqual(L.invariants(), [1, 3, 0])

        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3,4,5,6]))
        L.add_vector(Vector([7,3,8,10]))
        L.add_vector(Vector([14,5,6,7]))
        L.add_vector(Vector([2,2,10,9]))
        self.assertEqual(L.invariants(), [1, 1, 1, 687])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1,5,7]))
        L.add_vector(Vector([3,6,9]))
        L.add_vector(Vector([0,1,2]))
        self.assertEqual(L.invariants(), [1, 1, 6])

    def test_invariants_random(self):
        VALUES = list(self.VALUES)
        random.shuffle(VALUES)
        for a, b in zip(VALUES, VALUES[1:]):
            # do a lot of row ops
            A = Vector([a, 0])
            B = Vector([0, b])
            k1_list = random.choices(VALUES, k=10)
            k2_list = random.choices(VALUES, k=10)
            for k1, k2 in zip(k1_list, k2_list):
                A += k1 * B
                B += k2 * A
            # transpose so we can do "column ops"
            Alist = A.tolist()
            Blist = B.tolist()
            A = Vector([Alist[0], Blist[0]])
            B = Vector([Alist[1], Blist[1]])
            k1_list = random.choices(VALUES, k=10)
            k2_list = random.choices(VALUES, k=10)
            for k1, k2 in zip(k1_list, k2_list):
                A += k1 * B
                B += k2 * A

            if a == 0 and b == 0:
                expected = [0, 0]
            elif a == 0:
                expected = [abs(b), 0]
            elif b == 0:
                expected = [abs(a), 0]
            else:
                g = math.gcd(a, b)
                lcm = abs(a * b) // g
                expected = [g, lcm]

            L = Lattice(2, HNF_policy=self.HNF_POLICY)
            L.add_vector(A)
            L.add_vector(B)
            self.assertEqual(L.invariants(), expected)


class TestLatticeHNFPolicy0(LatticeTests, unittest.TestCase):
    HNF_POLICY = 0

class TestLatticeHNFPolicy1(LatticeTests, unittest.TestCase):
    HNF_POLICY = 1

class TestLatticeAPI(unittest.TestCase):

    def test_construction(self):
        with self.assertRaises(TypeError):
            Lattice()
        with self.assertRaises(TypeError):
            Lattice(N=2)
        with self.assertRaises(TypeError):
            Lattice(2, 0, 0)

        for N in range(5):
            L = Lattice(N)
            self.assertEqual(L.maxrank, N)
            self.assertEqual(L.HNF_policy, 1)
            L = Lattice(N, maxrank=-1)
            self.assertEqual(L.maxrank, N)
            self.assertEqual(L.HNF_policy, 1)

            with self.assertRaises(ValueError):
                Lattice(N, maxrank=-5)
            with self.assertRaises(ValueError):
                Lattice(N, maxrank=-5, HNF_policy=0)
            with self.assertRaises(ValueError):
                Lattice(N, maxrank=-5, HNF_policy=1)
            with self.assertRaises(ValueError):
                Lattice(N, maxrank=0, HNF_policy=10)

            for HNF_policy in [0, 1]:
                for maxrank in range(N+3):
                    L = Lattice(N, maxrank=maxrank, HNF_policy=HNF_policy)
                    self.assertEqual(L.maxrank, min(N, maxrank))
                    self.assertEqual(L.HNF_policy, HNF_policy)

    def test_maxrank(self):
        L = Lattice(3, maxrank=0)
        # should be able to add the zero vector
        L.add_vector(Vector([0, 0, 0]))
        with self.assertRaises(IndexError):
            # should not be able to add any other vector
            L.add_vector(Vector([0, 5, 0]))
        with self.assertRaises(RuntimeError):
            # After any attempt, should cause errors
            L.add_vector(Vector([0, 5, 0]))

        L = Lattice(3, maxrank=1)
        L.add_vector(Vector([2, 2, 2]))
        L.add_vector(Vector([0, 0, 0]))
        L.add_vector(Vector([4, 4, 4]))
        L.add_vector(Vector([3, 3, 3]))
        L.add_vector(Vector([3, 3, 3]))
        with self.assertRaises(IndexError):
            L.add_vector(Vector([0, 5, 0]))
        with self.assertRaises(RuntimeError):
            L.add_vector(Vector([0, 5, 0]))

        L = Lattice(3, maxrank=2)
        L.add_vector(Vector([2, 0, 0]))
        L.add_vector(Vector([0, 2, 0]))
        L.add_vector(Vector([10, 200000000, 0]))
        with self.assertRaises(IndexError):
            L.add_vector(Vector([0, 0, 1]))
        with self.assertRaises(RuntimeError):
            L.add_vector(Vector([0, 0, 1]))

        for N in range(5):
            basis = [Vector([0]*i + [1] + [0]*(N-1-i)) for i in range(N)]
            for maxrank in range(N+1):
                L = Lattice(N, maxrank=maxrank)
                for vec in basis[:maxrank]:
                    L.add_vector(vec)
                self.assertEqual(L.rank, maxrank)
                if maxrank < N:
                    with self.assertRaises(IndexError):
                        L.add_vector(basis[maxrank])
                    for vec in basis[maxrank:]:
                        with self.assertRaises(RuntimeError):
                            L.add_vector(vec)

    def test_is_full(self):
        for HNF_policy in [0, 1]:
            for N in range(5):
                basis = [Vector([0]*i + [1] + [2]*(N-1-i)) for i in range(N)]
                L = Lattice(N, HNF_policy=HNF_policy)
                for vec in basis[:-1]:
                    L.add_vector(vec)
                    self.assertIs(L.is_full(), False)
                for vec in basis[-1:]:
                    L.add_vector(vec)
                self.assertIs(L.is_full(), True)

    def test_full(self):
        Lattice.full(0)._assert_consistent()
        Lattice.full(1)._assert_consistent()
        Lattice.full(2)._assert_consistent()
        Lattice.full(3)._assert_consistent()
        self.assertEqual(Lattice.full(3).tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_tolist(self):
        L = Lattice(3)
        L.add_vector(Vector([1,0,0]))
        L.add_vector(Vector([0,2,3]))
        self.assertEqual(L.tolist(), [[1,0,0],[0,2,3]])

    def test_get_basis(self):
        vecs = [Vector([1,0,0]),
                Vector([0,2,3])]
        L = Lattice(3)
        for vec in vecs:
            L.add_vector(vec)
        self.assertEqual(L.get_basis(), vecs)

    def test_repr_eval(self):
        for expr in [
            "Lattice(0)",
            "Lattice(10)",
            "Lattice(10, maxrank=5)",
            "Lattice(10, HNF_policy=0)",
            "Lattice(10, maxrank=4, HNF_policy=0)",
            "Lattice(3, [[1, 2, 3]])",
            "Lattice(3, [[1, 2, 3], [0, 0, 5]])",
            "Lattice(3, [[1, 2, 3], [0, 3, 1], [0, 0, 5]])",
            "Lattice(3, [[1, 2, 3], [0, 1, 0], [0, 0, 1]], HNF_policy=0)",
            "Lattice(3, [[1, 2, 3], [0, 0, 1]], maxrank=2, HNF_policy=0)",
        ]:
            self.assertEqual(repr(eval(expr)), expr)

if __name__ == "__main__":
    unittest.main(exit=False)
