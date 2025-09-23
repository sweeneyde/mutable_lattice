import unittest
import itertools
import random
import math
import sys

from . import (
    Vector,
    row_op,
    generalized_row_op,
    Lattice,
    xgcd,
    relations_among,
    transpose,
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

    def test_Vector_tolist(self):
        self.assertEqual(Vector([]).tolist(), [])
        self.assertEqual(Vector([10]).tolist(), [10])
        self.assertEqual(Vector([10, 20, 30]).tolist(), [10, 20, 30])
        self.assertEqual(Vector([10**100]).tolist(), [10**100])
        self.assertEqual(Vector([-10*100, -10, -1, 0, 1, 10, 10**100]).tolist(),
                         [-10*100, -10, -1, 0, 1, 10, 10**100])
        self.assertEqual(Vector(self.VALUES).tolist(), self.VALUES)

    def test_Vector_num_bigints(self):
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

    def test_equality_simple(self):
        p = Vector([1,2])
        v = Vector([1,2,3])
        w = Vector([4,5,6])
        self.assertNotEqual(v, None)
        self.assertNotEqual(v, [1,2,3])
        self.assertNotEqual(p, v)
        self.assertNotEqual(v, w)
        self.assertEqual(v.copy(), v)
        self.assertEqual(v, v)

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
        with self.assertRaisesRegex(TypeError, "takes exactly one argument"):
            Vector()
        with self.assertRaisesRegex(TypeError, "takes no keyword arguments"):
            Vector([], foo="bar")
        with self.assertRaisesRegex(TypeError, "argument must be list"):
            Vector("hello")
        with self.assertRaisesRegex(TypeError, "argument must be list"):
            Vector(0)
        with self.assertRaisesRegex(TypeError, "takes exactly one argument"):
            Vector([1], [2], [3])
        with self.assertRaisesRegex(TypeError, "argument must be a list of int"):
            Vector([False])
        with self.assertRaises(TypeError):
            Vector.zero("foobar")
        with self.assertRaisesRegex(ValueError, "argument must be nonnegative"):
            Vector.zero(-1)
        p = Vector([1,2])
        v = Vector([1,2,3])
        w = Vector([4,5,6])
        with self.assertRaisesRegex(ValueError, "size mismatch.+addition"):
            p += v
        with self.assertRaisesRegex(ValueError, "size mismatch.+addition"):
            p + v
        with self.assertRaisesRegex(ValueError, "size mismatch.+subtraction"):
            p -= v
        with self.assertRaisesRegex(ValueError, "size mismatch.+subtraction"):
            p - v
        with self.assertRaises(TypeError): v *= w
        with self.assertRaises(TypeError): v * w
        with self.assertRaises(TypeError): v + 0
        with self.assertRaises(TypeError): v += 0
        with self.assertRaises(TypeError): w - 0
        with self.assertRaises(TypeError): v -= 0
        with self.assertRaises(TypeError): v < w
        with self.assertRaises(TypeError): v <= w

    def test_length(self):
        self.assertEqual(len(Vector([])), 0)
        self.assertEqual(len(Vector([10])), 1)
        self.assertEqual(len(Vector([10, 20, 30])), 3)

    def test_getitem(self):
        self.assertEqual(Vector([0, 10, 20, 30])[2], 20)
        self.assertEqual(Vector([0, 10, 20, 30])[-2], 20)

    def test_iter(self):
        data = [10, 20, 30]
        self.assertEqual([x for x in Vector(data)], data)

    def test_length(self):
        self.assertEqual(len(Vector([])), 0)
        self.assertEqual(len(Vector([10])), 1)
        self.assertEqual(len(Vector([10, 20, 30])), 3)

    def test_setitem_simple(self):
        v = Vector([10, 20, 30])
        v[1] = 999
        self.assertEqual(v.tolist(), [10, 999, 30])

    def test_setitem_random(self):
        old = Vector(self.VALUES)
        new = Vector(self.VALUES)
        perm = list(range(len(self.VALUES)))
        random.shuffle(perm)
        for i in range(len(self.VALUES)):
            new[i] = old[perm[i]]
        self.assertEqual(new.tolist(), [old[perm_i] for perm_i in perm])

    def test_sequence_errors(self):
        v = Vector([1, 2, 3])
        with self.assertRaises(IndexError): v[4]
        with self.assertRaises(IndexError): v[4] = 5
        with self.assertRaises(TypeError): v[None] = 5
        with self.assertRaises(TypeError): v[0] = None

    def test_shuffled_by_action_onearg(self):
        self.assertEqual(
            Vector([0, 10, 20, 30]).shuffled_by_action(Vector([1, 0, 3, 2])).tolist(),
            [10, 0, 30, 20],
        )
        VALUES = self.VALUES
        for N in [0, 1, 2, 3, 4, 5, 10, 20]:
            for _ in range(100):
                data = random.choices(VALUES, k=N)
                a = random.choices(range(N), k=N)
                expected = [0] * N
                for i, ai in enumerate(a):
                    expected[ai] += data[i]
                self.assertEqual(Vector(data).shuffled_by_action(Vector(a)).tolist(), expected)

    def test_shuffled_by_action_twoargs(self):
        self.assertEqual(
            Vector([100, 200, 300]).shuffled_by_action(Vector([0, 0, 0]), 1).tolist(),
            [600],
        )
        self.assertEqual(
            Vector([100, 200, 300, 400]).shuffled_by_action(Vector([1, 0, 1, 0]), 5).tolist(),
            [600, 400, 0, 0, 0],
        )
        VALUES = self.VALUES
        for N in range(5):
            for result_N in range(1, 5):
                with self.subTest(N=N, result_N=result_N):
                    for _ in range(10):
                        data = random.choices(VALUES, k=N)
                        a = random.choices(range(result_N), k=N)
                        expected = [0] * result_N
                        for i, ai in enumerate(a):
                            expected[ai] += data[i]
                        self.assertEqual(Vector(data).shuffled_by_action(Vector(a), result_N).tolist(), expected)
        self.assertEqual(Vector([]).shuffled_by_action(Vector([]), 0).tolist(), [])

    def test_shuffled_by_action_errors(self):
        v = Vector([0, 10, 20, 30])
        with self.assertRaisesRegex(TypeError, "takes 1 or 2 arguments"):
            v.shuffled_by_action()
        with self.assertRaisesRegex(TypeError, "first argument must be another Vector"):
            v.shuffled_by_action([0, 0, 0, 0])
        with self.assertRaisesRegex(TypeError, "second argument must be an int if present"):
            v.shuffled_by_action(Vector([0, 1, 2, 3]), 1.0)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            v.shuffled_by_action(Vector([0, 0]))
        with self.assertRaisesRegex(IndexError, "shuffle out of bounds"):
            v.shuffled_by_action(Vector([0, 4, 0, 0]))
        with self.assertRaisesRegex(IndexError, "shuffle out of bounds"):
            v.shuffled_by_action(Vector([0, 1, 2, 3]), 2)
        with self.assertRaisesRegex(IndexError, r"shuffle out of bounds \(got a big integer\)"):
            v.shuffled_by_action(Vector([0, 2**100, 0, 0]))


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

    def test_row_op_errors(self):
        with self.assertRaisesRegex(TypeError, "takes 3 arguments"):
            row_op(Vector([10]), Vector([20]), 1, 2)
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            row_op([10], [20], 3)
        with self.assertRaises(TypeError):
            row_op(Vector([10]), Vector([20]), 4.2)
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            row_op(Vector([10]), Vector([20, 30]), 4)

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

    def test_generalized_row_op_errors(self):
        with self.assertRaisesRegex(TypeError, "takes 6 arguments"):
            generalized_row_op(Vector([10]), Vector([20]), 1, 2, 3, 4, 5)
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            generalized_row_op([10], [20], 1, 2, 3, 4)
        with self.assertRaises(TypeError):
            generalized_row_op(Vector([10]), Vector([20]), 1, 2.2, 3, 4)
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            generalized_row_op(Vector([10]), Vector([20, 30]), 1, 2, 3, 4)

    def test_xgcd(self):
        for a, b in itertools.product(self.VALUES, repeat=2):
            x, y, g = xgcd(a, b)
            self.assertEqual(g, math.gcd(a, b))
            self.assertEqual(x*a + y*b, g)

    def test_xgcd_errors(self):
        with self.assertRaisesRegex(TypeError, "takes 2 arguments"):
            xgcd(1, 2, 3)
        with self.assertRaisesRegex(TypeError, "must be integers"):
            xgcd(1.0, 2.0)


class LatticeTests:

    # define in subclasses
    HNF_POLICY = None

    def setUp(self):
        assert self.HNF_POLICY is not None
        self.VALUES = make_some_values()

    def tearDown(self):
        self.VALUES.clear()
        del self.VALUES

    def test_add_vector(self):
        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1, 0, 0]))
        L.add_vector(Vector([0, 2, 0]))
        L.add_vector(Vector([0, 0, 3]))
        self.assertEqual(L.tolist(), [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def test_add_vector_errors(self):
        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        with self.assertRaises(TypeError): L.add_vector(None)
        with self.assertRaises(TypeError): L.add_vector([None])
        with self.assertRaises(TypeError): L.add_vector((1,2,3))
        with self.assertRaises(TypeError): L.add_vector([1, 2, 3])
        with self.assertRaises(ValueError): L.add_vector(Vector([1, 2, 3, 4]))

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

    def test_coefficients_of_random(self):
        VALUES = self.VALUES
        for N in range(10):
            for R in range(10):
                for _ in range(5):
                    L = Lattice(N, maxrank=R, HNF_policy=self.HNF_POLICY)
                    for _ in range(R):
                        L.add_vector(Vector(random.choices(VALUES, k=N)))
                    for _ in range(5):
                        w = Vector(random.choices(VALUES, k=L.rank))
                        v = L.linear_combination(w)
                        self.assertIn(v, L)
                        self.assertEqual(L.coefficients_of(v), w)
                    for _ in range(5):
                        v = Vector(random.choices(VALUES, k=N))
                        if v in L:
                            w = L.coefficients_of(v)
                            self.assertEqual(L.linear_combination(w), v)
                        else:
                            with self.assertRaises(ValueError):
                                L.coefficients_of(v)

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
        self.assertEqual(str(L), "<zero Lattice in Z^4>")
        L.add_vector(Vector([0, 1, 0, 1]))
        L.add_vector(Vector([0, 0, 2, 2]))
        self.assertEqual(str(L),
                        "[0 1 0 1]\n"
                        "[0 0 2 2]")

    def test_maxrank_raises(self):
        L = Lattice(3, maxrank=0, HNF_policy=self.HNF_POLICY)
        # should be able to add the zero vector
        L.add_vector(Vector([0, 0, 0]))
        with self.assertRaisesRegex(IndexError, "would exceed maxrank"):
            # should not be able to add any other vector
            L.add_vector(Vector([0, 5, 0]))
        with self.assertRaises(RuntimeError):
            # After any attempt, should cause errors
            L.add_vector(Vector([0, 5, 0]))

        L = Lattice(3, maxrank=1, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([2, 2, 2]))
        L.add_vector(Vector([0, 0, 0]))
        L.add_vector(Vector([4, 4, 4]))
        L.add_vector(Vector([3, 3, 3]))
        L.add_vector(Vector([3, 3, 3]))
        with self.assertRaisesRegex(IndexError, "would exceed maxrank"):
            L.add_vector(Vector([0, 5, 0]))
        with self.assertRaises(RuntimeError):
            L.add_vector(Vector([0, 5, 0]))

        L = Lattice(3, maxrank=2, HNF_policy=self.HNF_POLICY)
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
                L = Lattice(N, maxrank=maxrank, HNF_policy=self.HNF_POLICY)
                for vec in basis[:maxrank]:
                    L.add_vector(vec)
                self.assertEqual(L.rank, maxrank)
                if maxrank < N:
                    with self.assertRaises(IndexError):
                        L.add_vector(basis[maxrank])
                    for vec in basis[maxrank:]:
                        with self.assertRaises(RuntimeError):
                            L.add_vector(vec)
            self.assertEqual(Lattice(N, maxrank=N+5, HNF_policy=self.HNF_POLICY).maxrank, N)

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
        L = Lattice(2, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([2, 1]))
        L.add_vector(Vector([0, 2]))
        self.assertEqual(L.invariants(), [1, 4])
        self.assertEqual(L.nonzero_invariants(), [1, 4])

        # From Arne Storjohann (1998).
        # "Computing Hermite and Smith normal forms of triangular integer matrices"
        L = Lattice(5, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 113344, 95472, 42884, 12302]))
        L.add_vector(Vector([0,      2,  1576, 98594, 11872]))
        L.add_vector(Vector([0,      0,     2, 99206, 94692]))
        L.add_vector(Vector([0,      0,     0,  9456,  7080]))
        L.HNFify()
        self.assertEqual(L.invariants(), [1, 2, 6, 24, 0])
        self.assertEqual(L.nonzero_invariants(), [1, 2, 6, 24])

        # From https://github.com/sagemath/sage/blob/develop/src/sage/matrix/matrix_integer_dense.pyx
        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3, 0, 1]))
        L.add_vector(Vector([0, 1, 0]))
        self.assertEqual(L.invariants(), [1, 1, 0])
        self.assertEqual(L.nonzero_invariants(), [1, 1])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([0, 1, 2]))
        L.add_vector(Vector([3, 4, 5]))
        L.add_vector(Vector([6, 7, 8]))
        self.assertEqual(L.invariants(), [1, 3, 0])
        self.assertEqual(L.nonzero_invariants(), [1, 3])

        L = Lattice(4, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([3,4,5,6]))
        L.add_vector(Vector([7,3,8,10]))
        L.add_vector(Vector([14,5,6,7]))
        L.add_vector(Vector([2,2,10,9]))
        self.assertEqual(L.invariants(), [1, 1, 1, 687])
        self.assertEqual(L.nonzero_invariants(), [1, 1, 1, 687])

        L = Lattice(3, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([1,5,7]))
        L.add_vector(Vector([3,6,9]))
        L.add_vector(Vector([0,1,2]))
        self.assertEqual(L.invariants(), [1, 1, 6])
        self.assertEqual(L.nonzero_invariants(), [1, 1, 6])

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
            self.assertEqual(L.nonzero_invariants(), [x for x in expected if x])

    def test_error_after_corruption(self):
        # To avoid unexpected results, raise an error if we've already encountered
        # an error that might've corrupted the invariants of the Lattice.
        L = Lattice(3, maxrank=1, HNF_policy=self.HNF_POLICY)
        L.add_vector(Vector([2, 0, 0]))
        with self.assertRaises(IndexError):
            # should not be able to add any other vector
            L.add_vector(Vector([1, 1, 0]))
        self.assertEqual(str(L), "<corrupted Lattice>")
        self.assertEqual(repr(L), "<corrupted Lattice>")
        with self.assertRaises(RuntimeError): L.copy()
        with self.assertRaises(RuntimeError): L.add_vector(Vector([1, 2, 3]))
        with self.assertRaises(RuntimeError): L.get_basis()
        with self.assertRaises(RuntimeError): L.tolist()
        with self.assertRaises(RuntimeError): L._assert_consistent()
        with self.assertRaises(RuntimeError): L.HNFify()
        with self.assertRaises(RuntimeError): L.invariants()
        with self.assertRaises(RuntimeError): L.nonzero_invariants()
        with self.assertRaises(RuntimeError): L._unnormalized_nonzero_invariants()
        with self.assertRaises(RuntimeError): L.__getnewargs_ex__()
        with self.assertRaises(RuntimeError): L.coefficients_of(Vector([1, 1, 1]))
        with self.assertRaises(RuntimeError): L.linear_combination(Vector([2]))
        with self.assertRaises(RuntimeError): Vector([1, 2, 3]) in L
        with self.assertRaises(RuntimeError): L + L
        with self.assertRaises(RuntimeError): L += L
        with self.assertRaises(RuntimeError): L <= L
        with self.assertRaises(RuntimeError): L < Lattice(3)
        with self.assertRaises(RuntimeError): L == Lattice(3)
        # Make sure clearing resets the corruption state.
        L.clear()
        L.copy()


class TestLatticeHNFPolicy0(LatticeTests, unittest.TestCase):
    HNF_POLICY = 0

class TestLatticeHNFPolicy1(LatticeTests, unittest.TestCase):
    HNF_POLICY = 1

class TestLatticeAPI(unittest.TestCase):

    def test_ambient_dimension(self):
        for N in range(5):
            self.assertEqual(Lattice(N).ambient_dimension, N)

    def test_rank(self):
        for N in range(5):
            L = Lattice(N)
            for R in range(N):
                self.assertEqual(L.rank, R)
                L.add_vector(Vector([0]*R + [1] + [0]*(N-1-R)))
            self.assertEqual(L.rank, N)

    def test_maxrank(self):
        for N in range(5):
            for maxrank in range(N+1):
                self.assertEqual(Lattice(N, maxrank=maxrank).maxrank, maxrank)
            self.assertEqual(Lattice(N).maxrank, N)
            self.assertEqual(Lattice(N, maxrank=-1).maxrank, N)
            self.assertEqual(Lattice(N, maxrank=maxrank+5).maxrank, N)

    def test_HNF_policy(self):
        self.assertEqual(Lattice(10, HNF_policy=0).HNF_policy, 0)
        self.assertEqual(Lattice(10, HNF_policy=1).HNF_policy, 1)
        self.assertEqual(Lattice(10).HNF_policy, 1)

    def test__first_HNF_row(self):
        L = Lattice(3, [[1, 0, 0], [0, 0, 1]], HNF_policy=0)
        L.HNFify()
        self.assertEqual(L._first_HNF_row, 0)
        L.add_vector(Vector([0, 2, 2]))
        self.assertEqual(L._first_HNF_row, 2)
        self.assertEqual(L.tolist(), [[1, 0, 0], [0, 2, 2], [0, 0, 1]])
        L.HNFify()
        self.assertEqual(L._first_HNF_row, 0)
        self.assertEqual(L.tolist(), [[1, 0, 0], [0, 2, 0], [0, 0, 1]])

    def test_copy(self):
        L = Lattice(3, [[0, 1, 1]])
        L2 = L.copy()
        self.assertEqual(L2, L)
        L2._assert_consistent()
        self.assertEqual(repr(L2), repr(L))

    def test_construction(self):
        with self.assertRaises(TypeError):
            Lattice()
        with self.assertRaises(TypeError):
            Lattice(N=2)
        with self.assertRaises(TypeError):
            Lattice(2, 0, 0)
        with self.assertRaises(TypeError):
            Lattice(3, [0, 1, 1])
        with self.assertRaisesRegex(ValueError, "argument must be nonnegative"):
            Lattice(-3)

        for N in range(5):
            L = Lattice(N)
            self.assertEqual(L.maxrank, N)
            self.assertEqual(L.HNF_policy, 1)
            L = Lattice(N, maxrank=-1)
            self.assertEqual(L.maxrank, N)
            self.assertEqual(L.HNF_policy, 1)

            with self.assertRaisesRegex(ValueError, "maxrank must be >= -1"):
                Lattice(N, maxrank=-5, HNF_policy=1)
            with self.assertRaisesRegex(ValueError, "unknown HNF_policy"):
                Lattice(N, maxrank=0, HNF_policy=10)
            with self.assertRaisesRegex(TypeError, "must be list"):
                Lattice(N, None)

            for HNF_policy in [0, 1]:
                for maxrank in range(N+3):
                    L = Lattice(N, maxrank=maxrank, HNF_policy=HNF_policy)
                    self.assertEqual(L.maxrank, min(N, maxrank))
                    self.assertEqual(L.HNF_policy, HNF_policy)

    def test_constructor_with_data(self):
        for cls in [list, Vector]:
            for (N, data, expected) in [
                (0, [], []),
                (1, [], []),
                (1, [[0]], []),
                (1, [[1]], [[1]]),
                (1, [[2]], [[2]]),
                (2, [], []),
                (2, [[0, 0]], []),
                (2, [[1, 5]], [[1, 5]]),
                (2, [[1, 0], [0, 5]], [[1, 0], [0, 5]]),
                (2, [[2, 0], [0, 2], [-100, -200]], [[2, 0], [0, 2]]),
            ]:
                L = Lattice(N, list(map(cls, data)))
                self.assertEqual(L.tolist(), expected)

    def test_constructor_with_data_errors(self):
        with self.assertRaisesRegex(TypeError, "list or Vector"):
            L = Lattice(3, [(1,2,3)])

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

    def test_not_is_full_after_clear(self):
        L = Lattice.full(3)
        self.assertEqual(L.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(L.is_full())
        L.clear()
        self.assertEqual(L.tolist(), [])
        self.assertFalse(L.is_full())

    def test_clear(self):
        for N in range(5):
            L = Lattice(N)
            L.add_vector(Vector([2]*N))
            L.add_vector(Vector([3]*N))
            self.assertIsNone(L.clear())
            L._assert_consistent()
            self.assertEqual(L.tolist(), [])

    def test_full(self):
        Lattice.full(0)._assert_consistent()
        Lattice.full(1)._assert_consistent()
        Lattice.full(2)._assert_consistent()
        Lattice.full(3)._assert_consistent()
        self.assertEqual(Lattice.full(3).tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_full_errors(self):
        with self.assertRaisesRegex(TypeError, "must be integer"):
            Lattice.full(None)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Lattice.full(-1)

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

    def test_pickle(self):
        import pickle
        for L in [
            Lattice(0),
            Lattice(10),
            Lattice(10, maxrank=5),
            Lattice(10, HNF_policy=0),
            Lattice(10, maxrank=4, HNF_policy=0),
            Lattice(3, [[1, 2, 3]]),
            Lattice(3, [[1, 2, 3], [0, 0, 5]]),
            Lattice(3, [[1, 2, 3], [0, 3, 1], [0, 0, 5]]),
            Lattice(3, [[1, 2, 3], [0, 1, 0], [0, 0, 1]], HNF_policy=0),
            Lattice(3, [[1, 2, 3], [0, 0, 1]], maxrank=2, HNF_policy=0),
        ]:
            L2 = pickle.loads(pickle.dumps(L))
            self.assertEqual(L2, L)
            self.assertEqual(L2.maxrank, L.maxrank)
            self.assertEqual(L2.HNF_policy, L.HNF_policy)

    def test_can_construct_huge(self):
        # Can't construct N^2 ~ 800 TB,
        # but can construct N ~ 80 MB
        Lattice(10_000_000, maxrank=3)

    def test_raises_memoryerror(self):
        if sys.maxsize <= 10_000_000**2:
            raise unittest.SkipTest()
        with self.assertRaises(MemoryError):
            Lattice(10_000_000)

    def test_raises_overflowerror(self):
        with self.assertRaises(OverflowError):
            Lattice(sys.maxsize)

    def test_sizeof(self):
        # The size is:
        #   some overhead + 4 buffers of N words + (maxrank+1)*N words of data

        # All memory is allocated up front
        for n in range(10):
            self.assertEqual(Lattice(n).__sizeof__(), Lattice.full(n).__sizeof__())

        # Maxrank=N --> Quadratic growth
        sizes = [Lattice(n).__sizeof__() for n in range(10)]
        diffs = [b-a for a, b in zip(sizes, sizes[1:])]
        diffdiffs = {b-a for a, b in zip(diffs, diffs[1:])}
        self.assertIn(diffdiffs, ({2*8}, {2*4})) # 2 words per N per N

        # Bounded rank --> Linear growth in N
        sizes = [Lattice(n, maxrank=10).__sizeof__() for n in range(10, 20)]
        diffs = {b-a for a, b in zip(sizes, sizes[1:])}
        self.assertIn(diffs, ({8*(4+11)}, {4*(4+11)}))

        # Bounded N --> Linear growth in rank
        sizes = [Lattice(20, maxrank=n).__sizeof__() for n in range(10)]
        diffs = {b-a for a, b in zip(sizes, sizes[1:])}
        self.assertIn(diffs, ({8*20}, {4*20}))

    def test_contains(self):
        L = Lattice(3, [[3, 0, 0], [0, 0, 10]])
        self.assertIn(Vector([3, 0, 0]), L)
        self.assertIn(Vector([-6, 0, 0]), L)
        self.assertIn(Vector([0, 0, 10]), L)
        self.assertIn(Vector([0, 0, -70]), L)
        self.assertNotIn(Vector([1, 0, 0]), L)
        self.assertNotIn(Vector([0, 0, 1]), L)
        self.assertNotIn(Vector([2, 0, 2]), L)
        self.assertNotIn(Vector([0, 30, 0]), L)

    def test_contains_errors(self):
        L = Lattice(2, [[3, 0]])
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            [3, 0] in L
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            Vector([0, 0, 0, 0, 0, 0]) in L

    def test_coefficients_of(self):
        self.assertEqual(Lattice(2, [[2, 0], [0, 2]]).coefficients_of(Vector([2, 20])), Vector([1, 10]))
        self.assertEqual(Lattice(2, [[2, 0]]).coefficients_of(Vector([-2, 0])), Vector([-1]))
        self.assertEqual(Lattice(3, [[10, 10, 10], [0, 20, 20]]).coefficients_of(Vector([30, -10, -10])), Vector([3, -2]))
        with self.assertRaisesRegex(ValueError, "Vector not present in Lattice"):
            Lattice(3, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]).coefficients_of(Vector([2, 1, 0]))

    def test_coefficients_of_errors(self):
        L = Lattice(3, [[2,0,0],[0,2,0],[0,0,2]])
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            [1, 2, 3] in L
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            Vector([1, 2]) in L

    def test_linear_combination(self):
        self.assertEqual(Lattice(2, [[2, 0], [0, 2]]).linear_combination(Vector([1, 10])), Vector([2, 20]))
        self.assertEqual(Lattice(2, [[2, 0]]).linear_combination(Vector([-1])), Vector([-2, 0]))
        self.assertEqual(Lattice(3, [[10, 10, 10], [0, 20, 20]]).linear_combination(Vector([3, -2])), Vector([30, -10, -10]))

    def test_linear_combination_errors(self):
        L = Lattice(3, [[1,1,1]])
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            L.linear_combination([-1])
        with self.assertRaisesRegex(ValueError, "must have length L.rank"):
            L.linear_combination(Vector([1,2,3,4,5]))

    def test_add_vector(self):
        L = Lattice(3)
        self.assertEqual(L.rank, 0)
        L.add_vector(Vector([1, 2, 3]))
        self.assertEqual(L.rank, 1)

    def test_add_vector_errors(self):
        L = Lattice(3)
        with self.assertRaisesRegex(TypeError, "must be Vector"):
            L.add_vector([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            L.add_vector(Vector([1, 2]))

    def test_lattice_iadd(self):
        L1 = Lattice(3, [[2, 2, 2]])
        L2 = Lattice(3, [[0, 1, 1]])
        L1 += L2
        L1.HNFify()
        self.assertEqual(L1, Lattice(3, [[2, 0, 0], [0, 1, 1]]))
        self.assertEqual(L2, Lattice(3, [[0, 1, 1]]))

    def test_lattice_iadd_error(self):
        L1 = Lattice(3, [[2, 2, 2]])
        L2 = Lattice(2, [[1, 1]])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            L1 += L2

    def test_lattice_add(self):
        L1 = Lattice(3, [[2, 2, 2]])
        L2 = Lattice(3, [[0, 1, 1]])
        L3 = L1 + L2
        L3.HNFify()
        self.assertEqual(L1, Lattice(3, [[2, 2, 2]]))
        self.assertEqual(L2, Lattice(3, [[0, 1, 1]]))
        self.assertEqual(L3, Lattice(3, [[2, 0, 0], [0, 1, 1]]))

    def test_lattice_add(self):
        L1 = Lattice(3, [[2, 2, 2]])
        L2 = Lattice(2, [[1, 1]])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            L1 + L2

    def test_lattice_equal(self):
        L1 = Lattice(3, [[2, 2, 2], [0, 1, 1]], HNF_policy=0)
        L2 = Lattice(3, [[2, 0, 0], [0, 1, 1]], HNF_policy=1)
        self.assertEqual(L1, L2)
        self.assertNotEqual(L1.tolist(), L2.tolist())
        self.assertNotEqual(L1, Lattice(2))

    def test_lattice_leq(self):
        L = Lattice(3, [[2, 2, 2], [0, 1, 1]], HNF_policy=0)
        self.assertTrue(Lattice(3, [[0, 3, 3]]) <= L)
        self.assertTrue(L <= L)
        self.assertTrue(L.copy() <= L)
        self.assertFalse(Lattice(3, [[0, 3, 2]]) <= L)

    def test_lattice_lt(self):
        L = Lattice(3, [[2, 2, 2], [0, 1, 1]], HNF_policy=0)
        self.assertTrue(Lattice(3, [[0, 3, 3]]) < L)
        self.assertFalse(L < L)
        self.assertFalse(L.copy() < L)
        self.assertFalse(Lattice(3, [[0, 3, 2]]) < L)

    def test_lattice_compare_error(self):
        L1 = Lattice(1)
        L2 = Lattice(2)
        with self.assertRaisesRegex(ValueError, "different ambient dimensions"):
            L1 <= L2
        with self.assertRaisesRegex(ValueError, "different ambient dimensions"):
            L1 < L2

class TestKernels(unittest.TestCase):
    def setUp(self):
        self.VALUES = make_some_values()

    def tearDown(self):
        self.VALUES.clear()
        del self.VALUES

    def test_relations_among(self):
        self.assertEqual(
            relations_among([Vector([10]*100), Vector([20]*100)]),
            Lattice(2, [[2, -1]])
        )
        self.assertEqual(
            relations_among([Vector([60]), Vector([100]), Vector([150])]),
            Lattice(3, [[5, -3, 0], [0, 3, -2]])
        )
        self.assertEqual(
            relations_among([]),
            Lattice(0),
        )
        self.assertEqual(
            relations_among([Vector([]), Vector([]), Vector([])]),
            Lattice.full(3),
        ),
        self.assertEqual(
            relations_among([Vector([17])]),
            Lattice(1),
        )
        self.assertEqual(
            relations_among([Vector([17]), Vector([0])]),
            Lattice(2, [[0, 1]]),
        )
        self.assertEqual(
            relations_among([Vector([0, 0, 0])]*100),
            Lattice.full(100)
        )
        # From https://github.com/sagemath/sage/blob/develop/src/sage/matrix/matrix_integer_dense.pyx
        self.assertEqual(
            relations_among([
                Vector([4, 1, 0, 4]),
                Vector([7, 0, 1, 7]),
                Vector([9, 5, 0, 6]),
                Vector([7, 8, 1, 5]),
                Vector([5, 9, 9, 1]),
                Vector([0, 1, 7, 4]),
            ]),
            Lattice(6, [[26, -31, 30, -21, -2, 10], [-47, -13, 48, -14, -11, 18]]),
        )

    def test_relations_among_random(self):
        for values in [
            [-2, -1, 0, 0, 0, 0, 1, 2],
            self.VALUES,
        ]:
            for N in range(5):
                zero_N = Vector.zero(N)
                for R in range(5):
                    for _ in range(10):
                        vecs = [Vector(random.choices(values, k=N)) for _ in range(R)]
                        rank = Lattice(N, vecs, maxrank=R).rank
                        kernel = relations_among(vecs)
                        # Ensure the kernel has the correct rank
                        self.assertEqual(kernel.rank + rank, R)
                        # Ensure the computed kernel vectors are actually relations
                        for row in kernel.tolist():
                            lin_combo = sum((c*vec for (c, vec) in zip(row, vecs)), start=zero_N)
                            self.assertEqual(lin_combo, zero_N)
                        # Ensure the kernel is saturated
                        self.assertEqual(kernel.invariants(), [1]*kernel.rank + [0] * rank)

    def test_relations_among_errors(self):
        with self.assertRaisesRegex(TypeError, "must be a list"):
            relations_among((1, 2, 3))
        with self.assertRaisesRegex(TypeError, "must be a list of Vectors"):
            relations_among([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            relations_among([Vector([1,2,3]), Vector([4,5,6,7])])

    def test_transpose(self):
        self.assertEqual(transpose(2, [Vector([1, 2]), Vector([3, 4])]), [Vector([1, 3]), Vector([2, 4])])
        self.assertEqual(transpose(2, [Vector([2, 3]), Vector([20, 30]), Vector([200, 300])]),
                         [Vector([2, 20, 200]), Vector([3, 30, 300])])
        self.assertEqual(transpose(2, []), [Vector([]), Vector([])])

    def test_transpose_random(self):
        VALUES = self.VALUES
        for a in range(5):
            for b in range(5):
                flat = random.choices(VALUES, k=a*b)
                A = [Vector(flat[i*a:(i+1)*a]) for i in range(b)]
                B = [Vector(flat[j::a]) for j in range(a)]
                self.assertEqual(transpose(a, A), B)
                self.assertEqual(transpose(b, B), A)

    def test_transpose_errors(self):
        with self.assertRaisesRegex(TypeError, "takes 2 arguments"):
            transpose([Vector([1, 2]), Vector([3, 4])])
        with self.assertRaisesRegex(TypeError, "first argument must be integer"):
            transpose([Vector([1, 2]), Vector([3, 4])], 2)
        with self.assertRaisesRegex(TypeError, "second argument must be list"):
            transpose(2, (Vector([1, 2]), Vector([3, 4])))
        with self.assertRaisesRegex(TypeError, "second argument must be list of Vectors"):
            transpose(2, [[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "first argument cannot be negative"):
            transpose(-2, [Vector([1, 2]), Vector([3, 4])])
        with self.assertRaises(OverflowError):
            transpose(2**100, [])
        with self.assertRaisesRegex(ValueError, "vectors must have length N"):
            transpose(3, [Vector([1, 2]), Vector([3, 4, 5])])

if __name__ == "__main__":
    unittest.main(exit=False)
