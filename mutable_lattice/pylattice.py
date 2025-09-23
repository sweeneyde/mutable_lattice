from bisect import bisect_left
import itertools

class PyLattice:
    # The equivalent of mutable_lattice.Lattice, but
    # implemented entirely with Python lists.

    # This will be slower than the C version, but is useful for comparison.

    __slots__ = ["N",
                 "basis",
                 "pivot_location_in_column",
                 "pivot_location_in_row",
                 "zero_columns"]

    def __init__(self, ambient_dimension):
        self.N = N = ambient_dimension
        self.pivot_location_in_column = [None] * N
        self.pivot_location_in_row = []
        self.basis = []
        self.zero_columns = list(range(N))

    def copy(self):
        other = object.__new__(__class__)
        other.N = self.N
        other.pivot_location_in_column = self.pivot_location_in_column.copy()
        other.pivot_location_in_row = self.pivot_location_in_row.copy()
        other.basis = list(map(list.copy, self.basis))
        other.zero_columns = self.zero_columns.copy()
        return other

    def __contains__(self, vec):
        zc = self.zero_columns
        for _ in filter(vec.__getitem__, zc):
            return False
        col_piv = self.pivot_location_in_column
        N = self.N
        basis = self.basis
        vec = vec.copy()
        for j in filter(vec.__getitem__, range(N)):
            p = col_piv[j]
            if p is None:
                # can't zero this vec entry out
                # without disrupting previous parts
                return False
            a = basis[p][j]
            b = vec[j]
            if b % a != 0:
                # This pivot can't zero this entry
                return False
            else:
                q = b // a
                row = basis[p]
                for jj in range(j, N):
                    vec[jj] -= q * row[jj]
        return True

    def add_vector(self, vec0):
        col_piv = self.pivot_location_in_column
        row_piv = self.pivot_location_in_row
        N = self.N
        basis = self.basis
        assert len(vec0) == N
        vec = vec0.copy()
        for j in filter(vec.__getitem__, range(N)):
            p = col_piv[j]
            if p is None:
                # This vector gets inserted so that its first entry is a pivot.
                where = bisect_left(row_piv, j)
                basis.insert(where, vec)
                row_piv.insert(where, j)
                col_piv[j] = where
                for ii in range(where + 1, len(basis)):
                    assert col_piv[row_piv[ii]] == ii - 1
                    col_piv[row_piv[ii]] = ii

                zc = self.zero_columns
                s = bisect_left(zc, j)
                zc[s:] = itertools.filterfalse(vec0.__getitem__, zc[s:])
                return
            row = basis[p]
            a = row[j]
            b = vec[j]
            if b % a == 0:
                q = b // a
                for jj in range(j, N):
                    vec[jj] -= q * row[jj]
            elif a % b == 0:
                row[j:], vec[j:] = vec[j:], row[j:]
                q = a // b
                for jj in range(j, N):
                    vec[jj] -= q * row[jj]
            else:
                x, y, g = py_xgcd(a, b)
                ag = a//g
                mbg = -b//g
                for jj in range(j, N):
                    aa = row[jj]
                    bb = vec[jj]
                    row[jj] = x*aa + y*bb
                    vec[jj] = mbg*aa + ag*bb


def py_xgcd(a, b):
    # Maintain the invariants:
    #          x * a +      y * b ==      g
    #     next_x * a + next_y * b == next_g
    # Do the Euclidean algorithm to (g, next_g),
    # but carry the rest of the equations along for the ride.
    x, next_x = 1, 0
    y, next_y = 0, 1
    g, next_g = a, b
    while next_g:
        q = g // next_g
        x, next_x = next_x, x - q * next_x
        y, next_y = next_y, y - q * next_y
        g, next_g = next_g, g - q * next_g
    # if g < 0:
    #     x, y, g = -x, -y, -g
    return x, y, g