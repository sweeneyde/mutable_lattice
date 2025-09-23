#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifndef Py_READONLY
# include <structmember.h>
# define Py_READONLY READONLY
# define Py_T_PYSSIZET T_PYSSIZET
# define Py_T_INT T_INT
#endif

// This macro exists for the ease of testing the slow paths
#define USE_FAST_PATHS 1

static inline PyObject *
pylong_add(PyObject *a, PyObject *b)
{
    return PyLong_Type.tp_as_number->nb_add(a, b);
}

static inline PyObject *
pylong_subtract(PyObject *a, PyObject *b)
{
    return PyLong_Type.tp_as_number->nb_subtract(a, b);
}

static inline PyObject *
pylong_multiply(PyObject *a, PyObject *b)
{
    return PyLong_Type.tp_as_number->nb_multiply(a, b);
}

static inline PyObject *
pylong_floor_divide(PyObject *a, PyObject *b)
{
    return PyLong_Type.tp_as_number->nb_floor_divide(a, b);
}

static inline PyObject *
pylong_remainder(PyObject *a, PyObject *b)
{
    return PyLong_Type.tp_as_number->nb_remainder(a, b);
}

static inline PyObject *
pylong_negative(PyObject *a)
{
    return PyLong_Type.tp_as_number->nb_negative(a);
}

static inline int
pylong_lt(PyObject *a, PyObject *b)
{
    PyObject *res = PyLong_Type.tp_richcompare(a, b, Py_LT);
    if (res == NULL) {
        return -1;
    }
    assert(res == Py_True || res == Py_False);
    return (res == Py_True);
}

static inline int
pylong_bool(PyObject *x)
{
    int res = PyLong_Type.tp_as_number->nb_bool(x);
    assert(res >= 0);
    return res;
}

static inline PyObject *
pylong_repr(PyObject *x)
{
    assert(PyLong_CheckExact(x));
    return PyLong_Type.tp_repr(x);
}

static inline PyObject *
PyLong_FromIntptr(intptr_t x)
{
    static_assert(sizeof(intptr_t) == sizeof(Py_ssize_t), "bad C type sizes");
    return PyLong_FromSsize_t(x);
}


/*********************************************************************/
/* Compiler-specific magic to detect overflows                       */
/*********************************************************************/


#if !defined(_MSC_VER) && !defined(__GNUC__)
# error "Unsupported compiler"
#endif

#ifdef __GNUC__
# if INTPTR_WIDTH == LONG_WIDTH
#  define PyLong_AsIntptrAndOverflow PyLong_AsLongAndOverflow
#  define intptr_add_overflow __builtin_saddl_overflow
#  define intptr_sub_overflow __builtin_ssubl_overflow
#  define intptr_mul_overflow __builtin_smull_overflow
# elif INTPTR_WIDTH == LLONG_WIDTH
#  define PyLong_AsIntptrAndOverflow PyLong_AsLongLongAndOverflow
#  define intptr_add_overflow __builtin_saddll_overflow
#  define intptr_sub_overflow __builtin_ssubll_overflow
#  define intptr_mul_overflow __builtin_smulll_overflow
# else
#  error "INTPTR_WIDTH didn't make sense"
# endif
#endif

#ifdef _MSC_VER
# if _WIN64
# define PyLong_AsIntptrAndOverflow PyLong_AsLongLongAndOverflow

static inline bool
intptr_add_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    bool err = (b < 0) ? (a < INTPTR_MIN - b)
                       : (a > INTPTR_MAX - b);
    if (!err) {
        *res = a + b;
    }
    return err;
}

static inline bool
intptr_sub_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    bool err = (b < 0) ? (a > INTPTR_MAX + b)
                       : (a < INTPTR_MIN + b);
    if (!err) {
        *res = a - b;
    }
    return err;
}

static inline bool
intptr_mul_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "bad C type sizes");
    intptr_t high_word = __mulh(a, b);
    uintptr_t low_word = ((uintptr_t)a) * ((uintptr_t)b);
    bool negative = (bool)(low_word >> 63);
    if (high_word == 0) {
        if (negative) {
            return true;
        }
    } else if (high_word == -1) {
        if (!negative) {
            return true;
        }
    } else {
        return true;
    }
    *res = a * b;
    return false;
}

# elif _WIN32
#  define PyLong_AsIntptrAndOverflow PyLong_AsLongAndOverflow

static inline bool
intptr_add_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    int64_t aa = a;
    int64_t bb = b;
    int64_t cc = aa + bb;
    if (INTPTR_MIN <= cc && cc <= INTPTR_MAX) {
        *res = (intptr_t)cc;
        return false;
    }
    return true;
}

static inline bool
intptr_sub_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    int64_t aa = a;
    int64_t bb = b;
    int64_t cc = aa-bb;
    if (INTPTR_MIN <= cc && cc <= INTPTR_MAX) {
        *res = (intptr_t)cc;
        return false;
    }
    return true;
}

static inline bool
intptr_mul_overflow(intptr_t a, intptr_t b, intptr_t *res)
{
    int64_t aa = a;
    int64_t bb = b;
    int64_t cc = aa*bb;
    if (INTPTR_MIN <= cc && cc <= INTPTR_MAX) {
        *res = (intptr_t)cc;
        return false;
    }
    return true;
}
# else
#  error "_MSC_VER defined but neither _WIN32 nor _WIN64 are"
# endif
#endif

/*********************************************************************/
/* TagInt is a tagged union of:                                      */
/*     * signed intptr_t, packed inline with a 0 LSB appended        */
/*     * PyLongObject pointer, with a 1 LSB set                      */
/*********************************************************************/

typedef struct {
    intptr_t bits;
} TagInt;

static inline bool
TagInt_is_pointer(TagInt t)
{
    return (t.bits & 1);
}

static inline TagInt
_tag_pointer(PyObject *obj)
{
    static_assert(_Alignof(PyObject) % 2 == 0, "bad C type alignment");
    assert((((intptr_t)(void *)obj) & 1) == 0);
    return (TagInt) { .bits = ((intptr_t)(void *)obj) | 1 };
}

static inline PyObject *
untag_pointer(TagInt t)
{
    assert(TagInt_is_pointer(t));
    return (PyObject *)(void *)(t.bits & ~1);
}

static inline bool
is_packable_int(intptr_t x)
{
    return (INTPTR_MIN/2 <= x) && (x <= INTPTR_MAX/2);
}

static inline TagInt
pack_integer(intptr_t x)
{
    assert(is_packable_int(x));
    return (TagInt) {.bits = x * 2};
}

static inline intptr_t
unpack_integer(TagInt t)
{
    assert(!TagInt_is_pointer(t));
    return t.bits / 2;
}

#define TagInt_ONE ((TagInt) {.bits = 2})
#define TagInt_ZERO ((TagInt) {.bits = 0})

static inline bool
TagInt_is_zero(TagInt t) {
    return (t.bits == 0);
}

static inline bool
TagInt_is_one(TagInt t)
{
    return (t.bits == 2);
}

static inline bool
TagInt_is_negative_one(TagInt t)
{
    return (t.bits == -2);
}

// steals the obj reference to put in *t
static bool
object_to_TagInt_steal(PyObject *obj, TagInt *t)
{
    int overflow;
    intptr_t L = PyLong_AsIntptrAndOverflow(obj, &overflow);
    if (L == -1 && PyErr_Occurred()) {
        Py_DECREF(obj);
        return true;
    }
    if (overflow == 0 && is_packable_int(L)) {
        Py_DECREF(obj);
        *t = pack_integer(L);
        return false;
    }
    *t = _tag_pointer(obj);
    return false;
}

static inline TagInt
TagInt_copy(TagInt t)
{
    if (TagInt_is_pointer(t)) {
        Py_INCREF(untag_pointer(t));
    }
    return t;
}

static inline void
TagInt_clear(TagInt *t)
{
    TagInt tt = *t;
    t->bits = 0;
    if (TagInt_is_pointer(tt)) {
        Py_DECREF(untag_pointer(tt));
    }
}

static inline void
TagInt_setref(TagInt *t, TagInt val)
{
    TagInt tt = *t;
    t->bits = val.bits;
    if (TagInt_is_pointer(tt)) {
        Py_DECREF(untag_pointer(tt));
    }
}

static PyObject *
TagInt_to_object(TagInt t)
{
    if (TagInt_is_pointer(t)) {
        return Py_NewRef(untag_pointer(t));
    } else {
        return PyLong_FromIntptr(unpack_integer(t));
    }
}

// Puts a new reference in *c.
static bool
_TagInt_add_with_objects(TagInt a, TagInt b, TagInt *c)
{
    if (TagInt_is_zero(a)) {
        *c = TagInt_copy(b);
        return false;
    } else if (TagInt_is_zero(b)) {
        *c = TagInt_copy(a);
        return false;
    }
    PyObject *a_obj = TagInt_to_object(a);
    if (a_obj == NULL) {
        return true;
    }
    PyObject *b_obj = TagInt_to_object(b);
    if (b_obj == NULL) {
        Py_DECREF(a_obj);
        return true;
    }
    PyObject *c_obj = pylong_add(a_obj, b_obj);
    Py_DECREF(a_obj);
    Py_DECREF(b_obj);
    if (c_obj == NULL) {
        return true;
    }
    return object_to_TagInt_steal(c_obj, c);
}

static inline bool
TagInt_add(TagInt a, TagInt b, TagInt *c)
{
    if (!TagInt_is_pointer(a) && !TagInt_is_pointer(b)) {
        // Speed hack:
        // pack(unpack(a) + unpack(b)).bits
        // = 2*(a.bits/2 + b.bits/2)
        // = a.bits + b.bits
        if (!intptr_add_overflow(a.bits, b.bits, &c->bits)) {
            return false;
        }
    }
    // separate out this function to not inline.
    return _TagInt_add_with_objects(a, b, c);
}

static bool
_TagInt_negative_with_objects(TagInt a, TagInt *res)
{
    PyObject *a_obj = TagInt_to_object(a);
    if (a_obj == NULL) {
        return true;
    }
    PyObject *neg_a = pylong_negative(a_obj);
    Py_DECREF(a_obj);
    if (neg_a == NULL) {
        return true;
    }
    return object_to_TagInt_steal(neg_a, res);
}

static inline bool
TagInt_negative(TagInt a, TagInt *res)
{
    if (!TagInt_is_pointer(a) && a.bits != INTPTR_MIN) {
        // Speed hack:
        // pack(-unpack(a)).bits = 2*(-(a.bits/2)) = -a.bits
        res->bits = -a.bits;
        return false;
    }
    return _TagInt_negative_with_objects(a, res);
}

static inline int
TagInt_is_negative(TagInt a, PyObject *zero)
{
    if (!TagInt_is_pointer(a)) {
        // Speed hack:
        // unpack(a) < 0
        // iff a.bits/2 < 0
        // iff a.bits < 0
        return a.bits < 0;
    }
    return pylong_lt(untag_pointer(a), zero);
}

// Puts a new reference in *c.
static bool
_TagInt_sub_with_objects(TagInt a, TagInt b, TagInt *c)
{
    if (TagInt_is_zero(b)) {
        *c = TagInt_copy(a);
        return false;
    }
    PyObject *a_obj = TagInt_to_object(a);
    if (a_obj == NULL) {
        return true;
    }
    PyObject *b_obj = TagInt_to_object(b);
    if (b_obj == NULL) {
        Py_DECREF(a_obj);
        return true;
    }
    PyObject *c_obj = pylong_subtract(a_obj, b_obj);
    Py_DECREF(a_obj);
    Py_DECREF(b_obj);
    if (c_obj == NULL) {
        return true;
    }
    return object_to_TagInt_steal(c_obj, c);
}

static inline bool
TagInt_sub(TagInt a, TagInt b, TagInt *c)
{
    if (!TagInt_is_pointer(a) && !TagInt_is_pointer(b)) {
        // Speed hack:
        // pack(unpack(a) - unpack(b)).bits
        // = 2*(a.bits/2 - b.bits/2)
        // = a.bits - b.bits
        if (!intptr_sub_overflow(a.bits, b.bits, &c->bits)) {
            return false;
        }
    }
    // separate out this function to not inline.
    return _TagInt_sub_with_objects(a, b, c);
}

// Puts a new reference in *c.
static bool
TagInt_scale_with_objects(
    TagInt a, PyObject *m_obj, TagInt *c)
{
    if (TagInt_is_zero(a)) {
        (*c) = TagInt_ZERO;
        return false;
    }
    if (TagInt_is_one(a)) {
        return object_to_TagInt_steal(Py_NewRef(m_obj), c);
    }
    PyObject *a_obj = TagInt_to_object(a);
    if (a_obj == NULL) {
        return true;
    }
    PyObject *c_obj = pylong_multiply(a_obj, m_obj);
    Py_DECREF(a_obj);
    if (c_obj == NULL) {
        return true;
    }
    return object_to_TagInt_steal(c_obj, c);
}

static inline bool
TagInt_scale(
    TagInt a, intptr_t m, PyObject *m_obj, TagInt *c)
{
    if (!TagInt_is_pointer(a)) {
        // Speed hack:
        // pack(m*unpack(a))).bits
        // = 2*(m*a.bits/2)
        // = m*a.bits
        if (!intptr_mul_overflow(a.bits, m, &c->bits)) {
            return false;
        }
    }
    // separate this function to not inline
    return TagInt_scale_with_objects(a, m_obj, c);
}

// Do w += k*v in place
static bool
TagInt_row_op_with_objects(
    TagInt *v, TagInt *w, PyObject *k_obj)
{
    PyObject *v_obj = TagInt_to_object(*v);
    if (v_obj == NULL) {
        return true;
    }
    PyObject *kv_obj = pylong_multiply(v_obj, k_obj);
    Py_DECREF(v_obj);
    if (kv_obj == NULL) {
        return true;
    }
    PyObject *w_obj = TagInt_to_object(*w);
    if (w_obj == NULL) {
        Py_DECREF(kv_obj);
        return true;
    }
    PyObject *kv_plus_w_obj = pylong_add(kv_obj, w_obj);
    Py_DECREF(kv_obj);
    Py_DECREF(w_obj);
    if (kv_plus_w_obj == NULL) {
        return true;
    }
    TagInt_clear(w);
    return object_to_TagInt_steal(kv_plus_w_obj, w);
}

static inline bool
TagInt_row_op(
    TagInt *v, TagInt *w, intptr_t k, PyObject **k_obj_cache)
{
    if (!TagInt_is_pointer(*v) && !TagInt_is_pointer(*w)) {
        // Speed hack:
        // pack(unpack(w)+k*unpack(v)).bits
        // = 2*(w/2 + k*v/2)
        // = w + k*v
        intptr_t v2 = v->bits;
        intptr_t kv2;
        if (!intptr_mul_overflow(k, v2, &kv2)) {
            intptr_t w2 = w->bits;
            intptr_t kv2_plus_w2;
            if (!intptr_add_overflow(w2, kv2, &kv2_plus_w2)) {
                w->bits = kv2_plus_w2;
                return false;
            }
        }
    }
    if (*k_obj_cache == NULL) {
        *k_obj_cache = PyLong_FromIntptr(k);
        if (*k_obj_cache == NULL) {
            return false;
        }
    }
    // separate this function to not inline
    return TagInt_row_op_with_objects(v, w, *k_obj_cache);
}

// Do (v, w) = (av+bw, cv+dw)
static bool
TagInt_generalized_row_op_with_objects(
    TagInt *v, TagInt *w, PyObject *const *abcd_obj)
{
    PyObject *a_obj = abcd_obj[0], *b_obj = abcd_obj[1];
    PyObject *c_obj = abcd_obj[2], *d_obj = abcd_obj[3];
    PyObject *v_obj = NULL, *w_obj = NULL;
    PyObject *av = NULL, *bw = NULL, *av_bw = NULL;
    PyObject *cv = NULL, *dw = NULL, *cv_dw = NULL;
    TagInt new_v = TagInt_ZERO, new_w = TagInt_ZERO;

    if (!(v_obj = TagInt_to_object(*v))) { goto error; }
    if (!(w_obj = TagInt_to_object(*w))) { goto error; }
    if (!(av = pylong_multiply(a_obj, v_obj))) { goto error; }
    if (!(bw = pylong_multiply(b_obj, w_obj))) { goto error; }
    if (!(cv = pylong_multiply(c_obj, v_obj))) { goto error; }
    if (!(dw = pylong_multiply(d_obj, w_obj))) { goto error; }
    if (!(av_bw = pylong_add(av, bw))) { goto error; }
    if (!(cv_dw = pylong_add(cv, dw))) { goto error; }

    if (object_to_TagInt_steal(Py_NewRef(av_bw), &new_v)) { goto error; }
    if (object_to_TagInt_steal(Py_NewRef(cv_dw), &new_w)) { goto error; }

    TagInt_setref(v, new_v);
    TagInt_setref(w, new_w);

    Py_DECREF(v_obj); Py_DECREF(w_obj);
    Py_DECREF(av); Py_DECREF(bw); Py_DECREF(av_bw);
    Py_DECREF(cv); Py_DECREF(dw); Py_DECREF(cv_dw);
    return false;

error:
    Py_XDECREF(v_obj); Py_XDECREF(w_obj);
    Py_XDECREF(av); Py_XDECREF(bw); Py_XDECREF(av_bw);
    Py_XDECREF(cv); Py_XDECREF(dw); Py_XDECREF(cv_dw);
    TagInt_clear(&new_v); TagInt_clear(&new_w);
    return true;
}

static bool
TagInt_generalized_row_op(
    TagInt *v, TagInt *w, intptr_t *abcd, PyObject **abcd_obj)
{
    intptr_t av, bw, av_bw, cv, dw, cv_dw;
    if (TagInt_is_pointer(*v) || TagInt_is_pointer(*w)) { goto use_objects; }
    // Speed hack:
    // pack(a*unpack(v) + b*unpack(w)).bits
    // = 2*(a*v/2 + b*v/2)
    // = a*v.bits + b*w.bits
    // And similar for c*v+d*w
    if (intptr_mul_overflow(abcd[0], v->bits, &av)) { goto use_objects; }
    if (intptr_mul_overflow(abcd[1], w->bits, &bw)) { goto use_objects; }
    if (intptr_mul_overflow(abcd[2], v->bits, &cv)) { goto use_objects; }
    if (intptr_mul_overflow(abcd[3], w->bits, &dw)) { goto use_objects; }
    if (intptr_add_overflow(av, bw, &av_bw)) { goto use_objects; }
    if (intptr_add_overflow(cv, dw, &cv_dw)) { goto use_objects; }
    v->bits = av_bw;
    w->bits = cv_dw;
    return false;
use_objects:
    if (abcd_obj[0] == NULL) {
        for (int i = 0; i < 4; i++) {
            if (!(abcd_obj[i] = PyLong_FromIntptr(abcd[i]))) {
                for (i = i - 1; i >= 0; i--) {
                    Py_CLEAR(abcd_obj[i]);
                }
                return true;
            }
        }
    }
    return TagInt_generalized_row_op_with_objects(v, w, abcd_obj);
}

/*********************************************************************/
/* The Vector() type                                                 */
/*   A simple vector of TagInt.                                      */
/*********************************************************************/

static PyTypeObject Vector_Type;

typedef struct {
    PyObject_VAR_HEAD
    TagInt vec[1]; // No need for a separate buffer.
} Vec;

static inline TagInt *
Vector_get_vec(PyObject *v)
{
    assert(Py_TYPE(v) == &Vector_Type);
    return ((Vec *)v)->vec;
}

static Py_ssize_t
Vector_sq_length(PyObject *self)
{
    return Py_SIZE(self);
}

static PyObject *
Vector_sq_item(PyObject *self, Py_ssize_t j)
{
    Py_ssize_t N = Py_SIZE(self);
    if (!(0 <= j && j < N)) {
        PyErr_SetString(PyExc_IndexError, "Vector index out of range");
        return NULL;
    }
    TagInt *vec = Vector_get_vec(self);
    return TagInt_to_object(vec[j]);
}

static int
Vector_sq_ass_item(PyObject *self, Py_ssize_t j, PyObject *xo)
{
    Py_ssize_t N = Py_SIZE(self);
    if (!(0 <= j && j < N)) {
        PyErr_SetString(PyExc_IndexError, "Vector index out of range");
        return -1;
    }
    TagInt t;
    Py_INCREF(xo);
    if (object_to_TagInt_steal(xo, &t)) {
        Py_DECREF(xo);
        return -1;
    }
    TagInt *vec = Vector_get_vec(self);
    TagInt_setref(&vec[j], t);
    return 0;
}

static void
Vector_clear(PyObject *self)
{
    TagInt *self_vec = Vector_get_vec(self);
    for (Py_ssize_t i = 0; i < Py_SIZE(self); i++) {
        TagInt_clear(&self_vec[i]);
    }
}

static void
Vector_dealloc(PyObject *self)
{
    Vector_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *
Vector_zero_impl(Py_ssize_t N)
{
    return PyType_GenericAlloc(&Vector_Type, N);
}

static PyObject *
Vector_zero(PyTypeObject *type, PyObject *n_obj)
{
    Py_ssize_t N = PyLong_AsSsize_t(n_obj);
    if (N == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (N < 0) {
        PyErr_SetString(PyExc_ValueError, "Vector.zero() argument must be nonnegative");
        return NULL;
    }
    return Vector_zero_impl(N);
}

static PyObject *
Vector_new_impl(PyObject *data)
{
    if (!PyList_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "Vector() argument must be list");
        return NULL;
    }
    Py_ssize_t N = PyList_GET_SIZE(data);
    PyObject *self = PyType_GenericAlloc(&Vector_Type, N);
    if (self == NULL) {
        return NULL;
    }
    TagInt *self_vec = Vector_get_vec(self);
    for (Py_ssize_t i = 0; i < N && i < PyList_GET_SIZE(data); i++) {
        PyObject *x = PyList_GET_ITEM(data, i);
        if (!PyLong_CheckExact(x)) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_TypeError, "Vector() argument must be a list of int");
            return NULL;
        }
        if (object_to_TagInt_steal(Py_NewRef(x), &self_vec[i])) {
            Py_DECREF(self);
            return NULL;
        }
    }
    return self;
}

static PyObject *
Vector_new(PyTypeObject *Py_UNUSED(type), PyObject *args, PyObject *kwds)
{
    if (kwds != NULL && PyDict_GET_SIZE(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "Vector() takes no keyword arguments.");
        return NULL;
    }
    if (PyTuple_GET_SIZE(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "Vector() takes exactly one argument.");
        return NULL;
    }
    PyObject *data = PyTuple_GET_ITEM(args, 0);
    return Vector_new_impl(data);
}

static PyObject *
Vector_from_TagInts(TagInt *t, Py_ssize_t N)
{
    PyObject *result = Vector_zero_impl(N);
    if (result == NULL) {
        return NULL;
    }
    TagInt *vec = Vector_get_vec(result);
    for (Py_ssize_t j = 0; j < N; j++) {
        vec[j] = TagInt_copy(t[j]);
    }
    return result;
}

static PyObject *
Vector_copy(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    assert(Py_TYPE(self) == &Vector_Type);
    return Vector_from_TagInts(Vector_get_vec(self), Py_SIZE(self));
}

static bool
Vector_iadd_impl(TagInt *self_vec, TagInt *other_vec, Py_ssize_t N)
{
    for (Py_ssize_t i = 0; i < N; i++) {
        if (TagInt_is_zero(other_vec[i])) {
            continue;
        }
        TagInt t;
        if (TagInt_add(self_vec[i], other_vec[i], &t)) {
            return true;
        }
        TagInt_setref(&self_vec[i], t);
    }
    return false;
}

static PyObject *
Vector_iadd(PyObject *self, PyObject *other)
{
    if (Py_TYPE(other) != &Vector_Type || Py_TYPE(self) != &Vector_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Py_ssize_t N = Py_SIZE(self);
    if (N != Py_SIZE(other)) {
        PyErr_SetString(PyExc_ValueError, "size mismatch for Vector addition");
        return NULL;
    }
    if (Vector_iadd_impl(Vector_get_vec(self), Vector_get_vec(other), N)) {
        return NULL;
    }
    return Py_NewRef(self);
}

static PyObject *
Vector_add(PyObject *self, PyObject *other)
{
    PyObject *copy = Vector_copy(self, NULL);
    if (copy == NULL) {
        return NULL;
    }
    PyObject *result = Vector_iadd(copy, other);
    Py_DECREF(copy);
    return result;
}

static bool
Vector_isub_impl(TagInt *self_vec, TagInt *other_vec, Py_ssize_t N)
{
    for (Py_ssize_t i = 0; i < N; i++) {
        if (TagInt_is_zero(other_vec[i])) {
            continue;
        }
        TagInt t;
        if (TagInt_sub(self_vec[i], other_vec[i], &t)) {
            return true;
        }
        TagInt_setref(&self_vec[i], t);
    }
    return false;
}

static PyObject *
Vector_isub(PyObject *self, PyObject *other)
{
    if (Py_TYPE(other) != &Vector_Type || Py_TYPE(self) != &Vector_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Py_ssize_t N = Py_SIZE(self);
    if (N != Py_SIZE(other)) {
        PyErr_SetString(PyExc_ValueError, "size mismatch for Vector subtraction");
        return NULL;
    }
    if (Vector_isub_impl(Vector_get_vec(self), Vector_get_vec(other), N)) {
        return NULL;
    }
    return Py_NewRef(self);
}

static PyObject *
Vector_sub(PyObject *self, PyObject *other)
{
    PyObject *copy = Vector_copy(self, NULL);
    if (copy == NULL) {
        return NULL;
    }
    PyObject *result = Vector_isub(copy, other);
    Py_DECREF(copy);
    return result;
}

static PyObject *
Vector_imul(PyObject *self, PyObject *other)
{
    if (!PyLong_Check(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    TagInt *vec = Vector_get_vec(self);
    int overflow;
    intptr_t k = PyLong_AsIntptrAndOverflow(other, &overflow);
    if (k == -1 && PyErr_Occurred()) {
        return NULL;
    } else if (overflow) {
        for (Py_ssize_t i = 0; i < Py_SIZE(self); i++) {
            if (TagInt_is_zero(vec[i])) {
                continue;
            }
            TagInt t;
            if (TagInt_scale_with_objects(vec[i], other, &t)) {
                return NULL;
            }
            TagInt_setref(&vec[i], t);
        }
    } else if (k == 0) {
        Vector_clear(self);
    } else if (k == 1) {
        ; // do nothing
    } else {
        for (Py_ssize_t i = 0; i < Py_SIZE(self); i++) {
            if (TagInt_is_zero(vec[i])) {
                continue;
            }
            TagInt t;
            if (TagInt_scale(vec[i], k, other, &t)) {
                return NULL;
            }
            TagInt_setref(&vec[i], t);
        }
    }
    return Py_NewRef(self);
}

static PyObject *
Vector_mul(PyObject *self, PyObject *other)
{
    if (Py_TYPE(self) == &Vector_Type && PyLong_Check(other)) {
        ; // arguments as expected
    }
    else if (PyLong_Check(self) && Py_TYPE(other) == &Vector_Type) {
        PyObject *temp = self;
        self = other;
        other = temp;
    }
    else {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyObject *copy = Vector_copy(self, NULL);
    if (copy == NULL) {
        return NULL;
    }
    PyObject *result = Vector_imul(copy, other);
    Py_DECREF(copy);
    return result;
}

static bool
Vector_negate_impl(TagInt *vec, Py_ssize_t n)
{
    for (Py_ssize_t j = 0; j < n; j++) {
        if (!TagInt_is_zero(vec[j])) {
            TagInt t;
            if (TagInt_negative(vec[j], &t)) {
                return true;
            }
            TagInt_setref(&vec[j], t);
        }
    }
    return false;
}

static PyObject *
Vector_negative(PyObject *self)
{
    PyObject *copy = Vector_copy(self, NULL);
    if (copy == NULL) {
        return NULL;
    }
    if (Vector_negate_impl(Vector_get_vec(copy), Py_SIZE(copy))) {
        Py_DECREF(copy);
        return NULL;
    }
    return copy;
}

static PyObject *
Vector_tolist_impl(TagInt *vec, Py_ssize_t N)
{
    PyObject *result = PyList_New(N);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < N; i++) {
        PyObject *obj = TagInt_to_object(vec[i]);
        if (obj == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, obj);
    }
    return result;
}

static PyObject *
Vector_tolist(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    return Vector_tolist_impl(Vector_get_vec(self), Py_SIZE(self));
}

static PyObject *
Vector_richcompare(PyObject *a, PyObject *b, int op)
{
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (Py_TYPE(a) != &Vector_Type || Py_TYPE(b) != &Vector_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Py_ssize_t N = Py_SIZE(a);
    if (Py_SIZE(b) != N) {
        goto unequal;
    }
    TagInt *vec_a = Vector_get_vec(a);
    TagInt *vec_b = Vector_get_vec(b);
    for (Py_ssize_t i = 0; i < N; i++) {
        TagInt a = vec_a[i], b = vec_b[i];
        if (a.bits != b.bits) {
            if (!TagInt_is_pointer(a) && !TagInt_is_pointer(b)) {
                goto unequal;
            }
            if (TagInt_is_pointer(a) != TagInt_is_pointer(b)) {
                // assumes everything that can be packed is always packed.
                goto unequal;
            }
            int cmp = PyObject_RichCompareBool(untag_pointer(a), untag_pointer(b), Py_NE);
            if (cmp == -1) {
                return NULL;
            }
            if (cmp) {
                goto unequal;
            }
        }
    }
    // equal
    if (op == Py_NE) {
        Py_RETURN_FALSE;
    } else {
        Py_RETURN_TRUE;
    }
unequal:
    if (op == Py_NE) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static Py_ssize_t
Vector_num_pointers(TagInt *x, Py_ssize_t N)
{
    Py_ssize_t result = 0;
    for (Py_ssize_t i = 0; i < N; i++) {
        if (TagInt_is_pointer(x[i])) {
            result++;
        }
    }
    return result;
}

static PyObject *
Vector__num_bigints(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(Vector_num_pointers(Vector_get_vec(self), Py_SIZE(self)));
}

static PyObject *
Vector_shuffled_by_action_impl(PyObject *self, PyObject *other, Py_ssize_t result_N)
{
    PyObject *result = Vector_zero_impl(result_N);
    if (result == NULL) {
        return NULL;
    }
    Py_ssize_t N = Py_SIZE(other);
    TagInt *source = Vector_get_vec(self);
    TagInt *dest = Vector_get_vec(result);
    TagInt *action = Vector_get_vec(other);
    for (Py_ssize_t j = 0; j < N; j++) {
        if (!TagInt_is_zero(source[j])) {
            TagInt ajt = action[j];
            if (TagInt_is_pointer(ajt)) {
                PyErr_SetString(PyExc_IndexError, "shuffle out of bounds (got a big integer)");
                goto error;
            }
            intptr_t aj = unpack_integer(ajt);
            if (!(0 <= aj && aj < result_N)) {
                PyErr_SetString(PyExc_IndexError, "shuffle out of bounds");
                goto error;
            }
            TagInt t;
            if (TagInt_add(dest[aj], source[j], &t)) {
                goto error;
            }
            TagInt_setref(&dest[aj], t);
        }
    }
    return result;
error:
    Py_DECREF(result);
    return NULL;
}

static PyObject *
Vector_shuffled_by_action(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    assert(Py_TYPE(self) == &Vector_Type);
    Py_ssize_t result_N;
    if (nargs == 1) {
        result_N = Py_SIZE(self);
    } else if (nargs == 2) {
        if (!PyLong_CheckExact(args[1])) {
            PyErr_SetString(PyExc_TypeError, "v.shuffled_by_action(w, result_length) second argument must be an int if present");
            return NULL;
        }
        result_N = PyLong_AsSsize_t(args[1]);
        if (result_N == -1 && PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "v.shuffled_by_action(w[, result_length]) takes 1 or 2 arguments");
        return NULL;
    }
    PyObject *other = args[0];
    if (Py_TYPE(other) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "v.shuffled_by_action(w) first argument must be another Vector");
        return NULL;
    }
    if (Py_SIZE(other) != Py_SIZE(self)) {
        PyErr_SetString(PyExc_ValueError, "v.shuffled_by_action(w) length mismatch");
        return NULL;
    }
    return Vector_shuffled_by_action_impl(self, other, result_N);
}

static PyObject *
Vector_str_parts(TagInt *vec, Py_ssize_t N, Py_ssize_t *max_width)
{
    PyObject *result = PyList_New(N);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        PyObject *x = TagInt_to_object(vec[j]);
        if (x == NULL) {
            goto error;
        }
        PyObject *r = pylong_repr(x);
        Py_DECREF(x);
        if (r == NULL) {
            goto error;
        }
        Py_ssize_t w = PyUnicode_GET_LENGTH(r);
        if (w > *max_width) {
            *max_width = w;
        }
        PyList_SET_ITEM(result, j, r);
    }
    return result;
error:
    Py_DECREF(result);
    return NULL;
}

static PyObject *
Vector_str(PyObject *v)
{
    Py_ssize_t maxlen = 0;
    PyObject *reprs = Vector_str_parts(Vector_get_vec(v), Py_SIZE(v), &maxlen);
    if (reprs == NULL) {
        return NULL;
    }
    Py_ssize_t max_width = 0;
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(reprs); j++) {
        PyObject *r = PyList_GET_ITEM(reprs, j);
        Py_ssize_t w = PyUnicode_GET_LENGTH(r);
        if (w > max_width) {
            max_width = w;
        }
    }
    PyObject *empty=NULL, *space=NULL, *lbracket=NULL, *rbracket=NULL;
    PyObject *parts=NULL, *result=NULL;
    if (!(empty = PyUnicode_FromStringAndSize("", 0))) {
        goto error;
    }
    if (!(space = PyUnicode_FromStringAndSize(" ", 1))) {
        goto error;
    }
    if (!(lbracket = PyUnicode_FromStringAndSize("[", 1))) {
        goto error;
    }
    if (!(rbracket = PyUnicode_FromStringAndSize("]", 1))) {
        goto error;
    }
    if (!(parts = PyList_New(0))) {
        goto error;
    }
    if (PyList_Append(parts, lbracket) < 0) {
        goto error;
    }
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(reprs); j++) {
        PyObject *r = PyList_GET_ITEM(reprs, j);
        Py_ssize_t num_spaces = max_width + (j > 0) - PyUnicode_GET_LENGTH(r);
        while (num_spaces) {
            if (PyList_Append(parts, space) < 0) {
                goto error;
            }
            num_spaces--;
        }
        if (PyList_Append(parts, r) < 0) {
            goto error;
        }
    }
    if (PyList_Append(parts, rbracket) < 0) {
        goto error;
    }
    result = PyUnicode_Join(empty, parts);
error:
    Py_DECREF(reprs);
    Py_XDECREF(empty);
    Py_XDECREF(space);
    Py_XDECREF(lbracket);
    Py_XDECREF(rbracket);
    Py_XDECREF(parts);
    return result;
}

static PyObject *
Vector_repr(PyObject *self)
{
    PyObject *list = Vector_tolist(self, NULL);
    if (list == NULL) {
        return NULL;
    }
    PyObject *list_repr = PyObject_Repr(list);
    Py_DECREF(list);
    if (list_repr == NULL) {
        return NULL;
    }
    PyObject *start=NULL, *end=NULL, *empty=NULL;
    PyObject *parts=NULL, *result=NULL;
    if (!(start = PyUnicode_FromStringAndSize("Vector(", 7))) {
        goto error;
    }
    if (!(end = PyUnicode_FromStringAndSize(")", 1))) {
        goto error;
    }
    if (!(empty = PyUnicode_FromStringAndSize("", 0))) {
        goto error;
    }
    if (!(parts = PyTuple_Pack(3, start, list_repr, end))) {
        goto error;
    }
    result = PyUnicode_Join(empty, parts);
error:
    Py_DECREF(list_repr);
    Py_XDECREF(start);
    Py_XDECREF(end);
    Py_XDECREF(empty);
    Py_XDECREF(parts);
    return result;

}

static PyMethodDef Vector_methods[] = {
    {"tolist", (PyCFunction)Vector_tolist, METH_NOARGS,
        "Covert to list of int"},
    {"copy", (PyCFunction)Vector_copy, METH_NOARGS,
        "Make a new copy of this Vector"},
    {"zero", (PyCFunction)Vector_zero, METH_O | METH_CLASS,
        "Make a zero Vector of the given size"},
    {"_num_bigints", (PyCFunction)Vector__num_bigints, METH_NOARGS,
        "Count the number of boxed integers in this Vector"},
    {"shuffled_by_action", (PyCFunction)(void(*)(void))Vector_shuffled_by_action, METH_FASTCALL,
        "v.shuffled_by_action(a) does for each i: result[a[i]] += v[i]"},
    {NULL}
};

static PyNumberMethods Vector_as_number = {
    .nb_add = Vector_add,
    .nb_subtract = Vector_sub,
    .nb_multiply = Vector_mul,
    .nb_inplace_add = Vector_iadd,
    .nb_inplace_subtract = Vector_isub,
    .nb_inplace_multiply = Vector_imul,
    .nb_negative = Vector_negative,
};

static PySequenceMethods Vector_as_sequence = {
    .sq_length = Vector_sq_length,
    .sq_item = Vector_sq_item,
    .sq_ass_item = Vector_sq_ass_item,
};

static PyTypeObject Vector_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mutable_lattice.Vector",
    .tp_doc = PyDoc_STR("Vector mixing machine integers and bigints"),
    .tp_basicsize = sizeof(Vec) - sizeof(TagInt),
    .tp_itemsize = sizeof(TagInt),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Vector_new,
    .tp_dealloc = (destructor)Vector_dealloc,
    .tp_methods = Vector_methods,
    .tp_as_number = &Vector_as_number,
    .tp_as_sequence = &Vector_as_sequence,
    .tp_richcompare = Vector_richcompare,
    .tp_str = Vector_str,
    .tp_repr = Vector_repr,
};

/*********************************************************************/
/* Row operations                                                    */
/*********************************************************************/

static bool
row_op_impl_with_objects(TagInt *v, TagInt *w, Py_ssize_t n, PyObject *k_obj)
{
    // w[0:n] += k * v[0:n]
    for (Py_ssize_t i = 0; i < n; i++) {
        if (TagInt_is_zero(v[i])) {
            continue;
        }
        if (TagInt_row_op_with_objects(&v[i], &w[i], k_obj)) {
            return true;
        }
    }
    return false;
}

static bool
row_op_impl_with_intptr(TagInt *v, TagInt *w, Py_ssize_t n, intptr_t k, PyObject **k_obj_cache)
{
    // w[0:n] += k * v[0:n]
    if (k == 0) {
        return false;
    } else if (k == 1) {
        return Vector_iadd_impl(w, v, n);
    } else if (k == -1) {
        return Vector_isub_impl(w, v, n);
    } else {
        for (Py_ssize_t i = 0; i < n; i++) {
            if (TagInt_is_zero(v[i])) {
                continue;
            }
            if (TagInt_row_op(&v[i], &w[i], k, k_obj_cache)) {
                return true;
            }
        }
        return false;
    }
}

static bool
row_op_impl(TagInt *v, TagInt *w, Py_ssize_t n, PyObject *k_obj)
{
    int overflow;
    intptr_t k = PyLong_AsIntptrAndOverflow(k_obj, &overflow);
    if (k == -1 && PyErr_Occurred()) {
        return true;
    } else if (overflow) {
        return row_op_impl_with_objects(v, w, n, k_obj);
    } else {
        return row_op_impl_with_intptr(v, w, n, k, &k_obj);
    }
}

static PyObject *
row_op(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "row_op(v, w, k) takes 3 arguments.");
        return NULL;
    }
    PyObject *v = args[0];
    PyObject *w = args[1];
    PyObject *k = args[2];
    if (Py_TYPE(v) != &Vector_Type || Py_TYPE(w) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "First two arguments to row_op(v, w, k) must be Vector");
        return NULL;
    }
    if (Py_SIZE(v) != Py_SIZE(w)) {
        PyErr_SetString(PyExc_ValueError, "row_op vectors must have the same length");
        return NULL;
    }
    if (row_op_impl(Vector_get_vec(v), Vector_get_vec(w), Py_SIZE(v), k)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static bool
generalized_row_op_impl_with_intptr(TagInt *v, TagInt *w, Py_ssize_t n, intptr_t *abcd, PyObject **abcd_obj)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        if (TagInt_is_zero(v[i]) && TagInt_is_zero(w[i])) {
            continue;
        }
        if (TagInt_generalized_row_op(&v[i], &w[i], abcd, abcd_obj)) {
            return true;
        }
    }
    return false;
}

static bool
generalized_row_op_impl_with_objects(TagInt *v, TagInt *w, Py_ssize_t n, PyObject **abcd_obj)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        if (TagInt_is_zero(v[i]) && TagInt_is_zero(w[i])) {
            continue;
        }
        if (TagInt_generalized_row_op_with_objects(&v[i], &w[i], abcd_obj)) {
            return true;
        }
    }
    return false;
}

static bool
generalized_row_op_impl(TagInt *v, TagInt *w, Py_ssize_t n, PyObject **abcd_obj)
{
    intptr_t abcd[4];
    for (int i = 0; i < 4; i++) {
        int overflow;
        abcd[i] = PyLong_AsIntptrAndOverflow(abcd_obj[i], &overflow);
        if (abcd[i] == -1 && PyErr_Occurred()) {
            return true;
        }
        if (overflow) {
            goto use_objects;
        }
    }
    return generalized_row_op_impl_with_intptr(v, w, n, abcd, abcd_obj);
use_objects:
    return generalized_row_op_impl_with_objects(v, w, n, abcd_obj);
}

static PyObject *
generalized_row_op(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "generalized_row_op(v, w, a, b, c, d) takes 6 arguments.");
        return NULL;
    }
    PyObject *v = args[0];
    PyObject *w = args[1];
    if (Py_TYPE(v) != &Vector_Type || Py_TYPE(w) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "First two arguments to generalized_row_op(v, w, a, b, c, d) must be Vector");
        return NULL;
    }
    if (Py_SIZE(v) != Py_SIZE(w)) {
        PyErr_SetString(PyExc_ValueError, "generalized_row_op vectors must have the same length");
        return NULL;
    }
    PyObject *abcd[4] = {args[2], args[3], args[4], args[5]};
    if (generalized_row_op_impl(Vector_get_vec(v), Vector_get_vec(w), Py_SIZE(v), abcd)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/*********************************************************************/
/* xgcd                                                              */
/*********************************************************************/

static bool
xgcd_using_objects(PyObject *a, PyObject *b, PyObject **result)
{
    PyObject *x = NULL, *y = NULL, *g = Py_NewRef(a);
    PyObject *x1 = NULL, *y1 = NULL, *g1 = Py_NewRef(b);
    PyObject *q = NULL;
    PyObject *qx1 = NULL, *qy1 = NULL, *qg1 = NULL;
    PyObject *x2 = NULL, *y2 = NULL, *g2 = NULL;

    // Start with the two equations
    //  1*a + 0*b = a
    //  0*a + 1*b = b
    // Divide a and b back and forth, carrying the rest of
    // the equations along for the ride.
    if (!(x = PyLong_FromLong(1))) { goto error; }
    if (!(y = PyLong_FromLong(0))) { goto error; }
    y1 = Py_NewRef(x);
    x1 = Py_NewRef(y);
    while (pylong_bool(g1)) {
        if (!(q = pylong_floor_divide(g, g1))) { goto error; }
        if (!(qx1 = pylong_multiply(q, x1))) { goto error; }
        if (!(qy1 = pylong_multiply(q, y1))) { goto error; }
        if (!(qg1 = pylong_multiply(q, g1))) { goto error; }
        if (!(x2 = pylong_subtract(x, qx1))) { goto error; }
        if (!(y2 = pylong_subtract(y, qy1))) { goto error; }
        if (!(g2 = pylong_subtract(g, qg1))) { goto error; }
        Py_DECREF(q); q = NULL;
        Py_DECREF(qx1); qx1 = NULL;
        Py_DECREF(qy1); qy1 = NULL;
        Py_DECREF(qg1); qg1 = NULL;
        Py_DECREF(x); x = x1; x1 = x2; x2 = NULL;
        Py_DECREF(y); y = y1; y1 = y2; y2 = NULL;
        Py_DECREF(g); g = g1; g1 = g2; g2 = NULL;
    }
    int lt = pylong_lt(g, g1);
    if (lt == -1) {
        goto error;
    }
    if (lt) {
        PyObject *neg;
        if (!(neg = pylong_negative(g))) { goto error; }
        Py_DECREF(g); g = neg;
        if (!(neg = pylong_negative(x))) { goto error; }
        Py_DECREF(x); x = neg;
        if (!(neg = pylong_negative(y))) { goto error; }
        Py_DECREF(y); y = neg;
    }
    Py_DECREF(x1); Py_DECREF(y1); Py_DECREF(g1);
    result[0] = x; result[1] = y; result[2] = g;
    return false;
error:
    Py_XDECREF(x); Py_XDECREF(y); Py_XDECREF(g);
    Py_XDECREF(x1); Py_XDECREF(y1); Py_XDECREF(g1);
    Py_XDECREF(x2); Py_XDECREF(y2); Py_XDECREF(g2);
    Py_XDECREF(q);
    Py_XDECREF(qx1); Py_XDECREF(qy1); Py_XDECREF(qg1);
    return true;
}

static void
xgcd_using_intptr(intptr_t a, intptr_t b, intptr_t *result)
{
    // Start with the two equations
    //  1*a + 0*b = a
    //  0*a + 1*b = b
    // Divide a and b back and forth, carrying the rest of
    // the equations along for the ride.

    // The code below won't overflow!
    // This fact is included in the results of the nice short paper
    // "Computing Multiplicative Inverses in GF(p)" by George E. Collins.
    //    https://www.jstor.org/stable/2005073
    // A quick summary:
    //     * The last two equations are:
    //            x*a +  y*b == g
    //           x1*a + y1*b == 0
    //     * You can inductively show that x1 and y1 are coprime.
    //     * Dividing through by g, we conclude x1*(a/g) = -y1*(b/g)
    //     * Since a/g and b/g are coprime,
    //       we know x1=(+/-)(b/g) and y1=(+/-)(a/g)
    //     * The sequence of x-values (y-values) is increasing in absolute value,
    //       and the last one is at most b (at most a), so the whole sequence is.
    //     * The intermediate result q*x are also bounded: since x2 = x - q*x1,
    //           we know that |q*x1| = |x - x2| <= ||x|-|x2|| <= |b-0| = b.
    //     * The sequence of q-values and g-values are decreasing.

    // Collins assumes a and b are positive, and division with negatives
    // could be weird in C anyway, so let's make them nonnegative,
    // and account for that at the end.

    // Going over the cases Collins doesn't handle:
    // if b == 0 then the loop doesn't run at all.
    // if a == 0 then the first loop uses q=0 just swaps a and b,
    //   so then the argument still goes through after that.
    // if a == b then q = 1, and
    //     x2 = 1 - 1*0 = 1;
    //     y2 = 0 - 1*1 = -1;
    //     g2 = a - 1*b = 0;
    // Still no overflows.

    bool a_neg = (a < 0);
    if (a_neg) {
        assert(a != INTPTR_MIN);
        a = -a;
    }
    bool b_neg = (b < 0);
    if (b_neg) {
        assert(b != INTPTR_MIN);
        b = -b;
    }

    intptr_t x = 1, y = 0, g = a;
    intptr_t x1 = 0, y1 = 1, g1 = b;
    while (g1) {
        intptr_t q = g / g1;
        intptr_t x2 = x - q*x1;
        intptr_t y2 = y - q*y1;
        intptr_t g2 = g - q*g1;
        x = x1; x1 = x2;
        y = y1; y1 = y2;
        g = g1; g1 = g2;
    }
    assert(x != INTPTR_MIN);
    assert(y != INTPTR_MIN);
    assert(g != INTPTR_MIN);
    if (a_neg) {
        x = -x;
    }
    if (b_neg) {
        y = -y;
    }
    if (g < 0) {
        x = -x;
        y = -y;
        g = -g;
    }
    result[0] = x;
    result[1] = y;
    result[2] = g;
}


static PyObject *
xgcd(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "xgcd(a, b) takes 2 arguments.");
        return NULL;
    }
    PyObject *a_obj = args[0];
    PyObject *b_obj = args[1];
    PyObject *result_objects[3];
    if (!PyLong_CheckExact(a_obj) || !PyLong_CheckExact(b_obj)) {
        PyErr_SetString(PyExc_TypeError, "xgcd(a, b) arguments must be integers.");
        return NULL;
    }
    int overflow;
    intptr_t a = PyLong_AsIntptrAndOverflow(a_obj, &overflow);
    if (a == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (overflow || a == INTPTR_MIN) {
        goto use_objects;
    }
    intptr_t b = PyLong_AsIntptrAndOverflow(b_obj, &overflow);
    if (b == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (overflow || b == INTPTR_MIN) {
        goto use_objects;
    }
    intptr_t result_intptrs[3];
    xgcd_using_intptr(a, b, result_intptrs);
    return Py_BuildValue("nnn", result_intptrs[0], result_intptrs[1], result_intptrs[2]);
use_objects:
    if (xgcd_using_objects(a_obj, b_obj, result_objects)) {
        return NULL;
    }
    PyObject *tup = PyTuple_Pack(3, result_objects[0], result_objects[1], result_objects[2]);
    Py_DECREF(result_objects[0]); Py_DECREF(result_objects[1]); Py_DECREF(result_objects[2]);
    return tup;
}


/*********************************************************************/
/* The Lattice() type                                                */
/*********************************************************************/

static PyTypeObject Lattice_Type;

typedef struct {
    PyObject_VAR_HEAD
    Py_ssize_t N; // The ambient dimension; we're a sublattice of Z^N
    Py_ssize_t rank; // The number of basis vectors
    Py_ssize_t maxrank; // How many vectors we have space for
    Py_ssize_t num_zero_columns;
    Py_ssize_t *zero_columns;
    Py_ssize_t *col_to_pivot; // length=N. -1 if no pivot
    Py_ssize_t *row_to_pivot; // logical length=rank. Every row has a pivot.
    int HNF_policy; // 0 --> manual, 1 --> after every addition.
    bool corrupted; // set to true after an error occurs to avoid inconsistencies.
    bool is_full; // set to true once we have all of Z^N
    Py_ssize_t first_HNF_row;
    TagInt *buffer_for_tagints; // Space for (N+1)*N TagInts
    TagInt *basis[1]; // space for N Pointer-to-TagInts
    // The data hangs off the end of this struct.
    // Format:
    //    * The struct members leading up to L.basis
    //    * Then L.basis[N] (so we can shuffle rows around)
    //    * then space for 3*N*Py_ssize_t arrays above
    //    * then space for (maxrank+1)*N TagInts, pointed to by the above.
    // Technically we don't need to store all of the pointers above
    // and could recalculate, but I don't think we'll be starved
    // by this constant space overhead.
} Lattice;

static void
err_corrupted()
{
    PyErr_SetString(PyExc_RuntimeError, "Using a corrupted Lattice");
}

static TagInt *
Lattice_push_vector(Lattice *L, TagInt *vec, Py_ssize_t j0)
{
    // Naively push a vector to the end of the basis.
    Py_ssize_t N = L->N;
    assert(L->rank <= L->maxrank);
    TagInt *buf = L->buffer_for_tagints + N*L->rank;
#ifndef NDEBUG
    for (Py_ssize_t j = 0; j < j0; j++) {
        assert(TagInt_is_zero(buf[j]));
    }
#endif
    for (Py_ssize_t j = j0; j < N; j++) {
        assert(TagInt_is_zero(buf[j]));
        buf[j] = TagInt_copy(vec[j]);
    }
    return buf;
}

static void
Lattice_pop_vector(Lattice *L, Py_ssize_t j0)
{
    Py_ssize_t N = L->N;
    TagInt *vec = (TagInt *)(L->buffer_for_tagints + N*L->rank);
    for (Py_ssize_t j = j0; j < N; j++) {
        TagInt_clear(&vec[j]);
    }
}

static PyObject *
Lattice__assert_consistent(PyObject *self, PyObject *Py_UNUSED(other));

static void
Lattice_clear_impl(PyObject *self)
{
    Lattice *L = (Lattice *)self;
    Py_ssize_t N = L->N, R = L->rank;
    L->rank = 0;
    TagInt *t = L->buffer_for_tagints;
    TagInt *data_end = t + N * R;
    for (; t < data_end; t++) {
        TagInt_clear(t);
    }
    for (Py_ssize_t i = 0; i < N; i++) {
        L->zero_columns[i] = i;
        L->col_to_pivot[i] = -1;
    }
    L->num_zero_columns = N;
    L->corrupted = false;
    L->is_full = (N == 0);
    L->first_HNF_row = 0;
}

static PyObject *
Lattice_clear(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    Lattice_clear_impl(self);
    Py_RETURN_NONE;
}

static void
Lattice_dealloc(PyObject *self)
{
    Lattice_clear_impl(self);
    Py_TYPE(self)->tp_free(self);
}

static Py_ssize_t
bisect_left(Py_ssize_t *a, Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t x)
{
    while (lo < hi) {
        // The cast ensures there's no overflow.
        Py_ssize_t mid = ((size_t)lo + hi) / 2;
        assert(lo <= mid && mid <= hi);
        if (a[mid] < x) {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    return lo;
}

static PyObject *
Lattice__assert_consistent(PyObject *self, PyObject *Py_UNUSED(other))
{
    // printf("asserting consistent\n");
    Lattice *L = (Lattice *)self;
    Py_ssize_t N = L->N;
    Py_ssize_t R = L->rank;
    assert(R <= N);
    assert(R <= L->maxrank);
    assert(L->maxrank <= N);
    TagInt **basis = L->basis;

    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    {
        // assert the pool of free data to use is actually zeros.
        TagInt *t0 = L->buffer_for_tagints + N*R;
        TagInt *t1 = L->buffer_for_tagints + N*(L->maxrank + 1);
        for (TagInt *t = t0; t < t1; t++) {
            if (!TagInt_is_zero(*t)) {
                PyErr_SetString(PyExc_AssertionError, "unused TagInts were nonzero");
                return NULL;
            }
        }
    }

    Py_ssize_t nzc = L->num_zero_columns;
    Py_ssize_t *zero_columns = L->zero_columns;
    for (Py_ssize_t k = 1; k < nzc; k++) {
        if (!(zero_columns[k] > zero_columns[k-1])) {
            PyErr_SetString(PyExc_AssertionError, "zero_columns not increasing");
            return NULL;
        }
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        bool column_is_zero = true;
        for (Py_ssize_t i = 0; i < R; i++) {
            column_is_zero = column_is_zero && TagInt_is_zero(basis[i][j]);
        }
        Py_ssize_t zc_index = bisect_left(zero_columns, 0, nzc, j);
        bool column_is_zero_expected = (zc_index < nzc && zero_columns[zc_index] == j);
        if (column_is_zero != column_is_zero_expected) {
            PyErr_SetString(PyExc_AssertionError, "zero_columns inconsistent");
            return NULL;
        }
    }

    Py_ssize_t *row_to_pivot = L->row_to_pivot;
    for (Py_ssize_t i = 0; i < R; i++) {
        Py_ssize_t j = row_to_pivot[i];
        if (i > 0) {
            if (!(j > row_to_pivot[i-1])) {
                PyErr_SetString(PyExc_AssertionError, "row_to_pivot not increasing");
                return NULL;
            }
        }
        for (Py_ssize_t jj = 0; jj < j; jj++) {
            if (!TagInt_is_zero(basis[i][jj])) {
                PyErr_SetString(PyExc_AssertionError, "row_to_pivot too large");
                return NULL;
            }
        }
        TagInt x = basis[i][j];
        if (TagInt_is_zero(x)) {
            PyErr_SetString(PyExc_AssertionError, "row_to_pivot too small");
            return NULL;
        }
    }
    Py_ssize_t *col_to_pivot = L->col_to_pivot;
    Py_ssize_t prev_i = -1;
    for (Py_ssize_t j = 0; j < N; j++) {
        Py_ssize_t i = col_to_pivot[j];
        if (i == -1) {
            continue;
        }
        if (!(i > prev_i)) {
            PyErr_SetString(PyExc_AssertionError, "col_to_pivot nontrivials not increasing");
            return NULL;
        }
        prev_i = i;
        for (Py_ssize_t ii = i + 1; ii < R; ii++) {
            if (!TagInt_is_zero(basis[ii][j])) {
                PyErr_SetString(PyExc_AssertionError, "col_to_pivot too small");
                return NULL;
            }
        }
        if (TagInt_is_zero(basis[i][j])) {
            PyErr_SetString(PyExc_AssertionError, "col_to_pivot too large");
            return NULL;
        }
    }

    for (Py_ssize_t i = 0; i < R; i++) {
        if (col_to_pivot[row_to_pivot[i]] != i) {
            PyErr_SetString(PyExc_AssertionError, "col_to_pivot mismatches row_to_pivot");
            return NULL;
        }
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        if (col_to_pivot[j] != -1) {
            if (row_to_pivot[col_to_pivot[j]] != j) {
                PyErr_SetString(PyExc_AssertionError, "row_to_pivot mismatches col_to_pivot");
                return NULL;
            }
        }
    }

    for (Py_ssize_t i = 0; i < R; i++) {
        TagInt *row = basis[i];
        for (Py_ssize_t j = 0; j < N; j++) {
            if (TagInt_is_pointer(row[j])) {
                int overflow;
                intptr_t k = PyLong_AsIntptrAndOverflow(untag_pointer(row[j]), &overflow);
                if (k == -1 && PyErr_Occurred()) {
                    return NULL;
                }
                if (!overflow && is_packable_int(k)) {
                    PyErr_SetString(PyExc_AssertionError, "Packable int was left as object");
                    return NULL;
                }
            }
        }
    }

    PyObject *zero = PyLong_FromLong(0);
    for (Py_ssize_t i = L->first_HNF_row; i < R; i++) {
        TagInt *above_row = basis[i];
        for (Py_ssize_t ii = i + 1; ii < R; ii++) {
            Py_ssize_t jj = row_to_pivot[ii];
            PyObject *pivot = TagInt_to_object(basis[ii][jj]);
            if (pivot == NULL) {
                Py_DECREF(zero);
                return NULL;
            }
            int zero_lt_pivot = pylong_lt(zero, pivot);
            if (zero_lt_pivot != 1) {
                if (zero_lt_pivot == 0) {
                    PyErr_SetString(PyExc_AssertionError, "HNF pivot wasn't positive");
                }
                Py_DECREF(zero);
                Py_DECREF(pivot);
                return NULL;
            }
            PyObject *above = TagInt_to_object(above_row[jj]);
            if (above == NULL) {
                Py_DECREF(zero);
                Py_DECREF(pivot);
                return NULL;
            }
            int above_lt_zero = pylong_lt(above, zero);
            if (above_lt_zero != 0) {
                if (above_lt_zero == 1) {
                    PyErr_SetString(PyExc_AssertionError, "HNF entry above pivot was negative");
                }
                Py_DECREF(zero);
                Py_DECREF(pivot);
                Py_DECREF(above);
                return NULL;
            }
            int above_lt_pivot = pylong_lt(above, pivot);
            if (above_lt_pivot != 1) {
                if (above_lt_pivot == 0) {
                    PyErr_SetString(PyExc_AssertionError, "HNF entry above pivot was too large");
                }
                Py_DECREF(zero);
                Py_DECREF(pivot);
                Py_DECREF(above);
                return NULL;
            }
            Py_DECREF(pivot);
            Py_DECREF(above);
        }
    }
    Py_DECREF(zero);
    Py_RETURN_NONE;
}

static PyObject *
Lattice_new_impl(PyTypeObject *type, Py_ssize_t N, int HNF_policy, Py_ssize_t maxrank)
{
    static_assert(sizeof(TagInt) == sizeof(void *), "bad C type sizes");
    static_assert(sizeof(Py_ssize_t) == sizeof(void *), "bad C type sizes");
    // We need:
    //   3*N words for the Py_ssize_t data,
    // + N words for the basis
    // + N*(maxrank+1) words for the buffers
    // == N*(maxrank+5)
    assert(0 <= maxrank && maxrank <= N);
    if (N && maxrank > PY_SSIZE_T_MAX/N - 5) {
        PyErr_SetString(PyExc_OverflowError, "Lattice was too big to construct");
        return NULL;
    }
    Lattice *L = (Lattice *)PyType_GenericAlloc(type, N*(maxrank+5));
    if (!L) {
        return NULL;
    }
    // Divvy up the allocated block
    void **buffer = (void **)L->basis;
    Py_ssize_t *zero_columns = (Py_ssize_t *)&buffer[N];
    Py_ssize_t *col_to_pivot = (Py_ssize_t *)&buffer[2*N];
    Py_ssize_t *row_to_pivot = (Py_ssize_t *)&buffer[3*N];
    TagInt *buffer_for_tagints = (TagInt *)&buffer[4*N];
    assert(TagInt_is_zero(buffer_for_tagints[N*(maxrank+1)-1]));
    L->N = N;
    L->maxrank = maxrank;
    L->rank = 0;
    L->num_zero_columns = N;
    L->zero_columns = zero_columns;
    for (Py_ssize_t i = 0; i < N; i++) {
        zero_columns[i] = i;
        col_to_pivot[i] = -1;
    }
    L->row_to_pivot = row_to_pivot;
    L->col_to_pivot = col_to_pivot;
    L->HNF_policy = HNF_policy;
    L->corrupted = false;
    L->is_full = (N == 0);
    L->first_HNF_row = 0;
    L->buffer_for_tagints = buffer_for_tagints;
    return (PyObject *)L;
}

static bool
Lattice_add_vector_or_list_impl(PyObject *self, PyObject *other);

static PyObject *
Lattice_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"", "", "HNF_policy", "maxrank", NULL};
    Py_ssize_t N;
    PyObject *data = NULL;
    Py_ssize_t maxrank = -1;
    int HNF_policy = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|O$in", kwlist, &N, &data, &HNF_policy, &maxrank)) {
        return NULL;
    }
    if (N < 0) {
        PyErr_SetString(PyExc_ValueError, "Lattice(N) argument must be nonnegative");
        return NULL;
    }
    if (HNF_policy != 0 && HNF_policy != 1) {
        PyErr_SetString(PyExc_ValueError, "unknown HNF_policy");
        return NULL;
    }
    if (maxrank < -1) {
        PyErr_SetString(PyExc_ValueError, "maxrank must be >= -1");
        return NULL;
    }
    if (maxrank == -1 || maxrank > N) {
        maxrank = N;
    }
    PyObject *result = Lattice_new_impl(type, N, HNF_policy, maxrank);
    if (result == NULL) {
        return NULL;
    }
    if (data == NULL) {
        return result;
    }
    if (!PyList_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "Lattice(N, data) second argument must be list");
        Py_DECREF(result);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(data); i++) {
        if (Lattice_add_vector_or_list_impl(result, PyList_GET_ITEM(data, i))) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

static PyObject *
Lattice_full(PyObject *cls, PyObject *arg)
{
    if (!PyLong_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Lattice.full(N) argument must be integer");
        return NULL;
    }
    Py_ssize_t N = PyLong_AsSsize_t(arg);
    if (N == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (N < 0) {
        PyErr_SetString(PyExc_ValueError, "Lattice.full(N) argument cannot be negative");
        return NULL;
    }
    Lattice *L = (Lattice *)Lattice_new_impl((PyTypeObject *)cls, N, 1, N);
    if (L == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < N; i++) {
        L->row_to_pivot[i] = i;
        L->col_to_pivot[i] = i;
        L->basis[i] = L->buffer_for_tagints + N*i;
        L->basis[i][i] = TagInt_ONE;
    }
    L->rank = N;
    L->num_zero_columns = 0;
    L->is_full = true;
    L->first_HNF_row = 0;
    return (PyObject *)L;
}

static PyObject *
Lattice_copy(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    assert(Py_TYPE(self) == &Lattice_Type);
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    Py_ssize_t R = L->rank;
    Py_ssize_t N = L->N;
    Lattice *L_copy = (Lattice *)Lattice_new_impl(Py_TYPE(self), N, L->HNF_policy, L->maxrank);
    if (L_copy == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < R; i++) {
        L_copy->basis[i] = Lattice_push_vector(L_copy, L->basis[i], L->row_to_pivot[i]);
        L_copy->row_to_pivot[i] = L->row_to_pivot[i];
        L_copy->rank++;
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        L_copy->col_to_pivot[j] = L->col_to_pivot[j];
    }
    Py_ssize_t nzc = L->num_zero_columns;
    L_copy->num_zero_columns = nzc;
    for (Py_ssize_t k = 0; k < nzc; k++) {
        L_copy->zero_columns[k] = L->zero_columns[k];
    }
    L_copy->first_HNF_row = L->first_HNF_row;
    L_copy->is_full = L->is_full;
    return (PyObject *)L_copy;
}

static bool
Lattice_HNFify_impl(Lattice *L, Py_ssize_t first_row_to_fix);

static int
Lattice_nomutate_make_zero_at_entry_with_objects(
    TagInt *vecj, TagInt *rowj, Py_ssize_t N_j, PyObject **q_out)
{
    // Helper function for Lattice_contains.
    // Try to make "vecj" zero at its first entry
    //   (Use pointer arithmetic to make vec zero at its jth entry)
    // Returns: -1 if error,
    //           1 if successfully made zero (and assigns the coefficient to *q_out),
    //           0 if impossible to make zero.
    PyObject *a_obj = TagInt_to_object(rowj[0]);
    if (a_obj == NULL) {
        return -1;
    }
    PyObject *b_obj = TagInt_to_object(vecj[0]);
    if (b_obj == NULL) {
        Py_DECREF(a_obj);
        return -1;
    }
    PyObject *rem = pylong_remainder(b_obj, a_obj);
    if (rem == NULL) {
        Py_DECREF(a_obj);
        Py_DECREF(b_obj);
        return -1;
    }
    int not_a_multiple = pylong_bool(rem);
    Py_DECREF(rem);
    if (not_a_multiple) {
        // This pivot can't zero this entry
        Py_DECREF(a_obj);
        Py_DECREF(b_obj);
        return 0;
    }
    PyObject *q = pylong_floor_divide(b_obj, a_obj);
    Py_DECREF(a_obj);
    Py_DECREF(b_obj);
    if (q == NULL) {
        return -1;
    }
    PyObject *neg_q = pylong_negative(q);
    if (neg_q == NULL) {
        Py_DECREF(q);
        return -1;
    }
    // This is doing "vec -= q * row"
    if (row_op_impl_with_objects(rowj, vecj, N_j, neg_q)) {
        Py_DECREF(neg_q);
        Py_DECREF(q);
        return -1;
    }
    Py_DECREF(neg_q);
    *q_out = q;
    return 1;
}

static int
Lattice_contains_loop(Lattice *L, TagInt *vec, Py_ssize_t j)
{
    vec = Lattice_push_vector(L, vec, j);
    Py_ssize_t N = L->N;
    for (; j < N; j++) {
        if (TagInt_is_zero(vec[j])) {
            continue;
        }
        Py_ssize_t i = L->col_to_pivot[j];
        if (i == -1) {
            // No pivot here to zero out this nonzero vec entry.
            goto not_present;
        }
        TagInt *row = L->basis[i];
#if USE_FAST_PATHS
        if (!TagInt_is_pointer(row[j]) && !TagInt_is_pointer(vec[j])) {
            intptr_t rowj = unpack_integer(row[j]);
            intptr_t vecj = unpack_integer(vec[j]);
            assert(rowj != 0);
            assert(vecj != 0);
            if (vecj % rowj != 0) {
                goto not_present;
            }
            assert(vecj != INTPTR_MIN);
            intptr_t q = vecj / rowj;
            intptr_t neg_q = -q;
            PyObject *neg_q_obj = NULL;
            if (row_op_impl_with_intptr(&row[j], &vec[j], N-j, neg_q, &neg_q_obj)) {
                Py_XDECREF(neg_q_obj);
                goto error;
            }
            Py_XDECREF(neg_q_obj);
            assert(TagInt_is_zero(vec[j]));
            continue;
        }
#endif
        // Same thing, but slower: use PyObjects
        PyObject *q;
        switch (Lattice_nomutate_make_zero_at_entry_with_objects(&vec[j], &row[j], N-j, &q)) {
            case -1: goto error;
            case 0: goto not_present;
            case 1: break;
        }
        Py_DECREF(q);
        assert(TagInt_is_zero(vec[j]));
    }
    // Everything became zero, so the vector is present.
    // No need to manage those references either.
    return 1;
error:
    Lattice_pop_vector(L, j);
    return -1;
not_present:
    Lattice_pop_vector(L, j);
    return 0;
}

static int
Lattice_contains_impl(Lattice *L, TagInt *vec, Py_ssize_t j0)
{
    if (L->is_full) {
        return 1;
    }
    Py_ssize_t N = L->N;
    {
        Py_ssize_t nzc = L->num_zero_columns;
        Py_ssize_t *zero_columns = L->zero_columns;
        for (Py_ssize_t i = 0; i < nzc; i++) {
            if (!TagInt_is_zero(vec[zero_columns[i]])) {
                return 0;
            }
        }
    }
    Py_ssize_t j = j0;
    while (j < N && TagInt_is_zero(vec[j])) {
        j++;
    }
    if (j == N) {
        // The zero vector is always present
        return 1;
    }
    Py_ssize_t i = L->col_to_pivot[j];
    if (i == -1) {
        // No pivot here to zero out this nonzero vec entry.
        return 0;
    }
    TagInt *row = L->basis[i];
    if (memcmp(&row[j], &vec[j], (N-j)*sizeof(TagInt)) == 0) {
        // A pointer-for-pointer copy of the vector is already present.
        return 1;
    }
    // Make a copy so we can mutate as needed
    return Lattice_contains_loop(L, vec, j);
}

static int
Lattice_contains(PyObject *self, PyObject *other)
{
    assert(Py_TYPE(self) == &Lattice_Type);
    if (((Lattice *)self)->corrupted) {
        err_corrupted();
        return -1;
    }
    if (Py_TYPE(other) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "Lattice.__contains__ argument must be Vector");
        return -1;
    }
    Py_ssize_t N = ((Lattice *)self)->N;
    if (Py_SIZE(other) != N) {
        PyErr_SetString(PyExc_ValueError, "length mismatch in Lattice.__contains__");
        return -1;
    }
    return Lattice_contains_impl((Lattice *)self, Vector_get_vec(other), 0);
}

static PyObject *
Lattice_coefficients_of(PyObject *self, PyObject *other)
{
    if (Py_TYPE(other) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "L.coefficients_of(v) argument must be Vector");
        return NULL;
    }
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    Py_ssize_t N = L->N, R = L->rank;
    if (Py_SIZE(other) != L->N) {
        PyErr_SetString(PyExc_ValueError, "L.coefficients_of(v) argument length mismatch");
        return NULL;
    }
    PyObject *result = Vector_zero_impl(L->rank);
    if (result == NULL) {
        return NULL;
    }
    TagInt *result_vec = Vector_get_vec(result);
    TagInt *vec = Lattice_push_vector(L, Vector_get_vec(other), 0);
    for (Py_ssize_t i = 0; i < R; i++) {
        Py_ssize_t j = L->row_to_pivot[i];
        if (TagInt_is_zero(vec[j])) {
            continue;
        }
        TagInt *row = L->basis[i];
        PyObject *q_obj;
#if USE_FAST_PATHS
        if (!TagInt_is_pointer(row[j]) && !TagInt_is_pointer(vec[j])) {
            intptr_t rowj = unpack_integer(row[j]);
            intptr_t vecj = unpack_integer(vec[j]);
            assert(rowj != 0);
            assert(vecj != 0);
            if (vecj % rowj != 0) {
                goto not_present;
            }
            assert(vecj != INTPTR_MIN);
            intptr_t q = vecj / rowj;
            if (!is_packable_int(q)) {
                // Happends if vecj==INTPTR_MIN/2 and rowj==-1.
                goto slowpath;
            }
            intptr_t neg_q = -q;
            PyObject *neg_q_obj = NULL;
            if (row_op_impl_with_intptr(&row[j], &vec[j], N-j, neg_q, &neg_q_obj)) {
                Py_XDECREF(neg_q_obj);
                goto error;
            }
            Py_XDECREF(neg_q_obj);
            assert(TagInt_is_zero(vec[j]));
            result_vec[i] = pack_integer(q);
            continue;
        }
#endif
    slowpath:
        switch (Lattice_nomutate_make_zero_at_entry_with_objects(&vec[j], &row[j], N-j, &q_obj)) {
            case -1: goto error;
            case 0: goto not_present;
            case 1: break;
        }
        assert(TagInt_is_zero(vec[j]));
        if (object_to_TagInt_steal(q_obj, &result_vec[i])) {
            goto error;
        }
    }
    bool nonzero = false;
    for (Py_ssize_t j = 0; j < N; j++) {
        if (!TagInt_is_zero(vec[j])) {
            nonzero = true;
            TagInt_clear(&vec[j]);
        }
    }
    if (nonzero) {
        goto not_present;
    }
    // Already cleared out the pushed vector
    return result;
error:
    Py_DECREF(result);
    Lattice_pop_vector(L, 0);
    return NULL;
not_present:
    Py_DECREF(result);
    Lattice_pop_vector(L, 0);
    PyErr_SetString(PyExc_ValueError, "Vector not present in Lattice");
    return NULL;
}

static PyObject *
Lattice_linear_combination(PyObject *self, PyObject *other)
{
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    Py_ssize_t N = L->N, R = L->rank;
    if (Py_TYPE(other) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "L.linear_combination(w) argument must be Vector");
        return NULL;
    }
    if (Py_SIZE(other) != R) {
        PyErr_SetString(PyExc_ValueError, "L.linear_combination(w) argument must have length L.rank");
        return NULL;
    }
    TagInt *coefficients = Vector_get_vec(other);
    PyObject *result = Vector_zero_impl(N);
    if (result == NULL) {
        return result;
    }
    TagInt *vec = Vector_get_vec(result);
    for (Py_ssize_t i = 0; i < R; i++) {
        TagInt c = coefficients[i];
        if (TagInt_is_zero(c)) {
            continue;
        }
        Py_ssize_t j = L->row_to_pivot[i];
        TagInt *row = L->basis[i];
        if (TagInt_is_pointer(c)) {
            if (row_op_impl_with_objects(&row[j], &vec[j], N-j, untag_pointer(c))) {
                goto error;
            }
        }
        else {
            PyObject *c_object = NULL;
            if (row_op_impl_with_intptr(&row[j], &vec[j], N-j, unpack_integer(c), &c_object)) {
                Py_XDECREF(c_object);
                goto error;
            }
            Py_XDECREF(c_object);
        }
    }
    return result;
error:
    Py_DECREF(result);
    return NULL;
}

static Py_ssize_t
Lattice_insert_vector_with_pivot(Lattice *L, TagInt *vec, Py_ssize_t j)
{
    assert(vec == L->buffer_for_tagints + L->N * L->rank);
    Py_ssize_t R = L->rank;
    TagInt **basis = L->basis;
    Py_ssize_t *row_to_pivot = L->row_to_pivot;
    Py_ssize_t *col_to_pivot = L->col_to_pivot;
    Py_ssize_t where = bisect_left(row_to_pivot, 0, R, j);
    assert(0 <= where && where <= R);
    assert(where == 0 || row_to_pivot[where-1] < j);
    assert(where == R || row_to_pivot[where] > j);
    L->rank = R + 1;
    assert(L->rank <= L->N);

    // shift all the later rows down
    for (Py_ssize_t i = R; i > where; i--) {
        basis[i] = basis[i - 1];
        row_to_pivot[i] = row_to_pivot[i - 1];
        assert(col_to_pivot[row_to_pivot[i]] == i - 1);
        col_to_pivot[row_to_pivot[i]] = i;
    }
    basis[where] = vec;
    row_to_pivot[where] = j;
    col_to_pivot[j] = where;

    Py_ssize_t *zero_columns = L->zero_columns;
    Py_ssize_t nzc = L->num_zero_columns;
    Py_ssize_t dest, src;
    dest = src = bisect_left(zero_columns, 0, nzc, j);
    for (; src < nzc; src++) {
        if (TagInt_is_zero(vec[zero_columns[src]])) {
            zero_columns[dest++] = zero_columns[src];
        }
    }
    L->num_zero_columns = dest;
    assert(L->first_HNF_row <= R);
    L->first_HNF_row = Py_MAX(L->first_HNF_row, where) + 1;
    assert(L->first_HNF_row <= L->rank);

    return where;
}

static void
do_swap(TagInt *va, TagInt *vb, Py_ssize_t N)
{
    for (Py_ssize_t j = 0; j < N; j++) {
        TagInt tmp = va[j];
        va[j] = vb[j];
        vb[j] = tmp;
    }
}

static void
modified_row(Lattice *L, Py_ssize_t i)
{
    // Update first_HNF_row
    if (i >= L->first_HNF_row) {
        L->first_HNF_row = i + 1;
        assert(L->first_HNF_row <= L->rank);
    }
}

static bool
make_entry_zero(TagInt *vec, Lattice *L, Py_ssize_t i, Py_ssize_t j)
{
    // helper function for L.add_vector(v)
    Py_ssize_t N = L->N;
    TagInt *row = L->basis[i];
#if USE_FAST_PATHS
    if (!TagInt_is_pointer(row[j]) && !TagInt_is_pointer(vec[j])) {
        intptr_t rowj = unpack_integer(row[j]);
        intptr_t vecj = unpack_integer(vec[j]);
        assert(rowj != 0);
        assert(vecj != 0);
        assert(rowj != INTPTR_MIN);
        assert(vecj != INTPTR_MIN);
        if (vecj % rowj == 0) {
            intptr_t q = vecj / rowj;
            intptr_t neg_q = -q;
            PyObject *neg_q_obj = NULL;
            if (row_op_impl_with_intptr(&row[j], &vec[j], N - j, neg_q, &neg_q_obj)) {
                Py_XDECREF(neg_q_obj);
                return true;
            }
            Py_XDECREF(neg_q_obj);
            assert(TagInt_is_zero(vec[j]));
            return false;
        }
        if (rowj % vecj == 0) {
            do_swap(&row[j], &vec[j], N - j);
            intptr_t q = rowj / vecj;
            assert(q != INTPTR_MIN);
            intptr_t neg_q = -q;
            PyObject *neg_q_obj = NULL;
            if (row_op_impl_with_intptr(&row[j], &vec[j], N - j, neg_q, &neg_q_obj)) {
                Py_XDECREF(neg_q_obj);
                return true;
            }
            Py_XDECREF(neg_q_obj);
            assert(TagInt_is_zero(vec[j]));
            modified_row(L, i);
            assert(L->first_HNF_row <= L->rank);
            return false;
        }
        intptr_t xyg[3];
        xgcd_using_intptr(rowj, vecj, xyg);
        intptr_t x = xyg[0], y = xyg[1], g = xyg[2];
        intptr_t abcd[4] = {x, y, -(vecj/g), rowj/g};
        PyObject *abcd_obj[4] = {NULL, NULL, NULL, NULL};
        bool err = generalized_row_op_impl_with_intptr(&row[j], &vec[j], N - j, abcd, abcd_obj);
        for (int i = 0; i < 4; i++) {
            Py_XDECREF(abcd_obj[i]);
        }
        if (err) {
            return true;
        }
        assert(TagInt_is_zero(vec[j]));
        modified_row(L, i);
        return false;
    }
#endif
    PyObject *rowj = TagInt_to_object(row[j]);
    if (rowj == NULL) {
        return true;
    }
    PyObject *vecj = TagInt_to_object(vec[j]);
    if (vecj == NULL) {
        Py_DECREF(rowj);
        return true;
    }
    PyObject *vecj_mod_rowj = pylong_remainder(vecj, rowj);
    if (vecj_mod_rowj == NULL) {
        Py_DECREF(rowj);
        Py_DECREF(vecj);
        return true;
    }
    bool bool_vecj_mod_rowj = pylong_bool(vecj_mod_rowj);
    Py_DECREF(vecj_mod_rowj);
    if (!bool_vecj_mod_rowj) {
        PyObject *q = pylong_floor_divide(vecj, rowj);
        Py_DECREF(rowj);
        Py_DECREF(vecj);
        if (q == NULL) {
            return true;
        }
        PyObject *neg_q = pylong_negative(q);
        Py_DECREF(q);
        if (neg_q == NULL) {
            return true;
        }
        bool err = row_op_impl_with_objects(&row[j], &vec[j], N - j, neg_q);
        Py_DECREF(neg_q);
        return err;
    }
    PyObject *rowj_mod_vecj = pylong_remainder(rowj, vecj);
    if (rowj_mod_vecj == NULL) {
        Py_DECREF(rowj);
        Py_DECREF(vecj);
        return true;
    }
    bool bool_rowj_mod_vecj = pylong_bool(rowj_mod_vecj);
    Py_DECREF(rowj_mod_vecj);
    if (!bool_rowj_mod_vecj) {
        do_swap(&row[j], &vec[j], N - j);
        PyObject *q = pylong_floor_divide(rowj, vecj);
        Py_DECREF(rowj);
        Py_DECREF(vecj);
        if (q == NULL) {
            return true;
        }
        PyObject *neg_q = pylong_negative(q);
        Py_DECREF(q);
        if (neg_q == NULL) {
            return true;
        }
        bool err = row_op_impl_with_objects(&row[j], &vec[j], N - j, neg_q);
        Py_DECREF(neg_q);
        modified_row(L, i);
        return err;
    }
    // start using the error label here
    PyObject *xyg[3] = {NULL, NULL, NULL};
    PyObject *x=NULL, *y=NULL, *g=NULL;
    PyObject *vecj_g=NULL, *rowj_g=NULL, *neg_vecj_g=NULL;
    if (xgcd_using_objects(rowj, vecj, xyg)) {
        goto error;
    }
    x = xyg[0]; y = xyg[1]; g = xyg[2];
    if (!(vecj_g = pylong_floor_divide(vecj, g))) {
        goto error;
    }
    if (!(rowj_g = pylong_floor_divide(rowj, g))) {
        goto error;
    }
    if (!(neg_vecj_g = pylong_negative(vecj_g))) {
        goto error;
    }
    PyObject *abcd[4] = {x, y, neg_vecj_g, rowj_g};
    if (generalized_row_op_impl_with_objects(&row[j], &vec[j], N - j, abcd)) {
        goto error;
    }
    Py_DECREF(rowj); Py_DECREF(vecj);
    Py_DECREF(x); Py_DECREF(y); Py_DECREF(g);
    Py_DECREF(vecj_g); Py_DECREF(rowj_g); Py_DECREF(neg_vecj_g);
    modified_row(L, i);
    return false;
error:
    Py_DECREF(rowj); Py_DECREF(vecj);
    Py_XDECREF(x); Py_XDECREF(y); Py_XDECREF(g);
    Py_XDECREF(vecj_g); Py_DECREF(rowj_g); Py_DECREF(neg_vecj_g);
    return true;
}

static bool
Lattice_apply_HNF_policy(Lattice *L)
{
    switch (L->HNF_policy) {
    case 0: // NEVER
        return false;
    case 1: // ALWAYS
        return Lattice_HNFify_impl(L, 0);
    default:
        Py_UNREACHABLE();
    }
}

static PyObject *
Lattice_is_full(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyBool_FromLong(((Lattice *)self)->is_full);
}

static inline void
update_is_full(Lattice *L)
{
    Py_ssize_t N = L->N;
    if (L->rank == N) {
        for (Py_ssize_t i = N - 1; i >= 0; i--) {
            assert(L->row_to_pivot[i] == i);
            TagInt p = L->basis[i][i];
            if (!TagInt_is_one(p) && !TagInt_is_negative_one(p)) {
                return;
            }
        }
        L->is_full = true;
    }
}

static bool
Lattice_add_vector_impl(Lattice *L, TagInt *vec)
{
    if (L->is_full) {
        return false;
    }
    // Copy without an allocation!
    vec = Lattice_push_vector(L, vec, 0);
    Py_ssize_t N = L->N;
    for (Py_ssize_t j = 0; j < N; j++) {
        if (TagInt_is_zero(vec[j])) {
            continue;
        }
        Py_ssize_t i = L->col_to_pivot[j];
        if (i != -1) {
            if (make_entry_zero(vec, L, i, j)) {
                Lattice_pop_vector(L, j);
                L->corrupted = true;
                return true;
            }
            assert(TagInt_is_zero(vec[j]));
            continue;
        }
        if (L->rank == L->maxrank) {
            PyErr_SetString(PyExc_IndexError, "Lattice rank would exceed maxrank");
            L->corrupted = true;
            return true;
        }
        i = Lattice_insert_vector_with_pivot(L, vec, j);
        break;
    }
    // If break occurred, the new vector was added successfully.
    // If no break, the whole vector has been zero-ed out;
    // nothing to add, nor even to garbage collect.
    update_is_full(L);
    return Lattice_apply_HNF_policy(L);
}

static PyObject *
Lattice_add_vector(PyObject *self, PyObject *other)
{
    assert(Py_TYPE(self) == &Lattice_Type);
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    if (Py_TYPE(other) != &Vector_Type) {
        PyErr_SetString(PyExc_TypeError, "Lattice.add_vector(v) argument must be Vector");
        return NULL;
    }
    if (Py_SIZE(other) != L->N) {
        PyErr_SetString(PyExc_ValueError, "length mismatch in Lattice.add_vector");
        return NULL;
    }
    if (Lattice_add_vector_impl(L, Vector_get_vec(other))) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static bool
Lattice_add_vector_or_list_impl(PyObject *self, PyObject *other)
{
    if (Py_TYPE(other) == &PyList_Type) {
        other = Vector_new_impl(other);
        if (other == NULL) {
            return true;
        }
    }
    else if (Py_TYPE(other) == &Vector_Type) {
        Py_INCREF(other);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Lattice data entries must be list or Vector");
        return true;
    }
    PyObject *res = Lattice_add_vector(self, other);
    Py_DECREF(other);
    if (res == NULL) {
        return true;
    }
    Py_DECREF(res);
    return false;
}

static PyObject *
Lattice_get_basis(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    assert(Py_TYPE(self) == &Lattice_Type);
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    Py_ssize_t N = L->N;
    Py_ssize_t R = L->rank;
    PyObject *result = PyList_New(R);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < R; i++) {
        PyObject *v = Vector_from_TagInts(L->basis[i], N);
        if (v == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, v);
    }
    return result;
}

static PyObject *
Lattice_tolist(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    assert(Py_TYPE(self) == &Lattice_Type);
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    Py_ssize_t N = L->N;
    Py_ssize_t R = L->rank;
    PyObject *result = PyList_New(R);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < R; i++) {
        PyObject *v = Vector_tolist_impl(L->basis[i], N);
        if (v == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, v);
    }
    return result;
}

static bool
Lattice_inplace_add_impl(Lattice *L, Lattice *other)
{
    for (Py_ssize_t i = other->rank - 1; i >= 0; i--) {
        if (Lattice_add_vector_impl(L, other->basis[i])) {
            return true;
        }
    }
    return false;
}

static PyObject *
Lattice_inplace_add(PyObject *self, PyObject *other)
{
    assert(Py_TYPE(self) == &Lattice_Type);
    if (Py_TYPE(other) != &Lattice_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Lattice *L_self = (Lattice *)self;
    Lattice *L_other = (Lattice *)other;
    if (L_self->corrupted || L_other->corrupted) {
        err_corrupted();
        return NULL;
    }
    if (L_self->N != L_other->N) {
        PyErr_SetString(PyExc_ValueError, "length mismatch in Lattice.__iadd__");
        return NULL;
    }
    if (Lattice_inplace_add_impl(L_self, L_other)) {
        return NULL;
    }
    return Py_NewRef(self);
}

static PyObject *
Lattice_add(PyObject *self, PyObject *other)
{
    if (Py_TYPE(self) != &Lattice_Type || Py_TYPE(other) != &Lattice_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Lattice *L1 = (Lattice *)self;
    Lattice *L2 = (Lattice *)other;
    if (L1->N != L2->N) {
        PyErr_SetString(PyExc_ValueError, "length mismatch in Lattice.__add__");
        return NULL;
    }
    if (L1->corrupted || L2->corrupted) {
        err_corrupted();
        return NULL;
    }
    if (L2->rank > L1->rank) {
        Lattice *tmp = L1;
        L1 = L2;
        L2 = tmp;
    }
    Lattice *L1_copy = (Lattice *)Lattice_copy((PyObject *)L1, NULL);
    if (L1_copy == NULL) {
        return NULL;
    }
    if (Lattice_inplace_add_impl(L1_copy, L2)) {
        return NULL;
    }
    return (PyObject *)L1_copy;
}

static int
Lattice_leq_impl(Lattice *L1, Lattice *L2)
{
    if (L1->rank > L2->rank) {
        return 0;
    }
    if (L1->num_zero_columns < L2->num_zero_columns) {
        return 0;
    }
    for (Py_ssize_t j = 0; j < L1->N; j++) {
        if (L1->col_to_pivot[j] != -1 && L2->col_to_pivot[j] == -1) {
            return 0;
        }
    }
    for (Py_ssize_t i = 0; i < L1->rank; i++) {
        int contains = Lattice_contains_impl(L2, L1->basis[i], L1->row_to_pivot[i]);
        if (contains == -1) {
            return -1;
        }
        if (contains == 0) {
            return 0;
        }
    }
    return 1;
}

static int
Lattice_eq_impl(Lattice *L1, Lattice *L2)
{
    if (L1 == L2) {
        return 1;
    }
    if (L1->rank != L2->rank) {
        return 0;
    }
    if (L1->num_zero_columns != L2->num_zero_columns) {
        return 0;
    }
    if (0 != memcmp(L1->zero_columns,
                    L2->zero_columns,
                    L1->num_zero_columns * sizeof(L1->zero_columns[0]))) {
        return 0;
    }
    if (0 != memcmp(L1->row_to_pivot,
                    L2->row_to_pivot,
                    L1->rank * sizeof(L1->row_to_pivot[0]))) {
        return 0;
    }
    int leq = Lattice_leq_impl(L1, L2);
    if (leq == -1) {
        return -1;
    }
    if (leq == 0) {
        return 0;
    }
    return Lattice_leq_impl(L2, L1);
}

static PyObject *
Lattice_richcompare(PyObject *a, PyObject *b, int op)
{
    if (Py_TYPE(a) != &Lattice_Type || Py_TYPE(b) != &Lattice_Type) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Lattice *L1 = (Lattice *)a;
    Lattice *L2 = (Lattice *)b;
    if (L1->corrupted || L2->corrupted) {
        err_corrupted();
        return NULL;
    }
    if (L1->N != L2->N) {
        if (op == Py_EQ) {
            Py_RETURN_FALSE;
        }
        if (op == Py_NE) {
            Py_RETURN_TRUE;
        }
        PyErr_SetString(PyExc_ValueError, "Can't compare lattices with different ambient dimensions");
        return NULL;
    }
    if (op == Py_LE || op == Py_GE) {
        if (op == Py_GE) {
            Lattice *tmp = L1; L1 = L2; L2 = tmp;
        }
        int le = Lattice_leq_impl(L1, L2);
        if (le == -1) {
            return NULL;
        }
        return Py_NewRef(le ? Py_True : Py_False);
    }
    if (op == Py_EQ || op == Py_NE) {
        int eq = Lattice_eq_impl(L1, L2);
        if (eq == -1) {
            return NULL;
        }
        return Py_NewRef(eq ^ (op == Py_NE) ? Py_True : Py_False);
    }
    if (op == Py_LT || op == Py_GT) {
        if (op == Py_GT) {
            Lattice *tmp = L1; L1 = L2; L2 = tmp;
        }
        int le = Lattice_leq_impl(L1, L2);
        if (le == -1) {
            return NULL;
        }
        if (le == 0) {
            Py_RETURN_FALSE;
        }
        int eq = Lattice_eq_impl(L1, L2);
        if (eq == -1) {
            return NULL;
        }
        return Py_NewRef(eq ? Py_False : Py_True);
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *
Py_ssize_t_vec_to_list(Py_ssize_t *vec, Py_ssize_t N)
{
    PyObject *result = PyList_New(N);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        PyObject *x;
        if (vec[j] == -1) {
            x = Py_NewRef(Py_None);
        }
        else {
            x = PyLong_FromSsize_t(vec[j]);
            if (x == NULL) {
                Py_DECREF(result);
                return NULL;
            }
        }
        PyList_SET_ITEM(result, j, x);
    }
    return result;
}

static PyObject *
Lattice_get_zero_columns(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    Lattice *L = (Lattice *)self;
    return Py_ssize_t_vec_to_list(L->zero_columns, L->num_zero_columns);
}

static PyObject *
Lattice_get_row_to_pivot(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    Lattice *L = (Lattice *)self;
    return Py_ssize_t_vec_to_list(L->row_to_pivot, L->rank);
}

static PyObject *
Lattice_get_col_to_pivot(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    Lattice *L = (Lattice *)self;
    return Py_ssize_t_vec_to_list(L->col_to_pivot, L->N);
}

static bool
make_pivots_positive(Lattice *L)
{
    PyObject *zero = PyLong_FromIntptr(0);
    if (zero == NULL) {
        return true;
    }
    assert(L->first_HNF_row <= L->rank);
    for (Py_ssize_t i = 0; i < L->first_HNF_row; i++) {
        TagInt *row = L->basis[i];
        Py_ssize_t j = L->row_to_pivot[i];
        TagInt pivot = row[j];
        assert(!TagInt_is_zero(pivot));
        int neg = TagInt_is_negative(pivot, zero);
        if (neg == -1) {
            goto error;
        }
        if (neg) {
            if (Vector_negate_impl(&row[j], L->N - j)) {
                goto error;
            }
        }
    }
    return false;
error:
    Py_DECREF(zero);
    return true;
}

static bool
Lattice_HNFify_impl(Lattice *L, Py_ssize_t first_row_to_fix)
{
    if (first_row_to_fix >= L->first_HNF_row) {
        return false;
    }
    if (make_pivots_positive(L)) {
        L->corrupted = true;
        return true;
    }
    Py_ssize_t R = L->rank;

    // The following tries to keep the entries small:
    // Work from the bottom-right corner of the matrix,
    // converting just this corner to HNF.
    // To expand the size of this corner, incorporate the next row.
    // basis[0]   : 0 0 p ? ? ? ? ... ? ? ? ? ? ?
    // basis[1]   : 0 0 0 0 p ? ? ... ? ? ? ? ? ?
    //  .........................................
    // basis[R-3] : 0 0 0 0 0 0 0 ... p ? a ? ? a <====== reduce these two "a"s (row i)
    // basis[R-2] : 0 0 0 0 0 0 0 ... 0 0 p (HNF)   <-----using these two pivot rows
    // basis[R-1] : 0 0 0 0 0 0 0 ... 0 0 0 0 0 p   <--/

    Py_ssize_t N = L->N;
    for (Py_ssize_t i = L->first_HNF_row - 1; i >= first_row_to_fix; i--) {
        TagInt *row_to_reduce = L->basis[i];
        for (Py_ssize_t ii = i + 1; ii < R; ii++) {
            TagInt *pivot_row = L->basis[ii];
            Py_ssize_t jj = L->row_to_pivot[ii];
            TagInt pivot = pivot_row[jj];
            TagInt above = row_to_reduce[jj];
            if (TagInt_is_zero(above)) {
                continue;
            }
#if USE_FAST_PATHS
            if (!TagInt_is_pointer(above) && !TagInt_is_pointer(pivot)) {
                intptr_t p = unpack_integer(pivot);
                assert(p > 0);
                intptr_t a = unpack_integer(above);
                intptr_t q = a / p;
                if (a % p < 0) {
                    q -= 1; // floor division
                }
                intptr_t neg_q = -q;
                PyObject *neg_q_obj = NULL;
                if (row_op_impl_with_intptr(&pivot_row[jj], &row_to_reduce[jj], N-jj, neg_q, &neg_q_obj)) {
                    Py_XDECREF(neg_q_obj);
                    goto error;
                }
                Py_XDECREF(neg_q_obj);
                assert(!TagInt_is_pointer(row_to_reduce[jj]));
                assert(0 <= unpack_integer(row_to_reduce[jj]));
                assert(unpack_integer(row_to_reduce[jj]) < p);
                continue;
            }
#endif
            PyObject *pivot_obj = TagInt_to_object(pivot);
            if (pivot_obj == NULL) {
                goto error;
            }
            PyObject *above_obj = TagInt_to_object(above);
            if (above_obj == NULL) {
                Py_DECREF(pivot_obj);
                goto error;
            }
            PyObject *q_obj = pylong_floor_divide(above_obj, pivot_obj);
            Py_DECREF(pivot_obj);
            Py_DECREF(above_obj);
            if (q_obj == NULL) {
                goto error;
            }
            if (!pylong_bool(q_obj)) {
                Py_DECREF(q_obj);
                continue;
            }
            PyObject *neg_q_obj = pylong_negative(q_obj);
            Py_DECREF(q_obj);
            if (neg_q_obj == NULL) {
                goto error;
            }
            if (row_op_impl_with_objects(&pivot_row[jj], &row_to_reduce[jj], N-jj, neg_q_obj)) {
                Py_DECREF(neg_q_obj);
                goto error;
            }
            Py_DECREF(neg_q_obj);
        }
    }
    assert(first_row_to_fix <= L->rank);
    L->first_HNF_row = first_row_to_fix;
    return false;
error:
    L->corrupted = true;
    return true;
}

static PyObject *
Lattice_HNFify(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    if (Lattice_HNFify_impl(L, 0)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
Lattice_unnormalized_nonzero_invariants(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    // Just find the entries of the diagonal
    // This function doesn't ensure they have all the right divisibility.
    if (((Lattice *)self)->corrupted) {
        err_corrupted();
        return NULL;
    }
    Lattice *L = (Lattice *)Lattice_copy(self, NULL);
    if (L == NULL){
        return NULL;
    }
    PyObject *result = NULL;
    char *delete_row=NULL, *delete_col=NULL;
    TagInt *scratch = NULL;
    if (!(result = PyList_New(L->rank))) {
        goto error;
    }
    Py_ssize_t result_index = 0;
    if (!(delete_row = PyMem_Malloc(L->rank)) ) {
        PyErr_NoMemory();
        goto error;
    }
    if (!(delete_col = PyMem_Malloc(L->N))) {
        PyErr_NoMemory();
        goto error;
    }
    if (!(scratch = PyMem_Malloc(L->rank * sizeof(TagInt)))) {
        PyErr_NoMemory();
        goto error;
    }
    while (L->rank) {
        if (Lattice_HNFify_impl(L, 0)) {
            goto error;
        }
        // detect if each pivot is alone in its row and column.
        // if so, add it to the result and forget about it.
        memset(delete_row, 0, L->rank);
        memset(delete_col, 0, L->N);
        Py_ssize_t num_deletions = 0;
        for (Py_ssize_t i = 0; i < L->rank; i++) {
            Py_ssize_t j = L->row_to_pivot[i];
            bool alone = true;
            for (Py_ssize_t ii = 0; alone && ii < i; ii++) {
                alone = alone && TagInt_is_zero(L->basis[ii][j]);
            }
            for (Py_ssize_t jj = j + 1; alone && jj < L->N; jj++) {
                alone = alone && TagInt_is_zero(L->basis[i][jj]);
            }
            if (alone) {
                PyObject *p = TagInt_to_object(L->basis[i][j]);
                if (p == NULL) {
                    goto error;
                }
                PyList_SET_ITEM(result, result_index, p);
                result_index++;
                delete_row[i] = 1;
                delete_col[j] = 1;
                num_deletions++;
            }
        }
        // Transpose with these deletions
        Py_ssize_t next_R = L->rank - num_deletions;
        Lattice *next_L = (Lattice *)Lattice_new_impl(&Lattice_Type, next_R, L->HNF_policy, next_R);
        if (next_L == NULL) {
            goto error;
        }
        for (Py_ssize_t j = 0; j < L->N; j++) {
            if (delete_col[j]) {
                continue;
            }
            Py_ssize_t scratch_index = 0;
            for (Py_ssize_t i = 0; i < L->rank; i++) {
                if (delete_row[i]) {
                    continue;
                }
                scratch[scratch_index++] = L->basis[i][j];
            }
            assert(scratch_index == next_L->N);
            if (Lattice_add_vector_impl(next_L, scratch)) {
                Py_DECREF(next_L);
                goto error;
            }
        }
        Py_DECREF(L);
        L = next_L;
    }
    assert(result_index == PyList_GET_SIZE(result));
    PyMem_Free(delete_row);
    PyMem_Free(delete_col);
    PyMem_Free(scratch);
    Py_DECREF(L);
    return result;
error:
    Py_DECREF(L);
    Py_XDECREF(result);
    PyMem_Free(delete_row);
    PyMem_Free(delete_col);
    PyMem_Free(scratch);
    return NULL;
}

static PyObject *
Lattice_nonzero_invariants(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *mathmodule = NULL;
    PyObject *result = NULL;
    PyObject *gcdfunc = NULL;
    if (!(mathmodule = PyImport_ImportModule("math"))) {
        goto error;
    }
    if (!(gcdfunc = PyObject_GetAttrString(mathmodule, "gcd"))) {
        goto error;
    }
    if (!(result = Lattice_unnormalized_nonzero_invariants(self, NULL))) {
        goto error;
    }
    // "comb sort" but instead of swapping,
    // replace [a, b] with [gcd(a, b), lcm(a, b)]
    Py_ssize_t gap = PyList_GET_SIZE(result);
    bool sorted = false;
    while (!sorted) {
        gap = (Py_ssize_t) ((float)gap / 1.3);
        if (gap <= 1) {
            gap = 1;
            sorted = true;
        }
        for (Py_ssize_t i = 0; i + gap < PyList_GET_SIZE(result); i++) {
            PyObject *a = PyList_GET_ITEM(result, i);
            PyObject *b = PyList_GET_ITEM(result, i + gap);
            PyObject *r = pylong_remainder(b, a);
            bool need_to_mingle = pylong_bool(r);
            Py_DECREF(r);
            if (!need_to_mingle) {
                continue;
            }
            PyObject *g = PyObject_CallFunctionObjArgs(gcdfunc, a, b, NULL);
            if (g == NULL) {
                goto error;
            }
            PyObject *a_div_g = pylong_floor_divide(a, g);
            PyList_SET_ITEM(result, i, g);
            Py_DECREF(a);
            if (a_div_g == NULL) {
                goto error;
            }
            PyObject *lcm = pylong_multiply(a_div_g, b);
            PyList_SET_ITEM(result, i + gap, lcm);
            Py_DECREF(b);
            Py_DECREF(a_div_g);
            if (lcm == NULL) {
                goto error;
            }
            sorted = false;
        }
    }
    Py_DECREF(mathmodule);
    Py_DECREF(gcdfunc);
    return result;
error:
    Py_XDECREF(mathmodule);
    Py_XDECREF(gcdfunc);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *
Lattice_invariants(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *result = Lattice_nonzero_invariants(self, NULL);
    if (result == NULL) {
        return NULL;
    }
    PyObject *zero = PyLong_FromLong(0);
    if (zero == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    Py_ssize_t N = ((Lattice *)self)->N;
    while (PyList_GET_SIZE(result) < N) {
        if (PyList_Append(result, zero) < 0) {
            Py_DECREF(result);
            Py_DECREF(zero);
            return NULL;
        }
    }
    Py_DECREF(zero);
    return result;
}

static PyObject *
Lattice_repr(PyObject *self)
{
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        return PyUnicode_FromString("<corrupted Lattice>");
    }
    PyObject *result = NULL;
    PyObject *maxrank_str = NULL;
    PyObject *tolist=NULL, *tolist_str=NULL;
    if (L->maxrank != L->N) {
        if (!(maxrank_str = PyUnicode_FromFormat(", maxrank=%zd", L->maxrank))) { goto error; }
    }
    else {
        if (!(maxrank_str = PyUnicode_FromStringAndSize("", 0))) { goto error; }
    }
    if (L->rank) {
        if (!(tolist = Lattice_tolist(self, NULL))) { goto error; }
        if (!(tolist_str = PyUnicode_FromFormat(", %R", tolist))) { goto error; }
    }
    else {
        if (!(tolist_str = PyUnicode_FromStringAndSize("", 0))) { goto error; }
    }
    result = PyUnicode_FromFormat("Lattice(%zd%U%U%s)",
        L->N, tolist_str, maxrank_str,
        L->HNF_policy == 0 ? ", HNF_policy=0" : ""
    );
error:
    Py_XDECREF(maxrank_str);
    Py_XDECREF(tolist);
    Py_XDECREF(tolist_str);
    return result;
}

static PyObject *
Lattice_str(PyObject *self)
{
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        return PyUnicode_FromString("<corrupted Lattice>");
    }
    Py_ssize_t R = L->rank, N = L->N;
    if (R == 0) {
        return PyUnicode_FromFormat("<zero Lattice in Z^%zd>", N);
    }
    PyObject *reprs_by_row = PyList_New(R);
    if (reprs_by_row == NULL) {
        return NULL;
    }
    Py_ssize_t max_width = 0;
    for (Py_ssize_t i = 0; i < R; i++) {
        PyObject *row_reprs = Vector_str_parts(L->basis[i], L->N, &max_width);
        if (row_reprs == NULL) {
            Py_DECREF(reprs_by_row);
            return NULL;
        }
        PyList_SET_ITEM(reprs_by_row, i, row_reprs);
    }
    PyObject *empty=NULL, *space=NULL, *lbracket=NULL, *rbracket=NULL, *newline=NULL;
    PyObject *parts=NULL, *result=NULL;
    if (!(empty = PyUnicode_FromStringAndSize("", 0))) { goto error; }
    if (!(space = PyUnicode_FromStringAndSize(" ", 1))) { goto error; }
    if (!(lbracket = PyUnicode_FromStringAndSize("[", 1))) { goto error; }
    if (!(rbracket = PyUnicode_FromStringAndSize("]", 1))) { goto error; }
    if (!(newline = PyUnicode_FromStringAndSize("\n", 1))) { goto error; }
    if (!(parts = PyList_New(0))) { goto error; }
    for (Py_ssize_t i = 0; i < R; i++) {
        if (i > 0) {
            if (PyList_Append(parts, newline) < 0) {
                goto error;
            }
        }
        if (PyList_Append(parts, lbracket) < 0) {
            goto error;
        }
        PyObject *row_reprs = PyList_GET_ITEM(reprs_by_row, i);
        for (Py_ssize_t j = 0; j < N; j++) {
            PyObject *r = PyList_GET_ITEM(row_reprs, j);
            Py_ssize_t num_spaces = max_width + (j > 0) - PyUnicode_GET_LENGTH(r);
            while (num_spaces) {
                if (PyList_Append(parts, space) < 0) {
                    goto error;
                }
                num_spaces--;
            }
            if (PyList_Append(parts, r) < 0) {
                goto error;
            }
        }
        if (PyList_Append(parts, rbracket) < 0) {
            goto error;
        }
    }
    result = PyUnicode_Join(empty, parts);
error:
    Py_DECREF(reprs_by_row);
    Py_XDECREF(empty); Py_XDECREF(space); Py_XDECREF(newline);
    Py_XDECREF(lbracket); Py_XDECREF(rbracket);
    Py_XDECREF(parts);
    return result;
}

static PyObject *
Lattice___getnewargs_ex__(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    assert(Py_TYPE(self) == &Lattice_Type);
    Lattice *L = (Lattice *)self;
    if (L->corrupted) {
        err_corrupted();
        return NULL;
    }
    PyObject *tolist = Lattice_tolist(self, NULL);
    if (tolist == NULL) {
        return NULL;
    }
    return Py_BuildValue("(nN){sisn}",
        L->N, tolist,
        "HNF_policy", L->HNF_policy, "maxrank", L->maxrank);
}

static PyMethodDef Lattice_methods[] = {
    {"clear", Lattice_clear, METH_NOARGS,
     "L.clear() replaces the Lattice with the zero Lattice in the same ambient dimension"},
    {"copy", Lattice_copy, METH_NOARGS,
     "L.copy() makes a copy of the lattice L"},
    {"add_vector", Lattice_add_vector, METH_O,
     "L.add_vector(v) adds the vector v to the Lattice L"},
    {"get_basis", Lattice_get_basis, METH_NOARGS,
     "L.get_basis() returns a list of basis vectors for L"},
    {"tolist", Lattice_tolist, METH_NOARGS,
     "L.tolist() returns a python list of list of int representing the basis of this lattice."},
    {"_get_zero_columns", Lattice_get_zero_columns, METH_NOARGS,
     "L._get_zero_columns() returns a list of the indices of the zero columns of the lattice L."},
    {"_get_row_to_pivot", Lattice_get_row_to_pivot, METH_NOARGS,
     "L._get_row_to_pivot() returns a list mapping i-->j"},
    {"_get_col_to_pivot", Lattice_get_col_to_pivot, METH_NOARGS,
     "L._get_col_to_pivot() returns a list mapping j-->(i or None)"},
    {"_assert_consistent", Lattice__assert_consistent, METH_NOARGS,
     "assert the invariants of the Lattice data structure for C debugging"},
    {"HNFify", Lattice_HNFify, METH_NOARGS,
     "Apply row operations to convert the stored basis to Hermite normal form (HNF)"},
    {"invariants", Lattice_invariants, METH_NOARGS,
     "L.invariants() returns a the list of integer invariants, ordered by divisibility, potentially including both 0s and 1s."},
    {"nonzero_invariants", Lattice_nonzero_invariants, METH_NOARGS,
     "The same as L.invariants(), but exclude zeros."},
    {"_unnormalized_nonzero_invariants", Lattice_unnormalized_nonzero_invariants, METH_NOARGS,
     "The same as L.nonzero_invariants(), but don't guarantee any divisibility."},
    {"is_full", Lattice_is_full, METH_NOARGS,
     "Returns True iff the lattice is the entirety of Z^N"},
    {"full", Lattice_full, METH_O | METH_CLASS,
     "Returns the entire lattice Z^N"},
    {"__getnewargs_ex__", Lattice___getnewargs_ex__, METH_NOARGS,
     "get the arguments to reconstruct this Lattice"},
    {"coefficients_of", Lattice_coefficients_of, METH_O,
     "L.coefficients_of(v) returns the vector of coefficients needed to write L as linear combination of vectors in L."},
    {"linear_combination", Lattice_linear_combination, METH_O,
     "L.coefficients_of(w) returns the linear combination of the basis vectors of L with coefficients given by w."},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static PyMemberDef Lattice_members[] = {
    {"ambient_dimension", Py_T_PYSSIZET, offsetof(Lattice, N), Py_READONLY,
     "the dimension N of the ambient lattice Z^N"},
    {"maxrank", Py_T_PYSSIZET, offsetof(Lattice, maxrank), Py_READONLY,
     "the maximum basis size this Lattice has room to store"},
    {"rank", Py_T_PYSSIZET, offsetof(Lattice, rank), Py_READONLY,
     "the number of Vectors in a basis for this Lattice"},
    {"HNF_policy", Py_T_INT, offsetof(Lattice, HNF_policy), Py_READONLY,
     "whether this Lattice automatically maintains Hermite normal form"},
    {"_first_HNF_row", Py_T_PYSSIZET, offsetof(Lattice, first_HNF_row), Py_READONLY,
     "the first vector in this Lattice that is already in Hermite normal form"},
    {NULL}
};

static PySequenceMethods Lattice_as_sequence = {
    .sq_contains = Lattice_contains
};

static PyNumberMethods Lattice_as_number = {
    .nb_inplace_add = Lattice_inplace_add,
    .nb_add = Lattice_add,
};

static PyTypeObject Lattice_Type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mutable_lattice.Lattice",
    .tp_doc = PyDoc_STR("mutable lattice spanned by some integer vectors"),
    .tp_basicsize = sizeof(Lattice) - sizeof(PyObject *),
    .tp_itemsize = sizeof(PyObject *),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Lattice_new,
    .tp_dealloc = Lattice_dealloc,
    .tp_methods = Lattice_methods,
    .tp_members = Lattice_members,
    .tp_as_sequence = &Lattice_as_sequence,
    .tp_as_number = &Lattice_as_number,
    .tp_richcompare = Lattice_richcompare,
    .tp_str = Lattice_str,
    .tp_repr = Lattice_repr,
};

/*********************************************************************/
/* relations_among, transpose                                        */
/*********************************************************************/

static PyObject *
relations_among(PyObject *mod, PyObject *arg)
{
    if (!PyList_CheckExact(arg)) {
        PyErr_SetString(PyExc_TypeError, "relations_among(vecs) argument must be a list");
        return NULL;
    }
    Py_ssize_t R = PyList_GET_SIZE(arg);
    if (R == 0) {
        return Lattice_new_impl(&Lattice_Type, 0, 1, 0);
    }
    Py_ssize_t N = -1;
    for (Py_ssize_t i = 0; i < R; i++) {
        PyObject *v = PyList_GET_ITEM(arg, i);
        if (Py_TYPE(v) != &Vector_Type) {
            PyErr_SetString(PyExc_TypeError, "relations_among(vecs) argument must be a list of Vectors");
            return NULL;
        }
        if (i == 0) {
            N = Py_SIZE(v);
        } else {
            if (Py_SIZE(v) != N) {
                PyErr_SetString(PyExc_ValueError, "length mismatch in relations_among");
                return NULL;
            }
        }
    }
    if (N > PY_SSIZE_T_MAX - R || (size_t)N + (size_t)R > PY_SSIZE_T_MAX/sizeof(TagInt *)) {
        PyErr_SetNone(PyExc_OverflowError);
        return NULL;
    }
    TagInt *scratch = (TagInt *)PyMem_Malloc((N + R) * sizeof(TagInt *));
    if (scratch == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    Lattice *L = (Lattice *)Lattice_new_impl(&Lattice_Type, N + R, 1, R);
    if (L == NULL) {
        PyMem_Free(scratch);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(arg); i++) {
        PyObject *v = PyList_GET_ITEM(arg, i);
        memcpy(scratch, Vector_get_vec(v), N*sizeof(TagInt *));
        memset(scratch + N, 0, R*sizeof(TagInt *));
        scratch[N+i] = TagInt_ONE;
        if (Lattice_add_vector_impl(L, scratch)) {
            PyMem_Free(scratch);
            Py_DECREF(L);
            return NULL;
        }
    }
    PyMem_Free(scratch);
    Lattice *result = (Lattice *)Lattice_new_impl(&Lattice_Type, R, 1, R);
    if (result == NULL) {
        Py_DECREF(L);
        return NULL;
    }
    assert(L->rank == R);
    for (Py_ssize_t i = 0; i < R; i++) {
        if (L->row_to_pivot[i] >= N) {
            if (Lattice_add_vector_impl(result, L->basis[i] + N)) {
                Py_DECREF(L);
                Py_DECREF(result);
                return NULL;
            }
        }
    }
    Py_DECREF(L);
    return (PyObject *)result;
}

static PyObject *
transpose(PyObject *mod, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "transpose(N, [v1, ..., vk]) takes 2 arguments");
        return NULL;
    }
    if (!PyLong_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "transpose(N, [v1, ..., vk]) first argument must be integer");
        return NULL;
    }
    Py_ssize_t N = PyLong_AsSsize_t(args[0]);
    if (N == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (N < 0) {
        PyErr_SetString(PyExc_ValueError, "transpose(N, [v1, ..., vk]) first argument cannot be negative");
        return NULL;
    }
    PyObject *data = args[1];
    if (!PyList_CheckExact(data)) {
        PyErr_SetString(PyExc_TypeError, "transpose(N, [v1, ..., vk]) second argument must be list");
        return NULL;
    }
    Py_ssize_t R = PyList_GET_SIZE(data);
    PyObject *result = PyList_New(N);
    if (result == NULL) {
        return NULL;
    }
    for (Py_ssize_t j = 0; j < N; j++) {
        PyObject *result_vec = Vector_zero_impl(R);
        if (result_vec == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        assert(PyList_Check(result));
        PyList_SET_ITEM(result, j, result_vec);
    }
    for (Py_ssize_t i = 0; i < R; i++) {
        assert(PyList_Check(data));
        PyObject *v = PyList_GET_ITEM(data, i);
        if (Py_TYPE(v) != &Vector_Type) {
            PyErr_SetString(PyExc_TypeError, "transpose(N, [v1, ..., vk]) second argument must be list of Vectors");
            Py_DECREF(result);
            return NULL;
        }
        if (Py_SIZE(v) != N) {
            PyErr_SetString(PyExc_ValueError, "transpose(N, [v1, ..., vk]) vectors must have length N");
            Py_DECREF(result);
            return NULL;
        }
        TagInt *vec = Vector_get_vec(v);
        for (Py_ssize_t j = 0; j < N; j++) {
            assert(PyList_Check(result));
            Vector_get_vec(PyList_GET_ITEM(result, j))[i] = TagInt_copy(vec[j]);
        }
    }
    return result;
}

/*********************************************************************/
/* Module stuff                                                      */
/*********************************************************************/

static int
mutable_lattice_module_exec(PyObject *m)
{
    if (PyType_Ready(&Vector_Type) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(m, "Vector", (PyObject *)&Vector_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&Lattice_Type) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(m, "Lattice", (PyObject *)&Lattice_Type) < 0) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot mutable_lattice_module_slots[] = {
    {Py_mod_exec, mutable_lattice_module_exec},
#ifdef Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
    {0, NULL}
};

static PyMethodDef mutable_lattice_methods[] = {
    {"row_op", (PyCFunction)(void(*)(void))row_op, METH_FASTCALL,
     "row_op(v, w, k) does w += k*v when v and w are Vectors"},
    {"generalized_row_op", (PyCFunction)(void(*)(void))generalized_row_op, METH_FASTCALL,
     "generalized_row_op(v, w, a, b, c, d) does (v, w) = (a*v+b*w, c*v+d*w) when v and w are Vectors"},
    {"xgcd", (PyCFunction)(void(*)(void))xgcd, METH_FASTCALL,
     "xgcd(a, b) returns a triple (x, y, g) of integers with x*a + y*b == g == gcd(a, b)"},
    {"relations_among", relations_among, METH_O,
     "relations_among([v0, ..., vk]) returns the Lattice of coefficient vectors for linear dependencies among the given Vectors"},
    {"transpose", (PyCFunction)(void(*)(void))transpose, METH_FASTCALL,
     "transpose(N, [v0, ..., vk]) transposes a length-k list of legth-N vectors into a length-N list of length-k vectors"},
    {NULL, NULL, 0, NULL}   /* sentinel */
};


static PyModuleDef mutable_lattice_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_mutable_lattice",
    .m_doc = "integer linear algebra using mutable sublattices of Z^n",
    .m_size = 0,
    .m_slots = mutable_lattice_module_slots,
    .m_methods = mutable_lattice_methods,
};

PyMODINIT_FUNC
PyInit__mutable_lattice(void)
{
    return PyModuleDef_Init(&mutable_lattice_module);
}