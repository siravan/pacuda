// SPDX-License-Identifier: 0BSD
//
// pacman recurrence runner: one CUDA kernel runs a whole RustiCol Direct-Arena recurrence plan
// (contributions, finalizations, closures), given as an array of 64-bit row words.
//
// Each thread is one phase-space point and walks every word in order. Rows depend only on rows
// of the same point, so the kernel needs no synchronisation, and all threads of a warp take the
// same branch of the dispatch. The plan is data: one build serves every process.
//
// A word holds, from the least significant bit: the op (4 bits), then the op's fields, whose
// widths the caller chooses per plan (`width`, see `pacman_recurrence.h`):
//   P  current plane (component), A  amplitude plane, F  momentum form, X  factor index,
//   S  executor slot (its scale), K  small component count.
//
//   op 1  clear currents          P base, P count
//   op 2–6  contribution          1 initialize, S slot, P parent0, P parent1, P destination,
//                                 F form0, F form1, X exact factor
//           2 Weyl–vector→Weyl a, 3 b, 4 antisymmetric tensor–vector, 5 vector wedge vector,
//           6 colour-ordered three-vector
//   op 7–9  propagator            P base, F form, X exact factor
//           7 Weyl a, 8 Weyl b, 9 Feynman vector
//   op 10 identity finalization   P base, K count, X exact factor
//   op 11 closure-reduce          P parent0, P parent1, A amplitude, X exact factor,
//                                 X first component factor, K count
//   op 12 source                  P destination, F form, X exact factor, S mass slot, 10-bit
//                                 template: family (3: 0 scalar, 1 Weyl, 2 Dirac, 3 vector),
//                                 orientation (2: particle, antiparticle), helicity + 2 (3),
//                                 chirality + 1 (2)
//
//   op 13 symjit row              1 initialize, P parent0, P parent1, P destination, F form0,
//                                 F form1, X exact factor: a contribution or finalization of a
//                                 symjit-compiled executor, by the binding of the last op 14
//   op 14 symjit binding          16-bit kernel index, 8-bit input count n, 8-bit output pair count
//                                 m, 1-bit direct-capable; then ceil((n + m) / 2) data words of two
//                                 32-bit codes each (low half first): the n input codes, then the m
//                                 destination components of the output pairs (see `symjit_row`)
//
// Sources (op 12, before all other rows) evaluate RustiCol's external wavefunctions
// (`wavefunctions.rs`, transcribed) in f64 from the f64 momentum forms, in their own kernel before
// the df64/f32 conversions, so they equal the CPU's sources bit for bit in every kernel's layout.
// `pacman_recurrence_v1_capabilities()` reports them (bit 0).
//
// The arithmetic is RustiCol's (`recurrence_intrinsic_direct.rs`, `recurrence_closure_direct.rs`)
// operation for operation; built with -fmad=false, the `f64` kernel equals RustiCol's bit for bit.
// The default kernel, `df64`, emulates f64 with pairs of f32 (for GPUs whose FP64 is slow, e.g.
// RTX). `mixed` computes in f32 except the propagator denominators 1/p^2, which it computes in
// df64 from df64 momentum forms (`mixed64`: in f64 from the f64 forms, for GPUs with fast FP64)
// and rounds once to f32; the amplitudes are accumulated in df64. See `Options` below for
// PACMAN_RECURRENCE_KERNEL.
//
// Build: nvcc -O3 -fmad=false -Xcompiler -fPIC -shared -cudart static -o librecurrence.so
//        recurrence_runner.cu

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

// ---- the interface: pacman_recurrence.h ----------------------------------------------------

#include "pacman_recurrence.h"

// ---- device code ---------------------------------------------------------------------------

// df64: a double emulated by a pair of floats (high, low), for GPUs with slow FP64. Accurate
// double-word algorithms (Joldes, Muller, Popescu, ACM TOMS 44(2), 2017): + AccurateDWPlusDW
// (Alg. 6), x DWTimesDW3 (Alg. 12), / DWDivDW2 (Alg. 17) with DWTimesFP3 (Alg. 9). They need
// correctly rounded f32 operations that are not contracted (-fmad=false) and an explicit fma.
struct D {
    float hi, lo;
    __device__ __forceinline__ D() {}
    __device__ __forceinline__ D(float h, float l) : hi(h), lo(l) {}
    // for the constants of the formulas (0, +-1, 2): exact in f32
    __device__ __forceinline__ explicit D(double x) : hi((float)x), lo(0.0f) {}
};

__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
}
__device__ __forceinline__ void fast_two_sum(float a, float b, float &s, float &e) {
    s = __fadd_rn(a, b);
    e = __fsub_rn(b, __fsub_rn(s, a));
}
__device__ __forceinline__ D operator+(D x, D y) {
    float sh, sl, th, tl, vh, vl;
    two_sum(x.hi, y.hi, sh, sl);
    two_sum(x.lo, y.lo, th, tl);
    fast_two_sum(sh, __fadd_rn(sl, th), vh, vl);
    D z;
    fast_two_sum(vh, __fadd_rn(tl, vl), z.hi, z.lo);
    return z;
}
__device__ __forceinline__ D operator-(D x) { return D(-x.hi, -x.lo); }
__device__ __forceinline__ D operator-(D x, D y) { return x + (-y); }
__device__ __forceinline__ D operator*(D x, D y) {
    float ch = __fmul_rn(x.hi, y.hi);
    float cl1 = __fmaf_rn(x.hi, y.hi, -ch);
    float tl0 = __fmul_rn(x.lo, y.lo);
    float tl1 = __fmaf_rn(x.hi, y.lo, tl0);
    float cl2 = __fmaf_rn(x.lo, y.hi, tl1);
    D z;
    fast_two_sum(ch, __fadd_rn(cl1, cl2), z.hi, z.lo);
    return z;
}
__device__ __forceinline__ D operator/(D x, D y) {
    float th = __fdiv_rn(x.hi, y.hi);
    float rh = __fmul_rn(y.hi, th);
    float rl1 = __fmaf_rn(y.hi, th, -rh);
    float rl3 = __fmaf_rn(y.lo, th, rl1);
    float r_hi, r_lo;
    fast_two_sum(rh, rl3, r_hi, r_lo);
    float pi_h = __fsub_rn(x.hi, r_hi);
    float delta = __fadd_rn(pi_h, __fsub_rn(x.lo, r_lo));
    D z;
    fast_two_sum(th, __fdiv_rn(delta, y.hi), z.hi, z.lo);
    return z;
}

// complex numbers over double (`C`, RustiCol's arithmetic) or df64
template <class R> struct Cx {
    R re, im;
};
typedef Cx<double> C;

template <class R> __device__ __forceinline__ Cx<R> cx(R re, R im) { Cx<R> c; c.re = re; c.im = im; return c; }
template <class R> __device__ __forceinline__ Cx<R> add(Cx<R> a, Cx<R> b) { return cx(a.re + b.re, a.im + b.im); }
template <class R> __device__ __forceinline__ Cx<R> sub(Cx<R> a, Cx<R> b) { return cx(a.re - b.re, a.im - b.im); }
template <class R> __device__ __forceinline__ Cx<R> neg(Cx<R> a) { return cx(-a.re, -a.im); }
template <class R> __device__ __forceinline__ Cx<R> mul(Cx<R> a, Cx<R> b) {
    return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <class R> __device__ __forceinline__ Cx<R> mul_i(Cx<R> a) { return cx(-a.im, a.re); }
template <class R> __device__ __forceinline__ Cx<R> mul_real(Cx<R> a, R s) { return cx(a.re * s, a.im * s); }

// a value of the arena (planes of doubles; a df64 arena holds float2 in the same 8 bytes)
template <class R> __device__ __forceinline__ R ldr(const double *base, size_t at);
template <> __device__ __forceinline__ double ldr<double>(const double *base, size_t at) { return base[at]; }
template <> __device__ __forceinline__ D ldr<D>(const double *base, size_t at) {
    float2 v = reinterpret_cast<const float2 *>(base)[at];
    return D(v.x, v.y);
}
// mixed: an f32 value in the df64 layout (high = the value, low = 0), so that the conversions,
// the downloads and the reduction of the df64 kernel apply unchanged
template <> __device__ __forceinline__ float ldr<float>(const double *base, size_t at) {
    return reinterpret_cast<const float2 *>(base)[at].x;
}
__device__ __forceinline__ void str(double *base, size_t at, float v) {
    reinterpret_cast<float2 *>(base)[at] = make_float2(v, 0.0f);
}
__device__ __forceinline__ void str(double *base, size_t at, double v) { base[at] = v; }
__device__ __forceinline__ void str(double *base, size_t at, D v) {
    reinterpret_cast<float2 *>(base)[at] = make_float2(v.hi, v.lo);
}

struct Arena {
    double *cre, *cim, *are, *aim;
    float *fcre, *fcim;  // mixed: the currents as compact f32 planes (same plane-major indices)
    const double *mom;
    const double *sre, *sim, *fre, *fim;
    size_t ld;
    const uint32_t *bind;  // the symjit bindings of the program (op 14)
};

template <class R>
__device__ __forceinline__ R reciprocal_minkowski_norm(R p0, R p1, R p2, R p3) {
    return R(1.0) / (p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3);
}

// The arithmetic of a kernel: R for the values; Den for the propagator denominators and the
// momentum forms they are computed from; Amp for the amplitude accumulation. f64 and df64 use
// R throughout; mixed (R = float) uses df64 (or f64, mixed64) denominators and df64 amplitudes.
template <class R, class DenT = R> struct Kind {
    typedef R real;
    typedef DenT den;
    typedef R amp;
};
template <class DenT> struct Kind<float, DenT> {
    typedef float real;
    typedef DenT den;
    typedef D amp;
};

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(double x) { return __double2float_rn(x); }
__device__ __forceinline__ float to_float(D x) { return __fadd_rn(x.hi, x.lo); }

// a momentum-form component as R (for numerators): the form is stored in Den's layout
template <class R, class DenT> __device__ __forceinline__ R ld_form(const double *mom, size_t at) {
    return ldr<R>(mom, at);
}
template <> __device__ __forceinline__ float ld_form<float, double>(const double *mom, size_t at) {
    return __double2float_rn(mom[at]);
}
template <> __device__ __forceinline__ float ld_form<float, D>(const double *mom, size_t at) {
    return reinterpret_cast<const float2 *>(mom)[at].x;
}

// a denominator value as R: identity for f64/df64 (R = Den), rounded to f32 for mixed
template <class R, class S> struct ToReal {
    __device__ __forceinline__ static R of(S x) { return x; }
};
template <class S> struct ToReal<float, S> {
    __device__ __forceinline__ static float of(S x) { return to_float(x); }
};

// widening an f32 to the amplitude accumulator
__device__ __forceinline__ D widen(float x) { return D(x, 0.0f); }
__device__ __forceinline__ double widen(double x) { return x; }
__device__ __forceinline__ D widen(D x) { return x; }

template <class R> __device__ __forceinline__ Cx<R> minkowski_dot(const Cx<R> *l, const Cx<R> *r) {
    return sub(sub(sub(mul(l[0], r[0]), mul(l[1], r[1])), mul(l[2], r[2])), mul(l[3], r[3]));
}

template <class R> __device__ __forceinline__ Cx<R> minkowski_real_dot(const Cx<R> *l, const R *q) {
    return sub(sub(sub(mul_real(l[0], q[0]), mul_real(l[1], q[1])), mul_real(l[2], q[2])),
               mul_real(l[3], q[3]));
}

// ---- the executors' arithmetic, in f64 and df64: operands in, values out ----

// components read from parent0 and parent1 and written, per contribution op (2..6)
template <int OP> struct Shape {
    static constexpr int c0 = OP == 4 ? 6 : OP == 5 || OP == 6 ? 4 : 2;
    static constexpr int c1 = 4;
    static constexpr int cd = OP == 4 || OP == 6 ? 4 : OP == 5 ? 6 : 2;
};

__host__ __device__ __forceinline__ void contribution_shape(int op, int *c0, int *c1, int *cd) {
    *c0 = op == 4 ? 6 : op == 5 || op == 6 ? 4 : 2;
    *c1 = 4;
    *cd = op == 4 || op == 6 ? 4 : op == 5 ? 6 : 2;
}

// contribution values before the scale (`o`, cd of them); q0, q1: the momentum forms (op 6)
template <class R>
__device__ __forceinline__ void contribution_values(int op, const Cx<R> *l, const Cx<R> *r, const R *q0,
                                                    const R *q1, Cx<R> *o) {
    typedef Cx<R> C;
    if (op == 2) {
        o[0] = add(add(sub(neg(mul_i(mul(l[0], r[3]))), mul_i(mul(l[1], r[1]))), mul(l[1], r[2])),
                   mul_i(mul(l[0], r[0])));
        o[1] = add(add(sub(neg(mul(l[0], r[2])), mul_i(mul(l[0], r[1]))), mul_i(mul(l[1], r[0]))),
                   mul_i(mul(l[1], r[3])));
    } else if (op == 3) {
        o[0] = add(add(add(neg(mul(l[1], r[2])), mul_i(mul(l[0], r[0]))), mul_i(mul(l[0], r[3]))),
                   mul_i(mul(l[1], r[1])));
        o[1] = add(add(add(neg(mul_i(mul(l[1], r[3]))), mul(l[0], r[2])), mul_i(mul(l[0], r[1]))),
                   mul_i(mul(l[1], r[0])));
    } else if (op == 4) {
        o[0] = add(add(mul(l[0], r[1]), mul(l[1], r[2])), mul(l[2], r[3]));
        o[1] = add(add(mul(l[0], r[0]), mul(l[3], r[2])), mul(l[4], r[3]));
        o[2] = add(sub(mul(l[1], r[0]), mul(l[3], r[1])), mul(l[5], r[3]));
        o[3] = sub(sub(mul(l[2], r[0]), mul(l[4], r[1])), mul(l[5], r[2]));
    } else if (op == 5) {
        o[0] = sub(mul(l[0], r[1]), mul(l[1], r[0]));
        o[1] = sub(mul(l[0], r[2]), mul(l[2], r[0]));
        o[2] = sub(mul(l[0], r[3]), mul(l[3], r[0]));
        o[3] = sub(mul(l[1], r[2]), mul(l[2], r[1]));
        o[4] = sub(mul(l[1], r[3]), mul(l[3], r[1]));
        o[5] = sub(mul(l[2], r[3]), mul(l[3], r[2]));
    } else {  // op == 6
        C vector_dot = minkowski_dot(l, r);
        C left_dot_parent1 = minkowski_real_dot(l, q1);
        C right_dot_parent0 = minkowski_real_dot(r, q0);
        for (int k = 0; k < 4; k++) {
            C momentum_term = mul_real(vector_dot, q0[k] - q1[k]);
            C current_term =
                mul_real(sub(mul(left_dot_parent1, r[k]), mul(right_dot_parent0, l[k])), R(2.0));
            o[k] = add(momentum_term, current_term);
        }
    }
}

// propagator values (ops 7..9): the 2 (Weyl) or 4 (vector) components `c` become `o`, scaled
template <class R>
__device__ __forceinline__ int propagator_values(int op, const Cx<R> *c, const R *p, R reciprocal,
                                                 Cx<R> row_factor, Cx<R> *o) {
    typedef Cx<R> C;
    if (op == 9) {
        C scale = mul_real(mul(row_factor, cx(R(0.0), R(-1.0))), reciprocal);
        for (int k = 0; k < 4; k++) o[k] = mul(c[k], scale);
        return 4;
    }
    C scale = mul_real(row_factor, reciprocal);
    C o0, o1;
    if (op == 7) {
        o0 = add(add(sub(neg(mul_i(mul_real(c[0], p[3]))), mul_i(mul_real(c[1], p[1]))),
                     mul_real(c[1], p[2])),
                 mul_i(mul_real(c[0], p[0])));
        o1 = add(add(sub(neg(mul_real(c[0], p[2])), mul_i(mul_real(c[0], p[1]))),
                     mul_i(mul_real(c[1], p[0]))),
                 mul_i(mul_real(c[1], p[3])));
    } else {
        o0 = add(add(add(neg(mul_real(c[1], p[2])), mul_i(mul_real(c[0], p[0]))),
                     mul_i(mul_real(c[0], p[3]))),
                 mul_i(mul_real(c[1], p[1])));
        o1 = add(add(add(neg(mul_i(mul_real(c[1], p[3]))), mul_real(c[0], p[2])),
                     mul_i(mul_real(c[0], p[1]))),
                 mul_i(mul_real(c[1], p[0])));
    }
    o[0] = mul(o0, scale);
    o[1] = mul(o1, scale);
    return 2;
}

// `execute_identity_finalization_rows`: factor × value
template <class R> __device__ __forceinline__ Cx<R> identity_value(R fr, R fi, Cx<R> v) {
    return cx(fr * v.re - fi * v.im, fr * v.im + fi * v.re);
}

// `execute_closure_reduce_rows`, one component: the amplitude increment
template <class R>
__device__ __forceinline__ Cx<R> closure_increment(Cx<R> l, Cx<R> r, R cr, R ci, R rfr, R rfi) {
    R scale_re = cr * rfr - ci * rfi;
    R scale_im = cr * rfi + ci * rfr;
    R product_re = l.re * r.re - l.im * r.im;
    R product_im = l.re * r.im + l.im * r.re;
    return cx(product_re * scale_re - product_im * scale_im,
              product_re * scale_im + product_im * scale_re);
}

// ---- external wavefunctions: RustiCol's `wavefunctions.rs`, transcribed operation for operation ----
// (f64; built with -fmad=false, so equal to the Rust code bit for bit). The crossing of an
// incoming leg is a negative energy, as in RustiCol.

typedef Cx<double> W;
__device__ __forceinline__ W c64(double re, double im) { return cx(re, im); }
__device__ __forceinline__ W scale_w(W c, double s) { return cx(c.re * s, c.im * s); }
__device__ __forceinline__ double fortran_sign(double value, double sign_source) {
    return copysign(fabs(value), sign_source);
}
// Rust's f64::max/min (the other operand when one is NaN), as fmax/fmin
__device__ __forceinline__ double rmax(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ double rmin(double a, double b) { return fmin(a, b); }

__device__ void ext_quark_weyl_array(const double *m, int helicity, int chirality, W *o) {
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    if (energy > 0.0) {
        double sqp0p3 = (px == 0.0 && py == 0.0 && pz < 0.0) ? 0.0 : sqrt(rmax(energy + pz, 0.0));
        W chi1 = c64(sqp0p3, 0.0);
        W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * energy), 0.0)
                               : c64((double)helicity * px / sqp0p3, -py / sqp0p3);
        if (helicity == 1 && chirality == 1) { o[0] = chi1; o[1] = chi2; return; }
        if (helicity == -1 && chirality == -1) { o[0] = chi2; o[1] = chi1; return; }
        o[0] = c64(0.0, 0.0); o[1] = c64(0.0, 0.0);
        return;
    }
    double sqp0p3 = (px == 0.0 && py == 0.0 && pz > 0.0) ? 0.0 : -sqrt(rmax(-(energy + pz), 0.0));
    W chi1 = c64(sqp0p3, 0.0);
    W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * fabs(energy)), 0.0)
                           : c64(-(double)helicity * (-px) / sqp0p3, py / sqp0p3);
    if (helicity == -1 && chirality == 1) { o[0] = chi1; o[1] = chi2; }
    else if (helicity == 1 && chirality == -1) { o[0] = chi2; o[1] = chi1; }
    else { o[0] = c64(0.0, 0.0); o[1] = c64(0.0, 0.0); }
}

__device__ void ext_antiquark_weyl_array(const double *m, int helicity, int chirality, W *o) {
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    if (energy > 0.0) {
        double sqp0p3 = (px == 0.0 && py == 0.0 && pz < 0.0) ? 0.0 : -sqrt(rmax(energy + pz, 0.0));
        W chi1 = c64(sqp0p3, 0.0);
        W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * energy), 0.0)
                               : c64(-(double)helicity * px / sqp0p3, py / sqp0p3);
        if (helicity == 1 && chirality == 1) { o[0] = chi2; o[1] = chi1; return; }
        if (helicity == -1 && chirality == -1) { o[0] = chi1; o[1] = chi2; return; }
        o[0] = c64(0.0, 0.0); o[1] = c64(0.0, 0.0);
        return;
    }
    double sqp0p3 = (px == 0.0 && py == 0.0 && pz > 0.0) ? 0.0 : sqrt(rmax(-(energy + pz), 0.0));
    W chi1 = c64(sqp0p3, 0.0);
    W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * fabs(energy)), 0.0)
                           : c64((double)helicity * (-px) / sqp0p3, -py / sqp0p3);
    if (helicity == -1 && chirality == 1) { o[0] = chi2; o[1] = chi1; }
    else if (helicity == 1 && chirality == -1) { o[0] = chi1; o[1] = chi2; }
    else { o[0] = c64(0.0, 0.0); o[1] = c64(0.0, 0.0); }
}

__device__ void ext_quark_dirac(const double *m, int helicity, W *o) {
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    W z = c64(0.0, 0.0);
    if (energy > 0.0) {
        double sqp0p3 = (px == 0.0 && py == 0.0 && pz < 0.0) ? 0.0 : sqrt(rmax(energy + pz, 0.0));
        W chi1 = c64(sqp0p3, 0.0);
        W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * energy), 0.0)
                               : c64((double)helicity * px / sqp0p3, -py / sqp0p3);
        if (helicity == 1) { o[0] = chi1; o[1] = chi2; o[2] = z; o[3] = z; return; }
        o[0] = z; o[1] = z; o[2] = chi2; o[3] = chi1;
        return;
    }
    double sqp0p3 = (px == 0.0 && py == 0.0 && pz > 0.0) ? 0.0 : -sqrt(rmax(-(energy + pz), 0.0));
    W chi1 = c64(sqp0p3, 0.0);
    W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * fabs(energy)), 0.0)
                           : c64(-(double)helicity * (-px) / sqp0p3, py / sqp0p3);
    if (-helicity == 1) { o[0] = chi1; o[1] = chi2; o[2] = z; o[3] = z; }
    else { o[0] = z; o[1] = z; o[2] = chi2; o[3] = chi1; }
}

__device__ void ext_antiquark_dirac(const double *m, int helicity, W *o) {
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    W z = c64(0.0, 0.0);
    if (energy > 0.0) {
        double sqp0p3 = (px == 0.0 && py == 0.0 && pz < 0.0) ? 0.0 : -sqrt(rmax(energy + pz, 0.0));
        W chi1 = c64(sqp0p3, 0.0);
        W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * energy), 0.0)
                               : c64(-(double)helicity * px / sqp0p3, py / sqp0p3);
        if (-helicity == 1) { o[0] = z; o[1] = z; o[2] = chi1; o[3] = chi2; return; }
        o[0] = chi2; o[1] = chi1; o[2] = z; o[3] = z;
        return;
    }
    double sqp0p3 = (px == 0.0 && py == 0.0 && pz > 0.0) ? 0.0 : sqrt(rmax(-(energy + pz), 0.0));
    W chi1 = c64(sqp0p3, 0.0);
    W chi2 = sqp0p3 == 0.0 ? c64(-(double)helicity * sqrt(2.0 * fabs(energy)), 0.0)
                           : c64((double)helicity * (-px) / sqp0p3, -py / sqp0p3);
    if (helicity == 1) { o[0] = z; o[1] = z; o[2] = chi1; o[3] = chi2; }
    else { o[0] = chi2; o[1] = chi1; o[2] = z; o[3] = z; }
}

// ext_quark_dirac_massive (antiquark = false) and ext_antiquark_dirac_massive (true)
__device__ void ext_dirac_massive(const double *m, int helicity, double mass, bool antiquark, W *o) {
    if (fabs(mass) < 1.0e-8) {
        if (antiquark) ext_antiquark_dirac(m, helicity, o);
        else ext_quark_dirac(m, helicity, o);
        return;
    }
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    int nsf = antiquark ? (energy > 0.0 ? -1 : 1) : (energy > 0.0 ? 1 : -1);
    int nh = nsf * helicity;
    double pp = fabs(sqrt(px * px + py * py + pz * pz));
    double omega1 = sqrt(fabs(energy) + pp);
    double omega2 = mass / omega1;
    double omega[2] = {omega1, omega2};
    double sf1 = (double)(1 + nsf + (1 - nsf) * nh) * 0.5;
    double sf2 = (double)(1 + nsf - (1 - nsf) * nh) * 0.5;
    int ip = (3 + nh) / 2 - 1, im = (3 - nh) / 2 - 1;
    double sfomeg[2] = {sf1 * omega[ip], sf2 * omega[im]};
    double spx = energy > 0.0 ? px : -px, spy = energy > 0.0 ? py : -py, spz = energy > 0.0 ? pz : -pz;
    double pp3 = rmax(pp + spz, 0.0);
    W chi1 = pp == 0.0 ? c64(1.0, 0.0) : c64(sqrt(pp3 * 0.5 / pp), 0.0);
    W chi2;
    if (pp3 == 0.0 || pp == 0.0) chi2 = c64(-(double)nh, 0.0);
    else {
        double denom = sqrt(2.0 * pp * pp3);
        chi2 = c64((double)nh * spx / denom, (antiquark ? spy : -spy) / denom);
    }
    W chi[2] = {chi1, chi2};
    double first = antiquark ? sfomeg[0] : sfomeg[1], second = antiquark ? sfomeg[1] : sfomeg[0];
    o[0] = scale_w(chi[im], first);
    o[1] = scale_w(chi[ip], first);
    o[2] = scale_w(chi[im], second);
    o[3] = scale_w(chi[ip], second);
}

__device__ void ext_gluon(const double *m, int helicity, W *o) {
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    double sqh = sqrt(0.5);
    if (energy > 0.0) {
        double hel = (double)helicity, pp = energy;
        double pt = sqrt(px * px + py * py);
        W wf3 = c64(hel * pt / pp * sqh, 0.0), wf1, wf2;
        if (pt != 0.0) {
            double pzpt = pz / (pp * pt) * sqh * hel;
            wf1 = c64(-px * pzpt, -py / pt * sqh);
            wf2 = c64(-py * pzpt, px / pt * sqh);
        } else {
            wf1 = c64(-hel * sqh, 0.0);
            wf2 = c64(0.0, fortran_sign(sqh, pz));
        }
        o[0] = c64(0.0, 0.0); o[1] = wf1; o[2] = wf2; o[3] = wf3;
        return;
    }
    double hel = (double)(-helicity), pp = -energy;
    double pt = sqrt(px * px + py * py);
    W wf3 = c64(hel * pt / pp * sqh, 0.0), wf1, wf2;
    if (pt != 0.0) {
        double pzpt = -pz / (pp * pt) * sqh * hel;
        wf1 = c64(px * pzpt, py / pt * sqh);
        wf2 = c64(py * pzpt, -px / pt * sqh);
    } else {
        wf1 = c64(-hel * sqh, 0.0);
        wf2 = c64(0.0, -fortran_sign(sqh, pz));
    }
    o[0] = c64(0.0, 0.0); o[1] = wf1; o[2] = wf2; o[3] = wf3;
}

__device__ void ext_massive_vector(const double *m0, int helicity, double mass, W *o) {
    double m[4] = {m0[0], m0[1], m0[2], m0[3]};
    if (m[0] < 0.0) {  // ext_massive_vector(negate(momentum), -helicity, mass)
        for (int k = 0; k < 4; k++) m[k] = -m[k];
        helicity = -helicity;
    }
    double energy = m[0], px = m[1], py = m[2], pz = m[3];
    double sqh = sqrt(0.5);
    double hel = (double)helicity;
    double nsvahl = (double)abs(helicity);
    double pt2 = px * px + py * py;
    double pp = rmin(energy, sqrt(pt2 + pz * pz));
    double pt = rmin(pp, sqrt(pt2));
    double hel0 = 1.0 - fabs(hel);
    if (pp == 0.0) {
        o[0] = c64(0.0, 0.0); o[1] = c64(-hel * sqh, 0.0); o[2] = c64(0.0, nsvahl * sqh); o[3] = c64(hel0, 0.0);
        return;
    }
    double emp = energy / (mass * pp);
    W wf0 = c64(hel0 * pp / mass, 0.0);
    W wf3 = c64(hel0 * pz * emp + hel * pt / pp * sqh, 0.0);
    W wf1, wf2;
    if (pt != 0.0) {
        double pzpt = pz / (pp * pt) * sqh * hel;
        wf1 = c64(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh);
        wf2 = c64(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh);
    } else {
        wf1 = c64(-hel * sqh, 0.0);
        wf2 = c64(0.0, nsvahl * fortran_sign(sqh, pz));
    }
    o[0] = wf0; o[1] = wf1; o[2] = wf2; o[3] = wf3;
}

// the wavefunction of a source template (`execute_direct_source_rows`); its component count
__device__ int source_values(uint32_t code, double mass, const double *p, W *o) {
    int family = code & 7, orientation = (code >> 3) & 3;
    int helicity = (int)((code >> 5) & 7) - 2, chirality = (int)((code >> 8) & 3) - 1;
    if (family == 0) { o[0] = c64(1.0, 0.0); return 1; }
    if (family == 1) {
        if (orientation == 0) ext_quark_weyl_array(p, helicity, chirality, o);
        else ext_antiquark_weyl_array(p, helicity, chirality, o);
        return 2;
    }
    if (family == 2) { ext_dirac_massive(p, helicity, mass, orientation == 1, o); return 4; }
    if (mass == 0.0) ext_gluon(p, helicity, o);
    else ext_massive_vector(p, helicity, mass, o);
    return 4;
}

// an f64 value in a kernel's current layout, as the uploads convert it (`to_df64_planes`,
// `f64_to_f32_planes`)
template <class R> __device__ __forceinline__ R narrow(double x);
template <> __device__ __forceinline__ double narrow<double>(double x) { return x; }
template <> __device__ __forceinline__ float narrow<float>(double x) { return __double2float_rn(x); }
template <> __device__ __forceinline__ D narrow<D>(double x) {
    float hi = __double2float_rn(x);
    return D(hi, __double2float_rn(x - (double)hi));
}

// ---- symjit rows (op 13): pyAmpliCol's symjit-compiled executors -----------------------------
//
// The kernels are generated (`pacuda-gen cuda`) and included at build time with
// -DPACUDA_KERNELS='"file.cuh"': f64 functions `pk_*(in, out)`, the dispatch
// `pacuda_kernel(k, in, out)`, PACUDA_KERNEL_COUNT, PACUDA_MAX_IN, PACUDA_MAX_OUT and
// `pacuda_kernel_sha256[k]` (the hex SHA-256 of kernel k's `.plane.symjit`). With
// PACUDA_SIMD_LANES = L > 0 the points in whole blocks of L of a call take the kernels' symjit
// 2.27.3 AVX-order variants, the rest the scalar order, as RustiCol pinned to symjit 2.27.3 runs
// a tile (SIMD kernel per block of lanes from point 0, scalar kernel for the remainder). Without them the
// runner has no symjit kernels (capability bit 2 clear) and refuses op 13.
#ifdef PACUDA_KERNELS
#include PACUDA_KERNELS
#else
#define PACUDA_KERNEL_COUNT 0
#define PACUDA_MAX_IN 1
#define PACUDA_MAX_OUT 2
#define PACUDA_SIMD_LANES 0
static const char *const pacuda_kernel_sha256[1] = {""};
static __device__ void pacuda_kernel(uint32_t, bool, const double *, double *) {}
#endif

template <class T> __device__ __forceinline__ double to_f64(T x);
template <> __device__ __forceinline__ double to_f64<double>(double x) { return x; }
template <> __device__ __forceinline__ double to_f64<float>(float x) { return (double)x; }
template <> __device__ __forceinline__ double to_f64<D>(D x) { return __dadd_rn((double)x.hi, (double)x.lo); }

// binding input codes: bits 0-2 the kind, bit 3 the parent or momentum operand, bit 4 imaginary,
// bits 5- the component (or the scale slot)
enum { B_PARENT, B_MOMENTUM, B_DESTINATION, B_SCALE, B_FACTOR };

// ---- programs: the words rewritten by the host (`prepare` below), 32 bytes per instruction ----
//
// w[0] = kind | op << 8 | flags << 16; the fields follow in w[1..7]. The program is the plan's
// rows with the contribution runs stably sorted by destination (each destination keeps its order
// of rows, so its sum is unchanged) and the destination of a run of rows on one destination kept
// in registers (F_FIRST loads or starts it, F_LAST stores it).

enum { K_CLEAR = 1, K_CONTRIB, K_PROP, K_IDENTITY, K_CLOSURE, K_SOURCE, K_SYMJIT };
enum { F_INIT = 1, F_FIRST = 2, F_LAST = 4 };

struct IOp {
    uint32_t w[8];
};

__device__ __forceinline__ IOp load_op(const IOp *ops, uint32_t n) {
    const uint4 *q = reinterpret_cast<const uint4 *>(ops + n);
    uint4 x = __ldg(q), y = __ldg(q + 1);
    IOp o;
    o.w[0] = x.x; o.w[1] = x.y; o.w[2] = x.z; o.w[3] = x.w;
    o.w[4] = y.x; o.w[5] = y.y; o.w[6] = y.z; o.w[7] = y.w;
    return o;
}

// contributions w[1] scale slot, w[2] p0, w[3] p1, w[4] dst, w[5] f0, w[6] f1, w[7] factor;
// clear w[1] base, w[2] count; propagator w[1] base, w[2] form, w[3] factor; identity w[1] base,
// w[2] count, w[3] factor; closure w[1] p0, w[2] p1, w[3] amplitude, w[4] factor, w[5] first,
// w[6] count.
// the currents: f64 or df64 slots of the arena (`cre`, `cim`), or compact f32 planes for mixed
template <class R> __device__ __forceinline__ Cx<R> ld_cur(const Arena &a, size_t at) {
    return cx(ldr<R>(a.cre, at), ldr<R>(a.cim, at));
}
template <> __device__ __forceinline__ Cx<float> ld_cur<float>(const Arena &a, size_t at) {
    return cx(a.fcre[at], a.fcim[at]);
}
template <class R> __device__ __forceinline__ void st_cur(const Arena &a, size_t at, Cx<R> v) {
    str(a.cre, at, v.re);
    str(a.cim, at, v.im);
}
template <> __device__ __forceinline__ void st_cur<float>(const Arena &a, size_t at, Cx<float> v) {
    a.fcre[at] = v.re;
    a.fcim[at] = v.im;
}
template <class R>
__device__ __forceinline__ Cx<R> ld_cr(const Arena &a, uint32_t plane, size_t i) {
    return ld_cur<R>(a, plane * a.ld + i);
}

// A row of a symjit executor, as RustiCol's `execute_rows` runs it: the kernel's inputs gathered
// by its binding (`parameter_bindings`: arena planes and scalars), the kernel on them (in f64),
// then each output pair scaled by the row's exact factor into the destination component:
// `dest (+)= re fr - im fi, re fi + im fr` (`apply_complex_scaled`), or written as it is for a
// direct-capable finalization whose factor is exactly 1 (`direct_outputs_are_structurally_safe`).
// The values pass through f64 in every kernel (for the f64 kernel: RustiCol's arithmetic).
template <class K>
__device__ __noinline__ void symjit_row(const Arena &a, const IOp &o, int flags, size_t i, uint32_t points) {
    typedef typename K::real R;
    typedef typename K::den Den;
    const uint32_t *b = a.bind + o.w[7];
    uint32_t kernel = b[0], n_in = b[1], pairs = b[2], direct = b[3];
    const uint32_t *code = b + 4, *comp = b + 4 + n_in;
    uint32_t p[2] = {o.w[2], o.w[3]}, f[2] = {o.w[5] & 0xffff, o.w[5] >> 16}, dst = o.w[4];
    double fr = to_f64(ldr<R>(a.fre, o.w[6])), fi = to_f64(ldr<R>(a.fim, o.w[6]));
    double in[PACUDA_MAX_IN], out[PACUDA_MAX_OUT];
    for (uint32_t k = 0; k < n_in; k++) {
        uint32_t c = code[k], kind = c & 7, which = (c >> 3) & 1, im = (c >> 4) & 1, n = c >> 5;
        double v = 0.0;
        if (kind == B_PARENT || kind == B_DESTINATION) {
            Cx<R> x = ld_cr<R>(a, (kind == B_PARENT ? p[which] : dst) + n, i);
            v = to_f64(im ? x.im : x.re);
        } else if (kind == B_MOMENTUM) {
            v = to_f64(ldr<Den>(a.mom, (f[which] * 4 + n) * a.ld + i));
        } else if (kind == B_SCALE) {
            v = to_f64(ldr<R>(a.sre, n));
        } else {
            v = im ? fi : fr;
        }
        in[k] = v;
    }
    bool simd = PACUDA_SIMD_LANES > 0 && i < (size_t)(points / (PACUDA_SIMD_LANES > 0 ? PACUDA_SIMD_LANES : 1)) * PACUDA_SIMD_LANES;
    pacuda_kernel(kernel, simd, in, out);
    bool as_is = direct && __double_as_longlong(fr) == __double_as_longlong(1.0) &&
                 __double_as_longlong(fi) == 0;
    for (uint32_t j = 0; j < pairs; j++) {
        double re = out[2 * j], im = out[2 * j + 1];
        size_t at = (dst + comp[j]) * a.ld + i;
        double sr, si;
        if (as_is) {
            sr = re;
            si = im;
        } else {
            sr = __dsub_rn(__dmul_rn(re, fr), __dmul_rn(im, fi));
            si = __dadd_rn(__dmul_rn(re, fi), __dmul_rn(im, fr));
            if (!(flags & F_INIT)) {
                Cx<R> d = ld_cur<R>(a, at);
                sr = __dadd_rn(to_f64(d.re), sr);
                si = __dadd_rn(to_f64(d.im), si);
            }
        }
        st_cur<R>(a, at, cx(narrow<R>(sr), narrow<R>(si)));
    }
}

template <class K, int OP>
__device__ __forceinline__ void contribution(const Arena &a, const IOp &o, int flags,
                                             Cx<typename K::real> *acc, size_t i) {
    typedef typename K::real R;
    typedef typename K::den Den;
    typedef Cx<R> C;
    constexpr int op = OP, c0 = Shape<OP>::c0, c1 = Shape<OP>::c1, cd = Shape<OP>::cd;
    uint32_t p0 = o.w[2], p1 = o.w[3], dst = o.w[4];
    C scale = mul(cx(ldr<R>(a.sre, o.w[1]), ldr<R>(a.sim, o.w[1])), cx(ldr<R>(a.fre, o.w[7]), ldr<R>(a.fim, o.w[7])));
    C l[6], r[4], v[6];
    R q0[4], q1[4];
#pragma unroll
    for (int k = 0; k < c0; k++) l[k] = ld_cr<R>(a, p0 + k, i);
#pragma unroll
    for (int k = 0; k < c1; k++) r[k] = ld_cr<R>(a, p1 + k, i);
    if (op == 6)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q0[k] = ld_form<R, Den>(a.mom, (o.w[5] * 4 + k) * a.ld + i);
            q1[k] = ld_form<R, Den>(a.mom, (o.w[6] * 4 + k) * a.ld + i);
        }
    contribution_values(op, l, r, q0, q1, v);
#pragma unroll
    for (int k = 0; k < cd; k++) {
        C c = mul(v[k], scale);
        if (!(flags & F_FIRST)) acc[k] = add(acc[k], c);
        else if (flags & F_INIT) acc[k] = c;
        else acc[k] = add(ld_cr<R>(a, dst + k, i), c);
    }
    if (flags & F_LAST)
#pragma unroll
        for (int k = 0; k < cd; k++) st_cur<R>(a, (dst + k) * a.ld + i, acc[k]);
}

template <class K, int OP>
__device__ __forceinline__ void propagator(const Arena &a, const IOp &o, size_t i) {
    typedef typename K::real R;
    typedef typename K::den Den;
    typedef Cx<R> C;
    constexpr int m = OP == 9 ? 4 : 2;
    uint32_t base = o.w[1];
    R p[4];
    Den q[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        size_t at = (o.w[2] * 4 + k) * a.ld + i;
        q[k] = ldr<Den>(a.mom, at);
        p[k] = ld_form<R, Den>(a.mom, at);
    }
    // the denominator in Den, rounded once to R (f64/df64: R = Den, unchanged)
    R reciprocal = ToReal<R, Den>::of(reciprocal_minkowski_norm(q[0], q[1], q[2], q[3]));
    C c[4], v[4];
#pragma unroll
    for (int k = 0; k < m; k++) c[k] = ld_cr<R>(a, base + k, i);
    propagator_values(OP, c, p, reciprocal, cx(ldr<R>(a.fre, o.w[3]), ldr<R>(a.fim, o.w[3])), v);
#pragma unroll
    for (int k = 0; k < m; k++) st_cur<R>(a, (base + k) * a.ld + i, v[k]);
}

// The momentum forms from the external momenta, as RustiCol fills them (f64, in term order:
// 0.0 + c * p for the first term, then += c * p), into the f64 momentum planes.
__global__ void fill_forms(const double *__restrict__ external, uint32_t points, const uint32_t *form_terms,
                           uint32_t form_count, const uint32_t *slots, const double *coefficients,
                           double *mom, size_t ld) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    for (uint32_t f = 0; f < form_count; f++) {
        uint32_t first = form_terms[2 * f], n = form_terms[2 * f + 1];
        for (int k = 0; k < 4; k++) {
            double v = 0.0;
            for (uint32_t t = 0; t < n; t++) {
                double term = __dmul_rn(coefficients[first + t],
                                        external[((size_t)slots[first + t] * 4 + k) * points + i]);
                v = t == 0 ? __dadd_rn(0.0, term) : __dadd_rn(v, term);
            }
            mom[(f * 4 + k) * ld + i] = v;  // no term: 0.0, as RustiCol's fill(0.0)
        }
    }
}

// The sources (the leading K_SOURCE instructions): w[1] destination, w[2] form, w[3] factor,
// w[4] mass slot (a scale), w[5] template. Run before the conversions: the momentum forms, the
// scales and the factors are still f64, as on the CPU (`write_wavefunction`: factor × value).
template <class R>
__global__ void run_sources(const IOp *__restrict__ ops, uint32_t count, Arena a, uint32_t points) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    for (uint32_t n = 0; n < count; n++) {
        IOp o = load_op(ops, n);
        double p[4];
        for (int k = 0; k < 4; k++) p[k] = a.mom[(o.w[2] * 4 + k) * a.ld + i];
        double fr = a.fre[o.w[3]], fi = a.fim[o.w[3]], mass = a.sre[o.w[4]];
        W v[4];
        int c = source_values(o.w[5], mass, p, v);
        for (int k = 0; k < c; k++) {
            W s = cx(fr * v[k].re - fi * v[k].im, fr * v[k].im + fi * v[k].re);
            st_cur<R>(a, (o.w[1] + k) * a.ld + i, cx(narrow<R>(s.re), narrow<R>(s.im)));
        }
    }
}

// R = double: RustiCol's arithmetic; R = D: df64 on a df64 arena (all planes, momenta, scales and
// factors hold float2 (high, low) in place of their doubles; see `to_df64`, `to_f64_planes`).
template <class K>
__global__ void run_program(const IOp *__restrict__ ops, uint32_t count, Arena a, uint32_t points) {
    typedef typename K::real R;
    typedef typename K::amp Amp;
    typedef Cx<R> C;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    C acc[6];
    for (uint32_t n = 0; n < count; n++) {
        IOp o = load_op(ops, n);
        int kind = o.w[0] & 0xff, op = (o.w[0] >> 8) & 0xff, flags = o.w[0] >> 16;
        if (kind == K_CONTRIB) {
            switch (op) {
            case 2: contribution<K, 2>(a, o, flags, acc, i); break;
            case 3: contribution<K, 3>(a, o, flags, acc, i); break;
            case 4: contribution<K, 4>(a, o, flags, acc, i); break;
            case 5: contribution<K, 5>(a, o, flags, acc, i); break;
            default: contribution<K, 6>(a, o, flags, acc, i); break;
            }
        } else if (kind == K_SYMJIT) {
            symjit_row<K>(a, o, flags, i, points);
        } else if (kind == K_CLEAR) {
            for (uint32_t k = 0; k < o.w[2]; k++) st_cur<R>(a, (o.w[1] + k) * a.ld + i, cx(R(0.0), R(0.0)));
        } else if (kind == K_PROP) {
            if (op == 7) propagator<K, 7>(a, o, i);
            else if (op == 8) propagator<K, 8>(a, o, i);
            else propagator<K, 9>(a, o, i);
        } else if (kind == K_IDENTITY) {
            R fr = ldr<R>(a.fre, o.w[3]), fi = ldr<R>(a.fim, o.w[3]);
            for (uint32_t k = 0; k < o.w[2]; k++) {
                size_t at = (o.w[1] + k) * a.ld + i;
                C v = identity_value(fr, fi, ld_cur<R>(a, at));
                st_cur<R>(a, at, v);
            }
        } else if (kind == K_CLOSURE) {
            R rfr = ldr<R>(a.fre, o.w[4]), rfi = ldr<R>(a.fim, o.w[4]);
            size_t at = o.w[3] * a.ld + i;
            // the amplitudes accumulate in Amp (= R; df64 for mixed)
            Amp dr = ldr<Amp>(a.are, at), di = ldr<Amp>(a.aim, at);
            for (uint32_t k = 0; k < o.w[6]; k++) {
                C v = closure_increment(ld_cr<R>(a, o.w[1] + k, i), ld_cr<R>(a, o.w[2] + k, i),
                                        ldr<R>(a.fre, o.w[5] + k), ldr<R>(a.fim, o.w[5] + k), rfr, rfi);
                dr = dr + widen(v.re);
                di = di + widen(v.im);
            }
            str(a.are, at, dr);
            str(a.aim, at, di);
        }
    }
}

// in-place conversion of `count` doubles to df64 (float2 in the same 8 bytes)
__global__ void to_df64(double *p, size_t count) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < count; k += (size_t)gridDim.x * blockDim.x) {
        double x = p[k];
        float hi = __double2float_rn(x);
        float lo = __double2float_rn(x - (double)hi);
        reinterpret_cast<float2 *>(p)[k] = make_float2(hi, lo);
    }
}

__global__ void to_df64_planes(double *p, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        double x = p[at];
        float hi = __double2float_rn(x);
        float lo = __double2float_rn(x - (double)hi);
        reinterpret_cast<float2 *>(p)[at] = make_float2(hi, lo);
    }
}

// mixed: f64 planes to compact f32 planes (same indices) and back
__global__ void f64_to_f32_planes(const double *src, float *dst, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        dst[at] = __double2float_rn(src[at]);
    }
}
__global__ void f32_to_f64_planes(const float *src, double *dst, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        dst[at] = (double)src[at];
    }
}

__global__ void to_f64_planes(double *p, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        float2 v = reinterpret_cast<float2 *>(p)[at];
        p[at] = (double)v.x + (double)v.y;
    }
}

// ---- host side -----------------------------------------------------------------------------

namespace {

struct Device {
    double *cre = nullptr, *cim = nullptr, *are = nullptr, *aim = nullptr, *mom = nullptr;
    double *small = nullptr;  // scales, then factors
    float *fcre = nullptr, *fcim = nullptr;  // mixed: the compact f32 currents
    size_t fcur_capacity = 0;
    // forms on the device: the external momenta and the term table
    double *external = nullptr, *coefficients = nullptr;
    uint32_t *form_terms = nullptr, *slots = nullptr;
    size_t external_capacity = 0, form_capacity = 0, term_capacity = 0, slot_capacity = 0;
    size_t cre_capacity = 0, cim_capacity = 0, are_capacity = 0, aim_capacity = 0,
           momentum_capacity = 0, small_capacity = 0;
    // the last run with PACMAN_RECURRENCE_KEEP_AMPLITUDES (for a reduction)
    bool kept = false, kept_df64 = false;
    size_t kept_stride = 0, kept_points = 0, kept_planes = 0;
    // reduction buffers
    uint32_t *rplanes = nullptr;
    double *rweights = nullptr, *routput = nullptr;
    size_t rcount_capacity = 0, rweights_capacity = 0, routput_capacity = 0;
    // statistics (PACMAN_RUNNER_STATS=1)
    uint64_t calls = 0, points = 0, copies = 0, bytes = 0, reductions = 0;
    double seconds = 0;
    bool stats = false;
};

Device device;

// ---- programs (host) ---------------------------------------------------------------------------

struct Row {
    int op;           // the word's op (1..14)
    uint32_t f[8];    // its fields in word order
    std::vector<uint32_t> data;  // op 14: the binding's codes
};

std::vector<Row> decode(const uint64_t *words, uint32_t count, const uint8_t *width) {
    std::vector<Row> rows;
    rows.reserve(count);
    for (uint32_t n = 0; n < count; n++) {
        uint64_t w = words[n];
        if ((w & 15) == 14) {  // a symjit binding and its data words
            Row r{};
            r.op = 14;
            r.f[0] = (uint32_t)(w >> 4) & 0xffff;   // kernel
            r.f[1] = (uint32_t)(w >> 20) & 0xff;    // inputs
            r.f[2] = (uint32_t)(w >> 28) & 0xff;    // output pairs
            r.f[3] = (uint32_t)(w >> 36) & 1;       // direct-capable
            uint32_t codes = r.f[1] + r.f[2];
            for (uint32_t k = 0; k < codes; k++) {
                uint64_t d = n + 1 + k / 2 < count ? words[n + 1 + k / 2] : 0;
                r.data.push_back((uint32_t)(k % 2 ? d >> 32 : d));
            }
            n += (codes + 1) / 2;
            rows.push_back(r);
            continue;
        }
        auto take = [&](int bits) {
            uint32_t v = (uint32_t)(w & ((1ull << bits) - 1));
            w >>= bits;
            return v;
        };
        Row r{};
        r.op = (int)take(4);
        const int P = width[W_P], A = width[W_A], F = width[W_F], X = width[W_X], S = width[W_S],
                  K = width[W_K];
        int fields[8] = {0}, nf = 0;
        if (r.op == 1) { fields[0] = P; fields[1] = P; nf = 2; }
        else if (r.op >= 2 && r.op <= 6) {
            int f[8] = {1, S, P, P, P, F, F, X};
            for (int k = 0; k < 8; k++) fields[k] = f[k];
            nf = 8;
        } else if (r.op >= 7 && r.op <= 9) { fields[0] = P; fields[1] = F; fields[2] = X; nf = 3; }
        else if (r.op == 10) { fields[0] = P; fields[1] = K; fields[2] = X; nf = 3; }
        else if (r.op == 11) {
            int f[6] = {P, P, A, X, X, K};
            for (int k = 0; k < 6; k++) fields[k] = f[k];
            nf = 6;
        } else if (r.op == 12) {
            int f[5] = {P, F, X, S, 10};
            for (int k = 0; k < 5; k++) fields[k] = f[k];
            nf = 5;
        } else if (r.op == 13) {
            int f[7] = {1, P, P, P, F, F, X};
            for (int k = 0; k < 7; k++) fields[k] = f[k];
            nf = 7;
        }
        for (int k = 0; k < nf; k++) r.f[k] = take(fields[k]);
        rows.push_back(r);
    }
    return rows;
}

// The contribution runs stably sorted by destination, where that is safe: no row of the run reads
// a component the run writes, and the destinations of the run do not overlap. Each destination
// keeps its own order of rows, so its value is unchanged bit for bit.
void sort_contribution_runs(std::vector<Row> &rows) {
    size_t n = 0;
    while (n < rows.size()) {
        if (rows[n].op < 2 || rows[n].op > 6) { n++; continue; }
        size_t end = n;
        while (end < rows.size() && rows[end].op >= 2 && rows[end].op <= 6) end++;
        std::unordered_map<uint32_t, uint32_t> dst_base;  // component -> destination base
        bool safe = true;
        for (size_t k = n; k < end && safe; k++) {
            int c0, c1, cd;
            contribution_shape(rows[k].op, &c0, &c1, &cd);
            for (int q = 0; q < cd; q++) {
                auto it = dst_base.emplace(rows[k].f[4] + q, rows[k].f[4]);
                if (it.first->second != rows[k].f[4]) safe = false;  // overlapping destinations
            }
        }
        for (size_t k = n; k < end && safe; k++) {
            int c0, c1, cd;
            contribution_shape(rows[k].op, &c0, &c1, &cd);
            for (int q = 0; q < c0 && safe; q++) safe = !dst_base.count(rows[k].f[2] + q);
            for (int q = 0; q < c1 && safe; q++) safe = !dst_base.count(rows[k].f[3] + q);
        }
        if (safe)
            std::stable_sort(rows.begin() + n, rows.begin() + end,
                             [](const Row &x, const Row &y) { return x.f[4] < y.f[4]; });
        n = end;
    }
}

IOp make_op(int kind, int op, int flags) {
    IOp o{};
    o.w[0] = (uint32_t)kind | (uint32_t)op << 8 | (uint32_t)flags << 16;
    return o;
}

// `bindings`: the symjit bindings (op 14) as `symjit_row` reads them: kernel, inputs, output
// pairs, direct-capable, the input codes, the destination components
std::vector<IOp> program(const std::vector<Row> &rows, std::vector<uint32_t> &bindings) {
    std::vector<IOp> ops;
    int64_t binding = -1;
    for (size_t n = 0; n < rows.size(); n++) {
        const Row &r = rows[n];
        if (r.op == 14) {
            binding = (int64_t)bindings.size();
            bindings.insert(bindings.end(), {r.f[0], r.f[1], r.f[2], r.f[3]});
            bindings.insert(bindings.end(), r.data.begin(), r.data.end());
            continue;
        }
        if (r.op == 13) {
            IOp o = make_op(K_SYMJIT, 13, r.f[0] ? F_INIT : 0);
            o.w[1] = 0;
            o.w[2] = r.f[1]; o.w[3] = r.f[2]; o.w[4] = r.f[3];
            o.w[5] = r.f[4] | r.f[5] << 16;
            o.w[6] = r.f[6];
            o.w[7] = (uint32_t)binding;  // checked by `prepare`
            ops.push_back(o);
            continue;
        }
        if (r.op >= 2 && r.op <= 6) {
            bool first = n == 0 || rows[n - 1].op < 2 || rows[n - 1].op > 6 || rows[n - 1].f[4] != r.f[4];
            bool last = n + 1 == rows.size() || rows[n + 1].op < 2 || rows[n + 1].op > 6 ||
                        rows[n + 1].f[4] != r.f[4];
            IOp o = make_op(K_CONTRIB, r.op, (r.f[0] ? F_INIT : 0) | (first ? F_FIRST : 0) | (last ? F_LAST : 0));
            for (int k = 1; k < 8; k++) o.w[k] = r.f[k];
            ops.push_back(o);
        } else {
            int kind = r.op == 1 ? K_CLEAR : r.op <= 9 ? K_PROP : r.op == 10 ? K_IDENTITY
                     : r.op == 11 ? K_CLOSURE : K_SOURCE;
            IOp o = make_op(kind, r.op, 0);
            for (int k = 0; k < 7; k++) o.w[k + 1] = r.f[k];
            ops.push_back(o);
        }
    }
    return ops;
}

// One prepared program: its instructions on the device.
struct Prepared {
    uint32_t count;
    IOp *ops;
    uint32_t sources = 0;  // the leading K_SOURCE instructions (run by `run_sources`)
    uint32_t *bindings = nullptr;  // the symjit bindings (op 14), on the device
};

bool failed(cudaError_t e, const char *step) {
    if (e == cudaSuccess) return false;
    fprintf(stderr, "pacman recurrence runner: %s: %s\n", step, cudaGetErrorString(e));
    return true;
}

bool grow(double **p, size_t *capacity, size_t doubles, const char *what) {
    if (doubles <= *capacity) return true;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *capacity = 0;
    if (failed(cudaMalloc((void **)p, doubles * sizeof(double)), what)) return false;
    *capacity = doubles;
    return true;
}

// copy the plane runs of re and im between host and device
// `n` planes of `stride` doubles from `src` to `dst`: only their first `points` values (the tile
// can be much longer than the call's points)
bool copy_planes(void *dst, const void *src, size_t n, size_t stride, size_t points, cudaMemcpyKind kind) {
    size_t pitch = stride * sizeof(double);
    if (failed(cudaMemcpy2D(dst, pitch, src, pitch, points * sizeof(double), n, kind), "copy")) return false;
    device.copies++;
    device.bytes += n * points * sizeof(double);
    return true;
}

// copy the plane runs of re and im between host and device
bool copy_runs(double *dre, double *dim, double *hre, double *him, const uint32_t *runs,
               uint32_t count, uint32_t planes, size_t stride, size_t points, bool to_device) {
    for (uint32_t r = 0; r < count; r++) {
        uint32_t first = runs[2 * r], n = runs[2 * r + 1];
        if (first > planes || n > planes - first) return false;
        size_t offset = (size_t)first * stride;
        cudaMemcpyKind kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (to_device ? !copy_planes(dre + offset, hre + offset, n, stride, points, kind) ||
                            !copy_planes(dim + offset, him + offset, n, stride, points, kind)
                      : !copy_planes(hre + offset, dre + offset, n, stride, points, kind) ||
                            !copy_planes(him + offset, dim + offset, n, stride, points, kind))
            return false;
    }
    return true;
}

// The configuration (`pacman_recurrence_v1_configure`, before the first run; the defaults
// otherwise): the kernel (df64, the default: f32-pair arithmetic on a df64 copy of the arena, for
// GPUs whose FP64 is slow; f64: RustiCol's arithmetic bit for bit; mixed, mixed64), the threads
// per block (128), whether the contribution rows are sorted by destination (yes), statistics at
// exit (no). The runner reads no environment variables.
enum { KERNEL_DF64 = PACMAN_RECURRENCE_KERNEL_DF64, KERNEL_F64 = PACMAN_RECURRENCE_KERNEL_F64,
       KERNEL_MIXED = PACMAN_RECURRENCE_KERNEL_MIXED, KERNEL_MIXED64 = PACMAN_RECURRENCE_KERNEL_MIXED64 };
struct Options {
    int kernel = KERNEL_DF64;
    bool df64 = true;     // the arena (currents, amplitudes, constants) is in the df64 layout
    bool df64_forms = true;  // the momentum forms too (all but f64 and mixed64)
    int block = 128;
    bool sort = true;
};

Options current_options;
bool runs_started = false;

bool sort_option() { return current_options.sort; }

Options options() { return current_options; }

// the program of `call`'s words, built once per words key
std::unordered_map<uint64_t, Prepared> prepared_programs;

const Prepared *prepare(const pacman_recurrence_call *call) {
    static Prepared scratch{0, nullptr};
    if (call->words_key) {
        auto it = prepared_programs.find(call->words_key);
        if (it != prepared_programs.end()) return &it->second;
    }
    std::vector<Row> rows = decode(call->words, call->word_count, call->width);
    if (sort_option()) sort_contribution_runs(rows);
    for (const Row &r : rows)
        if ((r.op == 13 || r.op == 14) && PACUDA_KERNEL_COUNT == 0) {
            fprintf(stderr, "pacman recurrence runner: a symjit row, but this runner has no symjit kernels\n");
            return nullptr;
        }
    std::vector<uint32_t> bindings;
    std::vector<IOp> ops = program(rows, bindings);
    for (const IOp &o : ops)
        if ((o.w[0] & 0xff) == K_SYMJIT) {
            uint32_t b = o.w[7];
            if (b == UINT32_MAX || b + 4 > bindings.size() || bindings[b] >= (uint32_t)PACUDA_KERNEL_COUNT ||
                bindings[b + 1] > PACUDA_MAX_IN || 2 * bindings[b + 2] > PACUDA_MAX_OUT ||
                b + 4 + bindings[b + 1] + bindings[b + 2] > bindings.size()) {
                fprintf(stderr, "pacman recurrence runner: a symjit row without a valid binding\n");
                return nullptr;
            }
        }
    Prepared p{(uint32_t)ops.size(), nullptr};
    if (!bindings.empty() &&
        (failed(cudaMalloc((void **)&p.bindings, bindings.size() * sizeof(uint32_t)), "bindings") ||
         failed(cudaMemcpy(p.bindings, bindings.data(), bindings.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                "bindings")))
        return nullptr;
    while (p.sources < ops.size() && (ops[p.sources].w[0] & 0xff) == K_SOURCE) p.sources++;
    for (size_t n = p.sources; n < ops.size(); n++)
        if ((ops[n].w[0] & 0xff) == K_SOURCE) {
            fprintf(stderr, "pacman recurrence runner: a source row follows other rows\n");
            return nullptr;
        }
    if (failed(cudaMalloc((void **)&p.ops, std::max<size_t>(1, ops.size()) * sizeof(IOp)), "program") ||
        failed(cudaMemcpy(p.ops, ops.data(), ops.size() * sizeof(IOp), cudaMemcpyHostToDevice), "program"))
        return nullptr;
    if (call->words_key) return &(prepared_programs[call->words_key] = p);
    if (scratch.ops) cudaFree(scratch.ops);
    if (scratch.bindings) cudaFree(scratch.bindings);
    scratch = p;
    return &scratch;
}

bool launch(const Prepared &p, const Arena &arena, uint32_t points) {
    Options o = options();
    Arena a = arena;
    a.bind = p.bindings;
    unsigned block = (unsigned)o.block, grid = (points + block - 1) / block;
    const IOp *ops = p.ops + p.sources;
    uint32_t count = p.count - p.sources;
    switch (o.kernel) {
    case KERNEL_F64: run_program<Kind<double>><<<grid, block>>>(ops, count, a, points); break;
    case KERNEL_MIXED: run_program<Kind<float, D>><<<grid, block>>>(ops, count, a, points); break;
    case KERNEL_MIXED64: run_program<Kind<float, double>><<<grid, block>>>(ops, count, a, points); break;
    default: run_program<Kind<D>><<<grid, block>>>(ops, count, a, points); break;
    }
    return !failed(cudaGetLastError(), "launch");
}

// the sources, into the kernel's current layout, from the f64 forms, scales and factors
bool launch_sources(const Prepared &p, const Arena &a, uint32_t points) {
    if (p.sources == 0) return true;
    Options o = options();
    unsigned block = 128, grid = (points + block - 1) / block;
    switch (o.kernel) {
    case KERNEL_F64: run_sources<double><<<grid, block>>>(p.ops, p.sources, a, points); break;
    case KERNEL_MIXED:
    case KERNEL_MIXED64: run_sources<float><<<grid, block>>>(p.ops, p.sources, a, points); break;
    default: run_sources<D><<<grid, block>>>(p.ops, p.sources, a, points); break;
    }
    return !failed(cudaGetLastError(), "sources");
}

// in-place f64 <-> df64 conversions of `doubles` values, and of plane runs of re and im
bool convert(void (*f)(double *, size_t), double *p, size_t doubles) {
    if (doubles == 0) return true;
    unsigned grid = (unsigned)std::min<size_t>(4096, (doubles + 255) / 256);
    f<<<grid, 256>>>(p, doubles);
    return !failed(cudaGetLastError(), "conversion");
}

// the first `points` values of `planes` planes of `stride` doubles
bool convert_planes(bool to_df, double *p, size_t planes, size_t stride, size_t points) {
    size_t count = planes * points;
    if (count == 0) return true;
    unsigned grid = (unsigned)std::min<size_t>(4096, (count + 255) / 256);
    if (to_df) to_df64_planes<<<grid, 256>>>(p, planes, stride, points);
    else to_f64_planes<<<grid, 256>>>(p, planes, stride, points);
    return !failed(cudaGetLastError(), "conversion");
}

bool convert_runs(bool to_df, double *re, double *im, const uint32_t *runs, uint32_t count,
                  size_t stride, size_t points) {
    for (uint32_t r = 0; r < count; r++) {
        size_t offset = (size_t)runs[2 * r] * stride, n = runs[2 * r + 1];
        if (!convert_planes(to_df, re + offset, n, stride, points) ||
            !convert_planes(to_df, im + offset, n, stride, points))
            return false;
    }
    return true;
}

// mixed: the current runs between the f64 staging arena and the compact f32 planes
bool convert_current_runs(bool to_f32, const uint32_t *runs, uint32_t count, size_t stride, size_t points) {
    for (uint32_t r = 0; r < count; r++) {
        size_t offset = (size_t)runs[2 * r] * stride, n = runs[2 * r + 1];
        size_t total = n * points;
        if (total == 0) continue;
        unsigned grid = (unsigned)std::min<size_t>(4096, (total + 255) / 256);
        for (int part = 0; part < 2; part++) {
            double *d = (part ? device.cim : device.cre) + offset;
            float *f = (part ? device.fcim : device.fcre) + offset;
            if (to_f32) f64_to_f32_planes<<<grid, 256>>>(d, f, n, stride, points);
            else f32_to_f64_planes<<<grid, 256>>>(f, d, n, stride, points);
        }
        if (failed(cudaGetLastError(), "conversion")) return false;
    }
    return true;
}

bool grow_f32(float **p, size_t floats, const char *what) {
    if (*p) cudaFree(*p);
    *p = nullptr;
    return !failed(cudaMalloc((void **)p, floats * sizeof(float)), what);
}

void print_stats() {
    if (!device.stats || device.calls == 0) return;
    fprintf(stderr,
            "pacman recurrence runner: %llu calls, %llu points, %llu reductions, %llu copies (%.1f MB), "
            "%.3f s (%.2f us/point)\n",
            (unsigned long long)device.calls, (unsigned long long)device.points,
            (unsigned long long)device.reductions, (unsigned long long)device.copies, device.bytes / 1e6,
            device.seconds,
            device.points ? device.seconds * 1e6 / device.points : 0.0);
}

double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

}  // namespace

static void initialize() {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        atexit(print_stats);
    }
}

static int run_locked(const pacman_recurrence_call *in) {
    runs_started = true;
    initialize();
    const size_t v1_size = offsetof(pacman_recurrence_call, zero_amplitude);
    const size_t v2_size = offsetof(pacman_recurrence_call, external_momenta);
    if (!in || in->abi_version != PACMAN_RECURRENCE_ABI ||
        (in->struct_size != sizeof(pacman_recurrence_call) && in->struct_size != v1_size &&
         in->struct_size != v2_size))
        return PACMAN_RECURRENCE_INVALID;
    pacman_recurrence_call full;
    memset(&full, 0, sizeof full);
    memcpy(&full, in, in->struct_size);
    const pacman_recurrence_call *call = &full;
    if (call->points == 0 || call->points > call->stride || !call->words ||
        (call->zero_amplitude_runs && !call->zero_amplitude))
        return PACMAN_RECURRENCE_INVALID;
    device.kept = false;
    bool keep = call->flags & PACMAN_RECURRENCE_KEEP_AMPLITUDES;
    double start = device.stats ? now() : 0;
    size_t stride = call->stride;
    size_t current_doubles = (size_t)call->current_planes * stride;
    size_t amplitude_doubles = (size_t)call->amplitude_planes * stride;
    // a grown arena loses its contents: the caller uploads what the words read
    if (!grow(&device.cre, &device.cre_capacity, current_doubles, "current arena") ||
        !grow(&device.cim, &device.cim_capacity, current_doubles, "current arena") ||
        !grow(&device.are, &device.are_capacity, amplitude_doubles, "amplitudes") ||
        !grow(&device.aim, &device.aim_capacity, amplitude_doubles, "amplitudes"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    bool mixed = options().kernel == KERNEL_MIXED || options().kernel == KERNEL_MIXED64;
    if (mixed && current_doubles > device.fcur_capacity) {
        device.fcur_capacity = 0;
        if (!grow_f32(&device.fcre, current_doubles, "current arena (f32)") ||
            !grow_f32(&device.fcim, current_doubles, "current arena (f32)"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        device.fcur_capacity = current_doubles;
    }
    size_t momentum_doubles = (size_t)call->momentum_planes * stride;
    size_t small = 2 * ((size_t)call->scale_count + call->factor_count);
    if (!grow(&device.mom, &device.momentum_capacity, momentum_doubles ? momentum_doubles : 1,
              "momenta") ||
        !grow(&device.small, &device.small_capacity, small ? small : 1, "factors"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;

    // inputs: momenta, scales and factors, the current and amplitude runs named
    double *dscale_re = device.small, *dscale_im = dscale_re + call->scale_count;
    double *dfactor_re = dscale_im + call->scale_count, *dfactor_im = dfactor_re + call->factor_count;
    bool forms_on_device = call->flags & PACMAN_RECURRENCE_FORMS_ON_DEVICE;
    if (forms_on_device) {
        uint32_t form_count = call->momentum_planes / 4;
        size_t external_doubles = (size_t)call->external_count * 4 * call->points;
        if (!call->external_momenta || !call->form_terms || (call->term_count && (!call->term_slots ||
                                                                                   !call->term_coefficients)))
            return PACMAN_RECURRENCE_INVALID;
        for (uint32_t f = 0; f < form_count; f++)
            if (call->form_terms[2 * f] > call->term_count ||
                call->form_terms[2 * f + 1] > call->term_count - call->form_terms[2 * f])
                return PACMAN_RECURRENCE_INVALID;
        for (uint32_t t = 0; t < call->term_count; t++)
            if (call->term_slots[t] >= call->external_count) return PACMAN_RECURRENCE_INVALID;
        auto grow_raw = [](void **p, size_t *capacity, size_t bytes, const char *what) {
            if (bytes <= *capacity) return true;
            if (*p) cudaFree(*p);
            *p = nullptr;
            *capacity = 0;
            if (failed(cudaMalloc(p, bytes), what)) return false;
            *capacity = bytes;
            return true;
        };
        size_t form_bytes = std::max<size_t>(1, 2 * (size_t)form_count) * sizeof(uint32_t);
        size_t term_bytes = std::max<size_t>(1, call->term_count) * sizeof(double);
        if (!grow_raw((void **)&device.external, &device.external_capacity,
                      std::max<size_t>(1, external_doubles) * sizeof(double), "external momenta") ||
            !grow_raw((void **)&device.form_terms, &device.form_capacity, form_bytes, "form terms") ||
            !grow_raw((void **)&device.coefficients, &device.term_capacity, term_bytes, "form terms"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        if (!grow_raw((void **)&device.slots, &device.slot_capacity,
                      std::max<size_t>(1, call->term_count) * sizeof(uint32_t), "form terms"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        if (failed(cudaMemcpy(device.external, call->external_momenta, external_doubles * sizeof(double),
                              cudaMemcpyHostToDevice), "external momenta") ||
            failed(cudaMemcpy(device.form_terms, call->form_terms, 2 * (size_t)form_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice), "form terms") ||
            (call->term_count &&
             (failed(cudaMemcpy(device.slots, call->term_slots, call->term_count * sizeof(uint32_t),
                                cudaMemcpyHostToDevice), "form terms") ||
              failed(cudaMemcpy(device.coefficients, call->term_coefficients,
                                call->term_count * sizeof(double), cudaMemcpyHostToDevice), "form terms"))))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        device.copies += 1;
        device.bytes += external_doubles * sizeof(double);
        if (form_count) {
            unsigned block = 128, grid = (call->points + block - 1) / block;
            fill_forms<<<grid, block>>>(device.external, call->points, device.form_terms, form_count,
                                        device.slots, device.coefficients, device.mom, stride);
            if (failed(cudaGetLastError(), "forms")) return PACMAN_RECURRENCE_DEVICE_ERROR;
        }
    }
    if ((momentum_doubles && !forms_on_device &&
         !copy_planes(device.mom, call->momenta, call->momentum_planes, stride, call->points,
                      cudaMemcpyHostToDevice)) ||
        (call->scale_count &&
         (failed(cudaMemcpy(dscale_re, call->scale_re, call->scale_count * sizeof(double),
                            cudaMemcpyHostToDevice), "scales") ||
          failed(cudaMemcpy(dscale_im, call->scale_im, call->scale_count * sizeof(double),
                            cudaMemcpyHostToDevice), "scales"))) ||
        (call->factor_count &&
         (failed(cudaMemcpy(dfactor_re, call->factor_re, call->factor_count * sizeof(double),
                            cudaMemcpyHostToDevice), "factors") ||
          failed(cudaMemcpy(dfactor_im, call->factor_im, call->factor_count * sizeof(double),
                            cudaMemcpyHostToDevice), "factors"))))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!copy_runs(device.cre, device.cim, call->current_re, call->current_im, call->upload_current,
                   call->upload_current_runs, call->current_planes, stride, call->points, true) ||
        !copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->upload_amplitude, call->upload_amplitude_runs, call->amplitude_planes,
                   stride, call->points, true))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    // amplitudes the caller has cleared: zero on the device (zero is zero in df64 too)
    for (uint32_t r = 0; r < call->zero_amplitude_runs; r++) {
        uint32_t first = call->zero_amplitude[2 * r], n = call->zero_amplitude[2 * r + 1];
        if (first > call->amplitude_planes || n > call->amplitude_planes - first)
            return PACMAN_RECURRENCE_INVALID;
        size_t pitch = stride * sizeof(double), width = (size_t)call->points * sizeof(double);
        if (failed(cudaMemset2D(device.are + (size_t)first * stride, pitch, 0, width, n), "zero") ||
            failed(cudaMemset2D(device.aim + (size_t)first * stride, pitch, 0, width, n), "zero"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
    }

    Arena a;
    a.bind = nullptr;
    a.cre = device.cre;
    a.cim = device.cim;
    a.fcre = device.fcre;
    a.fcim = device.fcim;
    a.are = device.are;
    a.aim = device.aim;
    a.mom = device.mom;
    a.sre = dscale_re;
    a.sim = dscale_im;
    a.fre = dfactor_re;
    a.fim = dfactor_im;
    a.ld = stride;
    // df64: the uploaded planes, the momenta and the constants become df64 in place; the device
    // arena stays df64 between calls (planes copied back are converted to f64 and back again)
    // the sources first, while the forms, scales and factors are f64
    const Prepared *prepared = prepare(call);
    if (!prepared) return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!launch_sources(*prepared, a, call->points)) return PACMAN_RECURRENCE_DEVICE_ERROR;
    bool df64 = options().df64;
    // mixed: the uploaded currents go into the compact f32 planes (the f64 arena is the staging)
    if (mixed && !convert_current_runs(true, call->upload_current, call->upload_current_runs, stride,
                                       call->points))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (df64 && ((!mixed && !convert_runs(true, device.cre, device.cim, call->upload_current,
                                          call->upload_current_runs, stride, call->points)) ||
                 !convert_runs(true, device.are, device.aim, call->upload_amplitude,
                               call->upload_amplitude_runs, stride, call->points) ||
                 (options().df64_forms &&
                  !convert_planes(true, device.mom, call->momentum_planes, stride, call->points)) ||
                 !convert(to_df64, device.small, small)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!launch(*prepared, a, call->points))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    uint32_t amplitude_runs = keep ? 0 : call->download_amplitude_runs;
    if (df64 && (!convert_runs(false, device.are, device.aim, call->download_amplitude,
                               amplitude_runs, stride, call->points) ||
                 (mixed ? !convert_current_runs(false, call->download_current,
                                                call->download_current_runs, stride, call->points)
                        : !convert_runs(false, device.cre, device.cim, call->download_current,
                                        call->download_current_runs, stride, call->points))))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (failed(cudaDeviceSynchronize(), "kernel")) return PACMAN_RECURRENCE_DEVICE_ERROR;

    if (!copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->download_amplitude, amplitude_runs,
                   call->amplitude_planes, stride, call->points, false) ||
        !copy_runs(device.cre, device.cim, call->current_re, call->current_im,
                   call->download_current, call->download_current_runs, call->current_planes,
                   stride, call->points, false))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    // back to the device layouts (mixed: the f32 currents stay authoritative)
    if (df64 && (!convert_runs(true, device.are, device.aim, call->download_amplitude,
                               amplitude_runs, stride, call->points) ||
                 (!mixed && !convert_runs(true, device.cre, device.cim, call->download_current,
                                          call->download_current_runs, stride, call->points))))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (keep) {
        device.kept = true;
        device.kept_df64 = df64;
        device.kept_stride = stride;
        device.kept_points = call->points;
        device.kept_planes = call->amplitude_planes;
    }
    device.calls++;
    device.points += call->points;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}

// ---- the LC diagonal reduction on the device --------------------------------------------------

// a df64 value as the double it stands for: as `to_f64_planes` converts it, so the reduction of a df64 run
// equals the host's reduction of its copied-back amplitudes bit for bit
__device__ __forceinline__ double as_double(double x) { return x; }
__device__ __forceinline__ double as_double(D x) { return __dadd_rn((double)x.hi, (double)x.lo); }

template <class R>
__global__ void reduce_kernel(const double *are, const double *aim, size_t ld, const uint32_t *planes,
                              const double *weights, uint32_t count, double *output, uint32_t points) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    double acc = output[i];
    for (uint32_t k = 0; k < count; k++) {
        size_t at = (size_t)__ldg(planes + k) * ld + i;
        double re = as_double(ldr<R>(are, at)), im = as_double(ldr<R>(aim, at));
        // weight * real.mul_add(real, imaginary * imaginary), then +=
        acc = __dadd_rn(acc, __dmul_rn(__ldg(weights + k), __fma_rn(re, re, __dmul_rn(im, im))));
    }
    output[i] = acc;
}

// one thread per (point, output): its planes in order, as RustiCol's loop over the amplitudes
template <class R>
__global__ void reduce_resolved_kernel(const double *are, const double *aim, size_t ld, const uint32_t *planes,
                                       const double *weights, const uint32_t *offsets, double *output,
                                       uint32_t points, size_t output_stride, size_t point_stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t o = blockIdx.y;
    if (i >= points) return;
    size_t out = o * output_stride + i * point_stride;
    double acc = output[out];
    for (uint32_t k = __ldg(offsets + o), end = __ldg(offsets + o + 1); k < end; k++) {
        size_t at = (size_t)__ldg(planes + k) * ld + i;
        double re = as_double(ldr<R>(are, at)), im = as_double(ldr<R>(aim, at));
        acc = __dadd_rn(acc, __dmul_rn(__ldg(weights + k), __fma_rn(re, re, __dmul_rn(im, im))));
    }
    output[out] = acc;
}

// device buffers of at least `n` elements
template <class T> static bool reserve(T **p, size_t *capacity, size_t n, const char *what) {
    if (n <= *capacity) return true;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *capacity = 0;
    if (failed(cudaMalloc((void **)p, n * sizeof(T)), what)) return false;
    *capacity = n;
    return true;
}

static int reduce_resolved_locked(const pacman_recurrence_reduce_resolved *r) {
    initialize();
    if (!r || r->abi_version != PACMAN_RECURRENCE_ABI || r->struct_size != sizeof(pacman_recurrence_reduce_resolved) ||
        !r->offsets || !r->output || r->outputs == 0 || r->outputs > 65535)
        return PACMAN_RECURRENCE_INVALID;
    if (!device.kept || r->points != device.kept_points) return PACMAN_RECURRENCE_INVALID;
    uint32_t count = r->offsets[r->outputs];
    if (r->offsets[0] != 0 || (count && (!r->planes || !r->weights))) return PACMAN_RECURRENCE_INVALID;
    for (uint32_t o = 0; o < r->outputs; o++)
        if (r->offsets[o] > r->offsets[o + 1]) return PACMAN_RECURRENCE_INVALID;
    for (uint32_t k = 0; k < count; k++)
        if (r->planes[k] >= device.kept_planes) return PACMAN_RECURRENCE_INVALID;
    if (r->points == 0) return PACMAN_RECURRENCE_OK;
    // the output span (all the indices the call writes)
    size_t span = (size_t)(r->outputs - 1) * r->output_stride + (size_t)(r->points - 1) * r->point_stride + 1;
    bool zero = r->flags & PACMAN_RECURRENCE_OUTPUT_ZERO;
    double start = device.stats ? now() : 0;
    static uint32_t *offsets = nullptr;
    static size_t offsets_capacity = 0;
    if (!reserve(&device.rplanes, &device.rcount_capacity, std::max<size_t>(count, 1), "reduction") ||
        !reserve(&device.rweights, &device.rweights_capacity, std::max<size_t>(count, 1), "reduction") ||
        !reserve(&offsets, &offsets_capacity, r->outputs + 1, "reduction") ||
        !reserve(&device.routput, &device.routput_capacity, span, "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if ((count && (failed(cudaMemcpy(device.rplanes, r->planes, count * sizeof(uint32_t), cudaMemcpyHostToDevice), "reduction") ||
                   failed(cudaMemcpy(device.rweights, r->weights, count * sizeof(double), cudaMemcpyHostToDevice), "reduction"))) ||
        failed(cudaMemcpy(offsets, r->offsets, (r->outputs + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice), "reduction") ||
        (zero ? failed(cudaMemset(device.routput, 0, span * sizeof(double)), "reduction")
              : failed(cudaMemcpy(device.routput, r->output, span * sizeof(double), cudaMemcpyHostToDevice), "reduction")))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    dim3 block(128), grid((r->points + 127) / 128, r->outputs);
    if (device.kept_df64)
        reduce_resolved_kernel<D><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes, device.rweights,
                                                   offsets, device.routput, r->points, r->output_stride, r->point_stride);
    else
        reduce_resolved_kernel<double><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes,
                                                        device.rweights, offsets, device.routput, r->points,
                                                        r->output_stride, r->point_stride);
    if (failed(cudaGetLastError(), "reduction") ||
        failed(cudaMemcpy(r->output, device.routput, span * sizeof(double), cudaMemcpyDeviceToHost), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    device.copies += zero ? 1 : 2;
    device.bytes += (zero ? 1 : 2) * span * sizeof(double);
    device.reductions++;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}

static int reduce_locked(const pacman_recurrence_reduce *r) {
    initialize();
    if (!r || r->abi_version != PACMAN_RECURRENCE_ABI || r->struct_size != sizeof(pacman_recurrence_reduce) ||
        (r->count && (!r->planes || !r->weights)) || !r->output)
        return PACMAN_RECURRENCE_INVALID;
    if (!device.kept || r->points != device.kept_points) return PACMAN_RECURRENCE_INVALID;
    for (uint32_t k = 0; k < r->count; k++)
        if (r->planes[k] >= device.kept_planes) return PACMAN_RECURRENCE_INVALID;
    double start = device.stats ? now() : 0;
    if (!reserve(&device.rplanes, &device.rcount_capacity, std::max<uint32_t>(r->count, 1), "reduction") ||
        !reserve(&device.rweights, &device.rweights_capacity, std::max<uint32_t>(r->count, 1), "reduction") ||
        !reserve(&device.routput, &device.routput_capacity, std::max<uint32_t>(r->points, 1), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    size_t bytes = r->points * sizeof(double);
    if ((r->count && (failed(cudaMemcpy(device.rplanes, r->planes, r->count * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice), "reduction") ||
                      failed(cudaMemcpy(device.rweights, r->weights, r->count * sizeof(double),
                                        cudaMemcpyHostToDevice), "reduction"))) ||
        failed(cudaMemcpy(device.routput, r->output, bytes, cudaMemcpyHostToDevice), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    unsigned block = 128, grid = (r->points + block - 1) / block;
    if (device.kept_df64)
        reduce_kernel<D><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes,
                                          device.rweights, r->count, device.routput, r->points);
    else
        reduce_kernel<double><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes,
                                               device.rweights, r->count, device.routput, r->points);
    if (failed(cudaGetLastError(), "reduction") ||
        failed(cudaMemcpy(r->output, device.routput, bytes, cudaMemcpyDeviceToHost), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    device.copies += 2;
    device.bytes += 2 * bytes;
    device.reductions++;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}

// ---- Thread safety ------------------------------------------------------------------------
// The device state (the arena, the programs, the kept amplitudes) is one per process, shared
// by every caller: in mg7, every UMAMI library links its own RustiCol, and madspace calls them
// from many threads. Calls are serialised by one lock. A run that keeps its amplitudes on the
// device holds the lock until the same thread's reduction of them (RustiCol always reduces a
// kept run before its next one), so that no other call can overwrite the arena in between;
// a thread that runs again without reducing (an error in between) keeps holding it.
#include <mutex>
static std::mutex runner_lock;
static thread_local bool holding_runner_lock = false;

static void acquire_runner_lock() {
    if (!holding_runner_lock) {
        runner_lock.lock();
        holding_runner_lock = true;
    }
}

static void release_runner_lock() {
    if (holding_runner_lock) {
        holding_runner_lock = false;
        runner_lock.unlock();
    }
}

extern "C" int pacman_recurrence_v1_run(const pacman_recurrence_call *in) {
    acquire_runner_lock();
    int status = run_locked(in);
    if (!(status == PACMAN_RECURRENCE_OK && device.kept)) release_runner_lock();
    return status;
}

extern "C" int pacman_recurrence_v1_configure(const pacman_recurrence_options *o) {
    if (!o || o->abi_version != PACMAN_RECURRENCE_ABI || o->struct_size < sizeof(pacman_recurrence_options))
        return PACMAN_RECURRENCE_INVALID;
    if (o->kernel > PACMAN_RECURRENCE_KERNEL_MIXED64 || (o->block && (o->block < 32 || o->block > 1024)))
        return PACMAN_RECURRENCE_INVALID;
    acquire_runner_lock();
    int status = PACMAN_RECURRENCE_OK;
    Options n;
    n.kernel = (int)o->kernel;
    n.df64 = n.kernel != KERNEL_F64;
    n.df64_forms = n.kernel == KERNEL_DF64 || n.kernel == KERNEL_MIXED;
    n.block = o->block ? (int)o->block : 128;
    n.sort = !o->no_sort;
    bool same = n.kernel == current_options.kernel && n.block == current_options.block &&
                n.sort == current_options.sort;
    // the same configuration again is fine (several UMAMI libraries of one output share the
    // runner); another one only before the first run (the kernel fixes the arena's layout)
    if (runs_started && !same) {
        fprintf(stderr, "pacman recurrence runner: another configuration after the first run\n");
        status = PACMAN_RECURRENCE_INVALID;
    } else {
        current_options = n;
        device.stats = device.stats || o->stats != 0;
    }
    release_runner_lock();
    return status;
}

extern "C" uint32_t pacman_recurrence_v1_capabilities(void) {
    return PACMAN_RECURRENCE_CAP_SOURCES | PACMAN_RECURRENCE_CAP_FORMS |
           (PACUDA_KERNEL_COUNT > 0 ? PACMAN_RECURRENCE_CAP_SYMJIT : 0u);
}

// The index of the symjit kernel whose `.plane.symjit` has this SHA-256 (64 hex digits, either
// case), or -1 if the runner was built without it.
extern "C" int pacman_recurrence_v1_symjit_kernel(const char *sha256_hex) {
    if (!sha256_hex) return -1;
    for (int k = 0; k < PACUDA_KERNEL_COUNT; k++)
        if (!strcasecmp(pacuda_kernel_sha256[k], sha256_hex)) return k;
    return -1;
}

// Test only: the source wavefunctions of `count` (template code, mass, momentum) items, evaluated
// on the device by `source_values`; out_re/out_im get 4 components per item (unused ones 0).
// RustiCol's pacman test compares them with its `wavefunctions.rs` bit for bit.
__global__ void test_sources_kernel(const uint32_t *codes, const double *masses, const double *momenta,
                                    uint32_t count, double *out_re, double *out_im) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    W v[4] = {c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0)};
    source_values(codes[i], masses[i], momenta + 4 * i, v);
    for (int k = 0; k < 4; k++) { out_re[4 * i + k] = v[k].re; out_im[4 * i + k] = v[k].im; }
}

extern "C" int pacman_recurrence_v1_test_sources(const uint32_t *codes, const double *masses,
                                                 const double *momenta, uint32_t count,
                                                 double *out_re, double *out_im) {
    acquire_runner_lock();
    uint32_t *dc = nullptr;
    double *dm = nullptr, *dp = nullptr, *dre = nullptr, *dim = nullptr;
    bool ok = !failed(cudaMalloc((void **)&dc, count * sizeof(uint32_t)), "test") &&
              !failed(cudaMalloc((void **)&dm, count * sizeof(double)), "test") &&
              !failed(cudaMalloc((void **)&dp, 4 * count * sizeof(double)), "test") &&
              !failed(cudaMalloc((void **)&dre, 4 * count * sizeof(double)), "test") &&
              !failed(cudaMalloc((void **)&dim, 4 * count * sizeof(double)), "test") &&
              !failed(cudaMemcpy(dc, codes, count * sizeof(uint32_t), cudaMemcpyHostToDevice), "test") &&
              !failed(cudaMemcpy(dm, masses, count * sizeof(double), cudaMemcpyHostToDevice), "test") &&
              !failed(cudaMemcpy(dp, momenta, 4 * count * sizeof(double), cudaMemcpyHostToDevice), "test");
    if (ok) {
        test_sources_kernel<<<(count + 127) / 128, 128>>>(dc, dm, dp, count, dre, dim);
        ok = !failed(cudaGetLastError(), "test") &&
             !failed(cudaMemcpy(out_re, dre, 4 * count * sizeof(double), cudaMemcpyDeviceToHost), "test") &&
             !failed(cudaMemcpy(out_im, dim, 4 * count * sizeof(double), cudaMemcpyDeviceToHost), "test");
    }
    cudaFree(dc); cudaFree(dm); cudaFree(dp); cudaFree(dre); cudaFree(dim);
    release_runner_lock();
    return ok ? PACMAN_RECURRENCE_OK : PACMAN_RECURRENCE_DEVICE_ERROR;
}

extern "C" void *pacman_recurrence_v1_host_alloc(size_t bytes) {
    void *p = nullptr;
    if (cudaMallocHost(&p, bytes ? bytes : 1) != cudaSuccess) return nullptr;
    return p;
}

extern "C" void pacman_recurrence_v1_host_free(void *p) {
    if (p) cudaFreeHost(p);
}

extern "C" int pacman_recurrence_v1_reduce_resolved(const pacman_recurrence_reduce_resolved *r) {
    acquire_runner_lock();
    int status = reduce_resolved_locked(r);
    release_runner_lock();
    return status;
}

extern "C" int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *r) {
    acquire_runner_lock();
    int status = reduce_locked(r);
    release_runner_lock();
    return status;
}
