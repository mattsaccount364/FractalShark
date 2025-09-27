/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 /*
  * Release 1.2
  *
  * (1) Deployed new implementation of div_dblflt() and sqrt_dblflt() based on
  *     Newton-Raphson iteration, providing significant speedup.
  * (2) Added new function rsqrt_dblflt() which provides reciprocal square root.
  *
  * Release 1.1
  *
  * (1) Fixed a bug affecting add_dblflt() and sub_dblflt() that in very rare
  *     cases returned results with reduced accuracy.
  * (2) Replaced the somewhat inaccurate error bounds with the experimentally
  *     observed maximum relative error.
  */

#if !defined(dblflt_H_)
#define dblflt_H_

#include <math.h>       /* import sqrt() */
#include "dblflt.h"

  /* The head of a double-float number is stored in the most significant part
     of a double2 (the y-component). The tail is stored in the least significant
     part of the double2 (the x-component). All double-float operands must be
     normalized on both input to and return from all basic operations, i.e. the
     magnitude of the tail shall be <= 0.5 ulp of the head.
  */
  // TODO evaluate alignment issues with HDRFloat packing
  // If we move to single-exponent complex, we may be able to 
  // use float2 for dblflt?
  //typedef float2 dblflt;

  /* Create a double-float from two doubles. No normalization is performed,
     so the head and tail components passed in must satisfy the normalization
     requirement. To create a double-float from two arbitrary float-precision
     numbers, use add_float_to_dblflt().
  */
__device__ __forceinline__ dblflt make_dblflt(float head, float tail) {
    dblflt z;
    z.tail = tail;
    z.head = head;
    return z;
}

/* Return the head of a double-float number */
__device__ __forceinline__ float get_dblflt_head(dblflt a) {
    return a.head;
}

/* Return the tail of a double-float number */
__device__ __forceinline__ float get_dblflt_tail(dblflt a) {
    return a.tail;
}

/* Compute error-free sum of two unordered doubles. See Knuth, TAOCP vol. 2 */
__device__ __forceinline__ dblflt add_float_to_dblflt(float a, float b) {
    float t1, t2;
    dblflt z;
    z.head = __fadd_rn(a, b);
    t1 = __fadd_rn(z.head, -a);
    t2 = __fadd_rn(z.head, -t1);
    t1 = __fadd_rn(b, -t1);
    t2 = __fadd_rn(a, -t2);
    z.tail = __fadd_rn(t1, t2);
    return z;
}

/* Compute error-free product of two doubles. Take full advantage of FMA */
__device__ __forceinline__ dblflt mul_double_to_dblflt(float a, float b) {
    dblflt z;
    z.head = __fmul_rn(a, b);
    z.tail = __fmaf_rn(a, b, -z.head);
    return z;
}

/* Negate a double-float number, by separately negating head and tail */
__device__ __forceinline__ dblflt neg_dblflt(dblflt a) {
    dblflt z;
    z.head = -a.head;
    z.tail = -a.tail;
    return z;
}

/* Compute high-accuracy sum of two double-float operands. In the absence of
   underflow and overflow, the maximum relative error observed with 10 billion
   test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dblflt add_dblflt(dblflt a, dblflt b) {
    dblflt z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn(a.head, b.head);
    t2 = __fadd_rn(t1, -a.head);
    t3 = __fadd_rn(__fadd_rn(a.head, t2 - t1), __fadd_rn(b.head, -t2));
    t4 = __fadd_rn(a.tail, b.tail);
    t2 = __fadd_rn(t4, -a.tail);
    t5 = __fadd_rn(__fadd_rn(a.tail, t2 - t4), __fadd_rn(b.tail, -t2));
    t3 = __fadd_rn(t3, t4);
    t4 = __fadd_rn(t1, t3);
    t3 = __fadd_rn(t1 - t4, t3);
    t3 = __fadd_rn(t3, t5);
    z.head = e = __fadd_rn(t4, t3);
    z.tail = __fadd_rn(t4 - e, t3);
    return z;
}

/* Compute high-accuracy difference of two double-float operands. In the
   absence of underflow and overflow, the maximum relative error observed
   with 10 billion test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dblflt sub_dblflt(dblflt a, dblflt b) {
    dblflt z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn(a.head, -b.head);
    t2 = __fadd_rn(t1, -a.head);
    t3 = __fadd_rn(__fadd_rn(a.head, t2 - t1), -__fadd_rn(b.head, t2));
    t4 = __fadd_rn(a.tail, -b.tail);
    t2 = __fadd_rn(t4, -a.tail);
    t5 = __fadd_rn(__fadd_rn(a.tail, t2 - t4), -__fadd_rn(b.tail, t2));
    t3 = __fadd_rn(t3, t4);
    t4 = __fadd_rn(t1, t3);
    t3 = __fadd_rn(t1 - t4, t3);
    t3 = __fadd_rn(t3, t5);
    z.head = e = __fadd_rn(t4, t3);
    z.tail = __fadd_rn(t4 - e, t3);
    return z;
}

/* Compute high-accuracy product of two double-float operands, taking full
   advantage of FMA. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was 5.238480533564479e-32
   (~= 2**-103.9125).
*/
__device__ __forceinline__ dblflt mul_dblflt(dblflt a, dblflt b) {
    dblflt t, z;
    float e;
    t.head = __fmul_rn(a.head, b.head);
    t.tail = __fmaf_rn(a.head, b.head, -t.head);
    t.tail = __fmaf_rn(a.tail, b.tail, t.tail);
    t.tail = __fmaf_rn(a.head, b.tail, t.tail);
    t.tail = __fmaf_rn(a.tail, b.head, t.tail);
    z.head = e = __fadd_rn(t.head, t.tail);
    z.tail = __fadd_rn(t.head - e, t.tail);
    return z;
}

__device__ __forceinline__ dblflt mul_dblflt2x(dblflt a, dblflt b) {
    dblflt t, z;
    float e;
    t.head = __fmul_rn(a.head, b.head);
    t.tail = __fmaf_rn(a.head, b.head, -t.head);
    t.tail = __fmaf_rn(a.tail, b.tail, t.tail);
    t.tail = __fmaf_rn(a.head, b.tail, t.tail);
    t.tail = __fmaf_rn(a.tail, b.head, t.tail);
    z.head = e = __fadd_rn(t.head, t.tail);
    z.tail = __fadd_rn(t.head - e, t.tail);
    z.tail = __fmul_rn(z.tail, 2.0f);
    z.head = __fmul_rn(z.head, 2.0f);
    return z;
}

__device__ __forceinline__ dblflt sqr_dblflt(dblflt a) {
    dblflt t, z;
    float e;
    //t.head = __fmul_rn(a.head, a.head);
    //t.tail = __fmaf_rn(a.head, a.head, -t.head);
    //t.tail = __fmaf_rn(a.tail, a.tail, t.tail);
    //t.tail = __fmaf_rn(a.head, a.tail, t.tail);
    //t.tail = __fmaf_rn(a.tail, a.head, t.tail);
    //z.head = e = __fadd_rn(t.head, t.tail);
    //z.tail = __fadd_rn(t.head - e, t.tail);

    t.head = __fmul_rn(a.head, a.head);
    t.tail = __fmaf_rn(a.head, a.head, -t.head);
    t.tail = __fmaf_rn(a.tail, a.tail, t.tail);
    e = __fmul_rn(a.head, a.tail);
    t.tail = __fmaf_rn(2.0f, e, t.tail);
    z.head = e = __fadd_rn(t.head, t.tail);
    z.tail = __fadd_rn(t.head - e, t.tail);

    return z;
}

__device__ __forceinline__ dblflt shiftleft_dblflt(dblflt a) {
    dblflt z;
    z.tail = __fmul_rn(a.tail, 2.0f);
    z.head = __fmul_rn(a.head, 2.0f);
    return z;
}

/* Compute high-accuracy quotient of two double-float operands, using Newton-
   Raphson iteration. Based on: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
   Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
   In Proceedings of the 8th International Conference on Computational Science,
   ICCS '08, Part I, pp. 446-455. In the absence of underflow and overflow, the
   maximum relative error observed with 10 billion test cases was
   1.0161322480099059e-31 (~= 2**-102.9566).
*/
__device__ __forceinline__ dblflt div_dblflt(dblflt a, dblflt b) {
    dblflt t, z;
    float e, r;
    r = 1.0 / b.head;
    t.head = __fmul_rn(a.head, r);
    e = __fmaf_rn(b.head, -t.head, a.head);
    t.head = __fmaf_rn(r, e, t.head);
    t.tail = __fmaf_rn(b.head, -t.head, a.head);
    t.tail = __fadd_rn(a.tail, t.tail);
    t.tail = __fmaf_rn(b.tail, -t.head, t.tail);
    e = __fmul_rn(r, t.tail);
    t.tail = __fmaf_rn(b.head, -e, t.tail);
    t.tail = __fmaf_rn(r, t.tail, e);
    z.head = e = __fadd_rn(t.head, t.tail);
    z.tail = __fadd_rn(t.head - e, t.tail);
    return z;
}

/* Compute high-accuracy square root of a double-float number. Newton-Raphson
   iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
   High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
   1997, pp. 561-589. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was
   3.7564109505601846e-32 (~= 2**-104.3923).
*/
__device__ __forceinline__ dblflt sqrt_dblflt(dblflt a) {
    dblflt t, z;
    double e, y, s, r;
    r = rsqrt(a.head);
    if (a.head == 0.0f) r = 0.0;
    y = __fmul_rn(a.head, r);
    s = __fmaf_rn(y, -y, a.head);
    r = __fmul_rn(0.5, r);
    z.head = e = __fadd_rn(s, a.tail);
    z.tail = __fadd_rn(s - e, a.tail);
    t.head = __fmul_rn(r, z.head);
    t.tail = __fmaf_rn(r, z.head, -t.head);
    t.tail = __fmaf_rn(r, z.tail, t.tail);
    r = __fadd_rn(y, t.head);
    s = __fadd_rn(y - r, t.head);
    s = __fadd_rn(s, t.tail);
    z.head = e = __fadd_rn(r, s);
    z.tail = __fadd_rn(r - e, s);
    return z;
}

/* Compute high-accuracy reciprocal square root of a double-double number.
   Based on Newton-Raphson iteration. In the absence of underflow and overflow,
   the maximum relative error observed with 10 billion test cases was
   6.4937771666026349e-32 (~= 2**-103.6026)
*/
__device__ __forceinline__ dblflt rsqrt_dblflt(dblflt a) {
    dblflt z;
    double r, s, e;
    r = rsqrt(a.head);
    e = __fmul_rn(a.head, r);
    s = __fmaf_rn(e, -r, 1.0);
    e = __fmaf_rn(a.head, r, -e);
    s = __fmaf_rn(e, -r, s);
    e = __fmul_rn(a.tail, r);
    s = __fmaf_rn(e, -r, s);
    e = 0.5 * r;
    z.head = __fmul_rn(e, s);
    z.tail = __fmaf_rn(e, s, -z.head);
    s = __fadd_rn(r, z.head);
    r = __fadd_rn(r, -s);
    r = __fadd_rn(r, z.head);
    r = __fadd_rn(r, z.tail);
    z.head = e = __fadd_rn(s, r);
    z.tail = __fadd_rn(s - e, r);
    return z;
}

__device__ __forceinline__ double dblflt_to_double(dblflt a) {
    return (double)a.head + (double)a.tail;
}

__device__ __forceinline__ dblflt double_to_dblflt(double a) {
    // ?? 2^23 = 8388608.0
    dblflt res;
    res.head = (float)a;
    res.tail = (float)(a - (double)res.head);
    return res;
}

#endif /* dblflt_H_ */
