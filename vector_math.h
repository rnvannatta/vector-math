#ifndef SIMDVEC_H
#define SIMDVEC_H
#include <x86intrin.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdalign.h>
#include <limits.h>

/******************************************************************************
 *********************           S.1 definitions          *********************
 ******************************************************************************/

typedef struct vec4 { __m128 raw; } vec4;
typedef struct ivec4 { __m128i raw; } ivec4;
typedef struct uvec4 { __m128i raw; } uvec4;
typedef struct bvec4 { __m128i raw; } bvec4;

// Plan: floatify?

typedef struct vec3 { __m128 raw; } vec3;
typedef struct ivec3 { __m128i raw; } ivec3;
typedef struct uvec3 { __m128i raw; } uvec3;
typedef struct bvec3 { __m128i raw; } bvec3;

/* Storing a vec2 and even an ivec2 as a double
 *  a) keeps alignment at 8 (32 bit machines don't exist)
 *  b) results in the two components being passed in the lower half of the xmm registers (on SystemV anyway)
 *  c) allows for easy casting to __m128s and __m128is for simd
 */
typedef struct vec2 { double raw; } vec2;
typedef struct ivec2 { double raw; } ivec2;
typedef struct uvec2 { double raw; } uvec2;
typedef struct bvec2 { double raw; } bvec2;

vec2 _intrinsic_to_vec2(__m128 m) {
  return (vec2) { _mm_cvtsd_f64(_mm_castps_pd(m)) };
}
__m128 _vec2_to_intrinsic(vec2 v) {
  return _mm_castpd_ps(_mm_set_sd(v.raw));
}

typedef __attribute__ ((aligned (16))) float vec4a[4];
typedef __attribute__ ((aligned (16))) int ivec4a[4];
typedef __attribute__ ((aligned (16))) unsigned uvec4a[4];
typedef __attribute__ ((aligned (16))) int bvec4a[4];

typedef __attribute__ ((aligned (16))) float vec3a[3];
typedef __attribute__ ((aligned (16))) int ivec3a[3];
typedef __attribute__ ((aligned (16))) unsigned uvec3a[3];
typedef __attribute__ ((aligned (16))) int bvec3a[3];

typedef __attribute__ ((aligned (8))) float vec2a[2];
typedef __attribute__ ((aligned (8))) int ivec2a[2];
typedef __attribute__ ((aligned (8))) unsigned uvec2a[2];
typedef __attribute__ ((aligned (8))) int bvec2a[2];

/******************************************************************************
 *********************           S.0 macro hell           *********************
 ******************************************************************************/

#if 0
#define _num_args2(ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8, X4, X3, X2, X1, N, ...) N
#define _num_args(...) _num_args2(__VA_ARGS__ __VA_OPT__(,) error, error, error, error, error, error, error, error, 4, 3, 2, 1, 0)
#else
// __VA_OPT__ is C2X technically, gcc supports it in C but not clang
// FIXME this is not valid C11, find portable approach
#define _num_args2(X, ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8, X4, X3, X2, X1, N, ...) N
#define _num_args(...) _num_args2(X ,##__VA_ARGS__ , error, error, error, error, error, error, error, error, 4, 3, 2, 1, 0)
#endif

// Completed constructors
struct _c_vec4 { int _; };

struct _c_vec3_scalar { int _; };
struct _c_scalar_vec3 { int _; };

struct _c_vec2_vec2 { int _; };
struct _c_vec2_scalar_scalar { int _; };
struct _c_scalar_vec2_scalar { int _; };
struct _c_scalar_scalar_vec2 { int _; };

struct _c_scalar_scalar_scalar_scalar { int _; };

struct _c_error { int _; };

// Missing a float

struct _c_vec3 { struct _c_error Vec3; struct _c_error Vec2; struct _c_vec3_scalar Scalar; };
struct _c_scalar_scalar_scalar { struct _c_scalar_scalar_scalar_scalar Scalar; };

struct _c_vec2_scalar { struct _c_error Vec2; struct _c_vec2_scalar_scalar Scalar; };
struct _c_scalar_vec2 { struct _c_error Vec2; struct _c_scalar_vec2_scalar Scalar; };

// Missing two floats

struct _c_vec2 { struct _c_error Vec3; struct _c_vec2_vec2 Vec2; struct _c_vec2_scalar Scalar; };
struct _c_scalar_scalar { struct _c_scalar_scalar_vec2 Vec2; struct _c_scalar_scalar_scalar Scalar; };

// Missing three floats

struct _c_scalar { struct _c_scalar_scalar Scalar; struct _c_scalar_vec2 Vec2; struct _c_scalar_vec3 Vec3; };

#define _typeof1(x) \
  _Generic((x), \
    vec4: (struct _c_vec4) {0}, \
    vec3: (struct _c_vec3) {0}, \
    vec2: (struct _c_vec2) {0}, \
    default: (struct _c_scalar) {0})
#define _typeof2(x, y) \
  _Generic((y), \
    vec3: _typeof1(x).Vec3, \
    vec2: _typeof1(x).Vec2, \
    default: _typeof1(x).Scalar)
#define _typeof3(x, y, z) \
  _Generic((z), \
    vec2: _typeof2(x, y).Vec2, \
    default: _typeof2(x, y).Scalar)

#define _make_vec4_4_error(...) _Static_assert(0, "Too many arguments to make_vec4")
#define _make_vec4_4_4(x, y, z, w) make_vec4_4s(x, y, z, w)
#define _make_vec4_4_3(x, y, z) \
  _Generic(_typeof3(x, y, z), \
    struct _c_vec2_scalar_scalar: make_vec4_v2_2s, \
    struct _c_scalar_vec2_scalar: make_vec4_1s_v2_1s, \
    struct _c_scalar_scalar_vec2: make_vec4_2s_v2)(x, y, z)
#define _make_vec4_4_2(x, y) \
  _Generic(_typeof2(x, y), \
    struct _c_vec3_scalar: make_vec4_v3_1s, \
    struct _c_scalar_vec3: make_vec4_1s_v3, \
    struct _c_vec2_vec2: make_vec4_v2_v2)(x, y)
#define _make_vec4_4_1(x) \
  _Generic((x), \
    vec4: vec4_dup, \
    default: make_vec4_1s)(x)
#define _make_vec4_4_0() vec4_zero()

#define _make_vec4_3(N, ...) _make_vec4_4_ ## N(__VA_ARGS__)
#define _make_vec4_2(N, ...) _make_vec4_3(N, __VA_ARGS__)
#define make_vec4(...) _make_vec4_2(_num_args(__VA_ARGS__), __VA_ARGS__)

#define _make_vec3_4_error(...) _Static_assert(0, "Too many arguments to make_vec3")
#define _make_vec3_4_4(...) _Static_assert(0, "Too many arguments to make_vec3")
#define _make_vec3_4_3(x, y, z) make_vec3_3s(x, y, z)
#define _make_vec3_4_2(x, y) \
  _Generic(_typeof2(x, y), \
    struct _c_vec2_scalar: make_vec3_v2_1s, \
    struct _c_scalar_vec2: make_vec3_1s_v2)(x, y)
#define _make_vec3_4_1(x) \
  _Generic((x), \
    vec3: vec3_dup, \
    default: make_vec3_1s)(x)
#define _make_vec3_4_0() vec3_zero()
#define _make_vec3_3(N, ...) _make_vec3_4_ ## N(__VA_ARGS__)
#define _make_vec3_2(N, ...) _make_vec3_3(N, __VA_ARGS__)
#define make_vec3(...) _make_vec3_2(_num_args(__VA_ARGS__), __VA_ARGS__)

#define _make_vec2_4_error(...) _Static_assert(0, "Too many arguments to make_vec2")
#define _make_vec2_4_4(...) _Static_assert(0, "Too many arguments to make_vec2")
#define _make_vec2_4_3(...) _Static_assert(0, "Too many arguments to make_vec2")
#define _make_vec2_4_2(x, y) make_vec2_2s(x, y)
#define _make_vec2_4_1(x) \
  _Generic((x), \
    vec2: vec2_dup, \
    default: make_vec2_1s)(x)
#define _make_vec2_4_0() vec2_zero()
#define _make_vec2_3(N, ...) _make_vec2_4_ ## N(__VA_ARGS__)
#define _make_vec2_2(N, ...) _make_vec2_3(N, __VA_ARGS__)
#define make_vec2(...) _make_vec2_2(_num_args(__VA_ARGS__), __VA_ARGS__)

// quick method macros
#define make_gvec_trinary(g1, n1, g2, n2, fun) \
  inline g1##vec##n1 g1##vec##n1##fun(g1##vec##n1 a, g1##vec##n1 b, g1##vec##n1 c) { \
    return (g1##vec##n1) { g2##vec##n2##fun((g2##vec##n2){a.raw}, (g2##vec##n2){b.raw}, (g2##vec##n2){c.raw}).raw }; \
  }
#define make_gvec_binary(g1, n1, g2, n2, fun) \
  inline g1##vec##n1 g1##vec##n1##fun(g1##vec##n1 a, g1##vec##n1 b) { \
    return (g1##vec##n1) { g2##vec##n2##fun((g2##vec##n2){a.raw}, (g2##vec##n2){b.raw}).raw }; \
  }
#define make_gvec_unary(g1, n1, g2, n2, fun) \
  inline g1##vec##n1 g1##vec##n1##fun(g1##vec##n1 a) { \
    return (g1##vec##n1) { g2##vec##n2##fun((g2##vec##n2){a.raw}).raw }; \
  }
#define make_gvec_left_scalar(g1, n1, g2, n2, scalar, fun) \
  inline g1##vec##n1 g1##vec##n1##fun(scalar s, g1##vec##n1 a) { \
    return (g1##vec##n1) { g2##vec##n2##fun(s, (g2##vec##n2){a.raw}).raw }; \
  }
#define make_gvec_right_scalar(g1, n1, g2, n2, scalar, fun) \
  inline g1##vec##n1 g1##vec##n1##fun(g1##vec##n1 a, scalar s) { \
    return (g1##vec##n1) { g2##vec##n2##fun((g2##vec##n2){a.raw}, s).raw }; \
  }
#define make_gvec_compare(g1, n1, g2, n2, fun) \
  inline bvec##n1 g1##vec##n1##fun(g1##vec##n1 a, g1##vec##n1 b) { \
    return (bvec##n1) { g2##vec##n2##fun((g2##vec##n2){a.raw}, (g2##vec##n2){b.raw}).raw }; \
  }


/******************************************************************************
 *********************    S.2.1 small vec constructors    *********************
 ******************************************************************************/

inline vec2 vec2_zero() {
  return (vec2) { 0 };
}
inline vec2 vec2_dup(vec2 v) {
  return v;
}
inline vec2 make_vec2_2s(float x, float y) {
  return (vec2) { _mm_cvtsd_f64(_mm_castps_pd(_mm_setr_ps(x, y, 0, 0))) };
}
inline vec2 make_vec2_1s(float f) {
  return (vec2) { _mm_cvtsd_f64(_mm_castps_pd(_mm_set1_ps(f))) };
}

/** Writes a vec3 from register to an array in memory.
 */
inline void vec3_unpack(vec3a ret, vec3 v) {
  vec4a tmp;
  _mm_store_ps(tmp, v.raw);
  ret[0] = tmp[0];
  ret[1] = tmp[1];
  ret[2] = tmp[2];
}

/** Takes an aligned array and converts it to a vec3.
 */
inline vec3 vec3_pack(vec3a v) {
  // While this is technically reading beyond the limits of vec3a,
  // because vec3a is aligned to 16 bytes, this won't actually segfault
  __m128 tmp = _mm_load_ps(v);
  // Clear the out of bounds memory because I feel dirty
  __m128 ss = _mm_set_ss(0);
  __m128 ret = _mm_insert_ps(tmp, ss, (0 << 6) + (3 << 4));
  return (vec3) { ret };
}

inline vec3 vec3_zero() {
  return (vec3) { _mm_setzero_ps() };
}
inline vec3 vec3_dup(vec3 v) {
  return v;
}
inline vec3 make_vec3_3s(float x, float y, float z) {
  return (vec3) { _mm_setr_ps(x, y, z, 0) };
}
inline vec3 make_vec3_1s(float f) {
  return (vec3) { _mm_set1_ps(f) };
}
inline vec3 make_vec3_v2_1s(vec2 v, float f) {
  __m128 ss = _mm_set_ss(f);
  return (vec3) { _mm_movelh_ps(_vec2_to_intrinsic(v), ss) };
}
inline vec3 make_vec3_1s_v2(float f, vec2 v) {
  __m128 vi = _vec2_to_intrinsic(v);
  float v1 = _mm_cvtss_f32(_mm_shuffle_ps(vi, vi, 0));
  float v2 = _mm_cvtss_f32(_mm_shuffle_ps(vi, vi, 1));
  return (vec3) { _mm_setr_ps(f, v1, v2, 0) };
}

/******************************************************************************
 *********************      S.2.1 vec4 constructors       *********************
 ******************************************************************************/

/** Writes a vec4 from register to an array in memory.
 */
inline void vec4_unpack(vec4a ret, vec4 v) {
  _mm_store_ps(ret, v.raw);
}

/** Takes an aligned array and converts it to a vec4.
 */
inline vec4 vec4_pack(vec4a v) {
  return (vec4) { _mm_load_ps(v) };
}

inline vec4 vec4_zero() {
  return (vec4) { _mm_setzero_ps() };
}

inline vec4 make_vec4_4s(float x, float y, float z, float w) {
  return (vec4) { _mm_setr_ps(x, y, z, w) };
}

inline vec4 make_vec4_v3_1s(vec3 v, float f) {
  __m128 ss = _mm_set_ss(f);
  __m128 ret = _mm_insert_ps(v.raw, ss, (0 << 6) + (3 << 4));
  return (vec4) { ret };
}

inline vec4 make_vec4_1s_v3(float f, vec3 v) {
  __m128 ss = _mm_set_ss(f);
  __m128 ret = _mm_insert_ps(v.raw, ss, (0 << 6) + (3 << 4));
  return (vec4) { _mm_shuffle_ps(ret, ret, 3 | (0 << 2) | (1 << 4) | (2 << 6)) };
}

inline vec4 make_vec4_v2_v2(vec2 a, vec2 b) {
  return (vec4) { _mm_castpd_ps( _mm_setr_pd(a.raw, b.raw) ) };
}

inline vec4 make_vec4_v2_2s(vec2 a, float b, float c) {
  return (vec4) { _mm_castpd_ps( _mm_setr_pd(a.raw, make_vec2_2s(b, c).raw)) };
}

inline vec4 make_vec4_2s_v2(float a, float b, vec2 c) {
  return (vec4) { _mm_castpd_ps( _mm_setr_pd(make_vec2_2s(a, b).raw, c.raw)) };
}

inline vec4 make_vec4_1s_v2_1s(float a, vec2 b, float c) {
  __m128 bi = _vec2_to_intrinsic(b);
  float b1 = _mm_cvtss_f32(_mm_shuffle_ps(bi, bi, 0));
  float b2 = _mm_cvtss_f32(_mm_shuffle_ps(bi, bi, 1));
  return (vec4) { _mm_setr_ps(a, b1, b2, c) }; 
}

inline vec4 make_vec4_1s(float f) {
  return (vec4) { _mm_set1_ps(f) };
}

inline vec4 vec4_dup(vec4 v) {
  return (vec4) { v.raw };
}

/******************************************************************************
 *********************        S.2.2 vec4 shuffling        *********************
 ******************************************************************************/

inline vec4 vec4_min(vec4 a, vec4 b) {
  return (vec4) { _mm_min_ps(a.raw, b.raw) };
}

inline vec4 vec4_max(vec4 a, vec4 b) {
  return (vec4) { _mm_max_ps(a.raw, b.raw) };
}

inline vec4 vec4_blendv(vec4 a, vec4 b, bvec4 select) {
  __m128 ret = _mm_blendv_ps(a.raw, b.raw, _mm_castsi128_ps(select.raw));
  return (vec4) { ret };
}

/** Blend with set of 4 bools. Sad to say that clang can inline literal bools
 *    into blendps rather than blendvps, but gcc cannot. Clang is impressive.
 */
/* last args are imm */
inline vec4 vec4_blend(vec4 a, vec4 b, bool x, bool y, bool z, bool w) {
  __m128 ret = _mm_blendv_ps(a.raw, b.raw, _mm_castsi128_ps(_mm_set_epi32(-(int)w, -(int)z, -(int)y, -(int)x)));
  return (vec4) { ret };
}

#define make_blend(vec, x, y, z, w) \
  inline vec vec ## _blend ## x ## y ## z ## w(vec a, vec b) { \
    return vec ## _blend(a, b, #x[0] == 'B', #y[0] == 'B', #z[0] == 'B', #w[0] == 'B'); \
  }
//make_blend(vec4, A, A, B, B)


typedef enum vec_index {
  VEC_X,
  VEC_Y,
  VEC_Z,
  VEC_W
} vec_index;

/* last args are imm */
inline vec4 vec4_swizzle_4(vec4 v, vec_index x, vec_index y, vec_index z, vec_index w) {
  return (vec4) { _mm_shuffle_ps(v.raw, v.raw, x | (y << 2) | (z << 4) | (w << 6)) };
}
/* last args are imm */
inline vec3 vec4_swizzle_3(vec4 v, vec_index x, vec_index y, vec_index z) {
  return (vec3) { _mm_shuffle_ps(v.raw, v.raw, x | (y << 2) | (z << 4)) };
}
/* last args are imm */
inline vec2 vec4_swizzle_2(vec4 v, vec_index x, vec_index y) {
  return _intrinsic_to_vec2(_mm_shuffle_ps(v.raw, v.raw, x | (y << 2)));
}
/* last args are imm */
inline float vec4_extract(vec4 v, vec_index i) {
  return _mm_cvtss_f32(_mm_shuffle_ps(v.raw, v.raw, i));
}

#define make_vec_get(scalar, vector, comp) \
  inline scalar vector ## _get ## comp(vector v) { \
    return vec4_extract(v, VEC_ ## comp); \
  }
make_vec_get(float, vec4, X);
make_vec_get(float, vec4, Y);
make_vec_get(float, vec4, Z);
make_vec_get(float, vec4, W);

// vec3 shuffle

make_gvec_binary(, 3, , 4, _min)
make_gvec_binary(, 3, , 4, _max)

/******************************************************************************
 *********************         S.2.3 vec4 compare         *********************
 ******************************************************************************/

inline bvec4 vec4_greaterThan(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmpgt_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bvec4 vec4_lessThan(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmplt_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bvec4 vec4_lessThanEqual(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmple_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bvec4 vec4_greaterThanEqual(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmpge_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bvec4 vec4_equal(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmpeq_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bvec4 vec4_notEqual(vec4 a, vec4 b) {
  __m128 tmp = _mm_cmpneq_ps(a.raw, b.raw);
  return (bvec4) { _mm_castps_si128(tmp) };
}

inline bool vec4_allEqual(vec4 a, vec4 b) {
  return _mm_movemask_ps(_mm_cmpeq_ps(a.raw, b.raw)) == 15;
}

inline bool vec3_allEqual(vec3 a, vec3 b) {
  return (_mm_movemask_ps(_mm_cmpeq_ps(a.raw, b.raw)) & 7) == 7;
}

make_gvec_compare(, 3, , 4, _lessThan)
make_gvec_compare(, 3, , 4, _lessThanEqual)
make_gvec_compare(, 3, , 4, _greaterThan)
make_gvec_compare(, 3, , 4, _greaterThanEqual)
make_gvec_compare(, 3, , 4, _equal)
make_gvec_compare(, 3, , 4, _notEqual)

/******************************************************************************
 *********************        S.2.4 vec4 arithmetic       *********************
 ******************************************************************************/

inline vec4 vec4_add(vec4 a, vec4 b) {
  return (vec4) { _mm_add_ps(a.raw, b.raw) };
}

inline vec4 vec4_sub(vec4 a, vec4 b) {
  return (vec4) { _mm_sub_ps(a.raw, b.raw) };
}

inline vec4 vec4_mul(vec4 a, vec4 b) {
  return (vec4) { _mm_mul_ps(a.raw, b.raw) };
}

inline vec4 vec4_div(vec4 a, vec4 b) {
  return (vec4) { _mm_div_ps(a.raw, b.raw) };
}

inline vec4 vec4_scale(float f, vec4 v) {
  __m128 fv = _mm_set1_ps(f);
  return (vec4) { _mm_mul_ps(fv, v.raw) };
}

inline vec4 vec4_shrink(vec4 v, float f) {
  __m128 fv = _mm_set1_ps(f);
  return (vec4) { _mm_div_ps(v.raw, fv) };
}

inline vec4 vec4_abs(vec4 v) {
   __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(INT_MAX));
   return (vec4) { _mm_and_ps(v.raw, mask) };
}

inline vec4 vec4_sign(vec4 v) {
  vec4 normed = vec4_div(v, vec4_abs(v));
  bvec4 mask = vec4_notEqual(vec4_zero(), v);
  return (vec4) { _mm_and_ps(normed.raw, _mm_castsi128_ps(mask.raw)) };
}

// vec3 arithmetic

make_gvec_binary(, 3, , 4, _add)
make_gvec_binary(, 3, , 4, _sub)
make_gvec_binary(, 3, , 4, _mul)
make_gvec_binary(, 3, , 4, _div)

make_gvec_unary(, 3, , 4, _abs)
make_gvec_unary(, 3, , 4, _sign)

make_gvec_left_scalar(, 3, , 4, float, _scale)
make_gvec_right_scalar(, 3, , 4, float, _shrink)

/******************************************************************************
 *********************        S.2.5 vec4 irrational       *********************
 ******************************************************************************/

/******************************************************************************
 *********************         S.2.6 vec4 rounding        *********************
 ******************************************************************************/

inline vec4 vec4_floor(vec4 v) {
  return (vec4) { _mm_floor_ps(v.raw) };
}

inline vec4 vec4_ceil(vec4 v) {
  return (vec4) { _mm_ceil_ps(v.raw) };
}

inline vec4 vec4_clampv(vec4 v, vec4 a, vec4 b) {
  return vec4_min(vec4_max(v, a), b);
}

inline vec4 vec4_clamp(vec4 v, float a, float b) {
  return vec4_min(vec4_max(v, make_vec4_1s(a)), make_vec4_1s(b));
}

inline vec4 vec4_saturate(vec4 v) {
  return vec4_clamp(v, 0, 1);
}

inline vec4 vec4_round(vec4 v) {
  return (vec4) { _mm_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
}

inline vec4 vec4_trunc(vec4 v) {
  return (vec4) { _mm_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
}

inline vec4 vec4_fract(vec4 v) {
  return vec4_sub(v, vec4_floor(v));
}

inline vec4 vec4_modv(vec4 x, vec4 y) {
  return vec4_sub(x, vec4_mul(y, vec4_floor(vec4_div(x, y))));
}

inline vec4 vec4_mod(vec4 x, float y) {
  return vec4_modv(x, make_vec4_1s(y));
}

make_gvec_trinary(, 3, , 4, _clampv)
make_gvec_unary(, 3, , 4, _saturate)
make_gvec_unary(, 3, , 4, _floor)
make_gvec_unary(, 3, , 4, _ceil)
make_gvec_unary(, 3, , 4, _round)
make_gvec_unary(, 3, , 4, _trunc)
make_gvec_unary(, 3, , 4, _fract)
make_gvec_right_scalar(, 3, , 4, float, _mod)
make_gvec_binary(, 3, , 4, _modv)

inline vec3 vec3_clamp(vec3 v, float a, float b) {
  return (vec3) { vec4_clamp((vec4){v.raw}, a, b).raw };
}

/******************************************************************************
 *********************      S.2.7 vec4 interpolation      *********************
 ******************************************************************************/

inline vec4 vec4_mix(vec4 a, vec4 b, float x) {
  return vec4_add(vec4_scale(1 - x, a), vec4_scale(x, b));
}

inline vec4 vec4_mixv(vec4 a, vec4 b, vec4 x) {
  return vec4_add(vec4_mul(vec4_sub(make_vec4_1s(1), x), a), vec4_mul(x, b));
}

/** Returns the vector x such that mixv(a, b, x) = c. For the ambiguous case of
 *   a[i] == b[i], then the return component x[i] == 0.
 */
inline vec4 vec4_unmixVector(vec4 a, vec4 b, vec4 c) {
  vec4 ret = vec4_div(vec4_sub(c, a), vec4_sub(b, a));
  bvec4 eq = vec4_equal(a, b);
  return vec4_blendv(ret, vec4_zero(), eq);
}

/** Returns the float x such that mix(a, b, x) = c. It calculates the correct value even
 *   if some of the components are equal. If a == b, then 0 is returned. It is assumed
 *   that there does exist at least one scalar x such that mix(a, b, x) == c.
 */
inline float vec4_unmixScalar(vec4 a, vec4 b, vec4 c) {
  bvec4 neq = vec4_notEqual(a, b);
  int mask = _mm_movemask_epi8(neq.raw);
  int idx = _tzcnt_u32(mask) / 4;
  if(idx >= 4)
    return 0;
  __m128 ar = a.raw;
  __m128 br = b.raw;
  __m128 cr = c.raw;
  // Not sure whether it makes more sense to shuffle+extract then do the math or do the math then shuffle+extract
  // I'm sticking with the common case, where we hardly shuffle
  while(idx--) {
    ar = _mm_shuffle_ps(ar, ar, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
    br = _mm_shuffle_ps(br, br, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
    cr = _mm_shuffle_ps(cr, cr, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
  }
  float _a = _mm_cvtss_f32(ar);
  float _b = _mm_cvtss_f32(br);
  float _c = _mm_cvtss_f32(cr);
  return (_c - _a) / (_b - _a);
}

inline vec4 vec4_stepv(vec4 edge, vec4 x) {
  return vec4_blendv(make_vec4_1s(1), vec4_zero(), vec4_lessThan(x, edge));
}

inline vec4 vec4_step(float edge, vec4 x) {
  return vec4_stepv(make_vec4_1s(edge), x);
}

inline vec4 vec4_smoothstepv(vec4 edge0, vec4 edge1, vec4 x) {
  vec4 t = vec4_saturate(vec4_div(vec4_sub(x, edge0), vec4_sub(edge1, edge0)));
  return vec4_mul(t, vec4_mul(t, vec4_sub(make_vec4_1s(3), vec4_scale(2, t))));
}

inline vec4 vec4_smoothstep(float edge0, float edge1, vec4 x) {
  vec4 edge0v = make_vec4_1s(edge0);
  vec4 edge1v = make_vec4_1s(edge1);
  vec4 t = vec4_saturate(vec4_div(vec4_sub(x, edge0v), vec4_sub(edge1v, edge0v)));
  return vec4_mul(t, vec4_mul(t, vec4_sub(make_vec4_1s(3), vec4_scale(2, t))));
}

make_gvec_trinary(, 3, , 4, _mixv)
make_gvec_trinary(, 3, , 4, _unmixVector)
make_gvec_binary(, 3, , 4, _stepv)
make_gvec_left_scalar(, 3, , 4, float, _step)
make_gvec_trinary(, 3, , 4, _smoothstepv)

inline vec3 vec3_mix(vec3 a, vec3 b, float x) {
  return (vec3) { vec4_mix((vec4){a.raw}, (vec4){b.raw}, x).raw };
}
inline float vec3_unmixScalar(vec3 a, vec3 b, vec3 c) {
  bvec3 neq = vec3_notEqual(a, b);
  int mask = _mm_movemask_epi8(neq.raw);
  int idx = _tzcnt_u32(mask) / 4;
  if(idx >= 3)
    return 0;
  __m128 ar = a.raw;
  __m128 br = b.raw;
  __m128 cr = c.raw;
  // Not sure whether it makes more sense to shuffle+extract then do the math or do the math then shuffle+extract
  // I'm sticking with the common case, where we hardly shuffle
  while(idx--) {
    ar = _mm_shuffle_ps(ar, ar, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
    br = _mm_shuffle_ps(br, br, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
    cr = _mm_shuffle_ps(cr, cr, (0 << 6) | (3 << 4) | (2 << 2) | (1 << 0));
  }
  float _a = _mm_cvtss_f32(ar);
  float _b = _mm_cvtss_f32(br);
  float _c = _mm_cvtss_f32(cr);
  return (_c - _a) / (_b - _a);
}
inline vec3 vec3_smoothstep(float edge1, float edge2, vec3 v) {
  return (vec3) { vec4_smoothstep(edge1, edge2, (vec4){v.raw}).raw };
}

/******************************************************************************
 *********************        S.2.8 vec4 geometry         *********************
 ******************************************************************************/

inline float vec4_dot(vec4 a, vec4 b) {
  __m128 tmp = _mm_dp_ps(a.raw, b.raw, 0xFF);
  float ret;
  _mm_store_ss(&ret, tmp);
  return ret;
}

inline vec4 vec4_double_dot(vec4 a, vec4 b, vec4 x, vec4 y) {
  __m128 ret = _mm_mul_ps(a.raw, b.raw);
  __m128 tmp = _mm_mul_ps(x.raw, y.raw);
  ret = _mm_hadd_ps(ret, tmp);
  ret = _mm_hadd_ps(ret, vec4_zero().raw);
  return (vec4) { ret };
}

inline float vec4_length(vec4 v) {
  return sqrtf(fabs(vec4_dot(v, v)));
}

inline float vec4_distance(vec4 a, vec4 b) {
  return vec4_length(vec4_sub(a, b));
}

inline vec4 vec4_normalizeFast(vec4 v) {
  __m128 tmp = _mm_dp_ps(v.raw, v.raw, 0xFF);
  tmp = _mm_shuffle_ps(tmp, tmp, 0);
  tmp = _mm_rsqrt_ps(tmp);
  __m128 ret = _mm_mul_ps(tmp, v.raw);
  return (vec4) { ret };
}


inline vec4 vec4_normalize(vec4 v) {
  float k = fabs(1 / vec4_length(v));
  return vec4_scale(k, v);
}

inline vec4 vec4_faceforward(vec4 v, vec4 i, vec4 n) {
  return vec4_dot(i, n) < 0 ? v : vec4_scale(-1, v);
}

inline vec4 vec4_reflect(vec4 i, vec4 n) {
  return vec4_sub(i, vec4_scale(2 * vec4_dot(n, i), n));
}

// i and n must be normalized
// eta is n1 / n2 where i is leaving material 1 and entering material 2
inline vec4 vec4_refract(vec4 i, vec4 n, float eta) {
  float dp = vec4_dot(i, n);
  // like most things this is taken from the opengl docs
  // but this requires some splaining.
  // starkeffects.com/snells-law-vector.shtml shows a good one
  // though it involves cross products, but it can be simplified
  // to this form with vector algebra rules & i & n being normalized
  float k = 1 - eta * eta * (1 - dp * dp);
  // Total internal reflection
  if(k < 0)
    return vec4_zero();
  else
    return vec4_sub(vec4_scale(eta, i), vec4_scale(eta * dp + sqrt(k), n));
}

inline float vec3_dot(vec3 a, vec3 b) {
  __m128 tmp;
  tmp = _mm_dp_ps(a.raw, b.raw, 0x77);
  float ret;
  _mm_store_ss(&ret, tmp);
  return ret;
}

inline vec3 vec3_cross(vec3 a, vec3 b) {
  //3 0 2 1
  //x moves to z
  //y moves to x
  //z moves to y
  //w moves to w
  //Thank you Mr. Anonymous blog post comment
#define shuffle 0xC9
  __m128 t1 = _mm_mul_ps(b.raw, _mm_shuffle_ps(a.raw, a.raw, shuffle));
  __m128 t2 = _mm_mul_ps(a.raw, _mm_shuffle_ps(b.raw, b.raw, shuffle));
  __m128 ret = _mm_sub_ps(t2, t1);
  return (vec3) { _mm_shuffle_ps(ret, ret, shuffle) };
#undef shuffle
}


inline float vec3_length(vec3 v) {
  return sqrtf(fabs(vec3_dot(v, v)));
}

inline float vec3_distance(vec3 a, vec3 b) {
  return vec3_length(vec3_sub(a, b));
}

inline vec3 vec3_normalizeFast(vec3 v) {
  __m128 tmp = _mm_dp_ps(v.raw, v.raw, 0x77);
  tmp = _mm_shuffle_ps(tmp, tmp, 0);
  tmp = _mm_rsqrt_ps(tmp);
  return (vec3) { _mm_mul_ps(tmp, v.raw) };
}

inline vec3 vec3_normalize(vec3 v) {
  float k = fabs(1 / vec3_length(v));
  return vec3_scale(k, v);
}

inline vec3 vec3_faceforward(vec3 v, vec3 i, vec3 n) {
  return vec3_dot(i, n) < 0 ? v : vec3_scale(-1, v);
}

inline vec3 vec3_reflect(vec3 i, vec3 n) {
  return vec3_sub(i, vec3_scale(2 * vec3_dot(n, i), n));
}

// i and n must be normalized
// eta is n1 / n2 where i is leaving material 1 and entering material 2
inline vec3 vec3_refract(vec3 i, vec3 n, float eta) {
  float dp = vec3_dot(i, n);
  // like most things this is taken from the opengl docs
  // but this requires some splaining.
  // starkeffects.com/snells-law-vector.shtml shows a good one
  // though it involves cross products, but it can be simplified
  // to this form with vector algebra rules & i & n being normalized
  float k = 1 - eta * eta * (1 - dp * dp);
  // Total internal reflection
  if(k < 0)
    return vec3_zero();
  else
    return vec3_sub(vec3_scale(eta, i), vec3_scale(eta * dp + sqrt(k), n));
}


/******************************************************************************
 *********************      S.2.9 vec4 trigonometry       *********************
 ******************************************************************************/


// ivec4 stuff

inline ivec4 ivec4_from_scalar(int i) {
  return (ivec4) { _mm_set1_epi32(i) };
}

inline int ivec4_dot(ivec4 a, ivec4 b) {
  __m128i tmp = _mm_mullo_epi32(a.raw, b.raw);
  tmp = _mm_hadd_epi32(tmp, tmp);
  tmp = _mm_hadd_epi32(tmp, tmp);
  return _mm_cvtsi128_si32(tmp);
}

inline ivec4 ivec4_blendv(ivec4 a, ivec4 b, ivec4 select) {
  return (ivec4) { _mm_blendv_epi8(a.raw, b.raw, select.raw) };
}
/*
// GCC exclusive!
#define IVEC4_BLEND(a, b, x, y, z, w) \
  ({ivec4 ret = _mm_blend_epi16(a.raw, b.raw, w * (3 << 6) + z * (3 << 4) + y * (3 << 2) + x * 3); \
    ret;})
*/

inline void ivec4_unpack(ivec4a ret, ivec4 v) {
  _mm_store_si128((__m128i*)ret, v.raw);
}

inline ivec4 ivec4_pack(ivec4a v) {
  return (ivec4) { _mm_load_si128((__m128i*)v) };
}

inline bvec4 make_bvec4_4s(bool x, bool y, bool z, bool w) {
  return (bvec4) { _mm_set_epi32(-(int)w, -(int)z, -(int)y, -(int)x) };
}

inline bool bvec4_all(bvec4 b) {
  return _mm_movemask_epi8(b.raw) == 65535;
}
inline bool bvec4_allEqual(bvec4 a, bvec4 b) {
  return bvec4_all((bvec4) { _mm_cmpeq_epi32(a.raw, b.raw) });
}

inline bvec4 ivec4_equal(ivec4 a, ivec4 b) {
  return (bvec4) { _mm_cmpeq_epi32(a.raw, b.raw) };
}


typedef struct mat4_unpacked {
  union {
    float d[16];
    vec4a cols[4];
  };
} mat4_unpacked;

typedef struct mat4 {
  vec4 cols[4];
} mat4;
/*
// FIXME
inline vec4 mat4_mul_vec4(mat4 m, vec4 v) {
  vec4 row;
  row.raw = _mm_set1_ps(v.d[0]);
  vec4 ret;
  ret.raw = _mm_mul_ps(row.raw, m.cols[0].raw);
  for(int i = 1; i < 4; i++) {
    vec4 tmp;
    row.raw = _mm_set1_ps(v.d[i]);
    tmp.raw = _mm_mul_ps(row.raw, m.cols[i].raw);
    ret.raw = _mm_add_ps(ret.raw, tmp.raw);
  }
  return ret;
}

inline mat4 mat4_add(mat4 a, mat4 b) {
  mat4 ret;
  ret.cols[1] = _mm_add_ps(a.cols[1], b.cols[1]);
  ret.cols[2] = _mm_add_ps(a.cols[2], b.cols[2]);
  ret.cols[3] = _mm_add_ps(a.cols[3], b.cols[3]);
  ret.cols[4] = _mm_add_ps(a.cols[4], b.cols[4]);
  return ret;
}

inline mat4 mat4_mul(mat4 a, mat4 b) {
  mat4 ret;
  for(int j = 0; j < 4; j++)
    ret.cols[j] = mat4_mul_vec4(a, b.cols[j]);
  return ret;
}

// FIXME
inline mat4 mat4_id() {
  mat4 ret = { { {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 } }
  };
  return ret;
}

// FIXME
inline mat4 mat4_translate(float x, float y, float z) {
  mat4 ret = mat4_id();
  ret.d[12] = x;
  ret.d[13] = y;
  ret.d[14] = z;
  return ret;
}

// FIXME
inline mat4 mat4_ortho(float l, float r, float b, float t, float f, float n) {
  mat4 ret = { { {
      2/(r - l),           0,           0, 0,
              0,   2/(t - b),           0, 0,
              0,           0,   2/(n - f), 0,
    (l+r)/(l-r), (b+t)/(b-t), (n+f)/(n-f), 1 } }
  };
  return ret;
}

// FIXME
inline mat4 mat4_fov(float aspect, float vfov, float n, float f) {
  float tb = 1.0f/tanf(vfov/2);
  mat4 ret = { { {
    tb/aspect,  0,           0,  0,
       0,      tb,           0,  0,
       0,       0, (n+f)/(n-f), -1,
       0,       0, 2*n*f/(n-f),  0 } }
  };
  return ret;
}

// FIXME
inline mat4 mat4_fov_ideal(float aspect, float vfov, float n) {
  float epsilon = 1.0 / (1 << 22);
  float tb = 1.0/tanf(vfov/2);
  mat4 ret = { { {
    tb/aspect,  0,    0,  0,
    0,         tb,    0,  0,
    0,          0,   -1 + epsilon, -1,
    0,          0, -2*n,  0
  } } };
  return ret;
}

// FIXME
inline mat4 mat4_x_rot(float a) {
  float s = sinf(a);
  float c = cosf(a);
  //Recall that these are column major matrices
  //So each row is secretly a column. transpose
  //it in your mind
  mat4 ret = { { {
    1, 0, 0, 0,
    0, c, s, 0,
    0,-s, c, 0,
    0, 0, 0, 1 } }
  };
  return ret;
}

// FIXME
inline mat4 mat4_y_rot(float a) {
  float s = sinf(a);
  float c = cosf(a);
  mat4 ret = { { {
    c, 0, s, 0,
    0, 1, 0, 0,
   -s, 0, c, 0,
    0, 0, 0, 1 } }
  };
  return ret;
}

// FIXME
inline mat4 mat4_z_rot(float a) {
  float s = sinf(a);
  float c = cosf(a);
  mat4 ret = { { {
    c, s, 0, 0,
   -s, c, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 } }
  };
  return ret;
}

inline mat4 mat4_transpose(mat4 m) {
  mat4 tmp;
  mat4 ret;
  tmp.cols[0] = _mm_unpacklo_ps(m.cols[0], m.cols[1]);
  tmp.cols[1] = _mm_unpackhi_ps(m.cols[0], m.cols[1]);
  tmp.cols[2] = _mm_unpacklo_ps(m.cols[2], m.cols[3]);
  tmp.cols[3] = _mm_unpackhi_ps(m.cols[2], m.cols[3]);

  ret.cols[0] = _mm_movelh_ps(tmp.cols[0], tmp.cols[2]);
  ret.cols[1] = _mm_movehl_ps(tmp.cols[2], tmp.cols[0]);
  ret.cols[2] = _mm_movelh_ps(tmp.cols[1], tmp.cols[3]);
  ret.cols[3] = _mm_movehl_ps(tmp.cols[3], tmp.cols[1]);
  return ret;
}

inline mat4 mat4_rotate_to(vec4 dir, vec4 up) {
  vec4 z = vec4_scale(-1, dir);
  vec4 x = vec3_normalize(vec3_cross(up, z));
  vec4 y = vec3_cross(z, x);
  mat4 ret;
  ret.cols[0] = vec3_ideal(x);
  ret.cols[1] = vec3_ideal(y);
  ret.cols[2] = vec3_ideal(z);
  ret.cols[3] = make_vec4(0, 0, 0, 1);
  return mat4_transpose(ret);
}

inline mat4 mat4_lookat(vec4 eye, vec4 at, vec4 up) {
  vec4 z = vec3_normalize(vec4_sub(at, eye));
  vec4 x = vec3_normalize(vec3_cross(up, z));
  vec4 y = vec3_cross(z, x);
  mat4 ret;
  ret.cols[0] = vec3_ideal(x);
  ret.cols[1] = vec3_ideal(y);
  ret.cols[2] = vec3_ideal(z);
  ret.cols[3] = make_vec4_1s(0);
  ret = mat4_transpose(ret);
  //mat4 ret = mat4_rotate_to(vec3_normalize(vec4_sub(at, eye)), up);
  ret.cols[3] = make_vec4( -vec3_dot(x, eye), -vec3_dot(y, eye), -vec3_dot(z, eye), 1);
  return ret;
}

inline mat4 mat4_scale(float f, mat4 m) {
  vec4 fv;
  fv.raw = _mm_set1_ps(f);
  mat4 ret;
  ret.cols[0] = _mm_mul_ps(fv, m.cols[0]);
  ret.cols[1] = _mm_mul_ps(fv, m.cols[1]);
  ret.cols[2] = _mm_mul_ps(fv, m.cols[2]);
  ret.cols[3] = _mm_mul_ps(fv, m.cols[3]);
  return ret;
}

// FIXME
inline mat4 mat4_invert(mat4 m) {
  //TODO simd implementation
  mat4 t = mat4_transpose(m);
  mat4 co;
  mat4 tmp;
  //pairs for first 8 cofactors
  tmp.d[ 0] = t.d[10] * t.d[15];
  tmp.d[ 1] = t.d[11] * t.d[14];
  tmp.d[ 2] = t.d[ 9] * t.d[15];
  tmp.d[ 3] = t.d[11] * t.d[13];

  tmp.d[ 4] = t.d[ 9] * t.d[14];
  tmp.d[ 5] = t.d[10] * t.d[13];
  tmp.d[ 6] = t.d[ 8] * t.d[15];
  tmp.d[ 7] = t.d[11] * t.d[12];

  tmp.d[ 8] = t.d[ 8] * t.d[14];
  tmp.d[ 9] = t.d[10] * t.d[12];
  tmp.d[10] = t.d[ 8] * t.d[13];
  tmp.d[11] = t.d[ 9] * t.d[12];
  //first 8 cofactors
  co.d[0]  = tmp.d[ 0]*t.d[5] + tmp.d[ 3]*t.d[6] + tmp.d[ 4]*t.d[7];
  co.d[1]  = tmp.d[ 1]*t.d[4] + tmp.d[ 6]*t.d[6] + tmp.d[ 9]*t.d[7];
  co.d[2]  = tmp.d[ 2]*t.d[4] + tmp.d[ 7]*t.d[5] + tmp.d[10]*t.d[7];
  co.d[3]  = tmp.d[ 5]*t.d[4] + tmp.d[ 8]*t.d[5] + tmp.d[11]*t.d[6];

  co.d[0] -= tmp.d[ 1]*t.d[5] + tmp.d[ 2]*t.d[6] + tmp.d[ 5]*t.d[7];
  co.d[1] -= tmp.d[ 0]*t.d[4] + tmp.d[ 7]*t.d[6] + tmp.d[ 8]*t.d[7];
  co.d[2] -= tmp.d[ 3]*t.d[4] + tmp.d[ 6]*t.d[5] + tmp.d[11]*t.d[7];
  co.d[3] -= tmp.d[ 4]*t.d[4] + tmp.d[ 9]*t.d[5] + tmp.d[10]*t.d[6];

  co.d[4]  = tmp.d[ 1]*t.d[1] + tmp.d[ 2]*t.d[2] + tmp.d[ 5]*t.d[3];
  co.d[5]  = tmp.d[ 0]*t.d[0] + tmp.d[ 7]*t.d[2] + tmp.d[ 8]*t.d[3];
  co.d[6]  = tmp.d[ 3]*t.d[0] + tmp.d[ 6]*t.d[1] + tmp.d[11]*t.d[3];
  co.d[7]  = tmp.d[ 4]*t.d[0] + tmp.d[ 9]*t.d[1] + tmp.d[10]*t.d[2];

  co.d[4] -= tmp.d[ 0]*t.d[1] + tmp.d[ 3]*t.d[2] + tmp.d[ 4]*t.d[3];
  co.d[5] -= tmp.d[ 1]*t.d[0] + tmp.d[ 6]*t.d[2] + tmp.d[ 9]*t.d[3];
  co.d[6] -= tmp.d[ 2]*t.d[0] + tmp.d[ 7]*t.d[1] + tmp.d[10]*t.d[3];
  co.d[7] -= tmp.d[ 5]*t.d[0] + tmp.d[ 8]*t.d[1] + tmp.d[11]*t.d[2];
  //pairs for second 8 cofactors
  tmp.d[ 0] = t.d[2] * t.d[7];
  tmp.d[ 1] = t.d[3] * t.d[6];
  tmp.d[ 2] = t.d[1] * t.d[7];
  tmp.d[ 3] = t.d[3] * t.d[5];

  tmp.d[ 4] = t.d[1] * t.d[6];
  tmp.d[ 5] = t.d[2] * t.d[5];
  tmp.d[ 6] = t.d[0] * t.d[7];
  tmp.d[ 7] = t.d[3] * t.d[4];

  tmp.d[ 8] = t.d[0] * t.d[6];
  tmp.d[ 9] = t.d[2] * t.d[4];
  tmp.d[10] = t.d[0] * t.d[5];
  tmp.d[11] = t.d[1] * t.d[4];
  //second 8 cofactors
  co.d[ 8]  = tmp.d[ 0]*t.d[13] + tmp.d[ 3]*t.d[14] + tmp.d[ 4]*t.d[15];
  co.d[ 9]  = tmp.d[ 1]*t.d[12] + tmp.d[ 6]*t.d[14] + tmp.d[ 9]*t.d[15];
  co.d[10]  = tmp.d[ 2]*t.d[12] + tmp.d[ 7]*t.d[13] + tmp.d[10]*t.d[15];
  co.d[11]  = tmp.d[ 5]*t.d[12] + tmp.d[ 8]*t.d[13] + tmp.d[11]*t.d[14];

  co.d[ 8] -= tmp.d[ 1]*t.d[13] + tmp.d[ 2]*t.d[14] + tmp.d[ 5]*t.d[15];
  co.d[ 9] -= tmp.d[ 0]*t.d[12] + tmp.d[ 7]*t.d[14] + tmp.d[ 8]*t.d[15];
  co.d[10] -= tmp.d[ 3]*t.d[12] + tmp.d[ 6]*t.d[13] + tmp.d[11]*t.d[15];
  co.d[11] -= tmp.d[ 4]*t.d[12] + tmp.d[ 9]*t.d[13] + tmp.d[10]*t.d[14];

  co.d[12]  = tmp.d[ 2]*t.d[10] + tmp.d[ 5]*t.d[11] + tmp.d[ 1]*t.d[ 9];
  co.d[13]  = tmp.d[ 8]*t.d[11] + tmp.d[ 0]*t.d[ 8] + tmp.d[ 7]*t.d[10];
  co.d[14]  = tmp.d[ 6]*t.d[ 9] + tmp.d[11]*t.d[11] + tmp.d[ 3]*t.d[ 8];
  co.d[15]  = tmp.d[10]*t.d[10] + tmp.d[ 4]*t.d[ 8] + tmp.d[ 9]*t.d[ 9];

  co.d[12] -= tmp.d[ 4]*t.d[11] + tmp.d[ 0]*t.d[ 9] + tmp.d[ 3]*t.d[10];
  co.d[13] -= tmp.d[ 6]*t.d[10] + tmp.d[ 9]*t.d[11] + tmp.d[ 1]*t.d[ 8];
  co.d[14] -= tmp.d[10]*t.d[11] + tmp.d[ 2]*t.d[ 8] + tmp.d[ 7]*t.d[ 9];
  co.d[15] -= tmp.d[ 8]*t.d[ 9] + tmp.d[11]*t.d[10] + tmp.d[ 5]*t.d[ 8];
  //TODO inline
  float det = vec4_dot(t.cols[0], co.cols[0]);
  return mat4_scale(1/det, co);

}

inline ivec4 ivec4_zero() {
  return _mm_setzero_si128();
}

inline ivec4 make_ivec4(int x, int y, int z, int w) {
  return _mm_set_epi32(w, z, y, x);
}

inline int ivec3_dot(ivec4 a, ivec4 b) {
  ivec4 tmp;
  tmp = _mm_mullo_epi32(a, b);
  tmp = _mm_mullo_epi32(tmp, make_ivec4(1, 1, 1, 0));
  tmp = _mm_hadd_epi32(tmp, tmp);
  tmp = _mm_hadd_epi32(tmp, tmp);
  return _mm_cvtsi128_si32(tmp);
}

inline vec4 vec4_from_ivec4(ivec4 v) {
  return _mm_cvtepi32_ps(v);
}

inline ivec4 ivec4_from_vec4(vec4 v) {
  return _mm_cvtps_epi32(v);
}

// FIXME
inline ivec4 ivec4_from_ivec3(ivec4 v) {
  ivec4 ret = v;
  ret.d[3] = 1;
  return ret;
}

inline ivec4 ivec4_add(ivec4 a, ivec4 b) {
  return _mm_add_epi32(a, b);
}

inline ivec4 ivec4_sub(ivec4 a, ivec4 b) {
  return _mm_sub_epi32(a, b);
}

inline ivec4 ivec4_mul(ivec4 a, ivec4 b) {
  return _mm_mullo_epi32(a, b);
}

inline ivec4 ivec4_sar(ivec4 v, int amt) {
  return _mm_srai_epi32(v, amt);
}

inline ivec4 ivec4_slr(ivec4 v, int amt) {
  return _mm_srli_epi32(v, amt);
}

inline ivec4 ivec4_sll(ivec4 v, int amt) {
  return _mm_slli_epi32(v, amt);
}

inline ivec4 ivec4_and(ivec4 a, ivec4 b) {
  return _mm_and_si128(a, b);
}

inline ivec4 ivec4_or(ivec4 a, ivec4 b) {
  return _mm_or_si128(a, b);
}

inline ivec4 ivec4_greater(ivec4 a, ivec4 b) {
  return _mm_cmpgt_epi32(a, b);
}

inline ivec4 ivec4_less(ivec4 a, ivec4 b) {
  return ivec4_greater(b, a);
}

inline ivec4 ivec4_leq(ivec4 a, ivec4 b) {
  ivec4 ret = ivec4_greater(a, b);
  return _mm_xor_si128(ret, ivec4_from_scalar(-1));
}

inline ivec4 ivec4_geq(ivec4 a, ivec4 b) {
  return ivec4_leq(b, a);
}

inline ivec4 ivec4_equal(ivec4 a, ivec4 b) {
  return _mm_cmpeq_epi32(a.raw, b.raw);
}

// FIXME
inline bool ivec4_all_equal(ivec4 a, ivec4 b) {
  union {
    ivec4 i;
    vec4 v;
  } u;
  u.i.raw = _mm_cmpeq_epi32(a.raw, b.raw);
  return _mm_movemask_ps(u.v.raw) == 15;
}

inline ivec4 ivec4_min(ivec4 a, ivec4 b) {
  ivec4 mask = ivec4_greater(a, b);
  return ivec4_blendv(a, b, mask);
}

inline ivec4 ivec4_max(ivec4 a, ivec4 b) {
  ivec4 mask = ivec4_less(a, b);
  return ivec4_blendv(a, b, mask);
}

inline bool ivec4_allzero(ivec4 v) {
  return _mm_test_all_zeros(v, v);
}

// FIXME
inline bool ivec3_allzero(ivec4 v) {
  static ivec4 const test = { { { -1, -1, -1, 0 } } };
  return _mm_test_all_zeros(v, test);
}

inline bool ivec4_allone(ivec4 v) {
  ivec4 v2 = _mm_xor_si128(v, ivec4_from_scalar(-1));
  return ivec4_allzero(v2);
}

inline bool ivec3_allone(ivec4 v) {
  ivec4 v2 = _mm_xor_si128(v, ivec4_from_scalar(-1));
  return ivec3_allzero(v2);
}

inline ivec4 ivec4_div(ivec4 v, int amt) {
  ivec4 coeff = ivec4_from_scalar( (1 << amt) - 1 );
  ivec4 zero = ivec4_zero();
  ivec4 offset = ivec4_blendv(zero, coeff, ivec4_less(v, zero));
  return ivec4_sar(ivec4_add(v, offset), amt);
}

inline ivec4 ivec4_avg(ivec4 a, ivec4 b) {
  return ivec4_div(ivec4_add(a, b), 1);
}

inline ivec4 ivec4_mod(ivec4 v, int divisor) {
  int mask = ((1 << divisor) - 1);
  ivec4 pos_mod = ivec4_and(v, ivec4_from_scalar(mask));
  ivec4 neg_mod = ivec4_add(ivec4_or(v, ivec4_from_scalar(~mask)),
                            ivec4_from_scalar(1 << divisor));
  ivec4 selector = ivec4_less(v, ivec4_zero());
  return _mm_blendv_epi8(pos_mod.raw, neg_mod.raw, selector.raw);
}

inline ivec4 uvec4_greater(ivec4 a, ivec4 b) {
  return _mm_xor_si128(_mm_cmpeq_epi32(_mm_max_epu32(b, a), b), _mm_set1_epi32(-1));
}

inline ivec4 uvec4_less(ivec4 a, ivec4 b) {
  return uvec4_greater(b, a);
}

inline ivec4 uvec4_geq(ivec4 a, ivec4 b) {
  return _mm_cmpeq_epi32(_mm_max_epu32(a, b), a);
}

inline ivec4 uvec4_leq(ivec4 a, ivec4 b) {
  return uvec4_geq(b, a);
}

inline ivec4 ivec4_scale(int i, ivec4 v) {
  return _mm_mullo_epi32(_mm_set1_epi32(i), v);
}

inline int downbool(ivec4 v) {
  return _mm_movemask_epi8(v);
}

// FIXME
inline void vec4_print(vec4 v) {
  for(int i = 0; i < 4; i++)
    printf("%f ", v.d[i]);
  printf("\n");
}

// FIXME
inline void ivec4_print(ivec4 v) {
  for(int i = 0; i < 4; i++)
    printf("%d ", v.d[i]);
  printf("\n");
}
*/
#endif
