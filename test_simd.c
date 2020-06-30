#include <stdbool.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include "vector_math.h"
#include "inlines.h"

void vec4_base() {
  vec4a ref = { 1, 2, 3, 4 };
  assert(alignof(ref) == 16);
  vec4 v = make_vec4_4s(1, 2, 3, 4);
  vec4a vu;
  vec4_unpack(vu, v);
  for(int i = 0; i < 4; i++)
    assert(vu[i] == ref[i]);
  assert(vec4_allEqual(v, vec4_pack(ref)));
  assert(!vec4_allEqual(v, make_vec4_4s(1, 2, 2, 4)));
  assert(vec4_allEqual(make_vec4_4s(0, 0, 0, 0), vec4_zero()));
  assert(vec4_allEqual(make_vec4_4s(1, 1, 1, 1), make_vec4_1s(1)));

  vec3a ref3 = { 1, 2, 3 };
  assert(alignof(ref3) == 16);
  vec3 v3 = make_vec3_3s(1, 2, 3);
  vec3a vu3;
  vec3_unpack(vu3, v3);
  for(int i = 0; i < 2; i++)
    assert(vu3[i] == ref3[i]);
  assert(vec3_allEqual(v3, vec3_pack(ref3)));
  assert(!vec3_allEqual(v3, make_vec3_3s(1, 2, 4)));
  assert(vec3_allEqual(make_vec3_3s(1, 1, 1), (vec3) { make_vec4_4s(1, 1, 1, 4).raw }));
  assert(vec3_allEqual(make_vec3_3s(0, 0, 0), vec3_zero()));
  assert(vec3_allEqual(make_vec3_3s(1, 1, 1), make_vec3_1s(1)));
}

void vec4_constructor() {
  assert(vec4_allEqual(make_vec4(), vec4_zero()));
  vec4 v = make_vec4_4s(1, 2, 3, 4);
  assert(vec4_allEqual(make_vec4(v), v));
  assert(vec4_allEqual(make_vec4(3.0f), make_vec4_1s(3)));
  assert(vec4_allEqual(make_vec4(3), make_vec4_1s(3)));

  assert(vec4_allEqual(make_vec4(make_vec2_1s(1), make_vec2_1s(2)), make_vec4(1, 1, 2, 2)));
  assert(vec4_allEqual(make_vec4(vec2_zero(), make_vec2_1s(-2)), make_vec4(0, 0, -2, -2)));
  assert(vec4_allEqual(make_vec4(vec2_zero(), make_vec2_2s(-3, 2)), make_vec4(0, 0, -3, 2)));
  assert(vec4_allEqual(make_vec4(vec3_zero(), 1), make_vec4(0, 0, 0, 1)));
  assert(vec4_allEqual(make_vec4(make_vec3_3s(1, 2, 3), 0), make_vec4(1, 2, 3, 0)));
  assert(vec4_allEqual(make_vec4(-1.0, make_vec3_1s(0.5)), make_vec4(-1, 0.5, 0.5, 0.5)));

  assert(vec4_allEqual(make_vec4(make_vec2_2s(1, 2), 3, 4), make_vec4(1, 2, 3, 4)));
  assert(vec4_allEqual(make_vec4(4, 3, make_vec2_2s(2, 1)), make_vec4(4, 3, 2, 1)));
  assert(vec4_allEqual(make_vec4(-4, make_vec2_2s(-3, 2), 1), make_vec4(-4, -3, 2, 1)));

  assert(vec4_allEqual(make_vec4(make_vec3(), 4), make_vec4(0, 0, 0, 4)));
  assert(vec4_allEqual(make_vec4(make_vec3(1, 2, 3), 4), make_vec4(1, 2, 3, 4)));
  assert(vec4_allEqual(make_vec4(make_vec3(1), 4), make_vec4(1, 1, 1, 4)));
  vec3 v3 = make_vec3(2);
  assert(vec4_allEqual(make_vec4(make_vec3(v3), 4), make_vec4(2, 2, 2, 4)));
  assert(vec4_allEqual(make_vec4(make_vec3(1, make_vec2_1s(2)), 4), make_vec4(1, 2, 2, 4)));
  assert(vec4_allEqual(make_vec4(make_vec3(make_vec2_1s(1), 2), 4), make_vec4(1, 1, 2, 4)));

  assert(vec4_allEqual(make_vec4(make_vec2(), make_vec2(1)), make_vec4(0, 0, 1, 1)));
  vec2 v2 = make_vec2(7);
  assert(vec4_allEqual(make_vec4(make_vec2(v2), make_vec2(-1, 1)), make_vec4(7, 7, -1, 1)));
}

void vec4_shuffle() {
  assert(vec4_allEqual(make_vec4(-1, 1, 0, 1),
                        vec4_min(make_vec4(-1, 2, 0, 6),
                                  make_vec4(-1, 1, 0, 1))));
  assert(vec4_allEqual(make_vec4(-1, 1, 0, 1),
                        vec4_max(make_vec4(-1, -2, 0, -6),
                                  make_vec4(-1, 1, 0, 1))));
  assert(vec4_allEqual(make_vec4(1, 2, 3, 4),
                        vec4_blendv(make_vec4(1, 3, 3, 7),
                                    make_vec4(2, 2, 4, 4),
                                    make_bvec4_4s(false, true, false, true))));
  assert(vec4_allEqual(make_vec4(1, 2, 3, 4),
                        vec4_blend(make_vec4(1, 3, 3, 7),
                                    make_vec4(2, 2, 4, 4),
                                    false, true, false, true)));

  /*
  assert(vec4_allEqual(make_vec4(1, 2, -3, -4),
                       vec4_blendAABB(make_vec4(1, 2, 3, 4), make_vec4(-1, -2, -3, -4))));
                       */

  vec4 v = make_vec4(2, 1, 0.5, 0.25);
  assert(2 == vec4_extract(v, VEC_X));
  assert(1 == vec4_extract(v, VEC_Y));
  assert(0.5 == vec4_extract(v, VEC_Z));
  assert(0.25 == vec4_extract(v, VEC_W));

  assert(2 == vec4_getX(v));
  assert(1 == vec4_getY(v));
  assert(0.5 == vec4_getZ(v));
  assert(0.25 == vec4_getW(v));

  assert(vec4_allEqual(make_vec4(0.25, 0.5, 1, 2), vec4_swizzle_4(v, VEC_W, VEC_Z, VEC_Y, VEC_X)));
  assert(vec4_allEqual(make_vec4(1), vec4_swizzle_4(v, VEC_Y, VEC_Y, VEC_Y, VEC_Y)));
  assert(vec3_allEqual(make_vec3(1, 1, 0.25), vec4_swizzle_3(v, VEC_Y, VEC_Y, VEC_W)));
  assert(vec4_allEqual(make_vec4(1, 0.5, 0, 0), make_vec4(vec4_swizzle_2(v, VEC_Y, VEC_Z), vec2_zero())));

}

void vec4_compare() {
  assert(bvec4_all(make_bvec4_4s(true, true, true, true)));
  assert(!bvec4_all(make_bvec4_4s(true, true, false, true)));
  assert(bvec4_allEqual(make_bvec4_4s(false, false, true, true), make_bvec4_4s(false, false, true, true)));
  assert(!bvec4_allEqual(make_bvec4_4s(false, true, false, true), make_bvec4_4s(false, false, true, true)));

  assert(bvec4_allEqual(vec4_equal(make_vec4(1), make_vec4(1, 1, 0, 1)), make_bvec4_4s(true, true, false, true)));
  assert(bvec4_allEqual(vec4_notEqual(make_vec4(1), make_vec4(1, 1, 0, 1)), make_bvec4_4s(false, false, true, false)));

  assert(bvec4_allEqual(vec4_lessThan(make_vec4(1), make_vec4(1, 2, 0, -1)), make_bvec4_4s(false, true, false, false)));
  assert(bvec4_allEqual(vec4_lessThanEqual(make_vec4(1), make_vec4(1, 2, 0, -1)), make_bvec4_4s(true, true, false, false)));

  assert(bvec4_allEqual(vec4_greaterThan(make_vec4(1), make_vec4(1, 2, 0, -1)), make_bvec4_4s(false, false, true, true)));
  assert(bvec4_allEqual(vec4_greaterThanEqual(make_vec4(1), make_vec4(1, 2, 0, -1)), make_bvec4_4s(true, false, true, true)));
}

void vec4_arithmetic() {
  assert(vec4_allEqual(make_vec4(3), vec4_add(make_vec4(1), make_vec4(2))));
  assert(vec4_allEqual(make_vec4(-1), vec4_sub(make_vec4(1), make_vec4(2))));
  assert(vec4_allEqual(vec4_scale(0, make_vec4(1)), vec4_zero()));
  assert(vec4_allEqual(vec4_scale(-1, make_vec4(-1, -2, -3, -4)), make_vec4(1, 2, 3, 4)));
  assert(vec4_allEqual(vec4_shrink(make_vec4(1, 2, 3, 4), 1), make_vec4(1, 2, 3, 4)));
  assert(vec4_allEqual(vec4_shrink(make_vec4(1, 2, 3, 4), 2), make_vec4(0.5, 1, 1.5, 2)));

  assert(vec4_allEqual(vec4_mul(make_vec4(3, 3, 2, 2), make_vec4(1, 0.5, 3, -4)), make_vec4(3, 1.5, 6, -8)));
  assert(vec4_allEqual(vec4_div(make_vec4(2, 2, 4, 2), make_vec4(1, 0.5, 4, -4)), make_vec4(2, 4, 1, -0.5)));

  assert(vec4_allEqual(vec4_abs(make_vec4(1, -1, 0, -2)), make_vec4(1, 1, 0, 2)));
  assert(vec4_allEqual(vec4_sign(make_vec4(1, -1, 2, -2)), make_vec4(1, -1, 1, -1)));


}

void vec4_rounding() {
  assert(vec4_allEqual(vec4_floor(make_vec4(0, 0.6, -0.4, -1)), make_vec4(0, 0, -1, -1)));
  assert(vec4_allEqual(vec4_ceil(make_vec4(0, 0.6, -0.4, 1)), make_vec4(0, 1, 0, 1)));
  assert(vec4_allEqual(vec4_round(make_vec4(0, 0.6, -0.4, 1)), make_vec4(0, 1, 0, 1)));
  assert(vec4_allEqual(vec4_trunc(make_vec4(0, 0.6, -0.4, -1.6)), make_vec4(0, 0, 0, -1)));
  assert(vec4_allEqual(vec4_fract(make_vec4(0, 0.6, -0.25, -1.75)), make_vec4(0, 0.6, 0.75, 0.25)));

  assert(vec4_allEqual(vec4_mod(make_vec4(2, 1, 2.5, 5), 2), make_vec4(0, 1, 0.5, 1)));
  assert(vec4_allEqual(vec4_modv(make_vec4(2, 1, 2.5, 5), make_vec4(2, 2, 4, 4)), make_vec4(0, 1, 2.5, 1)));

  assert(vec4_allEqual(vec4_clamp(make_vec4(-1.5, -1, 0.6, 1), -1, 1), make_vec4(-1, -1, 0.6, 1)));
  assert(vec4_allEqual(vec4_clampv(make_vec4(-1.5, -1, 0.6,1), make_vec4(-1), make_vec4(1, 1, 0.5, 0.6)), make_vec4(-1, -1, 0.5, 0.6)));
  assert(vec4_allEqual(vec4_saturate(make_vec4(-1.5, -1, 0.6, 1.1)), make_vec4(0, 0, 0.6, 1)));
}

void vec4_interpolate() {
  assert(vec4_allEqual(vec4_mix(make_vec4(1, -1, 2, -2), make_vec4(1, -2, 4, 4), 0.5), make_vec4(1, -1.5, 3, 1)));
  assert(vec4_allEqual(vec4_mixv(make_vec4(1, -1, 2, -2), make_vec4(1, -2, 4, 4), make_vec4(0.5, 0.5, 0, 1)), make_vec4(1, -1.5, 2,4)));

  assert(vec4_unmixScalar(make_vec4(1, -1, 2, -2),
                          make_vec4(1, -2, 4, 4),
                          make_vec4(1, -1.5, 3, 1)) == 0.5);
  assert(vec4_allEqual(vec4_unmixVector(make_vec4(1, -1, 2, -2),
                                        make_vec4(1, -2, 4, 4),
                                        make_vec4(1, -1.5, 4, 1)),
                       make_vec4(0, 0.5, 1, 0.5)));
  assert(vec4_allEqual(vec4_step(1, make_vec4(0.1, 1, 1.1, -1)), make_vec4(0, 1, 1, 0)));
  assert(vec4_allEqual(vec4_stepv(make_vec4(0.5, 1, 0.7, -2), make_vec4(0.1, 1, 1.1, -1)), make_vec4(0, 1, 1, 1)));

  assert(vec4_allEqual(vec4_smoothstep(1, 2, make_vec4(0.1, 1, 1.5, 2)), make_vec4(0, 0, 0.5, 1)));
  assert(vec4_allEqual(vec4_smoothstepv(make_vec4(1), make_vec4(2, 2, 1.2, 3), make_vec4(0.1, 1, 1.5, 2)), make_vec4(0, 0, 1, 0.5)));

}

void vec4_geometry() {
  assert(0 == vec4_length(make_vec4(0)));
  assert(5 == vec4_length(make_vec4(0, 4, -3, 0)));

  assert(2 == vec4_distance(make_vec4(0), make_vec4(1)));
  assert(5 == vec4_distance(make_vec4(0, 0, 1, 1), make_vec4(4, -3, 1, 1)));

  assert(8 == vec4_dot(make_vec4(1), make_vec4(2)));
  assert(0 == vec4_dot(make_vec4(2, 0, 1, 1), make_vec4(-1, 0, 2, 0)));
  assert(vec4_allEqual(make_vec4(8, 0, 0, 0),
                        vec4_double_dot(make_vec4(1), make_vec4(2),
                                        make_vec4(2, 0, 1, 1), make_vec4(-1, 0, 2, 0))));

  assert(6 == vec3_dot(make_vec3(1), make_vec3(2)));
  assert(vec3_allEqual(make_vec3(0, 0, 1),
                          vec3_cross(make_vec3(1, 0, 0),
                                     make_vec3(0, 1, 0))));
  assert(vec3_allEqual(make_vec3(0, 0, -1),
                          vec3_cross(make_vec3(0, 1, 0),
                                     make_vec3(1, 0, 0))));
  assert(vec3_allEqual(make_vec3(2, -2, 0),
                          vec3_cross(make_vec3(1, 1, 0),
                                     make_vec3(0, 0, 2))));
  assert(vec4_allEqual(make_vec4(0.5), vec4_normalize(make_vec4(1))));
  assert(vec4_allEqual(make_vec4(0, -1, 0, 0), vec4_normalize(make_vec4(0, -3, 0, 0))));
  assert(vec4_distance(vec4_normalize(make_vec4(1, -1, 0, 0)), vec4_normalizeFast(make_vec4(1, -1, 0, 0))) < 0.01);

  assert(vec4_allEqual(make_vec4(-1, 1, -2, 2), vec4_faceforward(make_vec4(1, -1, 2, -2), make_vec4(0, 0, 1, 2), make_vec4(0, 0, 1, 0))));
  assert(vec4_allEqual(make_vec4(1, -1, 2, -2), vec4_faceforward(make_vec4(1, -1, 2, -2), make_vec4(0, 0, -1, 2), make_vec4(0, 0, 1, 0))));

  assert(vec4_distance(vec4_reflect(make_vec4(1, 0, 0, 0), vec4_normalize(make_vec4(-1, 1, 0, 0))), make_vec4(0, 1, 0, 0)) < 0.1);

  assert(vec4_distance(vec4_refract(make_vec4(1, 0, 0, 0), vec4_normalize(make_vec4(-1, 1, 0, 0)), 1), make_vec4(1, 0, 0, 0)) < 0.1);
  assert(vec4_distance(vec4_refract(make_vec4(1, 0, 0, 0), vec4_normalize(make_vec4(-1, 1, 0, 0)), 0.7), make_vec4(1, 0, 0, 0)) > 0.1);
  // Total internal reflection test of glass, as alleged by a random word problem from the interwebs.
  // This angle of total interal reflection is very close to 45 degrees
  assert(vec4_allEqual(vec4_refract(vec4_normalize(make_vec4(1, 0, -1, 0)), make_vec4(0, 0, 1, 0), 1.44), make_vec4(0)));
  assert(!vec4_allEqual(vec4_refract(vec4_normalize(make_vec4(0.9, 0, -1, 0)), make_vec4(0, 0, 1, 0), 1.44), make_vec4(0)));
}
