# vector-math
OpenGL Targeting Small Vector Library in C for sse4.2

This is my second attempt at this: it's in very early development stages. The goal is to have a small vector library capable of all the bells and whistles of GLSL + constructors for matrices. The first attempt didn't have any generics, and had a poor ABI due to my usage of unions with SSE intrinsics.

To use, simply `#include "vector-math.h"`. The other two files are for unit testing.

Excluding swizzling, the implementation of vec4's are complete and unit tested. Vec3's are also written, but not completely unit tested.

Example usage is

```
vec4 a = make_vec4(make_vec3(4, 5, 2), 1);
vec4 b = make_vec4(1);
b = vec4_sub(a, b);
float y = vec4_getY(b);
```

As you can see, `make_vec4()` is a fairly sophisticated macro using `_Generic` and struct metaprogramming. It works by counting the number of arguments, then incrementally building a struct that represents the sum type of all the inputs, and finally dispatches the arguments using that struct as a dispatcher with `_Generic`.

Unlike most small vector libraries, this library does not have an anonymous union over the raw type, hence why we have to do `vec4_getX(a)` instead of `a.x`. This is done because the ABI of such an implementation ended up passing the memory on the stack. By having `vec4` merely be `struct { __m128 raw; };`, the entire `vec4` can be passed in a single xmm register.

In this library, `vec3` and `vec4` are intended to be used as fast types for CPU side calculations. They should not be used for OpenGL buffers. The only way to have something that matches the ABI of std430 OpenGL is to use array types, which is provided with the `vec4a` and `vec3a` typedefs. Conversion between vec4 and vec4a is done with `vec4_pack()` and `vec4_unpack()`
