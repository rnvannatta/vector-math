# vector-math
OpenGL Targeting Small Vector Library for sse4.2

This is my second attempt at this: it's in very early development stages. The goal is to have a small vector library capable of all the bells and whistles of GLSL + constructors for matrices. The first attempt didn't have any generics, and had a poor ABI due to my usage of unions with SSE intrinsics.
