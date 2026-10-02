#pragma once
// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Compile the actual scalar headers, without pulling in the old renderer,
// TBB or Imath. The tiny adapters below supply only their scalar dependencies.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#define _EncinoWaves_Foundation_h_
#define _EncinoWaves_Basics_h_
#define M_TAU 6.28318530717958647693
namespace Imath {
template<class T> struct Vec2 {
    T x, y;
    Vec2(T const x_, T const y_) : x(x_), y(y_) {}
    T operator[](int const index) const { return index == 0 ? x : y; }
};
template<class T> T clamp(T const x,T const a,T const b) { return std::clamp(x,a,b); }
template<class T> T lerp(T const a,T const b,T const t) { return (1-t)*a+t*b; }
}
namespace EncinoWaves {
template<class T> constexpr T PI = static_cast<T>(3.14159265358979323846);
template<class T> constexpr T PI_2 = static_cast<T>(1.57079632679489661923);
template<class T> constexpr T TAU = static_cast<T>(6.28318530717958647693);
template<class T> T sqr(T const x) { return x*x; }
template<class T> T cube(T const x) { return x*x*x; }
}
