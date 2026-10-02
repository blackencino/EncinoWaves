// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Exercise the actual spatial helper without the legacy FFTW/viewer dependencies.
// Only scheduling and existing scalar dependencies are adapted; no filter is copied.
#include "scalar_adapters.h"
#include <functional>
#include <vector>

#define _EncinoWaves_FftwWrapper_h_
#define _EncinoWaves_SpectralSpatialField_h_
#define _EncinoWaves_InitialState_h_
#define _EncinoWaves_Stats_h_
#define EWAV_ASSERT(test, message) assert(test)

namespace tbb {
template<class T> struct blocked_range {
    T first, last;
    blocked_range(T const a, T const b) : first(a), last(b) {}
    T begin() const { return first; }
    T end() const { return last; }
};
template<class R, class F>
void parallel_for(R const& range, F const& function) { function(range); }
template<class R, class T, class F, class J>
T parallel_reduce(R const& range, T initial, F const& function, J const&) {
    return function(range, initial);
}
}

namespace EncinoWaves {
template<class T> class RealSpatialField2D;
template<class T> class ComplexSpectralField2D;
template<class T> class SpectralToPaddedSpatial2D;
template<class T> class InitialState;
template<class T, class A, class B> class SpectralIterationFunctor;
template<class T> T smoothstep(T const a, T const b, T const x) {
    T const t = std::clamp((x-a)/(b-a), T(0), T(1));
    return t*t*(T(3)-T(2)*t);
}
}
#include "EncinoWaves/Propagation.h"

template<class T>
void probe(int const n, double const domain, T const amount) {
    int const stride = n+1;
    std::vector<T> original(stride*stride), temporary(original.size()), smoothed(original.size());
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            double const wx = double(x%n)*domain/n;
            double const wy = double(y%n)*domain/n;
            double const tau = EncinoWaves::TAU<double>;
            original[y*stride+x] = T(std::cos(tau*wx/4)+.3*std::cos(tau*wy/8)
                +.02*std::cos(tau*wx/.25+.7)+.01*std::sin(tau*wy/.25));
        }
    }
    auto result = original;
    EncinoWaves::damp_trough_height(result.data(), temporary.data(), smoothed.data(),
                                  n, T(domain), T(.1), amount);
    for (int y = 0; y <= n; ++y) { assert(result[y*stride] == result[y*stride+n]); }
    for (int x = 0; x <= n; ++x) { assert(result[x] == result[n*stride+x]); }
    for (T const value : result) { assert(std::isfinite(value)); }
    if (amount == 0 || 5*.1 <= std::sqrt(2.)*domain/n) { assert(result == original); }
    for (int const y : {0, n/8, n/4, n/3, n/2, 3*n/4, n-1}) {
        for (int const x : {0, n/8, n/4, n/3, n/2, 3*n/4, n-1}) {
            int const index = y*stride+x;
            std::cout << 8*sizeof(T) << ' ' << n << ' ' << domain << ' ' << amount
                      << ' ' << y << ' ' << x << ' ' << original[index]
                      << ' ' << result[index] << '\n';
        }
    }
    std::fill(result.begin(), result.end(), T(2));
    EncinoWaves::damp_trough_height(result.data(), temporary.data(), smoothed.data(),
                                  n, T(domain), T(.1), T(1));
    for (T const value : result) { assert(value == T(2)); }
}

int main() {
    EncinoWaves::Parameters<double> const parameters;
    assert(parameters.troughDamping == .5 && parameters.troughSmoothingLength == .1);
    std::cout << std::setprecision(17);
    for (double const domain : {8., 64.}) {
        for (double const amount : {0., .5, 1.}) {
            probe<float>(128, domain, float(amount));
            probe<double>(128, domain, amount);
        }
    }
}
