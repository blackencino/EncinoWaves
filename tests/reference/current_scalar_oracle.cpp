// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Exercise the maintained C++ headers; Python supplies an independent reference.
#include "scalar_adapters.h"
#include "EncinoWaves/Parameters.h"
#include "EncinoWaves/Dispersion.h"
#include "EncinoWaves/Spectra.h"
#include "EncinoWaves/DirectionalSpreading.h"

namespace ew = EncinoWaves;

template<class T> struct Gamma_probe : ew::JONSWAPSpectrum<T> {
    using ew::JONSWAPSpectrum<T>::JONSWAPSpectrum;
    T gamma() const { return this->m_gamma; }
};

template<class T>
T direction(ew::Parameters<T> const& p, int const model, T const omega, T const theta) {
    switch (model) {
    case 0: return ew::HasselmannDirectionalSpreading<T>(p)(omega, theta, 0, 0);
    case 1: return ew::MitsuyasuDirectionalSpreading<T>(p)(omega, theta, 0, 0);
    case 2: return ew::DonelanBannerDirectionalSpreading<T>(p)(omega, theta, 0, 0);
    default: return ew::PosCosSquaredDirectionalSpreading<T>(p)(omega, theta, 0, 0);
    }
}

template<class T>
void probe() {
    int const bits = 8 * sizeof(T);
    ew::Parameters<T> p;
    p.depth = 20;
    // kind, model, three input values, two output values, for float and double.
    auto row = [bits](int kind, int model, T a, T b, T c, T x, T y) {
        std::cout << bits << ' ' << kind << ' ' << model << ' ' << a << ' '
                  << b << ' ' << c << ' ' << x << ' ' << y << '\n';
    };
    for (T const k : {T(0), T(1e-12), T(.001), T(.1), T(1), T(100), T(10000)}) {
        T omega, derivative;
        ew::DeepDispersion<T>{p}(k, omega, derivative);
        row(0, 0, k, 0, 0, omega, derivative);
        ew::FiniteDepthDispersion<T>{p}(k, omega, derivative);
        row(0, 1, k, 0, 0, omega, derivative);
        ew::CapillaryDispersion<T>{p}(k, omega, derivative);
        row(0, 2, k, 0, 0, omega, derivative);
    }
    for (T const omega : {T(0), T(1e-12), T(.01), T(.1), T(.5), T(1), T(10), T(1000)}) {
        T const gamma = Gamma_probe<T>(p).gamma();
        row(1, 0, omega, gamma, 0, ew::PiersonMoskowitzSpectrum<T>(p)(omega), 0);
        row(1, 1, omega, gamma, 0, ew::JONSWAPSpectrum<T>(p)(omega), 0);
        row(1, 2, omega, gamma, 0, ew::TMASpectrum<T>(p)(omega),
            ew::TMASpectrum<T>(p).kitaigorodskiiDepth(omega));
    }
    T const peak = ew::modalAngularFrequencyJONSWAP(p.gravity, p.windSpeed, p.fetch * T(1000));
    for (int model = 0; model < 4; ++model) {
        for (T const swell : {T(-1), T(-.5), T(0), T(.35), T(1), T(2)}) {
            p.directionalSpreading.swell = swell;
            for (T const ratio : {T(0), T(.01), T(.6), T(.95), T(1), T(1.6), T(3), T(10), T(100)}) {
                for (T const angle : {T(-1), T(-.75), T(-.5), T(-.1), T(0), T(.1), T(.5), T(.75), T(1)}) {
                    T const omega = peak * ratio;
                    T const theta = angle * ew::PI<T>;
                    row(2, model, swell, ratio, angle, direction(p, model, omega, theta), 0);
                }
            }
        }
        // Independent fine midpoint integral, including the broad low-frequency lobe
        // that the original half-circle normalization mishandled.
        for (T const swell : {T(-1), T(0), T(2)}) {
            p.directionalSpreading.swell = swell;
            for (T const ratio : {T(.01), T(1), T(10)}) {
                double sum = 0;
                int const n = 4096;
                for (int i = 0; i < n; ++i) {
                    T const angle = T(-1 + 2.0 * (i + .5) / n) * ew::PI<T>;
                    sum += direction(p, model, peak * ratio, angle);
                }
                row(3, model, swell, ratio, 0, T(sum * ew::TAU<double> / n), 0);
            }
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    probe<float>();
    probe<double>();
}
