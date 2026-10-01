// Copyright 2026 Christopher Jon Horvath. Apache-2.0.
// Compile the actual 2015 scalar headers, without pulling in the old renderer,
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
#include "../../src/EncinoWaves/Parameters.h"
#include "../../src/EncinoWaves/Dispersion.h"
#include "../../src/EncinoWaves/Spectra.h"
#include "../../src/EncinoWaves/DirectionalSpreading.h"

struct Gamma_probe : EncinoWaves::JONSWAPSpectrum<float> {
    using JONSWAPSpectrum::JONSWAPSpectrum;
    float gamma() const { return m_gamma; }
};

int main() {
    namespace ew=EncinoWaves;
    ew::Parameters<float> p;
    p.resolutionPowerOfTwo=4;
    p.domain=100;
    p.windSpeed=17;
    p.fetch=300;
    p.depth=20;
    p.directionalSpreading.swell=.35f;
    ew::CapillaryDispersion<float> const dispersion(p);
    ew::TMASpectrum<float> const spectrum(p);
    ew::HasselmannDirectionalSpreading<float> const spreading(p);
    ew::NormalRandom<float> random(p);
    std::cout << std::setprecision(10) << Gamma_probe(p).gamma() << '\n';
    auto const n=p.resolution();
    float const dk=ew::TAU<float>/p.domain;
    for (int j=0;j<n;++j) {
        int const real_j=j<=n/2 ? j : j-n;
        for (int i=0;i<=n/2;++i) {
            float const kx=float(i)*ew::TAU<float>/p.domain;
            float const ky=float(real_j)*ew::TAU<float>/p.domain;
            float const k=std::sqrt(kx*kx+ky*ky);
            if (k==0) { std::cout << "0 0 0 0 0 0 0 0 0 0 0\n"; continue; }
            float omega,derivative;
            dispersion(k,omega,derivative);
            float const theta_pos=std::atan2(-ky,kx);
            float const theta_neg=std::atan2(ky,-kx);
            float const dtheta=std::abs(std::atan2(dk,k));
            float const energy=spectrum(omega)*dk*dk*derivative/k;
            random.seed(Imath::Vec2<float>(kx,ky));
            float const a=random.nextAmp()*std::sqrt(2*energy*spreading(omega,theta_pos,k,dtheta));
            float const b=random.nextAmp()*std::sqrt(2*energy*spreading(omega,theta_neg,k,dtheta));
            float const pa=random.nextPhase();
            float const pb=random.nextPhase();
            std::cout << a*std::cos(pa) << ' ' << -a*std::sin(pa) << ' '
                      << b*std::cos(pb) << ' ' << -b*std::sin(pb) << ' ' << omega << ' '
                      << kx << ' ' << ky << ' ' << derivative << ' ' << spectrum(omega) << ' '
                      << spreading(omega,theta_pos,k,dtheta) << ' '
                      << spreading(omega,theta_neg,k,dtheta) << '\n';
        }
    }
}
