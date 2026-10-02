#include "scalar_adapters.h"
#include "EncinoWaves/Parameters.h"
#include "EncinoWaves/Dispersion.h"
#include "EncinoWaves/Spectra.h"
#include "EncinoWaves/DirectionalSpreading.h"

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
