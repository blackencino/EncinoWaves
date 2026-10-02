//-*****************************************************************************
// Copyright 2015 Christopher Jon Horvath
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//-*****************************************************************************

//-*****************************************************************************
// The basic architecture of these Waves is based on the TweakWaves application
// written by Chris Horvath for Tweak Films in 2001.  This, in turn, was based
// on the SIGGRAPH papers and courses by Jerry Tessendorf, and by the paper
// "A Simple Fluid Solver based on the FTT" by Jos Stam.
//
// The TMA, JONSWAP, and Pierson Moskowitz Wave Spectra, as well as the
// directional spreading functions are formulated based on the descriptions
// given in "Ocean Waves: The Stochastic Approach",
// by Michel K. Ochi, published by Cambridge Ocean Technology Series, 1998,2005.
//
// This library is written as a working implementation of the paper:
// Christopher J. Horvath. 2015.
// Empirical directional wave spectra for computer graphics.
// In Proceedings of the 2015 Symposium on Digital Production (DigiPro '15),
// Los Angeles, Aug. 8, 2015, pp. 29-39.
//-*****************************************************************************

#ifndef _EncinoWaves_DirectionalSpreading_h_
#define _EncinoWaves_DirectionalSpreading_h_

#include "Foundation.h"
#include "Basics.h"
#include "Parameters.h"

#include <array>

namespace EncinoWaves {

//------------------------------------------------------------------------------
template <typename T, typename FUNC>
T numericallyIntegrate(FUNC f, T a, T b, int n) {
  T nf  = static_cast<T>(n);
  T sum = 0;
  for (int k = 1; k < n; ++k) {
    sum += f(a + (k * (b - a) / nf));
  }
  return ((b - a) / nf) * ((f(a) / 2) + (f(b) / 2) + sum);
}

//------------------------------------------------------------------------------
template <typename T>
T swellShape(T omega, T modal_omega, T swell_amount) {
  return T(16) * (omega > T(0) ? std::tanh(modal_omega / omega) : T(1)) *
         sqr(swell_amount);
}

//------------------------------------------------------------------------------
template <typename T>
T swell(T theta, T omega, T modal_omega, T swell_amount) {
  T const shape = swellShape(omega, modal_omega, swell_amount);
  T const half_cos = std::sin((PI<T> - std::abs(theta)) / T(2));
  return std::pow(std::abs(half_cos), T(2) * shape);
}

//------------------------------------------------------------------------------
template <typename T, typename FUNCA, typename FUNCB>
T normalizedSwellDirectionalProduct(T theta, FUNCA A, FUNCB B,
                                    T extent = PI<T>) {
  auto product = [A, B](T x) -> T { return A(x) * B(x); };

  // Even 64-point Gauss-Legendre rule, over the complete support. The old
  // half-circle trapezoid over-normalized Donelan-Banner and under-resolved
  // narrow swell lobes. Accumulate in double even for float fields.
  static std::array<std::array<double, 2>, 32> const quadrature = {{
    {0.024350292663424436, 0.048690957009139689},
    {0.072993121787799042, 0.048575467441503428},
    {0.12146281929612056, 0.048344762234802933},
    {0.1696444204239928, 0.047999388596458303},
    {0.21742364374000708, 0.047540165714830343},
    {0.26468716220876742, 0.046968182816210007},
    {0.31132287199021097, 0.046284796581314375},
    {0.35722015833766813, 0.0454916279274181},
    {0.40227015796399157, 0.044590558163756601},
    {0.44636601725346409, 0.043583724529323471},
    {0.48940314570705296, 0.042473515123653542},
    {0.53127946401989457, 0.041262563242623486},
    {0.571895646202634, 0.039953741132720454},
    {0.61115535517239328, 0.038550153178615605},
    {0.64896547125465731, 0.037055128540240019},
    {0.68523631305423327, 0.035472213256882226},
    {0.71988185017161077, 0.033805161837141877},
    {0.75281990726053194, 0.032057928354851412},
    {0.78397235894334139, 0.030234657072402488},
    {0.81326531512279754, 0.028339672614259424},
    {0.84062929625258032, 0.026377469715054894},
    {0.86599939815409277, 0.024352702568711086},
    {0.88931544599511414, 0.022270173808382945},
    {0.91052213707850282, 0.020134823153530039},
    {0.92956917213193957, 0.017951715775697288},
    {0.94641137485840277, 0.01572603047602511},
    {0.96100879965205366, 0.01346304789671912},
    {0.97332682778991098, 0.011168139460131008},
    {0.98333625388462598, 0.0088467598263633606},
    {0.99101337147674429, 0.0065044579689784374},
    {0.99634011677195522, 0.0041470332605646485},
    {0.99930504173577217, 0.0017832807216942642},
  }};
  double integral = 0;
  for (auto const& point : quadrature) {
    integral += point[1] * double(product(T(point[0] * extent)));
  }
  T const denom = T(2 * double(extent) * integral);
  return product(theta) / denom;
}

//------------------------------------------------------------------------------
template <typename T>
T modalAngularFrequencyJONSWAP(T gravity, T meanWindSpeed, T fetchLength) {
  T dimensionlessFetch = gravity * fetchLength / sqr(meanWindSpeed);
  return TAU<T> * 3.5 * (gravity / meanWindSpeed) *
         std::pow(dimensionlessFetch, -0.33);
}

//------------------------------------------------------------------------------
// Gamma duplication reduces the cosine-power normalization to a stable ratio.
// The double intermediate also limits cancellation for float exponents.
template <typename T>
T cosine_power_spreading(T const theta, T const shape) {
  double const s = double(shape);
  double const log_q = std::lgamma(s + 1) - std::lgamma(s + .5) -
                       std::log(2 * std::sqrt(PI<double>));
  // Exactly zero at +/- pi, even when cos(pi/2) rounds away from zero.
  T const half_cos = std::sin((PI<T> - std::abs(theta)) / T(2));
  return T(std::exp(log_q)) * std::pow(std::abs(half_cos), T(2) * shape);
}

//------------------------------------------------------------------------------
template <typename T>
class DonelanBannerDirectionalSpreading {
public:
  DonelanBannerDirectionalSpreading(const Parameters<T>& params)
      : m_modalAngularFrequency(modalAngularFrequencyJONSWAP(
          params.gravity, params.windSpeed, params.fetch * T(1000)))
      , m_swell(params.directionalSpreading.swell) {}

  T operator()(T i_omega, T i_theta, T i_kMag, T i_dTheta) const {
    T omega_over_modal_omega = i_omega / m_modalAngularFrequency;
    T beta_s;
    if (omega_over_modal_omega < 0.95) {
      beta_s = 2.61 * std::pow(omega_over_modal_omega, 1.3);
    } else if (omega_over_modal_omega < 1.6) {
      beta_s = 2.28 * std::pow(omega_over_modal_omega, -1.3);
    } else {
      T expo =
        -0.4 +
        0.8393 * std::exp(-0.567 * std::log(sqr(omega_over_modal_omega)));
      beta_s = std::pow(10, expo);
    }

    // We need to do a numerical integration to determine the
    // normalization factor for the product of the original function (B)
    // with the swell elongation (A).
    auto A = [this, i_omega](T x) -> T {
      return swell(x, i_omega, m_modalAngularFrequency, m_swell);
    };
    auto B = [beta_s](T const x) -> T {
      T const e = std::exp(-T(2) * std::abs(beta_s * x));
      return T(4) * e / sqr(T(1) + e);
    };

    if (m_swell > 0.0) {
      return normalizedSwellDirectionalProduct(i_theta, A, B);
    } else {
      // beta/tanh(pi*beta) has a removable singularity at beta=0.
      T const integral = beta_s > T(0) ?
        T(2) * std::tanh(beta_s * PI<T>) / beta_s : TAU<T>;
      T const d = B(i_theta) / integral;
      return Imath::lerp(d, T(1) / TAU<T>,
                         Imath::clamp(-m_swell, T(0), T(1)));
    }
  }

protected:
  T m_modalAngularFrequency;
  T m_swell;
};

//------------------------------------------------------------------------------
template <typename T>
class MitsuyasuDirectionalSpreading {
public:
  MitsuyasuDirectionalSpreading(const Parameters<T>& params)
      : m_modalAngularFrequency(modalAngularFrequencyJONSWAP(
          params.gravity, params.windSpeed, params.fetch * T(1000)))
      , m_modalShape(11.5 * std::pow(m_modalAngularFrequency *
                                       params.windSpeed / params.gravity,
                                     -2.5))
      , m_modalCelerity(params.gravity / m_modalAngularFrequency)
      , m_windSpeedOverCelerity(params.windSpeed / m_modalCelerity)
      , m_swell(params.directionalSpreading.swell) {}

  T operator()(T i_omega, T i_theta, T i_kMag, T i_dTheta) const {
    T shape_bias = 0.0;
    if (m_swell >= 0.0) {
      shape_bias = swellShape(i_omega, m_modalAngularFrequency, m_swell);
    }

    T shape_exp = i_omega <= m_modalAngularFrequency ? 5.0 : -2.5;
    T shape =
      m_modalShape * std::pow(i_omega / m_modalAngularFrequency, shape_exp);

    shape += shape_bias;

    T const direction = cosine_power_spreading(i_theta, shape);
    if (m_swell < 0) {
      return Imath::lerp(direction, T(1) / T(TAU<T>),
                         Imath::clamp(-m_swell, T(0), T(1)));
    } else {
      return direction;
    }
  }

protected:
  T m_modalAngularFrequency;
  T m_modalShape;
  T m_modalCelerity;
  T m_windSpeedOverCelerity;
  T m_swell;
};

//------------------------------------------------------------------------------
template <typename T>
class HasselmannDirectionalSpreading {
public:
  HasselmannDirectionalSpreading(const Parameters<T>& params)
      : m_modalAngularFrequency(modalAngularFrequencyJONSWAP(
          params.gravity, params.windSpeed, params.fetch * T(1000)))
      , m_modalShape(11.5 * std::pow(m_modalAngularFrequency *
                                       params.windSpeed / params.gravity,
                                     -2.5))
      , m_modalCelerity(params.gravity / m_modalAngularFrequency)
      , m_windSpeedOverCelerity(params.windSpeed / m_modalCelerity)
      , m_swell(params.directionalSpreading.swell) {}

  T operator()(T i_omega, T i_theta, T i_kMag, T i_dTheta) const {
    T shape_bias = 0.0;
    if (m_swell >= 0.0) {
      shape_bias = swellShape(i_omega, m_modalAngularFrequency, m_swell);
    }

    T shape;
    if (i_omega > m_modalAngularFrequency) {
      shape =
        9.77 * std::pow(i_omega / m_modalAngularFrequency,
                        -2.33 - (1.45 * (m_windSpeedOverCelerity - 1.17)));
    } else {
      shape = 6.97 * std::pow(i_omega / m_modalAngularFrequency, 4.06);
    }
    shape += shape_bias;

    T const direction = cosine_power_spreading(i_theta, shape);
    if (m_swell < 0) {
      return Imath::lerp(direction, T(1) / T(TAU<T>),
                         Imath::clamp(-m_swell, T(0), T(1)));
    } else {
      return direction;
    }
  }

protected:
  T m_modalAngularFrequency;
  T m_modalShape;
  T m_modalCelerity;
  T m_windSpeedOverCelerity;
  T m_swell;
};

//------------------------------------------------------------------------------
template <typename T>
class PosCosSquaredDirectionalSpreading {
protected:
  static T modalAngularFrequencyJONSWAP(T gravity, T meanWindSpeed,
                                        T fetchLength) {
    T dimensionlessFetch = gravity * fetchLength / sqr(meanWindSpeed);
    return TAU<T> * 3.5 * (gravity / meanWindSpeed) *
           std::pow(dimensionlessFetch, -0.33);
  }

public:
  PosCosSquaredDirectionalSpreading(const Parameters<T>& params)
      : m_modalAngularFrequency(modalAngularFrequencyJONSWAP(
          params.gravity, params.windSpeed, params.fetch * T(1000)))
      , m_swell(params.directionalSpreading.swell) {}

  T operator()(T i_omega, T i_theta, T i_kMag, T i_dTheta) const {
    auto A = [this, i_omega](T x) -> T {
      return swell(x, i_omega, m_modalAngularFrequency,
                   std::max(T(0), m_swell));
    };
    auto B = [](T x) -> T {
      if (x < -PI_2<T> || x > PI_2<T>) {
        return T{0};
      } else {
        return sqr(std::cos(x));
      }
    };

    T const direction = normalizedSwellDirectionalProduct(i_theta, A, B, PI_2<T>);
    return m_swell < T(0) ?
      Imath::lerp(direction, T(1) / TAU<T>, Imath::clamp(-m_swell, T(0), T(1))) :
      direction;
  }

protected:
  T m_modalAngularFrequency;
  T m_swell;
};

}  // namespace EncinoWaves

#endif
