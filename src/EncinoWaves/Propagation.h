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

#ifndef _EncinoWaves_Propagation_h_
#define _EncinoWaves_Propagation_h_

#include "Foundation.h"
#include "FftwWrapper.h"
#include "SpectralSpatialField.h"
#include "Basics.h"
#include "Parameters.h"
#include "InitialState.h"
#include "Stats.h"
#include <algorithm>

namespace EncinoWaves {

//-*****************************************************************************
template <typename T> struct PropagatedState {
  RealSpatialField2D<T> Height;
  RealSpatialField2D<T> Dx;
  RealSpatialField2D<T> Dy;
  RealSpatialField2D<T> MinE;

  explicit PropagatedState(const Parameters<T> &i_params)
      : Height(i_params.resolutionPowerOfTwo, 1),
        Dx(i_params.resolutionPowerOfTwo, 1),
        Dy(i_params.resolutionPowerOfTwo, 1),
        MinE(i_params.resolutionPowerOfTwo, 1) {}

  explicit PropagatedState(int i_resolutionPowerOfTwo)
      : Height(i_resolutionPowerOfTwo, 1), Dx(i_resolutionPowerOfTwo, 1),
        Dy(i_resolutionPowerOfTwo, 1), MinE(i_resolutionPowerOfTwo, 1) {}
};

//-*****************************************************************************
template <typename T> struct Propagation {
  ComplexSpectralField2D<T> HSpec;
  ComplexSpectralField2D<T> TempSpec;
  RealSpatialField2D<T> TempSpat;

  RealSpatialField2D<T> FiltHeight;

  SpectralToPaddedSpatial2D<T> Converter;

  T Domain;

  explicit Propagation(const Parameters<T> &i_params, int i_nthreads = -1)
      : HSpec(i_params.resolutionPowerOfTwo),
        TempSpec(i_params.resolutionPowerOfTwo),
        TempSpat(i_params.resolutionPowerOfTwo, 1),
        FiltHeight(i_params.resolutionPowerOfTwo, 1),
        Converter(HSpec, TempSpat, i_nthreads), Domain(i_params.domain) {}

  void propagate(const Parameters<T> &i_params, const InitialState<T> &i_istate,
                 PropagatedState<T> &o_pstate, T i_time);
};

//-*****************************************************************************
//-*****************************************************************************
// TYPEDEFS
//-*****************************************************************************
//-*****************************************************************************
typedef PropagatedState<float> PropagatedStatef;
typedef PropagatedState<double> PropagatedStated;

typedef Propagation<float> Propagationf;
typedef Propagation<double> Propagationd;

//-*****************************************************************************
//-*****************************************************************************
// FUNCTORS FOR EVALUATION OF PROPAGATION
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
template <typename T> struct HSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecPos;
  const complex_type *HSpecNeg;
  const real_type *Omega;

  complex_type *HSpecProp;

  real_type Time;

  void operator()(std::size_t i_index) {
    HSpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    const real_type omega = Omega[i_index];
    const real_type cosOmegaT = std::cos(omega * Time);
    const real_type sinOmegaT = std::sin(omega * Time);

    const complex_type fwd(cosOmegaT, -sinOmegaT);
    const complex_type bkwd(cosOmegaT, sinOmegaT);

    complex_type hs = (HSpecPos[i_index] * fwd) + (HSpecNeg[i_index] * bkwd);
    HSpecProp[i_index] = hs;

    EWAV_ASSERT(std::isfinite(hs.real()) && std::isfinite(hs.imag()),
                     "Bad hspec: " << hs << " at index: " << i_index);
  }
};

//-*****************************************************************************
// Note that this is not the derivative, but rather, is the derivative at
// each wavelength scaled by the inverse of the wave number magnitude.
template <typename T> struct DXSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecProp;
  complex_type *DxSpecProp;

  void operator()(std::size_t i_index) {
    DxSpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    DxSpecProp[i_index] =
        complex_type(0.0, -i_k[0] / i_kMag) * HSpecProp[i_index];
  }
};

//-*****************************************************************************
// Note that this is not the derivative, but rather, is the derivative at
// each wavelength scaled by the inverse of the wave number magnitude.
template <typename T> struct DYSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecProp;
  complex_type *DySpecProp;

  void operator()(std::size_t i_index) {
    DySpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    DySpecProp[i_index] =
        complex_type(0.0, -i_k[1] / i_kMag) * HSpecProp[i_index];
  }
};

//-*****************************************************************************
template <typename T> struct DXXSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecProp;
  complex_type *DxxSpecProp;

  void operator()(std::size_t i_index) {
    DxxSpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    DxxSpecProp[i_index] =
        complex_type(sqr(i_k[0]) / i_kMag, 0.0) * HSpecProp[i_index];
  }
};

//-*****************************************************************************
template <typename T> struct DYYSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecProp;
  complex_type *DyySpecProp;

  void operator()(std::size_t i_index) {
    DyySpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    DyySpecProp[i_index] =
        complex_type(sqr(i_k[1]) / i_kMag, 0.0) * HSpecProp[i_index];
  }
};

//-*****************************************************************************
template <typename T> struct DXYSPEC {
  typedef T real_type;
  typedef std::complex<T> complex_type;
  typedef Imath::Vec2<real_type> vec_type;

  const complex_type *HSpecProp;
  complex_type *DxySpecProp;

  void operator()(std::size_t i_index) {
    DxySpecProp[i_index] = complex_type(0.0, 0.0);
  }

  void operator()(const vec_type &i_k, real_type i_kMag, real_type i_dK,
                  std::size_t i_index) {
    DxySpecProp[i_index] =
        complex_type((i_k[0] * i_k[1]) / i_kMag, 0.0) * HSpecProp[i_index];
  }
};

//-*****************************************************************************
template <typename T> struct ComputeMinE {
  const T *Dxx;
  const T *Dyy;
  T *Dxy_and_MinE;
  T Pinch;

  void operator()(const tbb::blocked_range<std::size_t> &i_range) const {
    for (std::size_t i = i_range.begin(); i != i_range.end(); ++i) {
      const T Jxx = T(1) - Pinch * Dxx[i];
      const T Jyy = T(1) - Pinch * Dyy[i];
      const T Jxy = -Pinch * (Dxy_and_MinE[i]);

      const T A = (Jxx + Jyy) / T(2.0);
      const T B = std::sqrt(sqr(Jxx - Jyy) + T(4.0) * sqr(Jxy)) / T(2.0);

      Dxy_and_MinE[i] = -(A - B);
    }
  }
};

//-*****************************************************************************
// Periodic, separable spatial height smoothing. No additional transforms and
// no changes to horizontal displacement or the original crest source.
template <typename T> struct gaussian_height_pass {
  T const* source;
  T* destination;
  std::vector<T> const* weights;
  int width;
  int radius;
  bool vertical;

  void operator()(tbb::blocked_range<std::size_t> const& range) const {
    int const stride = width + 1;
    for (std::size_t index = range.begin(); index != range.end(); ++index) {
      int const x = int(index % stride) % width;
      int const y = int(index / stride) % width;
      T sum = T(0);
      for (int offset = -radius; offset <= radius; ++offset) {
        int const sx = vertical ? x : ((x + offset) % width + width) % width;
        int const sy = vertical ? ((y + offset) % width + width) % width : y;
        sum += (*weights)[offset + radius] * source[sy * stride + sx];
      }
      destination[index] = sum;
    }
  }
};

template <typename T>
void damp_trough_height(T* height, T* temporary, T* smoothed, int const n,
                       T const domain, T const sigma_metres, T const amount) {
  T const spacing = domain / T(n);
  // The ~5-sigma ripple scale must be represented even on the diagonal.
  if (amount <= T(0) || sigma_metres <= T(0) ||
      T(5) * sigma_metres <= std::sqrt(T(2)) * spacing) {
    return;
  }
  // Three-sigma support, at most 33 taps per axis. Extremely fine grids reduce
  // the effective sigma; they never silently widen the physical smoothing.
  int const max_radius = 16;
  T const sigma = std::min(sigma_metres / spacing, T(max_radius) / T(3));
  int const radius = std::min(max_radius, std::max(1, int(std::ceil(T(3) * sigma))));
  std::vector<T> weights(2 * radius + 1);
  T total = T(0);
  for (int offset = -radius; offset <= radius; ++offset) {
    T const value = std::exp(T(-0.5) * sqr(T(offset) / sigma));
    weights[offset + radius] = value;
    total += value;
  }
  for (T &weight : weights) { weight /= total; }
  std::size_t const size = std::size_t(n + 1) * std::size_t(n + 1);
  gaussian_height_pass<T> pass{height, temporary, &weights, n, radius, false};
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), pass);
  pass.source = temporary;
  pass.destination = smoothed;
  pass.vertical = true;
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), pass);

  // Count each periodic texel once: duplicated last rows/columns must not bias
  // the height mask or make it depend on where the patch boundary was placed.
  int const stride = n + 1;
  double const count = double(n) * double(n);
  double const mean = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, n), 0.0,
      [&](tbb::blocked_range<int> const& rows, double sum) {
        for (int y = rows.begin(); y != rows.end(); ++y) {
          for (int x = 0; x < n; ++x) { sum += double(smoothed[y * stride + x]); }
        }
        return sum;
      }, std::plus<double>()) / count;
  double const variance = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, n), 0.0,
      [&](tbb::blocked_range<int> const& rows, double sum) {
        for (int y = rows.begin(); y != rows.end(); ++y) {
          for (int x = 0; x < n; ++x) {
            double const delta = double(smoothed[y * stride + x]) - mean;
            sum += delta * delta;
          }
        }
        return sum;
      }, std::plus<double>()) / count;
  T const deviation = T(std::sqrt(variance));
  if (deviation <= T(1e-8)) { return; }
  T const scale = std::sqrt(T(2)) * deviation;
  T const strength = Imath::clamp(amount, T(0), T(1));
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size),
      [&](tbb::blocked_range<std::size_t> const& range) {
        for (std::size_t i = range.begin(); i != range.end(); ++i) {
          T const depth = (T(mean) - smoothed[i]) / scale;
          T const wet = strength * smoothstep(T(0), T(1), depth);
          height[i] += wet * (smoothed[i] - height[i]);
        }
      });
}

//-*****************************************************************************
template <typename T>
void Propagation<T>::propagate(const Parameters<T> &i_params,
                               const InitialState<T> &i_istate,
                               PropagatedState<T> &o_pstate, T i_time) {
  // Check sizes.
  int N = o_pstate.Height.unpaddedWidth();
  std::size_t dataSize = o_pstate.Height.size();
  std::size_t grainSize = 1024;
  EWAV_ASSERT(
      i_istate.HSpectralPos.height() == N &&
          i_istate.HSpectralNeg.height() == N && i_istate.Omega.height() == N &&
          o_pstate.Dx.width() == (N + 1) && o_pstate.Dy.width() == (N + 1) &&
          o_pstate.MinE.width() == (N + 1),
      "Mismatched sizes in wave propagation.");

  // Make Hspec.
  {
    HSPEC<T> F;
    F.HSpecPos = i_istate.HSpectralPos.cdata();
    F.HSpecNeg = i_istate.HSpectralNeg.cdata();
    F.Omega = i_istate.Omega.cdata();
    F.HSpecProp = HSpec.data();
    F.Time = i_time;
    SpectralIterationFunctor<T, HSPEC<T>, HSPEC<T>> SIF(&F, Domain, N);
  }

  // Make DxxSpec from Hspec
  {
    DXXSPEC<T> F;
    F.HSpecProp = HSpec.cdata();
    F.DxxSpecProp = TempSpec.data();
    SpectralIterationFunctor<T, DXXSPEC<T>, DXXSPEC<T>> SIF(&F, Domain, N);
  }

  // Compute Dxx, temporarily put into Dx.
  { Converter.execute(TempSpec, o_pstate.Dx); }

  // Make DyySpec from Hspec
  {
    DYYSPEC<T> F;
    F.HSpecProp = HSpec.cdata();
    F.DyySpecProp = TempSpec.data();
    SpectralIterationFunctor<T, DYYSPEC<T>, DYYSPEC<T>> SIF(&F, Domain, N);
  }

  // Compute Dyy, temporarily put into Dy.
  { Converter.execute(TempSpec, o_pstate.Dy); }

  // Make DxySpec from Hspec
  {
    DXYSPEC<T> F;
    F.HSpecProp = HSpec.cdata();
    F.DxySpecProp = TempSpec.data();
    SpectralIterationFunctor<T, DXYSPEC<T>, DXYSPEC<T>> SIF(&F, Domain, N);
  }

  // Compute Dxy, temporarily put into MinE.
  { Converter.execute(TempSpec, o_pstate.MinE); }

  // Compute MinE from Dxx, Dyy, Dxy.
  {
    ComputeMinE<T> F;
    F.Dxx = o_pstate.Dx.cdata();
    F.Dyy = o_pstate.Dy.cdata();
    F.Dxy_and_MinE = o_pstate.MinE.data();
    F.Pinch = T(1.25);
    // CJH HACK
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, dataSize /*, grainSize*/), F);
  }

  // Make DxSpec from Hspec
  {
    DXSPEC<T> F;
    F.HSpecProp = HSpec.cdata();
    F.DxSpecProp = TempSpec.data();
    SpectralIterationFunctor<T, DXSPEC<T>, DXSPEC<T>> SIF(&F, Domain, N);
  }

  // Compute Dx
  { Converter.execute(TempSpec, o_pstate.Dx); }

  // Make DySpec from Hspec
  {
    DYSPEC<T> F;
    F.HSpecProp = HSpec.cdata();
    F.DySpecProp = TempSpec.data();
    SpectralIterationFunctor<T, DYSPEC<T>, DYSPEC<T>> SIF(&F, Domain, N);
  }

  // Compute Dy
  { Converter.execute(TempSpec, o_pstate.Dy); }

  // Synthesize height once, then quiet only small ripples below the mean.
  // Dx, Dy and MinE above remain exactly as originally synthesized.
  Converter.execute(HSpec, o_pstate.Height);
  damp_trough_height(o_pstate.Height.data(), TempSpat.data(), FiltHeight.data(),
                   N, Domain, i_params.troughSmoothingLength, i_params.troughDamping);
}

} // namespace EncinoWaves

#endif
