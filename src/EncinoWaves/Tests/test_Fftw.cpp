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

#include <EncinoWaves/All.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace ewav = EncinoWaves;

typedef ewav::CSpectralField2Df::value_type Cf;

//-*****************************************************************************
struct RandFillFunctor {
  Cf* Spectral;
  std::size_t StrideJ;
  int N;
  float Domain;
  ewav::seed_type Seed;

  void operator()(tbb::blocked_range2d<int> i_range) const {
    ewav::Rand48_Engine r48;
    std::normal_distribution<float> gdist{0.0f, 1.0f};

    const float maxKmag = float(N / 2) * M_TAU / Domain;

    const float dk = float(1) * M_TAU / Domain;

    for (int j = i_range.rows().begin(); j != i_range.rows().end(); ++j) {
      // kj is the wave number in the j direction.
      int realJ = j <= (N / 2) ? j : j - N;
      float kj  = float(realJ) * M_TAU / Domain;

      for (int i = i_range.cols().begin(); i != i_range.cols().end(); ++i) {
        // ki is the wave number in the i direction
        float ki = float(i) * M_TAU / Domain;

        // Magnitude of the wave vector.
        float kMag = std::hypot(ki, kj);

        // Index into the array.
        std::size_t index = (std::size_t(j) * StrideJ) + i;

        // If the wavenumber is zero, or we're in the outer ring,
        // set it to zero.
        if ((i == 0 && realJ == 0) || (kMag > maxKmag)) {
          Spectral[index] = Cf(0.0, 0.0);
        } else {
          r48.seed(ewav::SeedFromWavenumber(Imath::V2f(ki, kj), Seed));
          Spectral[index] = dk * dk * Cf(gdist(r48), gdist(r48));
        }
      }
    }
  }
};

//-*****************************************************************************
int main(int argc, char* argv[]) {
  int powerOfTwo = 12;
  ewav::RSpatialField2Df spatial(powerOfTwo);
  int N = spatial.width();
  std::cout << "Made " << N << " x " << N << " spatial field." << std::endl;

  ewav::CSpectralField2Df spectral(powerOfTwo);
  std::cout << "Made " << N << " x " << N << " spectral field." << std::endl;

  ewav::SpectralToSpatial2Df convert(spectral, spatial);
  std::cout << "Made " << N << " x " << N << " converter." << std::endl;

  // Fill spectral
  {
    RandFillFunctor F;
    F.Spectral = spectral.data();
    F.StrideJ  = spectral.stride();
    F.N        = N;
    F.Domain   = 1000.0f;
    F.Seed     = 54321;

    // Rows, then columns.
    tbb::blocked_range2d<int> range{0, spectral.height(), 1,
                                    0, spectral.width(),  512};
    tbb::parallel_for(range, F);
  }
  std::cout << "Filled spectral array with gaussian random numbers"
            << std::endl;

  // Convert.
  convert.execute(spectral, spatial);
  std::cout << "Converted to spatial." << std::endl
            << "Spatial midpoint: " << spatial[N / 2][N / 2] << std::endl;

  return 0;
}