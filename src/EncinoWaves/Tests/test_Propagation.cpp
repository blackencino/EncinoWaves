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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace ewav = EncinoWaves;

//-*****************************************************************************
int main(int argc, char* argv[]) {
  ewav::Parametersf params;
  params.resolutionPowerOfTwo = 11;

  ewav::InitialStatef istate(params);
  int w = istate.HSpectralPos.width();
  int h = istate.HSpectralPos.height();
  int N = h;
  std::cout << "Computed initial state." << std::endl
            << "Size: " << w << " by " << h << std::endl
            << "HspecPos: " << istate.HSpectralPos[N / 4][N / 4] << std::endl
            << "HspecNeg: " << istate.HSpectralNeg[N / 4][N / 4] << std::endl
            << "Omega: " << istate.Omega[N / 4][N / 4] << std::endl;

  ewav::PropagatedStatef pstate(params);
  std::cout << "Created propagated state." << std::endl;

  ewav::Propagationf prop(params);
  std::cout << "Created propagation." << std::endl;

  for (int frame = 1; frame < 24; ++frame) {
    float ftime = float(frame) / 24.0f;
    prop.propagate(params, istate, pstate, ftime);
    std::cout << "Propagated to frame: " << frame << std::endl
              << "H: " << pstate.Height[N / 4][N / 4] << std::endl
              << "Dx: " << pstate.Dx[N / 4][N / 4] << std::endl
              << "Dy: " << pstate.Dy[N / 4][N / 4] << std::endl
              << "MinE: " << pstate.MinE[N / 4][N / 4] << std::endl;
  }

  return 0;
}