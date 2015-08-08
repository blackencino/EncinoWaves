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

#include <boost/program_options.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace ewav = EncinoWaves;
namespace po   = boost::program_options;

//-*****************************************************************************
void doTest(const ewav::Parametersf& i_params) {
  ewav::InitialStatef istate(i_params);
  int w = istate.HSpectralPos.width();
  int h = istate.HSpectralPos.height();
  int N = h;
  std::cout << "A: Computed initial state." << std::endl
            << "Size: " << w << " by " << h << std::endl
            << "HspecPos: " << istate.HSpectralPos[N / 4][N / 4] << std::endl
            << "HspecNeg: " << istate.HSpectralNeg[N / 4][N / 4] << std::endl
            << "Omega: " << istate.Omega[N / 4][N / 4] << std::endl;
}

//-*****************************************************************************
int main(int argc, char* argv[]) {
  ewav::Parametersf params;

  int threads              = -1;
  int dispersion           = 2;
  int spectrum             = 2;
  int directionalSpreading = 0;
  int filter               = 0;
  int random               = 0;
  int seed                 = 54321;

  po::options_description desc("Tweak Waves 2013 : Initial State Test");
  desc.add_options()

    // clang-format off

    ( "help,h", "prints this help message" )


    ( "threads",
      po::value<int>( &threads )
      ->default_value( threads ),
      "Threads to use. Default of -1 means use all." )

    ( "resolution",
      po::value<int>( &params.resolutionPowerOfTwo )
      ->default_value( params.resolutionPowerOfTwo ),
      "Power of Two of the sim resolution" )

    ( "domain",
      po::value<float>( &params.domain )
      ->default_value( params.domain ),
      "Size, in meters, of largest wave" )

    ( "gravity",
      po::value<float>( &params.gravity )
      ->default_value( params.gravity ),
      "Gravitational constant, in meters per second squared" )

    ( "surfaceTension",
      po::value<float>( &params.surfaceTension )
      ->default_value( params.surfaceTension ),
      "Surface tension constant, in Newtons per meter" )

    ( "density",
      po::value<float>( &params.density )
      ->default_value( params.density ),
      "Water density, in kilograms per meter cubed" )

    ( "depth",
      po::value<float>( &params.depth )
      ->default_value( params.depth ),
      "Average depth of the ocean, in meters" )

    ( "windSpeed",
      po::value<float>( &params.windSpeed )
      ->default_value( params.windSpeed ),
      "Average wind speed, in meters per second" )

    ( "fetch",
      po::value<float>( &params.fetch )
      ->default_value( params.fetch ),
      "Wind fetch, in KILOMETERS" )

    ( "pinch",
      po::value<float>( &params.pinch )
      ->default_value( params.pinch ),
      "Lateral displacement, normalized" )

    ( "amplitudeGain",
      po::value<float>( &params.amplitudeGain )
      ->default_value( params.amplitudeGain ),
      "Gain on the wave height" )

    ( "dispersion",
      po::value<int>( &dispersion )
      ->default_value( dispersion ),
      "Dispersion: 0 for Deep, 1 for Finite Depth, 2 for Capillary" )

    ( "spectrum",
      po::value<int>( &spectrum )
      ->default_value( spectrum ),
      "Spectrum: 0 for Pierson-Moskowitz, 1 for JONSWAP, 2 for TMA" )

    ( "directionalSpreading",
      po::value<int>( &directionalSpreading )
      ->default_value( directionalSpreading ),
      "Directional Spreading: 0 for Balanced Cos2 Theta, "
      "1 for Mitsuyasu, 2 for Hasselmann, 3 for Donelan-Banner" )

    ( "swell",
      po::value<float>( &params.directionalSpreading.swell )
      ->default_value( params.directionalSpreading.swell ),
      "The mix between a wind-driven local sea and a swell caused by a "
      "distant storm." )

    ( "filter",
      po::value<int>( &filter )
      ->default_value( filter ),
      "Filter: 0 for nullptr, 1 for Smoothed Invertible Band-Pass" )

    ( "filterSoftWidth",
      po::value<float>( &params.filter.softWidth )
      ->default_value( params.filter.softWidth ),
      "Size in meters of the softness of wavelength filter falloff." )

    ( "filterSmall",
      po::value<float>( &params.filter.smallWavelength )
      ->default_value( params.filter.smallWavelength ),
      "Size in meters of the smallest kept wavelengths." )

    ( "filterBig",
      po::value<float>( &params.filter.bigWavelength )
      ->default_value( params.filter.bigWavelength ),
      "Size in meters of the biggest kept wavelengths." )

    ( "filterMin",
      po::value<float>( &params.filter.min )
      ->default_value( params.filter.min ),
      "Minimum value of filter." )

    ( "filterInvert", "Invert the filter" )

    ( "random",
      po::value<int>( &random )
      ->default_value( random ),
      "Random Distribution: 0 for Normal, 1 for Log-Normal" )

    ( "seed",
      po::value<int>( &params.random.seed )
      ->default_value( params.random.seed ),
      "Random Seed" )

    ;

  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  //-*************************************************************************
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm.count("filterInvert")) {
    params.filter.invert = true;
  }

  params.dispersion.type = (ewav::DispersionType)dispersion;
  params.spectrum.type = (ewav::SpectrumType)spectrum;
  params.directionalSpreading.type =
    (ewav::DirectionalSpreadingType)directionalSpreading;
  params.filter.type = (ewav::FilterType)filter;
  params.random.type = (ewav::RandomType)random;
  params.random.seed = seed;

  doTest(params);

  return 0;
}