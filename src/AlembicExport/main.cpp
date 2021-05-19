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

#include "EncinoWaves/All.h"

#include <Alembic/Abc/TypedArraySample.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <OpenEXR/ImathVec.h>
#include <fmt/format.h>
#include <boost/program_options.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace po = boost::program_options;
using EncinoWaves::wrap;
using Imath::V3f;

int main(int argc, char* argv[]) {
    EncinoWaves::Parametersf params;

    int threads = -1;
    int dispersion = 2;
    int spectrum = 2;
    int directionalSpreading = 3;
    int filter = 0;
    int random = 0;
    int num_frames = 100;
    std::string out_file_base;

    po::options_description desc("Encino Waves Alembic Export 2021");
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

    ( "troughDamping",
     po::value<float>( &params.troughDamping )
     ->default_value( params.troughDamping ),
     "Trough damping." )

    ( "troughDampingSmallWavelength",
      po::value<float>( &params.troughDampingSmallWavelength )
      ->default_value( params.troughDampingSmallWavelength ),
      "Trough damping small wavelength." )

    ( "troughDampingBigWavelength",
      po::value<float>( &params.troughDampingBigWavelength )
      ->default_value( params.troughDampingBigWavelength ),
      "Trough damping big wavelength." )

    ( "troughDampingSoftWidth",
      po::value<float>( &params.troughDampingSoftWidth )
      ->default_value( params.troughDampingSoftWidth ),
      "Trough damping soft width." )

    ( "random",
      po::value<int>( &random )
      ->default_value( random ),
      "Random Distribution: 0 for Normal, 1 for Log-Normal" )

    ( "seed",
      po::value<int>( &params.random.seed )
      ->default_value( params.random.seed ),
      "Random Seed" )

    ( "num_frames",
      po::value<int>( &num_frames )
      ->default_value( num_frames ),
      "Num Frames" )

    ( "out_file_base",
       po::value<std::string>(),
       "The output file name base" )
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

    if (vm.count("filterInvert")) { params.filter.invert = true; }

    if (vm.count("out_file_base")) {
        out_file_base = vm["out_file_base"].as<std::string>();
    }

    params.dispersion.type =
      static_cast<EncinoWaves::DispersionType>(dispersion);
    params.spectrum.type = static_cast<EncinoWaves::SpectrumType>(spectrum);
    params.directionalSpreading.type =
      static_cast<EncinoWaves::DirectionalSpreadingType>(directionalSpreading);
    params.filter.type = static_cast<EncinoWaves::FilterType>(filter);
    params.random.type = static_cast<EncinoWaves::RandomType>(random);

    // Create initial state...
    auto wavesInitialState =
      std::make_unique<EncinoWaves::InitialStatef>(params);
    auto const N = wavesInitialState->HSpectralPos.height();
    std::cout << "Created Initial State. " << std::endl
              << "Resolution: " << N << " x " << N << std::endl;

    // Create propagated state.
    auto wavesPropagatedState =
      std::make_unique<EncinoWaves::PropagatedStatef>(params);
    std::cout << "Created Propagated State. " << std::endl;

    // Create propagation
    auto wavesPropagation = std::make_unique<EncinoWaves::Propagationf>(params);
    std::cout << "Created Propagation." << std::endl;

    std::vector<V3f> verts;
    std::vector<int32_t> indices;
    std::vector<int32_t> counts;
    std::vector<V3f> normals;

    normals.resize((N + 1) * (N + 1));
    verts.resize((N + 1) * (N + 1));
    int numTris = 2 * (N * N);
    indices.resize(3 * numTris);

    V3f originXYZ(-0.5f * params.domain, -0.5f * params.domain, 0.0f);
    V3f sizeXYZ(params.domain, params.domain, 0.0f);

    for (int frame = 1; frame <= num_frames; ++frame) {
        counts.clear();
        auto const wave_time = 24.0 * static_cast<double>(frame);

        // Init propagated state.
        wavesPropagation->propagate(
          params, *wavesInitialState, *wavesPropagatedState, wave_time);
        std::cout << "Propagated to frame" << frame << std::endl;

        //   vec4 vtx = vec4(
        //            ( g_vertex.x - g_pinch * g_dx ),
        //            ( g_vertex.y - g_pinch * g_dy ),
        //            ( g_amplitude * g_h ),
        //            1 );

        // Compute normals.
        EncinoWaves::ComputeNormals<float>(
          params, *wavesPropagatedState, normals.data());
        std::cout << "Computed normals." << std::endl;

        // Gather stats.
        auto wavesStats = std::make_unique<EncinoWaves::Statsf>(
          wavesPropagatedState->Height, wavesPropagatedState->MinE);
        std::cout << "Gathered stats." << std::endl;

        //-*************************************************************************
        // STATIC ARRAY COMPUTATION
        //-*************************************************************************
        // Make an XY array for vertices to match sim data.

        // Fill with the correct size.
        auto const* heights = wavesPropagatedState->Height.cdata();
        auto const* dxs = wavesPropagatedState->Dx.cdata();
        auto const* dys = wavesPropagatedState->Dy.cdata();
        std::size_t index = 0;
        for (int j = 0; j < N + 1; ++j) {
            float fj = float(j) / float(N);

            for (int i = 0; i < N + 1; ++i, ++index) {
                float fi = float(i) / float(N);

                verts[index] = originXYZ + (sizeXYZ * V3f(fi, fj, 0.0f)) +
                               V3f{-params.pinch * dxs[index],
                                   -params.pinch * dys[index],
                                   params.amplitudeGain * heights[index]};
            }
        }

        // Do indices. (TRIANGLES NOW)
        index = 0;
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i, index += 6) {
                counts.push_back(3);
                counts.push_back(3);

                // First triangle.
                indices[index] = wrap(i + 1, N + 1) + (N + 1) * wrap(j, N + 1);
                indices[index + 1] = wrap(i, N + 1) + (N + 1) * wrap(j, N + 1);
                indices[index + 2] =
                  wrap(i + 1, N + 1) + (N + 1) * wrap(j + 1, N + 1);

                // Second triangle.
                indices[index + 3] = wrap(i, N + 1) + (N + 1) * wrap(j, N + 1);
                indices[index + 4] =
                  wrap(i, N + 1) + (N + 1) * wrap(j + 1, N + 1);
                indices[index + 5] =
                  wrap(i + 1, N + 1) + (N + 1) * wrap(j + 1, N + 1);
            }
        }

        // Namespace scope
        if (out_file_base != "") {
            auto const filename =
              fmt::format("{}.{:04}.abc", out_file_base, frame);

            using namespace Alembic::AbcGeom;

            P3fArraySample points_array_sample{
              reinterpret_cast<Alembic::Abc::V3f const*>(verts.data()),
              verts.size()};

            Int32ArraySample indices_array_sample{
              reinterpret_cast<int32_t const*>(indices.data()), indices.size()};

            Int32ArraySample counts_array_sample{
              reinterpret_cast<int32_t const*>(counts.data()), counts.size()};

            ON3fGeomParam::Sample normals_geom_sample{
              N3fArraySample{
                reinterpret_cast<Alembic::Abc::V3f const*>(normals.data()),
                normals.size()},
              kVertexScope};

            OV2fGeomParam::Sample uvs_geom_sample{};

            OPolyMeshSchema::Sample mesh_sample{points_array_sample,
                                                indices_array_sample,
                                                counts_array_sample,
                                                uvs_geom_sample,
                                                normals_geom_sample};

            OArchive archive{Alembic::AbcCoreOgawa::WriteArchive(),
                             filename.c_str()};

            OObject top_object{archive, kTop};

            OPolyMesh mesh_object{top_object, "fluid_mesh"};
            auto& mesh = mesh_object.getSchema();

            mesh.set(mesh_sample);
        }

        std::cout << "Done frame: " << frame << std::endl;
    }

    return 0;
}
