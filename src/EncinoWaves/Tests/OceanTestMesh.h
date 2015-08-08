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

#ifndef _EncinoWaves_OceanTestMesh_h_
#define _EncinoWaves_OceanTestMesh_h_

#include "OceanTestFoundation.h"
#include "OceanTestSky.h"
#include "OceanTestTextureSky.h"
#include "OceanTestEnvSphere.h"

namespace OceanTest {

//-*****************************************************************************
class Mesh {
public:
  struct DrawParameters {
    int repeat = 2;
    float wind_rotation = 45.0f;
    DrawParameters() = default;
  };

  Mesh(const ewav::Parametersf& i_wparams, const Sky::Parameters& i_sparams,
       const DrawParameters& i_dparams);
  ~Mesh();

  void step();

  void draw(const EncinoWaves::SimpleSimViewer::GLCamera& i_camera);

  Box3d getBounds() const {
    return Box3d(V3d(-0.125 * m_params.domain, -0.125 * m_params.domain,
                     -0.125 * m_params.domain),
                 V3d(0.125 * m_params.domain));
  }

  const ewav::Parametersf& wavesParams() const { return m_params; }
  void setWavesParams(const ewav::Parametersf& i_params);
  void setDrawParams(const DrawParameters& dparams) { m_drawParams = dparams; }

protected:
  void createProgram();
  void setCameraUniforms(
    const EncinoWaves::SimpleSimViewer::GLCamera& i_camera);
  void setWavesUniforms();
  void domainChange();
  void propagateAtFrame();

  // Waves parameters - sky params will be stored in sky
  ewav::Parametersf m_params;
  DrawParameters m_drawParams;

  // Waves system itself.
  std::unique_ptr<ewav::InitialStatef> m_wavesInitialState;
  std::unique_ptr<ewav::PropagatedStatef> m_wavesPropagatedState;
  std::unique_ptr<ewav::Propagationf> m_wavesPropagation;
  std::unique_ptr<ewav::Statsf> m_wavesStats;
  int N;

  // Non-changing arrays of the wave system.
  std::vector<V2f> m_vertsXY;
  std::vector<GLuint> m_indices;
  std::vector<V3f> m_normals;

  // The time
  int m_frame;

  // xy, h, dx, dy, minE, normals, quadIndices.
  enum VertexBuffer {
    XY_BUFFER,
    H_BUFFER,
    DX_BUFFER,
    DY_BUFFER,
    MINE_BUFFER,
    NORMALS_BUFFER,
    INDICES_BUFFER,

    NUM_VERTEX_BUFFERS
  };

  // The VAO and VBOs
  GLuint m_vertexArrayObject;
  GLuint m_vertexBuffers[NUM_VERTEX_BUFFERS];

  // The Program.
  std::unique_ptr<GeepGLFW::Program> m_program;

  // The Sky
  std::unique_ptr<Sky> m_sky;
  std::unique_ptr<TextureSky> m_texture_sky;

  // Env Sphere
  std::unique_ptr<EnvSphere> m_env_sphere;
};

}  // namespace OceanTest

#endif
