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

#include "OceanTestMesh.h"
#include "OceanTestShaders.h"

namespace OceanTest {

//-*****************************************************************************
using namespace GeepGLFW;

//-*****************************************************************************
Mesh::Mesh(const ewav::Parametersf& i_wparams, const Sky::Parameters& i_sparams,
           const DrawParameters& i_dparams)
  : m_params(i_wparams)
  , m_drawParams(i_dparams)
  , m_frame(1)
  , m_vertexArrayObject(0) {
  //-*************************************************************************
  // WAVES INITIALIZATION
  //-*************************************************************************

  // Create initial state...
  m_wavesInitialState.reset(new ewav::InitialStatef(m_params));
  N = m_wavesInitialState->HSpectralPos.height();
  std::cout << "Created Initial State. " << std::endl
            << "Resolution: " << N << " x " << N << std::endl;

  // Create propagated state.
  m_wavesPropagatedState.reset(new ewav::PropagatedStatef(m_params));
  std::cout << "Created Propagated State. " << std::endl;

  // Create propagation
  m_wavesPropagation.reset(new ewav::Propagationf(m_params));
  std::cout << "Created Propagation." << std::endl;

  // Init propagated state.
  float time = float(m_frame) / 24.0;
  m_wavesPropagation->propagate(m_params, *m_wavesInitialState,
                                *m_wavesPropagatedState, time);
  std::cout << "Propagated to start frame" << std::endl;

  // Compute normals.
  m_normals.resize((N + 1) * (N + 1));
  ewav::ComputeNormals<float>(m_params, *m_wavesPropagatedState,
                              &(m_normals.front()));
  std::cout << "Computed normals." << std::endl;

  // Gather stats.
  m_wavesStats.reset(new ewav::Statsf(m_wavesPropagatedState->Height,
                                      m_wavesPropagatedState->MinE));
  std::cout << "Gathered stats." << std::endl;

  //-*************************************************************************
  // STATIC ARRAY COMPUTATION
  //-*************************************************************************
  // Make an XY array for vertices to match sim data.
  m_vertsXY.resize((N + 1) * (N + 1));
  int numTris = 2 * (N * N);
  m_indices.resize(3 * numTris);

  V2f originXY(-0.5f * m_params.domain, -0.5f * m_params.domain);
  V2f sizeXY(m_params.domain, m_params.domain);

  // Fill with the correct size.
  std::size_t index = 0;
  for (int j = 0; j < N + 1; ++j) {
    float fj = float(j) / float(N);

    for (int i = 0; i < N + 1; ++i, ++index) {
      float fi = float(i) / float(N);

      m_vertsXY[index] = originXY + (sizeXY * V2f(fi, fj));
    }
  }

  // Do indices. (TRIANGLES NOW)
  index = 0;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i, index += 6) {
      // First triangle.
      m_indices[index] =
        wrap(i + 1, N + 1) + (N + 1) * wrap(j, N + 1);
      m_indices[index + 1] =
        wrap(i, N + 1) + (N + 1) * wrap(j, N + 1);
      m_indices[index + 2] =
        wrap(i + 1, N + 1) + (N + 1) * wrap(j + 1, N + 1);

      // Second triangle.
      m_indices[index + 3] =
        wrap(i, N + 1) + (N + 1) * wrap(j, N + 1);
      m_indices[index + 4] =
        wrap(i, N + 1) + (N + 1) * wrap(j + 1, N + 1);
      m_indices[index + 5] =
        wrap(i + 1, N + 1) + (N + 1) * wrap(j + 1, N + 1);
    }
  }

  //-*************************************************************************
  // SKY INIT
  //-*************************************************************************
  if (i_sparams.filename != "") {
    m_texture_sky.reset(new TextureSky(i_sparams.filename));
    m_env_sphere.reset(new EnvSphere{});
  } else {
    m_sky.reset(new Sky(i_sparams));
  }

  //-*************************************************************************
  // OPENGL INIT
  //-*************************************************************************
  UtilGL::CheckErrors("mesh init before anything");

  // Create and bind VAO
  glGenVertexArrays(1, &m_vertexArrayObject);
  UtilGL::CheckErrors("glGenVertexArrays");
  EWAV_ASSERT(m_vertexArrayObject > 0, "Failed to create VAO");

  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray");

  // CJH HACK
  // DO I CREATE THE PROGRAM HERE?
  // createProgram();

  // Create vertex buffers.
  glGenBuffers(NUM_VERTEX_BUFFERS, m_vertexBuffers);
  UtilGL::CheckErrors("glGenBuffers");
  EWAV_ASSERT(m_vertexBuffers[0] > 0, "Failed to create VBOs");

  // Data size.
  std::size_t dataSize = m_vertsXY.size();

  // xy buffer
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[XY_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer XY");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V2f) * dataSize, &(m_vertsXY.front()),
               GL_STATIC_DRAW);
  UtilGL::CheckErrors("glBufferData XY");
  glVertexAttribPointer(XY_BUFFER, 2, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer XY");
  glEnableVertexAttribArray(XY_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray XY");

  // h buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[H_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer H");
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize,
               m_wavesPropagatedState->Height.cdata(), GL_DYNAMIC_DRAW);
  UtilGL::CheckErrors("glBufferData H");
  glEnableVertexAttribArray(H_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray H");
  glVertexAttribPointer(H_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer H");

  // dx buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[DX_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer DX");
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize,
               m_wavesPropagatedState->Dx.cdata(), GL_DYNAMIC_DRAW);
  UtilGL::CheckErrors("glBufferData DX");
  glEnableVertexAttribArray(DX_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray DX");
  glVertexAttribPointer(DX_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer DX");

  // dy buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[DY_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer DY");
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize,
               m_wavesPropagatedState->Dy.cdata(), GL_DYNAMIC_DRAW);
  UtilGL::CheckErrors("glBufferData DY");
  glEnableVertexAttribArray(DY_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray DY");
  glVertexAttribPointer(DY_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer DY");

  // MinE buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[MINE_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer MINE");
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * dataSize,
               m_wavesPropagatedState->MinE.cdata(), GL_DYNAMIC_DRAW);
  UtilGL::CheckErrors("glBufferData MINE");
  glEnableVertexAttribArray(MINE_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray MINE");
  glVertexAttribPointer(MINE_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer MINE");

  // Normals buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[NORMALS_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer NORMALS");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V3f) * dataSize, &(m_normals.front()),
               GL_DYNAMIC_DRAW);
  UtilGL::CheckErrors("glBufferData NORMALS");
  glEnableVertexAttribArray(NORMALS_BUFFER);
  UtilGL::CheckErrors("glEnableVertexAttribArray NORMALS");
  glVertexAttribPointer(NORMALS_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
  UtilGL::CheckErrors("glVertexAttribPointer NORMALS");

  // Indices buffer.
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexBuffers[INDICES_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer INDICES");
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_indices.size(),
               &(m_indices.front()), GL_STATIC_DRAW);
  UtilGL::CheckErrors("glBufferData INDICES");

  // CJH HACK
  // or do I create the program HERE?
  createProgram();

  // Unbind vao.
  glBindVertexArray(0);
  UtilGL::CheckErrors("Unbind VAO");
}

//-*****************************************************************************
Mesh::~Mesh() {
  if (m_vertexArrayObject > 0) {
    glDeleteVertexArrays(1, &m_vertexArrayObject);
    m_vertexArrayObject = 0;
  }

  if (m_vertexBuffers[0] > 0) {
    glDeleteBuffers(NUM_VERTEX_BUFFERS, m_vertexBuffers);
    for (int i = 0; i < NUM_VERTEX_BUFFERS; ++i) {
      m_vertexBuffers[i] = 0;
    }
  }
}

//-*****************************************************************************
void Mesh::step() {
  ++m_frame;
  propagateAtFrame();
}

//-*****************************************************************************
// Separating this out so that we can reorder when it happens, since I'm not
// sure which is correct.
void Mesh::createProgram() {
  // Make the bindings
  Program::Bindings vtxBindings;
  vtxBindings.push_back(Program::Binding(XY_BUFFER, "g_vertex"));
  vtxBindings.push_back(Program::Binding(H_BUFFER, "g_h"));
  vtxBindings.push_back(Program::Binding(DX_BUFFER, "g_dx"));
  vtxBindings.push_back(Program::Binding(DY_BUFFER, "g_dy"));
  vtxBindings.push_back(Program::Binding(MINE_BUFFER, "g_minE"));
  vtxBindings.push_back(Program::Binding(NORMALS_BUFFER, "g_normal"));

  Program::Bindings frgBindings;
  frgBindings.push_back(Program::Binding(0, "g_fragmentColor"));

  m_program.reset(
    new Program("OceanTestDraw", VertexShader(), GeometryShader(),
                FragmentShader("gs", static_cast<bool>(m_texture_sky)),
                vtxBindings, frgBindings, m_vertexArrayObject));
}

//-*****************************************************************************
void Mesh::setCameraUniforms(
  const EncinoWaves::SimpleSimViewer::GLCamera& i_cam) {
  M44d pm  = i_cam.projectionMatrix();
  M44d mvm = i_cam.modelViewMatrix();
  (*m_program)(Uniform("projection_matrix", pm));
  (*m_program)(Uniform("modelview_matrix", mvm));

  // Eye location.
  const V3d& eye = i_cam.translation();
  (*m_program)(Uniform("g_eyeWld", eye));
}

//-*****************************************************************************
void Mesh::setWavesUniforms() {
  float gainMinE      = 1.0f / (2.0f * m_wavesStats->StdDevMinE);
  float biasMinE      = -m_wavesStats->MeanMinE / (2.0f * m_wavesStats->StdDevMinE);
  float bigWaveHeight = std::max(std::abs(m_wavesStats->MinHeight),
                                 std::abs(m_wavesStats->MaxHeight));
  float minClipE = 0.5;  // Crap.
  float maxClipE = 1.1;

  (*m_program)(Uniform("g_pinch", m_params.pinch));
  (*m_program)(Uniform("g_amplitude", m_params.amplitudeGain));
  (*m_program)(Uniform("g_gainMinE", gainMinE));
  (*m_program)(Uniform("g_biasMinE", biasMinE));
  (*m_program)(Uniform("g_BigHeight", 1.5f * bigWaveHeight));
  (*m_program)(Uniform("g_domain", m_params.domain));
  (*m_program)(Uniform("g_minClipE", minClipE));
  (*m_program)(Uniform("g_maxClipE", maxClipE));

  (*m_program)(Uniform("g_repeat", m_drawParams.repeat));
}

//-*****************************************************************************
void Mesh::setWavesParams(const ewav::Parametersf& i_wparams) {
  // We don't take ALL the parameters, since some we don't allow to change.
  bool reInit   = false;
  bool domainCh = false;

#define CHECK_INIT_CHANGE(PARAM)           \
  if (m_params.PARAM != i_wparams.PARAM) { \
    reInit         = true;                 \
    m_params.PARAM = i_wparams.PARAM;      \
  }

  if (m_params.domain != i_wparams.domain) {
    reInit          = true;
    domainCh        = true;
    m_params.domain = i_wparams.domain;
  }

  CHECK_INIT_CHANGE(depth);
  CHECK_INIT_CHANGE(windSpeed);
  CHECK_INIT_CHANGE(fetch);
  CHECK_INIT_CHANGE(dispersion.type);
  CHECK_INIT_CHANGE(spectrum.type);
  CHECK_INIT_CHANGE(directionalSpreading.type);
  CHECK_INIT_CHANGE(directionalSpreading.swell);
  CHECK_INIT_CHANGE(filter.type);
  CHECK_INIT_CHANGE(filter.smallWavelength);
  CHECK_INIT_CHANGE(filter.bigWavelength);
  CHECK_INIT_CHANGE(filter.min);
  CHECK_INIT_CHANGE(filter.invert);
  CHECK_INIT_CHANGE(random.type);
  CHECK_INIT_CHANGE(random.seed);

#undef CHECK_INIT_CHANGE

  bool reProp = false;
#define CHECK_PROP_CHANGE(PARAM)           \
  if (m_params.PARAM != i_wparams.PARAM) { \
    reProp         = true;                 \
    m_params.PARAM = i_wparams.PARAM;      \
  }

  m_params.amplitudeGain = i_wparams.amplitudeGain;
  m_params.pinch         = i_wparams.pinch;

  if (reInit) {
    if (domainCh) {
      domainChange();
    }

    // Create initial state...
    m_wavesInitialState.reset(new ewav::InitialStatef(m_params));
    std::cout << "Reset Initial State. " << std::endl;

    propagateAtFrame();

    // Gather stats.
    m_wavesStats.reset(new ewav::Statsf(m_wavesPropagatedState->Height,
                                        m_wavesPropagatedState->MinE));
    // std::cout << "Gathered stats." << std::endl;
  } else {
    CHECK_PROP_CHANGE(troughDamping);
    CHECK_PROP_CHANGE(troughDampingSmallWavelength);
    CHECK_PROP_CHANGE(troughDampingBigWavelength);
    CHECK_PROP_CHANGE(troughDampingSoftWidth);

    if (reProp) {
      propagateAtFrame();
      // Gather stats.
      m_wavesStats.reset(new ewav::Statsf(m_wavesPropagatedState->Height,
                                          m_wavesPropagatedState->MinE));
    }
  }

#undef CHECK_PROP_CHANGE
}

//-*****************************************************************************
void Mesh::domainChange() {
  V2f originXY(-0.5f * m_params.domain, -0.5f * m_params.domain);
  V2f sizeXY(m_params.domain, m_params.domain);

  // Fill with the correct size.
  std::size_t index = 0;
  for (int j = 0; j < N + 1; ++j) {
    float fj = float(j) / float(N);

    for (int i = 0; i < N + 1; ++i, ++index) {
      float fi = float(i) / float(N);

      m_vertsXY[index] = originXY + (sizeXY * V2f(fi, fj));
    }
  }

  // Activate VAO
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray step");

  std::size_t dataSize = m_vertsXY.size();

  // xy buffer
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[XY_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer XY");
  glBufferData(GL_ARRAY_BUFFER, sizeof(V2f) * dataSize, &(m_vertsXY.front()),
               GL_STATIC_DRAW);
  UtilGL::CheckErrors("glBufferData XY");

  // Unbind
  glBindVertexArray(0);
}

//-*****************************************************************************
void Mesh::propagateAtFrame() {
  float time = float(m_frame) / 24.0;
  m_wavesPropagation->propagate(m_params, *m_wavesInitialState,
                                *m_wavesPropagatedState, time);

  ewav::ComputeNormals<float>(m_params, *m_wavesPropagatedState,
                              &(m_normals.front()));

  // Activate VAO
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray step");

  std::size_t dataSize = m_vertsXY.size();

  // h buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[H_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer H");
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * dataSize,
                  m_wavesPropagatedState->Height.cdata());
  UtilGL::CheckErrors("glBufferData H");

  // dx buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[DX_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer DX");
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * dataSize,
                  m_wavesPropagatedState->Dx.cdata());
  UtilGL::CheckErrors("glBufferData DX");

  // dy buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[DY_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer DY");
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * dataSize,
                  m_wavesPropagatedState->Dy.cdata());
  UtilGL::CheckErrors("glBufferData DY");

  // MinE buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[MINE_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer MINE");
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * dataSize,
                  m_wavesPropagatedState->MinE.cdata());
  UtilGL::CheckErrors("glBufferData MINE");

  // Normals buffer.
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffers[NORMALS_BUFFER]);
  UtilGL::CheckErrors("glBindBuffer NORMALS");
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(V3f) * dataSize,
                  &(m_normals.front()));
  UtilGL::CheckErrors("glBufferData NORMALS");

  // Unbind
  glBindVertexArray(0);
}

//-*****************************************************************************
void Mesh::draw(const EncinoWaves::SimpleSimViewer::GLCamera& i_cam) {
  glDisable(GL_DEPTH_TEST);
  if (m_env_sphere) {
    m_env_sphere->draw(i_cam, *m_texture_sky);
  }
  glEnable(GL_DEPTH_TEST);

  // Bind the vertex array
  glBindVertexArray(m_vertexArrayObject);
  UtilGL::CheckErrors("glBindVertexArray draw");

  // Use the program
  m_program->use();

  // Set the uniforms
  if (m_texture_sky) {
    m_texture_sky->bind(m_program->id());
  } else {
    m_sky->setUniforms(*m_program);
  }
  setCameraUniforms(i_cam);
  setWavesUniforms();
  m_program->setUniforms();

  // Draw the elements
  glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
  UtilGL::CheckErrors("glDrawElements");

  // Unuse the program
  m_program->unuse();

  // Unbind the vertex array
  glBindVertexArray(0);
  UtilGL::CheckErrors("glBindVertexArray 0 draw");
}

}  // namespace OceanTest
