//-*****************************************************************************
// Copyright (c) 2001-2013, Christopher Jon Horvath. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of Christopher Jon Horvath nor the names of his
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//-*****************************************************************************

#include <SimpleSimViewer/All.h>
#include <Util/All.h>
#include <ImathMatrixAlgo.h>

using namespace EncinoWaves::SimpleSimViewer;
using namespace EncinoWaves::Util;

//-*****************************************************************************
class SimplePointsSim : public Sim3D {
protected:
  void setTime(double i_time) {
    m_time            = i_time;
    m_angleOfRotation = m_time * m_rotationRate;
    m_objectToWorld.setAxisAngle(m_axisOfRotation, m_angleOfRotation);
    m_worldSpaceBounds = Imath::transform(m_objectSpaceBounds, m_objectToWorld);
  }

  virtual void init(int w, int h) override {
    Sim3D::init(w, h);

    m_pointsDrawHelper.reset(new PointsDrawHelper(false, m_vtxPosData.size(),
                                                  m_vtxPosData.data(),
                                                  m_vtxNormData.data(),
                                                  // NULL,
                                                  NULL, NULL));

    Program::Bindings vtxBindings;
    vtxBindings.push_back(
      Program::Binding(m_pointsDrawHelper->posVboIdx(), "g_Pobj"));
    vtxBindings.push_back(
      Program::Binding(m_pointsDrawHelper->normVboIdx(), "g_Nobj"));

    Program::Bindings frgBindings;
    frgBindings.push_back(Program::Binding(0, "g_fragmentColor"));

    m_program.reset(new Program(
      "SimplePointsDraw", SimpleVertexShader(), SimplePointsGeometryShader(),
      KeyFillFragmentShader(),
      // ConstantRedFragmentShader(),
      vtxBindings, frgBindings, m_pointsDrawHelper->vertexArrayObject()));

    // Sun, moon.
    float sunAltitude = radians(45.0f);
    float sunAzimuth = radians(35.0f);
    V3f toSun(cosf(sunAltitude) * cosf(sunAzimuth),
              cosf(sunAltitude) * sinf(sunAzimuth), sinf(sunAltitude));
    V3f sunColor(1.0f, 1.0f, 1.0f);

    float moonAltitude = radians(65.0f);
    float moonAzimuth = sunAzimuth - radians(180.0f);
    V3f toMoon(cosf(moonAltitude) * cosf(moonAzimuth),
               cosf(moonAltitude) * sinf(moonAzimuth), sinf(moonAltitude));
    V3f moonColor(0.1f, 0.1f, 0.3f);

    SetKeyFillLights(*m_program, toSun, sunColor, toMoon, moonColor);

    // Materials.
    SetStdMaterial(*m_program, V3f(0.18f), V3f(0.1f), 25.0f);
  }

public:
  SimplePointsSim()
    : Sim3D()
    , m_axisOfRotation(0.0, 0.0, 1.0)
    , m_angleOfRotation(0.0)
    , m_rotationRate(0.75)
    , m_time(0.0) {
    V3d c000(-0.5, -0.5, -0.5);
    V3d c100(0.5, -0.5, -0.5);
    V3d c010(-0.5, 0.5, -0.5);
    V3d c110(0.5, 0.5, -0.5);
    V3d c001(-0.5, -0.5, 0.5);
    V3d c101(0.5, -0.5, 0.5);
    V3d c011(-0.5, 0.5, 0.5);
    V3d c111(0.5, 0.5, 0.5);

    m_vtxPosData.push_back(c000);
    m_vtxNormData.push_back(c000.normalized());

    m_vtxPosData.push_back(c100);
    m_vtxNormData.push_back(c100.normalized());

    m_vtxPosData.push_back(c010);
    m_vtxNormData.push_back(c010.normalized());

    m_vtxPosData.push_back(c110);
    m_vtxNormData.push_back(c110.normalized());

    m_vtxPosData.push_back(c001);
    m_vtxNormData.push_back(c001.normalized());

    m_vtxPosData.push_back(c101);
    m_vtxNormData.push_back(c101.normalized());

    m_vtxPosData.push_back(c011);
    m_vtxNormData.push_back(c011.normalized());

    m_vtxPosData.push_back(c111);
    m_vtxNormData.push_back(c111.normalized());

    m_objectSpaceBounds.min = c000;
    m_objectSpaceBounds.max = c111;

    setTime(m_time);
  }

  virtual std::string getName() const override { return "SimplePointsSim"; }

  virtual void step() override { setTime(m_time + (1.0 / 24.0)); }

  virtual Box3d getBounds() const override { return m_worldSpaceBounds; }

  //! This draws, assuming a camera matrix has already been set.
  //! ...
  virtual void draw() override {
    glEnable(GL_PROGRAM_POINT_SIZE);

    SetStdMatrices(*m_program, m_camera, m_objectToWorld);
    m_program->use();

    (*m_program)(Uniform("g_pointSize", (float)25.0,
                         (Uniform::Requirement)Uniform::kRequireOptional));
    m_program->setUniforms();

    m_pointsDrawHelper->draw(m_camera);
    m_program->unuse();
  }

protected:
  std::vector<V3f> m_vtxPosData;
  std::vector<V3f> m_vtxNormData;
  M44d m_objectToWorld;
  V3d m_axisOfRotation;
  double m_angleOfRotation;
  double m_rotationRate;
  double m_time;
  Box3d m_objectSpaceBounds;
  Box3d m_worldSpaceBounds;

  std::unique_ptr<PointsDrawHelper> m_pointsDrawHelper;
  std::unique_ptr<Program> m_program;
};

//-*****************************************************************************
int main(int argc, char* argv[]) {
  std::shared_ptr<BaseSim> sptr(new SimplePointsSim);
  SimpleViewSim(sptr, true);
  return 0;
}
