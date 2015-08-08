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

#include "Sim.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
BaseSim::~BaseSim() {}

//-*****************************************************************************
void BaseSim::outerDraw() {
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  UtilGL::CheckErrors("outerDraw glClearColor");

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  UtilGL::CheckErrors("outerDraw glClear");

  draw();
}

//-*****************************************************************************
Sim3D::~Sim3D() {}

//-*****************************************************************************
void Sim3D::init(int w, int h) {
  m_camera.setSize(w, h);
  m_camera.lookAt(V3d(24, 18, 24), V3d(0.0));
  m_camera.frame(getBounds());
}

//-*****************************************************************************
void Sim3D::reshape(int w, int h) { m_camera.setSize(w, h); }

//-*****************************************************************************
void Sim3D::frame() { m_camera.frame(getBounds()); }

//-*****************************************************************************
void Sim3D::dolly(float dx, float dy) { m_camera.dolly(V2d(dx, dy)); }

//-*****************************************************************************
void Sim3D::track(float dx, float dy) { m_camera.track(V2d(dx, dy)); }

//-*****************************************************************************
void Sim3D::rotate(float dx, float dy) { m_camera.rotate(V2d(dx, dy)); }

//-*****************************************************************************
void Sim3D::outputCamera() {
  std::cout << "# Camera\n" << m_camera.RIB() << std::endl;
}

//-*****************************************************************************
void Sim3D::outerDraw() {
#if OSX_GLFW_VIEWPORT_BUG
  glViewport(0, 0, 2 * (GLsizei)m_camera.width(),
             2 * (GLsizei)m_camera.height());
#else
  glViewport(0, 0, (GLsizei)m_camera.width(), (GLsizei)m_camera.height());
#endif

  double n, f;
  m_camera.autoSetClippingPlanes(getBounds());
  if (overrideClipping(n, f)) {
    m_camera.setClippingPlanes(n, f);
  }

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  UtilGL::CheckErrors("outerDraw glClearColor");

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  UtilGL::CheckErrors("outerDraw glClear");

  draw();
}

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves
