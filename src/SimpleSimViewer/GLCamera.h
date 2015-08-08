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

#ifndef _EncinoWaves_SimpleSimViewer_GLCamera_h_
#define _EncinoWaves_SimpleSimViewer_GLCamera_h_

#include "Foundation.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
class GLCamera {
public:
  GLCamera();
  ~GLCamera(){};

  // Executes OpenGL commands to show the current camera
  void apply() const;

  //-*************************************************************************
  // LOCAL TRANSFORM: Get, Set
  M44d modelViewMatrix() const;
  M44d projectionMatrix() const;

  const V3d &rotation() const { return m_rotation; }
  void setRotation(const V3d &r) { m_rotation = r; }

  const V3d &scale() const { return m_scale; }
  void setScale(const V3d &s) { m_scale = s; }

  const V3d &translation() const { return m_translation; }
  void setTranslation(const V3d &t) { m_translation = t; }

  double centerOfInterest() const { return m_centerOfInterest; }
  void setCenterOfInterest(double coi) {
    m_centerOfInterest = std::max(coi, 0.1);
  }

  double fovy() const { return m_fovy; }
  void setFovy(double fvy) { m_fovy = fvy; }

  int width() const { return m_size.x; }
  int height() const { return m_size.y; }

  const V2d &clippingPlanes() const { return m_clip; }
  void autoSetClippingPlanes(const Box3d &bounds);
  void setClippingPlanes(double near, double far) {
    m_clip.x = near;
    m_clip.y = far;
  }

  //-*************************************************************************
  // UI Actions
  void track(const V2d &point);
  void dolly(const V2d &point, double dollySpeed = 5.0);
  void rotate(const V2d &point, double rotateSpeed = 400.0);

  void frame(const Box3d &bounds);

  void lookAt(const V3d &eye, const V3d &at);

  void setSize(int w, int h) {
    m_size.x = w;
    m_size.y = h;
  }
  void setSize(const V2i &sze) { m_size = sze; }

  //-*************************************************************************
  // RIB STUFF
  std::string RIB() const;

  //-*************************************************************************
  // RAY STUFF - convert a screen space point, in raster space, to a
  // ray from the eye.
  Imath::Line3d getRayThroughRasterPoint(const V2d &pt_raster) const;

protected:
  //-*************************************************************************
  // DATA
  V3d m_rotation;
  V3d m_scale;
  V3d m_translation;

  double m_centerOfInterest;
  double m_fovy;
  V2d m_clip;
  V2i m_size;
  double m_aspect;
};

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
