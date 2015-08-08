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

#ifndef _EncinoWaves_SimpleSimViewer_Sim_h_
#define _EncinoWaves_SimpleSimViewer_Sim_h_

#include "Foundation.h"
#include "GLCamera.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
class BaseSim {
public:
  //! Virtual base class for simulation.
  //! ...
  BaseSim(void) {}

  //! Virtual destructor
  //! ...
  virtual ~BaseSim();

  //! Return a name
  //! ...
  virtual std::string getName() const { return "BaseSim"; }

  //! Preferred window size.
  //! ...
  virtual V2i preferredWindowSize() const { return V2i(800, 600); }

  //! Init draw
  //! ...
  virtual void init(int w, int h) {}

  //! Reshape
  //! ...
  virtual void reshape(int w, int h) {}

  //! Just step forward.
  //! ...
  virtual void step() {}

  //! frame
  //! ...
  virtual void frame() {}

  //! Dolly
  //! ...
  virtual void dolly(float dx, float dy) {}

  //! Track
  //! ...
  virtual void track(float dx, float dy) {}

  //! Rotate
  //! ...
  virtual void rotate(float dx, float dy) {}

  //! Output camera
  //! ...
  virtual void outputCamera() {}

  //! This draws, assuming a camera matrix has already been set.
  //! ...
  virtual void draw() {}

  //! This outer draw function sets up the drawing environment.
  //! ...
  virtual void outerDraw();

  //! This calls the character function
  virtual void character(unsigned int i_char, int x, int y) {}

  //! This calls the keyboard function.
  virtual void keyboard(int i_key, int i_scancode, int i_action, int i_mods,
                        int i_x, int i_y) {}

  virtual void mouse(int i_button, int i_action, int i_mods, double x, double y,
                     double lastX, double lastY) {}

  virtual void mouseDrag(double x, double y, double lastX, double lastY) {}
};

//-*****************************************************************************
class Sim3D : public BaseSim {
public:
  //! Virtual base class for simulation.
  //! ...
  Sim3D(void)
    : BaseSim() {}

  //! Virtual destructor
  //! ...
  virtual ~Sim3D();

  //! Return a name
  //! ...
  virtual std::string getName() const override { return "Sim3D"; }

  //! Init draw
  //! ...
  virtual void init(int w, int h) override;

  //! Reshape
  //! ...
  virtual void reshape(int w, int h) override;

  //! frame
  //! ...
  virtual void frame() override;

  //! Dolly
  //! ...
  virtual void dolly(float dx, float dy) override;

  //! Track
  //! ...
  virtual void track(float dx, float dy) override;

  //! Rotate
  //! ...
  virtual void rotate(float dx, float dy) override;

  //! Output camera
  //! ...
  virtual void outputCamera() override;

  //! Return the bounds at the current time.
  //! ...
  virtual Box3d getBounds() const { return Box3d(V3d(-0.1), V3d(0.1)); }

  //! Override clipping
  //! ...
  virtual bool overrideClipping(double& o_near, double& o_far) const {
    return false;
  }

  //! This outer draw function sets up the drawing environment.
  //! ...
  virtual void outerDraw() override;

  //! Return the camera
  const GLCamera& camera() const { return m_camera; }

protected:
  GLCamera m_camera;
};

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
