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

#ifndef _EncinoWaves_SimpleSimViewer_ViewerGLFW_h_
#define _EncinoWaves_SimpleSimViewer_ViewerGLFW_h_

#include "Foundation.h"
#include "Sim.h"
#include "GLCamera.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
class Viewer;

//-*****************************************************************************
class GLFWGlobal {
public:
  GLFWGlobal();
  ~GLFWGlobal();

  void registerWindowAndViewer(GLFWwindow* i_window, Viewer* i_viewer);

  void unregisterWindowAndAllViewers(GLFWwindow* i_window);

  Viewer* viewerFromWindow(GLFWwindow* i_window, bool i_throwIfNotFound = true);

protected:
  typedef std::map<GLFWwindow*, Viewer*> WindowViewerMap;
  WindowViewerMap m_wvMap;
};

//-*****************************************************************************
class Viewer {
public:
  static const int BMASK_LEFT   = 0x1 << 0;  // binary 001
  static const int BMASK_MIDDLE = 0x1 << 1;  // binary 010
  static const int BMASK_RIGHT  = 0x1 << 2;  // binary 100

  explicit Viewer(std::shared_ptr<BaseSim> i_sim, bool i_anim = false);
  ~Viewer();

  void init();
  void tick(bool i_force);
  void display();
  void reshape(int i_width, int i_height);
  void keyboard(int i_key, int i_scancode, int i_action, int i_mods);
  void character(unsigned int i_char);
  void mouse(int i_button, int i_action, int i_mods);
  void mouseDrag(double x, double y);
  GLFWwindow* window() { return m_window; }

protected:
  std::shared_ptr<BaseSim> m_sim;

  GLFWwindow* m_window;

  int m_buttonMask;
  double m_mouseX;
  double m_mouseY;
  double m_lastX;
  double m_lastY;
  int m_keyMods;

  bool m_animating;

  // GLCamera m_camera;
  Timer m_playbackTimer;
};

//-*****************************************************************************
void SimpleViewSim(std::shared_ptr<BaseSim> i_sim, bool i_playing = false);

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
