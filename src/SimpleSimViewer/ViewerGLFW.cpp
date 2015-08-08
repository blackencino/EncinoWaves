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

#include "ViewerGLFW.h"

//-*****************************************************************************
namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
//-*****************************************************************************
// GLFW GLOBAL THING
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
static void error_callback(int error, const char* description) {
  std::cerr << description << std::endl;
}

//-*****************************************************************************
GLFWGlobal::GLFWGlobal() {
  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    EWAV_THROW("glfwInit returned false");
  }
}

//-*****************************************************************************
GLFWGlobal::~GLFWGlobal() { glfwTerminate(); }

//-*****************************************************************************
void GLFWGlobal::registerWindowAndViewer(GLFWwindow* i_window,
                                         Viewer* i_viewer) {
  EWAV_ASSERT(i_window != NULL && i_viewer != NULL,
              "Cannot register null window or viewer.");

  // Check to make sure it's not already registered.
  WindowViewerMap::iterator fiter = m_wvMap.find(i_window);
  if (fiter != m_wvMap.end()) {
    EWAV_THROW("Can't register GLFWwindow twice.");
  }

  // Register!
  m_wvMap[i_window] = i_viewer;
}

//-*****************************************************************************
void GLFWGlobal::unregisterWindowAndAllViewers(GLFWwindow* i_window) {
  // Map can only have one viewer per window anyway.
  m_wvMap.erase(i_window);
}

//-*****************************************************************************
Viewer* GLFWGlobal::viewerFromWindow(GLFWwindow* i_window,
                                     bool i_throwIfNotFound) {
  EWAV_ASSERT(i_window != NULL, "Cannot get viewer from null window.");

  WindowViewerMap::iterator fiter = m_wvMap.find(i_window);
  if (fiter == m_wvMap.end()) {
    if (i_throwIfNotFound) {
      EWAV_THROW("Could not find viewer for window.");
    } else {
      return NULL;
    }
  }

  if ((*fiter).second->window() != i_window) {
    EWAV_THROW("Corrupt window-viewer map in GLFWGlobal");
  }

  return (*fiter).second;
}

//-*****************************************************************************
// STATIC GLOBAL SINGLETON, AS A UNIQUE PTR.
//-*****************************************************************************
std::unique_ptr<GLFWGlobal> g_glfwGlobal;

//-*****************************************************************************
//-*****************************************************************************
// GLOBAL GLFW FUNCTIONS
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
void g_reshape(GLFWwindow* i_window, int i_width, int i_height) {
  EWAV_ASSERT(i_window != NULL, "NULL Viewer");
  EWAV_ASSERT(g_glfwGlobal.get() != NULL, "NO GLFW GLOBAL");

  // This will throw if viewer is NULL.
  Viewer* viewer = g_glfwGlobal->viewerFromWindow(i_window, true);

  viewer->reshape(i_width, i_height);
}

//-*****************************************************************************
void g_character(GLFWwindow* i_window, unsigned int i_char) {
  EWAV_ASSERT(i_window != NULL, "NULL Viewer");
  EWAV_ASSERT(g_glfwGlobal.get() != NULL, "NO GLFW GLOBAL");

  // This will throw if viewer is NULL.
  Viewer* viewer = g_glfwGlobal->viewerFromWindow(i_window, true);

  viewer->character(i_char);
}

//-*****************************************************************************
void g_keyboard(GLFWwindow* i_window, int i_key, int i_scancode, int i_action,
                int i_mods) {
  EWAV_ASSERT(i_window != NULL, "NULL Viewer");
  EWAV_ASSERT(g_glfwGlobal.get() != NULL, "NO GLFW GLOBAL");

  // This will throw if viewer is NULL.
  Viewer* viewer = g_glfwGlobal->viewerFromWindow(i_window, true);

  viewer->keyboard(i_key, i_scancode, i_action, i_mods);
}

//-*****************************************************************************
void g_mouse(GLFWwindow* i_window, int i_button, int i_action, int i_mods) {
  EWAV_ASSERT(i_window != NULL, "NULL Viewer");
  EWAV_ASSERT(g_glfwGlobal.get() != NULL, "NO GLFW GLOBAL");

  // This will throw if viewer is NULL.
  Viewer* viewer = g_glfwGlobal->viewerFromWindow(i_window, true);

  viewer->mouse(i_button, i_action, i_mods);
}

//-*****************************************************************************
void g_mouseDrag(GLFWwindow* i_window, double i_fx, double i_fy) {
  EWAV_ASSERT(i_window != NULL, "NULL Viewer");
  EWAV_ASSERT(g_glfwGlobal.get() != NULL, "NO GLFW GLOBAL");

  // This will throw if viewer is NULL.
  Viewer* viewer = g_glfwGlobal->viewerFromWindow(i_window, true);

  viewer->mouseDrag(i_fx, i_fy);
}

//-*****************************************************************************
//-*****************************************************************************
// VIEWER
//-*****************************************************************************
//-*****************************************************************************

//-*****************************************************************************
Viewer::Viewer(std::shared_ptr<BaseSim> i_simPtr, bool i_anim)
  : m_sim(i_simPtr)
  , m_window(NULL)
  , m_buttonMask(0)
  , m_mouseX(0)
  , m_mouseY(0)
  , m_lastX(0)
  , m_lastY(0)
  , m_keyMods(0)
  , m_animating(i_anim)
  //, m_camera()
  , m_playbackTimer() {
  // Initialize the global, if needed. This is so that the global doesn't
  // get created unless a viewer is requested.
  if (!g_glfwGlobal) {
    g_glfwGlobal.reset(new GLFWGlobal);
  }

  // Make it all modern-like.
  // glfwWindowHint( GLFW_CLIENT_API, GLFW_OPENGL_API );
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#ifdef DEBUG
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Make a window for this sim!
  V2i preferredWindowSize = m_sim->preferredWindowSize();
  m_window = glfwCreateWindow(preferredWindowSize.x, preferredWindowSize.y,
                              m_sim->getName().c_str(),
                              NULL,  // glfwGetPrimaryMonitor(),
                              NULL);
  if (!m_window) {
    EWAV_THROW("Null Window returned from glfwCreateWindow");
  }

  // Associate the window with the viewer in the global.
  g_glfwGlobal->registerWindowAndViewer(m_window, this);

  // Make context current.
  glfwMakeContextCurrent(m_window);

  // Init GLEW and other stuff.
  EncinoWaves::GeepGLFW::UtilGL::Init(true);
  init();

  // Register callbacks.
  glfwSetKeyCallback(m_window, g_keyboard);
  glfwSetCharCallback(m_window, g_character);
  glfwSetCursorPosCallback(m_window, g_mouseDrag);
  glfwSetMouseButtonCallback(m_window, g_mouse);
  glfwSetWindowSizeCallback(m_window, g_reshape);

  // Main loop.
  while (!glfwWindowShouldClose(m_window)) {
    tick(false);
    display();
    glFlush();
    glfwSwapBuffers(m_window);
    glfwPollEvents();

    glfwSetWindowTitle(m_window, m_sim->getName().c_str());
  }

  glfwDestroyWindow(m_window);
  g_glfwGlobal->unregisterWindowAndAllViewers(m_window);
  m_window = NULL;
}

//-*****************************************************************************
Viewer::~Viewer() {
  if (m_window) {
    glfwDestroyWindow(m_window);
    if (g_glfwGlobal) {
      g_glfwGlobal->unregisterWindowAndAllViewers(m_window);
    }
    m_window = NULL;
  }
}

//-*****************************************************************************
void Viewer::init(void) {
  UtilGL::CheckErrors("Viewer::init() begin");

  glClearColor(0.0, 0.0, 0.0, 0.0);
  UtilGL::CheckErrors("Viewer::init() glClearColor");

  glEnable(GL_DEPTH_TEST);
  UtilGL::CheckErrors("Viewer::init() glEnable GL_DEPTH_TEST");

  glEnable(GL_BLEND);
  UtilGL::CheckErrors("Viewer::init() glEnable GL_BLEND");
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  UtilGL::CheckErrors("Viewer::init() glBlendFunc");

  glDisable(GL_CULL_FACE);
  UtilGL::CheckErrors("Viewer::init() glDisable GL_CULL_FACE");

  m_buttonMask = 0;

  // m_camera.setSize( 800, 600 );
  // m_camera.lookAt( V3d( 24, 18, 24 ), V3d( 0.0 ) );
  // m_camera.frame( m_sim->getBounds() );
  V2i pws = m_sim->preferredWindowSize();
  m_sim->init(pws.x, pws.y);
}

//-*****************************************************************************
void Viewer::tick(bool i_force) {
  if (i_force || m_animating) {
    if (i_force || m_playbackTimer.elapsed() > 1.0 / 60.0) {
      m_playbackTimer.stop();
      m_playbackTimer.start();
      m_sim->step();
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
    }
  }
}

//-*****************************************************************************
void Viewer::display() {
  // #if OSX_GLFW_VIEWPORT_BUG
  //     glViewport( 0, 0,
  //                 2*( GLsizei )m_camera.width(),
  //                 2*( GLsizei )m_camera.height() );
  // #else
  //     glViewport( 0, 0,
  //                 ( GLsizei )m_camera.width(),
  //                 ( GLsizei )m_camera.height() );
  // #endif

  //     glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  //     UtilGL::CheckErrors( "Viewer::display() glClear" );

  //     double n, f;
  //     if ( m_sim->overrideClipping( m_camera, n, f ) )
  //     {
  //         m_camera.setClippingPlanes( n, f );
  //     }
  //     else
  //     {
  //         m_camera.autoSetClippingPlanes( m_sim->getBounds() );
  //    }
  m_sim->outerDraw();

  glFlush();
  UtilGL::CheckErrors("Viewer::display() glFlush");
}

//-*****************************************************************************
void Viewer::reshape(int i_width, int i_height) {
  m_sim->reshape(i_width, i_height);
  // m_camera.setSize( i_width, i_height );
  // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
}

//-*****************************************************************************
void Viewer::keyboard(int i_key, int i_scancode, int i_action, int i_mods) {
  // Shift
  if (i_mods & GLFW_MOD_SHIFT) {
    m_keyMods |= GLFW_MOD_SHIFT;
  } else {
    m_keyMods &= ~GLFW_MOD_SHIFT;
  }

  // Control
  if (i_mods & GLFW_MOD_CONTROL) {
    m_keyMods |= GLFW_MOD_CONTROL;
  } else {
    m_keyMods &= ~GLFW_MOD_CONTROL;
  }

  // Alt
  if (i_mods & GLFW_MOD_ALT) {
    m_keyMods |= GLFW_MOD_ALT;
  } else {
    m_keyMods &= ~GLFW_MOD_ALT;
  }

  // Super
  if (i_mods & GLFW_MOD_SUPER) {
    m_keyMods |= GLFW_MOD_SUPER;
  } else {
    m_keyMods &= ~GLFW_MOD_SUPER;
  }

  // std::cout << "Key hit: " << ( int )i_key << std::endl;
  if (i_key == GLFW_KEY_ESCAPE && i_action == GLFW_PRESS) {
    glfwSetWindowShouldClose(m_window, GL_TRUE);
    return;
  }

  double DBLx;
  double DBLy;
  glfwGetCursorPos(m_window, &DBLx, &DBLy);
  int x = (int)DBLx;
  int y = (int)DBLy;

  // m_sim->keyboard( m_camera, i_key, i_scancode, i_action, i_mods, x, y );
  m_sim->keyboard(i_key, i_scancode, i_action, i_mods, x, y);
}

//-*****************************************************************************
void Viewer::character(unsigned int i_char) {
  switch (i_char) {
  // case '`':
  // case '~':
  //    glutFullScreen();
  //    break;
  case 'f':
  case 'F':
    m_sim->frame();
    // m_camera.frame( m_sim->getBounds() );
    // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
    break;
  case ' ':
    tick(true);
    break;

  case '>':
  case '.':
    m_animating = !m_animating;
    break;
  case 'c':
  case 67:
    m_sim->outputCamera();
    // std::cout << "# Camera\n" << m_camera.RIB() << std::endl;
    break;
  default:
    break;
  }

  double DBLx;
  double DBLy;
  glfwGetCursorPos(m_window, &DBLx, &DBLy);
  int x = (int)DBLx;
  int y = (int)DBLy;
  // m_sim->character( m_camera, i_char, x, y );
  m_sim->character(i_char, x, y);
}

//-*****************************************************************************
void Viewer::mouse(int i_button, int i_action, int i_mods) {
  m_lastX = m_mouseX;
  m_lastY = m_mouseY;
  glfwGetCursorPos(m_window, &m_mouseX, &m_mouseY);

  // Shift
  if (i_mods & GLFW_MOD_SHIFT) {
    m_keyMods |= GLFW_MOD_SHIFT;
  } else {
    m_keyMods &= ~GLFW_MOD_SHIFT;
  }

  // Control
  if (i_mods & GLFW_MOD_CONTROL) {
    m_keyMods |= GLFW_MOD_CONTROL;
  } else {
    m_keyMods &= ~GLFW_MOD_CONTROL;
  }

  // Alt
  if (i_mods & GLFW_MOD_ALT) {
    m_keyMods |= GLFW_MOD_ALT;
  } else {
    m_keyMods &= ~GLFW_MOD_ALT;
  }

  // Super
  if (i_mods & GLFW_MOD_SUPER) {
    m_keyMods |= GLFW_MOD_SUPER;
  } else {
    m_keyMods &= ~GLFW_MOD_SUPER;
  }

  if (i_action == GLFW_PRESS) {
    switch (i_button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      m_buttonMask = m_buttonMask | BMASK_LEFT;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      m_buttonMask = m_buttonMask | BMASK_MIDDLE;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      m_buttonMask = m_buttonMask | BMASK_RIGHT;
      break;
    }
  } else {
    switch (i_button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      m_buttonMask = m_buttonMask & ~BMASK_LEFT;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      m_buttonMask = m_buttonMask & ~BMASK_MIDDLE;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      m_buttonMask = m_buttonMask & ~BMASK_RIGHT;
      break;
    }
  }
  m_sim->mouse(i_button, i_action, i_mods, m_mouseX, m_mouseY, m_lastX,
               m_lastY);
}

//-*****************************************************************************
void Viewer::mouseDrag(double x, double y) {
  m_lastX   = m_mouseX;
  m_lastY   = m_mouseY;
  m_mouseX  = x;
  m_mouseY  = y;
  double dx = m_mouseX - m_lastX;
  double dy = m_mouseY - m_lastY;

  if (m_keyMods & GLFW_MOD_ALT) {
    if (((m_buttonMask & BMASK_LEFT) && (m_buttonMask & BMASK_MIDDLE)) ||
        (m_buttonMask & BMASK_RIGHT)) {
      // m_camera.dolly( V2d( dx, dy ) );
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
      m_sim->dolly(dx, dy);
    } else if (m_buttonMask & BMASK_LEFT) {
      // m_camera.rotate( V2d( dx, dy ) );
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
      m_sim->rotate(dx, dy);
    } else if (m_buttonMask & BMASK_MIDDLE) {
      // m_camera.track( V2d( dx, dy ) );
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
      m_sim->track(dx, dy);
    }
  }
  // Some scroll-wheel mice refuse to give middle button signals,
  // so we allow for 'ctrl+left' for tracking.
  else if (m_keyMods & GLFW_MOD_CONTROL) {
    if (m_buttonMask & BMASK_LEFT) {
      // m_camera.track( V2d( dx, dy ) );
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
      m_sim->track(dx, dy);
    }
  }
  // Just to make mac usage easier... we'll use SHIFT for dolly also.
  else if (m_keyMods & GLFW_MOD_SHIFT) {
    if (m_buttonMask & BMASK_LEFT) {
      // m_camera.dolly( V2d( dx, dy ) );
      // m_camera.autoSetClippingPlanes( m_sim->getBounds() );
      m_sim->dolly(dx, dy);
    }
  } else {
    m_sim->mouseDrag(m_mouseX, m_mouseY, m_lastX, m_lastY);
  }
}

//-*****************************************************************************
//-*****************************************************************************
//-*****************************************************************************

void SimpleViewSim(std::shared_ptr<BaseSim> i_sim, bool i_playing) {
  Viewer v(i_sim, i_playing);
}

}  // namespace SimpleSimViewer
}  // namespace Emld
