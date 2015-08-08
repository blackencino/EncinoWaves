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

#include "UtilGL.h"

namespace EncinoWaves {
namespace GeepGLFW {
namespace UtilGL {

//-*****************************************************************************
void Init(bool i_experimental) {
  CheckErrors("GeepGLFW::init before anything");

// On Mac, GLEW stuff is not necessary.
#ifndef PLATFORM_DARWIN
  glewExperimental = i_experimental ? GL_TRUE : GL_FALSE;
  glewInit();
#endif
  // Reset errors.
  glGetError();

  printf("OPEN GL VERSION: %s\n", glGetString(GL_VERSION));

  CheckErrors("GeepGLFW::init glGetString");
}

//-*****************************************************************************
void CheckErrors(const std::string &i_label) {
  GLenum errCode;

#ifndef DEBUG
  EWAV_ASSERT((errCode = glGetError()) == GL_NO_ERROR,
              "OpenGL Error: "
                << "Code = " << (int)errCode << " ( Label: " << i_label
                << " )");

#else

  if ((errCode = glGetError()) != GL_NO_ERROR) {
    std::cerr << "OpenGL Error: "
              << "Code = " << (int)errCode << " ( Label: " << i_label << " )"
              << std::endl;
    abort();
  }

#endif
}

//-*****************************************************************************
void CheckFramebuffer() {
  GLenum status;
  status = (GLenum)glCheckFramebufferStatus(GL_FRAMEBUFFER);
  switch (status) {
  case GL_FRAMEBUFFER_COMPLETE:
    return;
  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    EWAV_THROW("Framebuffer incomplete, incomplete attachment");
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED:
    EWAV_THROW("Unsupported framebuffer format");
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    EWAV_THROW("Framebuffer incomplete, missing attachment");
    break;
  // case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
  //    EWAV_THROW(
  //        "Framebuffer incomplete, attached images "
  //        "must have same dimensions" );
  //    break;
  // case GL_FRAMEBUFFER_INCOMPLETE_FORMATS:
  //    EWAV_THROW(
  //        "Framebuffer incomplete, attached images "
  //        "must have same format" );
  //    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    EWAV_THROW("Framebuffer incomplete, missing draw buffer");
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    EWAV_THROW("Framebuffer incomplete, missing read buffer");
    break;
  }
  EWAV_THROW("Unknown GL Framebuffer error");
}

}  // namespace UtilGL
}  // namespace GeepGLFW
}  // namespace EncinoWaves
