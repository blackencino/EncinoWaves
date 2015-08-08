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

#ifndef _EncinoWaves_GeepGLFW_Foundation_h_
#define _EncinoWaves_GeepGLFW_Foundation_h_

#define GLFW_INCLUDE_GLU 1

//-*****************************************************************************
// MAC INCLUDES
//-*****************************************************************************
#ifndef PLATFORM_DARWIN

#include <GL/glew.h>

#else

//#include <OpenGL/gl3.h>

#endif  // ifdef PLATFORM_DARWIN

#define GLFW_INCLUDE_GLCOREARB
#undef GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#ifdef PLATFORM_DARWIN
#include <OpenGL/glext.h>
#endif

#include <Util/All.h>

#include <vector>
#include <iostream>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace EncinoWaves {
namespace GeepGLFW {

//-*****************************************************************************
// NOTHING
//-*****************************************************************************

}  // namespace GeepGLFW
}  // namespace EncinoWaves

#endif
