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

#ifndef _EncinoWaves_GeepGLFW_UtilGL_h_
#define _EncinoWaves_GeepGLFW_UtilGL_h_

#include "Foundation.h"

namespace EncinoWaves {
namespace GeepGLFW {

//-*****************************************************************************
//! Notational grouping namespace for OpenGL-related functions
//
//! The functions declared in the UtilGL namespace are global and not part
//! of any particular class.
namespace UtilGL {

//-*****************************************************************************
//! OpenGL does not need extension initialization on Mac OSX, but does
//! require initialization via glewInit on non-mac systems. This function
//! abstracts that.
void Init(bool i_experimental = true);

//-*****************************************************************************
//! This function will throw an exception with an attached label if the OpenGL
//! Error flag is set. It will get the error string from OpenGL and attach that
//! to the exception text.
void CheckErrors(const std::string &i_label);

//-*****************************************************************************
//! Checks framebuffer status.
//! Copied directly out of the spec, modified to throw an exception
//! for any failed checks.
void CheckFramebuffer();

}  // namespace UtilGL
}  // namespace GeepGLFW
}  // namespace EncinoWaves

#endif  // ifndef _EncinoWaves_GeepGLFW_UtilGL_h_
