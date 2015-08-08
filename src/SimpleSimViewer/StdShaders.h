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

#ifndef _EncinoWaves_SimpleSimViewer_StdShaders_h_
#define _EncinoWaves_SimpleSimViewer_StdShaders_h_

#include "Foundation.h"
#include "GLCamera.h"

namespace EncinoWaves {
namespace SimpleSimViewer {

//-*****************************************************************************
std::string StdShaderHeader();
std::string StdMatrices();
std::string StdTransformFunctions();
std::string StdSpecDiffuseGammaFunctions();
std::string SimpleVertexShader();
std::string SimplePointsGeometryShader();
std::string SimpleTrianglesGeometryShader();
std::string SimpleTrianglesWireframeGeometryShader();
std::string KeyFillFragmentShader();
std::string ConstantRedFragmentShader();
std::string ConstantWhiteFragmentShader();
void SetStdMatrices(Program& o_program, const GLCamera& i_cam,
                    const M44d& i_objectToWorld);
void SetKeyFillLights(Program& o_program, const V3f& i_toKey,
                      const V3f& i_keyColor, const V3f& i_toFill,
                      const V3f& i_fillColor);
void SetStdMaterial(Program& o_program, const V3f& i_diffColor,
                    const V3f& i_specColor, float i_specExponent);

}  // namespace SimpleSimViewer
}  // namespace EncinoWaves

#endif
